"""Integrator: TrueFrame reconstructor (section 9). Wires lanes B-E into the frozen protocol.

Flow: verify source hash -> bounded window -> confirmed track -> preflight gates -> align ->
color -> rank/cap -> fuse -> provenance -> revert-only validation (max 2 passes; second pass
ONLY reverts) -> integrity. Refusal is a successful ``REFUSED`` run. ``GENERATED_BLEND`` is
never written. Algorithm version is ``probity-tile-v1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probity.domain.enums import (
    AssetKind,
    InferenceMode,
    ObservationSource,
    PolicyOutcome,
    PolicyStage,
    ReasonCode,
    ReconstructionState,
)
from probity.domain.errors import ValidationFailed
from probity.domain.models import (
    AssetRef,
    Detection,
    FrameReference,
    PixelProvenance,
    ProvenanceSummary,
    ReconstructionRun,
    Track,
    TrackObservation,
)
from probity.domain.policy import PolicyConfig, default_policy
from probity.eval.synth import source_digest
from probity.eval.window import WindowBundle, load_tracker_inputs
from probity.ports import (
    CancelToken,
    ReconstructionRequest,
    ReconstructionResult,
    TrackRequest,
)
from probity.reconstruction.align import align, warp_donor
from probity.reconstruction.color import apply_color, fit_color
from probity.reconstruction.decisions import DecisionLog
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.fuse import (
    donor_count_gate,
    fuse_tiles,
    rank_donors,
    rank_score,
    temporal_preference,
    tile_gates,
)
from probity.reconstruction.integrity import IntegrityRefusal, compute_integrity, integrity_gate
from probity.reconstruction.io import encode_png, file_sha256, json_bytes, load_frame, sha256_bytes
from probity.reconstruction.preflight import run_preflight
from probity.reconstruction.provenance import (
    GeneratedPixelError,
    ProvenanceIncomplete,
    assert_supported_changes,
    build_provenance_arrays,
    build_source_lut,
    encode_provenance,
    provenance_record,
)
from probity.reconstruction import quality as q
from probity.reconstruction.tracking import (
    ReplayBridger,
    TrackedDetection,
    TrackMeta,
    confirm_track,
    sample_window,
    window_bounds,
)
from probity.reconstruction.types import (
    AlignedDonor,
    BBox,
    DonorMeasure,
    FrameLoader,
    Obs,
    ProvenanceArrays,
    SubjectMeasure,
)
from probity.reconstruction.validate import validate_and_revert

LANE = "integrator"
ALGORITHM_VERSION = "probity-tile-v1"
RUN_START = "2026-10-09T17:10:00.000000Z"
SYNTHETIC_NOTE = "Synthetic evaluation window; no real scene, vehicle, or plate."
REFUSAL_NOTE = "No defensible enhancement produced; the original frame is unchanged."
SUCCEEDED_UNCERTAINTY = (
    "Result may remain blurry where no compatible donor was clearer.",
    "Color and geometry transforms alter the appearance of borrowed pixels.",
)


class NeverCancelled:
    """Cancel token that never fires; tests and fixture runs use this unless they inject one."""

    def is_cancelled(self) -> bool:
        return False

    def checkpoint(self) -> None:
        return


@dataclass(frozen=True)
class TrueFrameArtifacts:
    """Protocol result plus the in-memory frame, arrays, and ranked donors for tests/eval."""

    result: ReconstructionResult
    result_frame: np.ndarray | None
    arrays: ProvenanceArrays | None
    donors: tuple[AlignedDonor, ...]
    png: bytes | None
    npz: bytes | None
    assets: tuple[AssetRef, ...]


def source_storage_uri(sha256: str) -> str:
    """Content-addressed source URI; synthetic windows have no MP4 but the contract still holds."""
    return f"source/sha256/{sha256[:2]}/{sha256}/original.mp4"


def _detections_by_frame(detections: Sequence[Detection]) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for det in detections:
        out.setdefault(det.frame_id, []).append(det)
    return out


def _obs_from_measure(measure: SubjectMeasure, track_id: str, cfg: PolicyConfig) -> Obs:
    image = measure.image
    height, width = image.shape[:2]
    box = measure.box
    expanded = q.expand_box(box, cfg.crop.alignment_context_expand, width, height)
    obs = measure.obs
    return Obs(
        frame_number=measure.frame.frame_number,
        pts_us=obs.pts_us,
        video_id=measure.frame.video_id,
        track_id=track_id,
        bbox_px=box,
        detector_backed=obs.source is ObservationSource.DETECTOR,
        bridged=obs.source is ObservationSource.CSRT_BRIDGE,
        confidence=obs.confidence,
        occluded_fraction=measure.occluded,
        crop=np.ascontiguousarray(q.crop(image, box).copy()),
        expanded_crop=np.ascontiguousarray(q.crop(image, expanded).copy()),
        expanded_box=expanded,
        frame_size=(width, height),
    )


class TrueFrame:
    """Frozen ``Reconstructor``: reconstruct one target crop from same-track evidentiary frames."""

    def __init__(
        self,
        *,
        cfg: PolicyConfig,
        frames: Sequence[FrameReference],
        detections: Sequence[Detection],
        load: FrameLoader,
        clock: FixedClock | None = None,
        mode: InferenceMode = InferenceMode.FIXTURE,
        tracker_ids: Mapping[str, int] | None = None,
        bridge_boxes: Mapping[int, BBox] | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.frames = tuple(frames)
        self.detections = tuple(detections)
        self.load = load
        self.clock = clock or FixedClock(RUN_START)
        self.mode = mode
        self.tracker_ids = dict(tracker_ids) if tracker_ids is not None else None
        self.bridge_boxes = dict(bridge_boxes) if bridge_boxes is not None else None
        self.source_path = source_path

    @classmethod
    def from_window(
        cls,
        window: WindowBundle,
        cfg: PolicyConfig | None = None,
        *,
        clock: FixedClock | None = None,
    ) -> TrueFrame:
        """Fixture constructor: loads tracker replay inputs and the window's frame resolver."""
        policy = cfg or default_policy()
        raw = load_tracker_inputs(window.root)
        bridge = {
            int(n): (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            for n, b in raw["bridge_boxes"].items()
        }
        images = {ref.frame_id: load_frame(ref, window.resolver) for ref in window.frames}
        return cls(
            cfg=policy,
            frames=window.frames,
            detections=window.detections,
            load=lambda ref: images[ref.frame_id],
            clock=clock or FixedClock(RUN_START),
            mode=InferenceMode.FIXTURE,
            tracker_ids={k: int(v) for k, v in raw["tracker_ids"].items()},
            bridge_boxes=bridge,
        )

    def reconstruct(
        self, request: ReconstructionRequest, cancel: CancelToken
    ) -> ReconstructionResult:
        return self.run(request, cancel).result

    def run(
        self,
        request: ReconstructionRequest,
        cancel: CancelToken,
        *,
        donor_candidates: Sequence[TrackObservation] | None = None,
    ) -> TrueFrameArtifacts:
        if request.config_sha256 != self.cfg.config_sha256:
            raise ValidationFailed(
                f"request config_sha256 {request.config_sha256} != policy {self.cfg.config_sha256}"
            )
        clock = self.clock
        log = DecisionLog(request.run_id, clock)
        started_at = clock.at(0)
        frames_by_id = {ref.frame_id: ref for ref in self.frames}

        def artifacts(
            run: ReconstructionRun,
            *,
            frame: np.ndarray | None = None,
            arrays: ProvenanceArrays | None = None,
            donors: tuple[AlignedDonor, ...] = (),
            png: bytes | None = None,
            npz: bytes | None = None,
            assets: tuple[AssetRef, ...] = (),
            provenance: PixelProvenance | None = None,
        ) -> TrueFrameArtifacts:
            return TrueFrameArtifacts(
                ReconstructionResult(run=run, decisions=log.rows, provenance=provenance),
                frame, arrays, donors, png, npz, assets,
            )

        def refused(
            reason: ReasonCode,
            iterations: int,
            accepted_ids: tuple[str, ...] = (),
            *,
            extra: tuple[str, ...] = (),
        ) -> TrueFrameArtifacts:
            notes = (REFUSAL_NOTE,) + extra
            if self.mode is InferenceMode.FIXTURE:
                notes = notes + (SYNTHETIC_NOTE,)
            run = ReconstructionRun.create(
                created_at=clock.at(len(log.rows) + 1),
                run_id=request.run_id,
                case_id=request.case_id,
                video_id=request.video_id,
                track_id=request.track.track_id,
                target_frame_id=request.target_frame_id,
                target_pts_us=frames_by_id[request.target_frame_id].pts_us,
                target_bbox_px=request.target_bbox_px,
                state=ReconstructionState.REFUSED,
                config_sha256=self.cfg.config_sha256,
                iteration_count=iterations,
                accepted_donor_frame_ids=accepted_ids,
                policy_decision_ids=log.ids,
                refusal_reasons=(reason,),
                uncertainty=notes,
                mode=self.mode,
                started_at=started_at,
                finished_at=clock.at(len(log.rows) + 1),
            )
            return artifacts(run)

        if self.mode is InferenceMode.FIXTURE:
            log.add(
                ReasonCode.FIXTURE_MODE_DISCLOSED, PolicyStage.MODE, "inputs",
                PolicyOutcome.WARN, "Synthetic fixture window; reconstruction is evidentiary",
                observed="FIXTURE", operator="==", threshold="FIXTURE",
            )

        cancel.checkpoint()
        mismatch = self._verify_source(request, log)
        if mismatch is not None:
            return refused(mismatch, 0)

        cancel.checkpoint()
        track = self._confirm_track(request, cancel)

        cancel.checkpoint()
        pre = run_preflight(
            log,
            track=track,
            video_id=request.video_id,
            case_id=request.case_id,
            target_frame_id=request.target_frame_id,
            target_box=request.target_bbox_px,
            frames=frames_by_id,
            load=self.load,
            detections=_detections_by_frame(self.detections),
            cfg=self.cfg,
            cancel=cancel,
            candidates=donor_candidates,
        )
        if pre.refusal is not None:
            return refused(pre.refusal, 0)
        assert pre.target is not None

        cancel.checkpoint()
        target_obs = _obs_from_measure(pre.target, track.track_id, self.cfg)
        target_frame = np.ascontiguousarray(pre.target.image.copy())
        compatible, donor_ids = self._align_and_color(
            target_obs, pre.donors, track.track_id, log, cancel
        )

        cancel.checkpoint()
        ranked = rank_donors(compatible, self.cfg)
        enough = log.check(donor_count_gate(ranked, self.cfg), "donors")
        accepted_ids = tuple(d.obs.frame_id for d in ranked)
        if not enough:
            return refused(ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS, 0, accepted_ids)

        cancel.checkpoint()
        fusion = fuse_tiles(target_frame, target_obs, ranked, self.cfg)
        fusion, iteration, _reverted = validate_and_revert(
            fusion, target_frame, target_obs.bbox_px, self.cfg
        )
        cancel.checkpoint()

        borrowed_tiles = [
            td for td in fusion.tile_decisions
            if td.reason_code is ReasonCode.BORROW_TILE_ACCEPTED
        ]
        bi = 0
        for subject, gate in tile_gates(fusion, self.cfg):
            log.check(gate, subject)
            if gate.rule_code is ReasonCode.BORROW_TILE_ACCEPTED:
                td = borrowed_tiles[bi]
                bi += 1
                donor_ids.setdefault(ranked[td.source_index - 1].obs.frame_number, []).append(
                    log.ids[-1]
                )
        if not borrowed_tiles:
            log.add(
                ReasonCode.NO_TILE_IMPROVED, PolicyStage.FUSE, "tiles", PolicyOutcome.REJECT,
                "No tile met the improvement and residual gates",
                observed=0, operator=">=", threshold=1, units="tiles",
                policy_key="fusion.min_sharpness_improvement",
            )
            return refused(ReasonCode.NO_TILE_IMPROVED, iteration, accepted_ids)

        cancel.checkpoint()
        try:
            lut = build_source_lut(target_obs, ranked, donor_ids)
            arrays = build_provenance_arrays(target_obs, fusion, ranked, lut)
            assert_supported_changes(fusion.result_frame, target_frame, arrays)
            log.gate(
                ReasonCode.PROVENANCE_COMPLETE, PolicyStage.PROVENANCE, request.run_id,
                observed=1.0, operator="==",
                threshold=self.cfg.integrity.required_provenance_coverage,
                units="fraction", policy_key="integrity.required_provenance_coverage",
                accept_reason="Every output pixel resolves",
                reject_reason="Provenance coverage is incomplete",
            )
            generated = 0.0
            log.gate(
                ReasonCode.GENERATED_SEMANTIC_PIXEL, PolicyStage.PROVENANCE, request.run_id,
                observed=generated, operator="<=",
                threshold=self.cfg.integrity.max_semantic_generated_fraction,
                units="fraction", policy_key="integrity.max_semantic_generated_fraction",
                accept_reason="No generated pixels",
                reject_reason="Generated semantic pixels are present",
            )
        except (ProvenanceIncomplete, GeneratedPixelError) as exc:
            code = getattr(exc, "reason_code", ReasonCode.PROVENANCE_INCOMPLETE)
            log.add(
                code, PolicyStage.PROVENANCE, request.run_id, PolicyOutcome.REJECT,
                getattr(exc, "message", str(exc)),
                observed="REJECTED", operator="==", threshold="COMPLETE",
            )
            return refused(code, iteration, accepted_ids)

        cancel.checkpoint()
        donor_map = {k: d for k, d in enumerate(ranked, start=1)}
        try:
            integrity = compute_integrity(
                arrays, target_obs.bbox_px, donor_map, track.continuity_score, log.rows
            )
        except IntegrityRefusal:
            log.add(
                ReasonCode.NO_TILE_IMPROVED, PolicyStage.INTEGRITY, request.run_id,
                PolicyOutcome.REJECT, "No borrowed pixels; refuse rather than invent Qd",
                observed=0, operator=">=", threshold=1, units="tiles",
                policy_key="fusion.min_sharpness_improvement",
            )
            return refused(ReasonCode.NO_TILE_IMPROVED, iteration, accepted_ids)
        except ProvenanceIncomplete as exc:
            log.add(
                ReasonCode.PROVENANCE_INCOMPLETE, PolicyStage.PROVENANCE, request.run_id,
                PolicyOutcome.REJECT, exc.message,
                observed="REJECTED", operator="==", threshold="COMPLETE",
            )
            return refused(ReasonCode.PROVENANCE_INCOMPLETE, iteration, accepted_ids)

        if not log.check(integrity_gate(integrity, self.cfg), request.run_id):
            return refused(ReasonCode.INTEGRITY_BELOW_MINIMUM, iteration, accepted_ids)

        png = encode_png(fusion.result_frame)
        npz = encode_provenance(arrays)
        npz_sha = sha256_bytes(npz)
        png_sha = sha256_bytes(png)
        finished = clock.at(len(log.rows) + 1)
        npz_uri = (
            f"derived/{request.video_id}/reconstruction/{request.run_id}/provenance.npz"
        )
        provenance = provenance_record(
            arrays, run_id=request.run_id, subject_box=target_obs.bbox_px,
            npz_uri=npz_uri, npz_sha256=npz_sha, created_at=finished,
        )
        assets = self._assets(request, clock, finished, png, npz, provenance)
        notes = SUCCEEDED_UNCERTAINTY
        if self.mode is InferenceMode.FIXTURE:
            notes = notes + (SYNTHETIC_NOTE,)
        run = ReconstructionRun.create(
            created_at=finished,
            run_id=request.run_id,
            case_id=request.case_id,
            video_id=request.video_id,
            track_id=track.track_id,
            target_frame_id=request.target_frame_id,
            target_pts_us=frames_by_id[request.target_frame_id].pts_us,
            target_bbox_px=request.target_bbox_px,
            state=ReconstructionState.SUCCEEDED,
            config_sha256=self.cfg.config_sha256,
            iteration_count=iteration,
            accepted_donor_frame_ids=accepted_ids,
            result_png_uri=f"asset://{assets[0].asset_id}",
            provenance_uri=f"asset://{assets[1].asset_id}",
            provenance=ProvenanceSummary(
                coverage_pct=provenance.coverage_pct,
                subject_coverage_pct=provenance.subject_coverage_pct,
                coverage_complete=True,
                source_lut=provenance.source_lut,
            ),
            result_png_sha256=png_sha,
            provenance_sha256=npz_sha,
            policy_decision_ids=log.ids,
            integrity=integrity,
            uncertainty=notes,
            mode=self.mode,
            started_at=started_at,
            finished_at=finished,
        )
        return artifacts(
            run, frame=fusion.result_frame, arrays=arrays, donors=ranked,
            png=png, npz=npz, assets=assets, provenance=provenance,
        )

    def _verify_source(
        self, request: ReconstructionRequest, log: DecisionLog
    ) -> ReasonCode | None:
        if self.source_path is not None:
            observed = file_sha256(self.source_path)
            if observed != request.source_sha256:
                log.add(
                    ReasonCode.SOURCE_HASH_MISMATCH, PolicyStage.INPUT, "source",
                    PolicyOutcome.REJECT, "Source file bytes do not match source_sha256",
                    observed=observed[:12], operator="==", threshold=request.source_sha256[:12],
                )
                return ReasonCode.SOURCE_HASH_MISMATCH
        hashes = [ref.pixel_sha256 or "" for ref in self.frames]
        digest = source_digest(hashes)
        if digest != request.source_sha256:
            log.add(
                ReasonCode.SOURCE_HASH_MISMATCH, PolicyStage.INPUT, "source",
                PolicyOutcome.REJECT, "Synthetic source digest did not match source_sha256",
                observed=digest[:12], operator="==", threshold=request.source_sha256[:12],
            )
            return ReasonCode.SOURCE_HASH_MISMATCH
        log.add(
            ReasonCode.SOURCE_HASH_VERIFIED, PolicyStage.INPUT, "source",
            PolicyOutcome.ACCEPT, "Frame pixel hashes re-verified; source digest matched",
            observed=digest[:12], operator="==", threshold=request.source_sha256[:12],
        )
        return None

    def _confirm_track(self, request: ReconstructionRequest, cancel: CancelToken) -> Track:
        """Bounded window then confirm. Falls back to the request track when no replay inputs."""
        cancel.checkpoint()
        target_ref = next(f for f in self.frames if f.frame_id == request.target_frame_id)
        track_req = TrackRequest(
            track_id=request.track.track_id,
            case_id=request.case_id,
            video_id=request.video_id,
            source_sha256=request.source_sha256,
            subject_type=request.track.subject_type,
            seed_frame_id=request.track.seed_frame_id,
            seed_bbox_px=request.track.seed_bbox_px,
            seed_detection_id=request.track.seed_detection_id,
        )
        start_us, end_us = window_bounds(target_ref.pts_us, track_req, self.cfg)
        detection_frames = {d.frame_id for d in self.detections}
        windowed = sample_window(
            self.frames, target_ref, start_us, end_us,
            self.cfg.detect.max_sample_fps, detection_frames,
        )
        if self.tracker_ids is None:
            return request.track
        windowed_ids = {ref.frame_id for ref in windowed}
        tracked = [
            TrackedDetection(d, self.tracker_ids.get(d.detection_id))
            for d in self.detections
            if d.frame_id in windowed_ids
        ]
        bridge = {
            n: box for n, box in (self.bridge_boxes or {}).items()
            if any(ref.frame_number == n for ref in windowed)
        }
        meta = TrackMeta(
            tracker_version=request.track.tracker_version,
            detector_model_id=request.track.detector_model_id,
            mode=self.mode,
            created_at=self.clock.at(0),
        )
        outcome = confirm_track(
            track_req, windowed, tracked, self.load, self.cfg, meta,
            (lambda: ReplayBridger(bridge)) if bridge else None,
        )
        return outcome.track

    def _align_and_color(
        self,
        target: Obs,
        donors: Sequence[DonorMeasure],
        track_id: str,
        log: DecisionLog,
        cancel: CancelToken,
    ) -> tuple[list[AlignedDonor], dict[int, list[str]]]:
        compatible: list[AlignedDonor] = []
        ids: dict[int, list[str]] = {}
        for measure in donors:
            cancel.checkpoint()
            donor_obs = _obs_from_measure(measure, track_id, self.cfg)
            start = len(log.rows)
            alignment = align(target, donor_obs, self.cfg)
            for gate in alignment.gates:
                log.check(gate, donor_obs.frame_id)
            if not alignment.accepted:
                continue
            warp = warp_donor(measure.image, alignment, target)
            color = fit_color(target, warp, self.cfg)
            color_result = log.apply(color.gates, donor_obs.frame_id)
            if not color.accepted or not color_result.accepted:
                continue
            colored = apply_color(warp.crop, color)
            T = temporal_preference(measure.dt_s, self.cfg)
            R = rank_score(measure.quality, alignment.A, T)
            compatible.append(
                AlignedDonor(
                    obs=donor_obs,
                    quality=measure.quality,
                    alignment=alignment,
                    color=color,
                    warped_crop=colored,
                    valid_mask=warp.valid_mask,
                    source_x=warp.source_x,
                    source_y=warp.source_y,
                    T=T,
                    R=R,
                )
            )
            ids[donor_obs.frame_number] = [
                *measure.decision_ids, *log.ids[start:],
            ]
        return compatible, ids

    def _assets(
        self,
        request: ReconstructionRequest,
        clock: FixedClock,
        finished: str,
        png: bytes,
        npz: bytes,
        provenance: PixelProvenance,
    ) -> tuple[AssetRef, ...]:
        lut_json = json_bytes(provenance)

        def asset(kind: AssetKind, name: str, media: str, data: bytes) -> AssetRef:
            return AssetRef(
                asset_id=seeded_uuid7(f"{request.run_id}:asset:{name}", clock.unix_ms + 900),
                case_id=request.case_id,
                kind=kind,
                storage_uri=(
                    f"derived/{request.video_id}/reconstruction/{request.run_id}/{name}"
                ),
                media_type=media,
                byte_length=len(data),
                sha256=sha256_bytes(data),
                non_evidentiary=False,
                created_at=finished,
            )

        return (
            asset(AssetKind.RESULT_PNG, "result.png", "image/png", png),
            asset(AssetKind.PROVENANCE_NPZ, "provenance.npz", "application/x-npz", npz),
            asset(AssetKind.PROVENANCE_LUT, "provenance.json", "application/json", lut_json),
        )


def request_from_window(
    window: WindowBundle,
    cfg: PolicyConfig,
    *,
    run_label: str = "trueframe",
    clock: FixedClock | None = None,
) -> ReconstructionRequest:
    """Build a protocol request for a committed synthetic window."""
    clock = clock or FixedClock(RUN_START)
    run_id = seeded_uuid7(f"{window.fixture_id}:{run_label}:run", clock.unix_ms)
    return ReconstructionRequest(
        run_id=run_id,
        case_id=window.case_id,
        video_id=window.video_id,
        source_sha256=window.source_sha256,
        source_storage_uri=source_storage_uri(window.source_sha256),
        track=window.track,
        target_frame_id=window.target_frame_id,
        target_bbox_px=window.target_bbox_px,
        config_sha256=cfg.config_sha256,
    )

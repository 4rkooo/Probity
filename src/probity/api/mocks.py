"""Fixture-backed mock job handlers for TRACK / RECONSTRUCT / REPORT.

Handlers load ``fixtures/contracts`` samples, re-identify IDs through
``Model.create`` (valid content hashes), and store via ``Repository``.
They never mutate the loaded fixture files. Mode is always FIXTURE.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from probity.config import Settings
from probity.domain.enums import (
    AssetKind,
    InferenceMode,
    JobKind,
    JobStage,
    JobState,
    ReasonCode,
    ReconstructionState,
    TrackState,
)
from probity.domain.ids import parse_frame_id, sha256_hex
from probity.domain.models import (
    AssetRef,
    EvidenceReport,
    PixelProvenance,
    PolicyDecision,
    ReconstructionRun,
    Track,
)
from probity.domain.policy import PolicyConfig
from probity.ports import (
    ClaimedJob,
    JobContext,
    JobHandler,
    JobOutcome,
    ReconstructPayload,
    ReportPayload,
    Repository,
    SourceStore,
    TrackPayload,
)

UUID7_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
PLATE_FRAME = 417
PLATE_BBOX = (216, 470, 292, 496)
CONTRACTS_REL = Path("fixtures/contracts")


def contracts_dir(fixture_root: Path) -> Path:
    return fixture_root.resolve().parent / "contracts"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_uuids(value: Any, found: set[str]) -> None:
    if isinstance(value, str):
        found.update(UUID7_RE.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_uuids(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_uuids(item, found)


def _rewrite(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            return mapping.get(token, token)

        return UUID7_RE.sub(repl, value)
    if isinstance(value, dict):
        return {key: _rewrite(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, mapping) for item in value]
    return value


def reidentify(
    payload: Any,
    *,
    new_id: Callable[..., str],
    forced: dict[str, str],
    mapping: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """Deep-copy ``payload``, remap every UUIDv7, drop Record hash/timestamp fields."""

    data = copy.deepcopy(payload)
    found: set[str] = set()
    _collect_uuids(data, found)
    table = dict(mapping or {})
    table.update(forced)
    for token in found:
        table.setdefault(token, new_id())
    rewritten = _rewrite(data, table)
    return _strip_record_meta(rewritten), table


def _strip_record_meta(value: Any) -> Any:
    if isinstance(value, dict):
        out = {
            key: _strip_record_meta(item)
            for key, item in value.items()
            if key not in {"content_sha256", "created_at", "schema_version"}
        }
        return out
    if isinstance(value, list):
        return [_strip_record_meta(item) for item in value]
    return value


def _bbox_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def seed_confirms_plate(seed_frame_id: str, seed_bbox_px: tuple[int, int, int, int]) -> bool:
    _, frame_number = parse_frame_id(seed_frame_id)
    return frame_number == PLATE_FRAME and _bbox_overlap(seed_bbox_px, PLATE_BBOX)


class TrackMockHandler:
    """Load confirmed vs not-confirmed track fixtures from seed overlap with f417 plate."""

    kind = JobKind.TRACK

    def __init__(
        self,
        *,
        repository: Repository,
        fixture_root: Path,
        clock: Callable[[], str],
        new_id: Callable[..., str],
    ) -> None:
        self.repository = repository
        self.fixture_root = fixture_root
        self.clock = clock
        self.new_id = new_id

    async def run(self, claimed: ClaimedJob, ctx: JobContext) -> JobOutcome:
        assert isinstance(claimed.payload, TrackPayload)
        req = claimed.payload.request
        ctx.progress(JobStage.TRACK, 0, 1)
        confirmed = seed_confirms_plate(req.seed_frame_id, req.seed_bbox_px)
        name = "track_confirmed.json" if confirmed else "track_not_confirmed.json"
        raw = load_json(contracts_dir(self.fixture_root) / name)
        data, _mapping = reidentify(
            raw,
            new_id=self.new_id,
            forced={
                raw["track_id"]: req.track_id,
                raw["case_id"]: req.case_id,
                raw["video_id"]: req.video_id,
            },
        )
        data["track_id"] = req.track_id
        data["case_id"] = req.case_id
        data["video_id"] = req.video_id
        data["seed_frame_id"] = req.seed_frame_id
        data["seed_bbox_px"] = list(req.seed_bbox_px)
        data["seed_detection_id"] = req.seed_detection_id
        data["subject_type"] = req.subject_type.value
        data["mode"] = InferenceMode.FIXTURE.value
        track = Track.create(**data)
        self.repository.put_track(track)
        ctx.progress(JobStage.TRACK, 1, 1)
        if track.state is TrackState.CONFIRMED:
            return JobOutcome(state=JobState.SUCCEEDED)
        return JobOutcome(
            state=JobState.REFUSED,
            refusal_reasons=track.reason_codes or (ReasonCode.TRACK_NOT_CONFIRMED,),
        )


class ReconstructMockHandler:
    """Target frame 417 succeeds from the golden fixture; any other target is refused."""

    kind = JobKind.RECONSTRUCT

    def __init__(
        self,
        *,
        repository: Repository,
        fixture_root: Path,
        clock: Callable[[], str],
        new_id: Callable[..., str],
        policy: PolicyConfig,
    ) -> None:
        self.repository = repository
        self.fixture_root = fixture_root
        self.clock = clock
        self.new_id = new_id
        self.policy = policy

    async def run(self, claimed: ClaimedJob, ctx: JobContext) -> JobOutcome:
        assert isinstance(claimed.payload, ReconstructPayload)
        req = claimed.payload.request
        ctx.progress(JobStage.ALIGN, 0, 1)
        _, frame_number = parse_frame_id(req.target_frame_id)
        succeeded = frame_number == PLATE_FRAME
        contracts = contracts_dir(self.fixture_root)
        run_name = (
            "reconstruction_run_succeeded.json" if succeeded else "reconstruction_run_refused.json"
        )
        decisions_name = (
            "policy_decisions_succeeded.json" if succeeded else "policy_decisions_refused.json"
        )
        raw_run = load_json(contracts / run_name)
        raw_decisions = load_json(contracts / decisions_name)
        forced = {
            raw_run["run_id"]: req.run_id,
            raw_run["case_id"]: req.case_id,
            raw_run["video_id"]: req.video_id,
            raw_run["track_id"]: req.track.track_id,
        }
        run_data, mapping = reidentify(raw_run, new_id=self.new_id, forced=forced)
        run_data["run_id"] = req.run_id
        run_data["case_id"] = req.case_id
        run_data["video_id"] = req.video_id
        run_data["track_id"] = req.track.track_id
        run_data["target_frame_id"] = req.target_frame_id
        run_data["target_bbox_px"] = list(req.target_bbox_px)
        run_data["config_sha256"] = req.config_sha256
        run_data["mode"] = InferenceMode.FIXTURE.value
        if succeeded:
            result_asset_id = self.new_id()
            provenance_asset_id = self.new_id()
            run_data["result_png_uri"] = f"asset://{result_asset_id}"
            run_data["provenance_uri"] = f"asset://{provenance_asset_id}"
            raw_prov = load_json(contracts / "pixel_provenance.json")
            prov_data, mapping = reidentify(
                raw_prov, new_id=self.new_id, forced=forced, mapping=mapping
            )
            prov_data["run_id"] = req.run_id
            provenance = PixelProvenance.create(**prov_data)
            self.repository.put_provenance(provenance)
            result_path = contracts / "artifacts" / "result.png"
            npz_path = contracts / "artifacts" / "provenance.npz"
            now = self.clock()
            self.repository.put_asset(
                AssetRef(
                    asset_id=result_asset_id,
                    case_id=req.case_id,
                    kind=AssetKind.RESULT_PNG,
                    storage_uri="fixtures/contracts/artifacts/result.png",
                    media_type="image/png",
                    byte_length=result_path.stat().st_size,
                    sha256=sha256_hex(result_path.read_bytes()),
                    non_evidentiary=False,
                    created_at=now,
                )
            )
            self.repository.put_asset(
                AssetRef(
                    asset_id=provenance_asset_id,
                    case_id=req.case_id,
                    kind=AssetKind.PROVENANCE_NPZ,
                    storage_uri="fixtures/contracts/artifacts/provenance.npz",
                    media_type="application/octet-stream",
                    byte_length=npz_path.stat().st_size,
                    sha256=sha256_hex(npz_path.read_bytes()),
                    non_evidentiary=False,
                    created_at=now,
                )
            )
        run = ReconstructionRun.create(**run_data)
        self.repository.put_run(run)
        decisions = []
        for raw in raw_decisions:
            item, mapping = reidentify(raw, new_id=self.new_id, forced=forced, mapping=mapping)
            item["run_id"] = req.run_id
            decisions.append(PolicyDecision.create(**item))
        self.repository.put_decisions(decisions)
        ctx.progress(JobStage.FUSE, 1, 1)
        if run.state is ReconstructionState.SUCCEEDED:
            return JobOutcome(state=JobState.SUCCEEDED)
        return JobOutcome(
            state=JobState.REFUSED,
            refusal_reasons=run.refusal_reasons or (ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS,),
        )


class ReportMockHandler:
    """Materialize the contract evidence report against the approved run."""

    kind = JobKind.REPORT

    def __init__(
        self,
        *,
        repository: Repository,
        source_store: SourceStore,
        fixture_root: Path,
        clock: Callable[[], str],
        new_id: Callable[..., str],
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.source_store = source_store
        self.fixture_root = fixture_root
        self.clock = clock
        self.new_id = new_id
        self.settings = settings

    async def run(self, claimed: ClaimedJob, ctx: JobContext) -> JobOutcome:
        assert isinstance(claimed.payload, ReportPayload)
        payload = claimed.payload
        ctx.progress(JobStage.REPORT, 0, 1)
        run = self.repository.get_run(payload.run_id)
        review = self.repository.latest_review(payload.run_id)
        video = self.repository.get_video(run.video_id)
        raw = load_json(contracts_dir(self.fixture_root) / "evidence_report.json")
        html_id = self.new_id()
        json_id = self.new_id()
        data, _mapping = reidentify(
            raw,
            new_id=self.new_id,
            forced={
                raw["report_id"]: payload.report_id,
                raw["case_id"]: run.case_id,
                raw["run_id"]: run.run_id,
                raw["review_id"]: payload.review_id,
            },
        )
        data["report_id"] = payload.report_id
        data["case_id"] = run.case_id
        data["run_id"] = run.run_id
        data["review_id"] = payload.review_id
        data["source_sha256"] = video.sha256
        data["source_sha256_verified_at_export"] = video.sha256
        data["source_verified"] = True
        data["html_uri"] = f"asset://{html_id}"
        data["json_uri"] = f"asset://{json_id}"
        data["generated_at"] = self.clock()
        data["mode"] = InferenceMode.FIXTURE.value
        html_rel = f"derived/{run.video_id}/report/{payload.report_id}/report.html"
        json_rel = f"derived/{run.video_id}/report/{payload.report_id}/report.json"
        html_path = self.settings.data_dir / html_rel
        json_path = self.settings.data_dir / json_rel
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_bytes = b"<html><body>Probity evidence report</body></html>"
        json_bytes = b'{"report":"probity-demo"}'
        html_path.write_bytes(html_bytes)
        json_path.write_bytes(json_bytes)
        now = self.clock()
        self.repository.put_asset(
            AssetRef(
                asset_id=html_id,
                case_id=run.case_id,
                kind=AssetKind.REPORT_HTML,
                storage_uri=html_rel,
                media_type="text/html",
                byte_length=len(html_bytes),
                sha256=sha256_hex(html_bytes),
                non_evidentiary=False,
                created_at=now,
            )
        )
        self.repository.put_asset(
            AssetRef(
                asset_id=json_id,
                case_id=run.case_id,
                kind=AssetKind.REPORT_JSON,
                storage_uri=json_rel,
                media_type="application/json",
                byte_length=len(json_bytes),
                sha256=sha256_hex(json_bytes),
                non_evidentiary=False,
                created_at=now,
            )
        )
        report = EvidenceReport.create(**data)
        self.repository.put_report(report)
        _ = review
        ctx.progress(JobStage.REPORT, 1, 1)
        return JobOutcome(state=JobState.SUCCEEDED)


def mock_handlers(
    *,
    repository: Repository,
    source_store: SourceStore,
    fixture_root: Path,
    clock: Callable[[], str],
    new_id: Callable[..., str],
    policy: PolicyConfig,
    settings: Settings,
) -> dict[JobKind, JobHandler]:
    return {
        JobKind.TRACK: TrackMockHandler(
            repository=repository, fixture_root=fixture_root, clock=clock, new_id=new_id
        ),
        JobKind.RECONSTRUCT: ReconstructMockHandler(
            repository=repository,
            fixture_root=fixture_root,
            clock=clock,
            new_id=new_id,
            policy=policy,
        ),
        JobKind.REPORT: ReportMockHandler(
            repository=repository,
            source_store=source_store,
            fixture_root=fixture_root,
            clock=clock,
            new_id=new_id,
            settings=settings,
        ),
    }

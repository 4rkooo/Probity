"""Hand-built lane D inputs: seeded frames, translated donors, a fusion result, a decision log.

Nothing here calls lane B or C code. Donor warps follow the ``types.py`` convention directly:
``cv2.remap(donor, source_x, source_y, INTER_LANCZOS4, BORDER_REFLECT_101)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from probity.domain.enums import AlignmentMethod, PolicyOutcome, PolicyStage, ReasonCode
from probity.domain.ids import frame_id
from probity.domain.models import PolicyDecision
from probity.domain.policy import PolicyConfig
from probity.reconstruction import quality as q
from probity.reconstruction.decisions import DecisionLog
from probity.reconstruction.determinism import FixedClock, seeded_uuid7
from probity.reconstruction.types import (
    AlignedDonor,
    Alignment,
    BBox,
    ColorFit,
    FusionResult,
    Obs,
    QualityScores,
    TileDecision,
)

UNIX_MS = 1_760_000_000_000
VIDEO = seeded_uuid7("lane-d:video", UNIX_MS)
OTHER_VIDEO = seeded_uuid7("lane-d:other-video", UNIX_MS)
TRACK = seeded_uuid7("lane-d:track", UNIX_MS)
OTHER_TRACK = seeded_uuid7("lane-d:other-track", UNIX_MS)
RUN = seeded_uuid7("lane-d:run", UNIX_MS)
START = "2026-10-09T17:10:00.000000Z"
W, H = 96, 64
SUBJECT: BBox = (24, 20, 72, 44)
TARGET_N = 30
FPS_US = 66_667


def pts(frame_number: int) -> int:
    return frame_number * FPS_US


def scene(rng: np.random.Generator, pad: int = 12) -> np.ndarray:
    """A textured scene larger than the frame so translated views stay meaningful."""
    return rng.integers(0, 256, size=(H + 2 * pad, W + 2 * pad, 3), dtype=np.uint8)


def view(world: np.ndarray, dx: int, dy: int, pad: int = 12) -> np.ndarray:
    """Frame whose pixel (x, y) is world (x + pad + dx, y + pad + dy)."""
    return np.ascontiguousarray(world[pad + dy:pad + dy + H, pad + dx:pad + dx + W])


def make_obs(frame: np.ndarray, box: BBox, frame_number: int, cfg: PolicyConfig, *,
             video_id: str = VIDEO, track_id: str = TRACK, detector_backed: bool = True,
             bridged: bool = False) -> Obs:
    h, w = frame.shape[:2]
    expanded = q.expand_box(box, cfg.crop.alignment_context_expand, w, h)
    return Obs(frame_number=frame_number, pts_us=pts(frame_number), video_id=video_id,
               track_id=track_id, bbox_px=box, detector_backed=detector_backed, bridged=bridged,
               confidence=0.9, occluded_fraction=0.0, crop=q.crop(frame, box).copy(),
               expanded_crop=q.crop(frame, expanded).copy(), expanded_box=expanded,
               frame_size=(w, h))


def quality(Q: float) -> QualityScores:  # noqa: N803 - section 9 symbol
    return QualityScores(D=0.9, S=0.8, Z=0.5, E=1.0, O=1.0, P=1.0, Q=Q)


def translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_donor(target: Obs, frame: np.ndarray, frame_number: int, cfg: PolicyConfig, *,
               tx: float, ty: float, R: float, Q: float = 0.8, A: float = 0.9,  # noqa: N803
               method: AlignmentMethod = AlignmentMethod.AKAZE_HOMOGRAPHY,
               matrix: np.ndarray | None = None, video_id: str = VIDEO, track_id: str = TRACK,
               detector_backed: bool = True, bridged: bool = False,
               gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
               bias: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> AlignedDonor:
    """A donor related to the target by ``target = donor + (tx, ty)`` (donor->target)."""
    x1, y1, x2, y2 = target.bbox_px
    box = (round(x1 - tx), round(y1 - ty), round(x2 - tx), round(y2 - ty))
    obs = make_obs(frame, box, frame_number, cfg, video_id=video_id, track_id=track_id,
                   detector_backed=detector_backed, bridged=bridged)
    ex1, ey1, ex2, ey2 = target.expanded_box
    xs, ys = np.meshgrid(np.arange(ex1, ex2, dtype=np.float32),
                         np.arange(ey1, ey2, dtype=np.float32))
    source_x = (xs - np.float32(tx)).astype(np.float32)
    source_y = (ys - np.float32(ty)).astype(np.float32)
    h, w = frame.shape[:2]
    valid = (source_x >= 0) & (source_x <= w - 1) & (source_y >= 0) & (source_y <= h - 1)
    warped = cv2.remap(frame, source_x, source_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT_101)
    reason = (ReasonCode.ALIGNMENT_AKAZE_ACCEPTED if method is AlignmentMethod.AKAZE_HOMOGRAPHY
              else ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED)
    alignment = Alignment(method=method,
                          matrix=translation(tx, ty) if matrix is None else matrix,
                          inlier_ratio=1.0, median_reproj_px=0.0,
                          valid_coverage=float(valid.mean()), corner_outside_fraction=0.0, A=A,
                          accepted=True, reason=reason)
    color = ColorFit(gain=gain, bias=bias, mean_abs_residual=1.0, samples=500, accepted=True,
                     reason=ReasonCode.PHOTOMETRIC_INCOMPATIBLE)
    return AlignedDonor(obs=obs, quality=quality(Q), alignment=alignment, color=color,
                        warped_crop=warped, valid_mask=valid, source_x=source_x,
                        source_y=source_y, T=1.0, R=R)


def make_fusion(target_frame: np.ndarray, target: Obs, donors: Sequence[AlignedDonor],
                borrow: Mapping[int, int], cfg: PolicyConfig) -> FusionResult:
    """``borrow`` maps tile index -> LUT row (>= 1); every other tile keeps the original."""
    result = target_frame.copy()
    ex1, ey1 = target.expanded_box[:2]
    decisions = []
    for i, (x1, y1, x2, y2) in enumerate(q.tiles(target.bbox_px, cfg.fusion.tile_px)):
        k = borrow.get(i, 0)
        if k:
            d = donors[k - 1]
            result[y1:y2, x1:x2] = d.warped_crop[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1]
            decisions.append(TileDecision(i, (x1, y1, x2, y2), k, 1.0,
                                          ReasonCode.BORROW_TILE_ACCEPTED, 0.5, 2.0))
        else:
            decisions.append(TileDecision(i, (x1, y1, x2, y2), 0, 0.0,
                                          ReasonCode.KEEP_ORIGINAL_CLEAR))
    return FusionResult(result_frame=result, tile_decisions=tuple(decisions),
                        donor_lut_order=tuple(d.obs.frame_number for d in donors))


@dataclass
class Scenario:
    world: np.ndarray
    target_frame: np.ndarray
    target: Obs
    donors: tuple[AlignedDonor, ...]


DONOR_SHIFTS = ((26, 3.0, 1.0, 0.9), (34, -2.0, 0.5, 0.8), (38, 4.5, -1.5, 0.7))
# Two 8x8 tiles from LUT row 1, one from row 2; the rest stay ORIGINAL.
DEFAULT_BORROW: dict[int, int] = {0: 1, 1: 1, 6: 2}


def scenario(rng: np.random.Generator, cfg: PolicyConfig) -> Scenario:
    """Target frame 30 plus three same-track donors in LUT order (R descending)."""
    world = scene(rng)
    target_frame = view(world, 0, 0)
    target = make_obs(target_frame, SUBJECT, TARGET_N, cfg)
    donors = []
    for n, tx, ty, r in DONOR_SHIFTS:
        frame = view(world, round(tx), round(ty))
        donors.append(make_donor(target, frame, n, cfg, tx=tx, ty=ty, R=r, Q=0.6 + r / 4))
    return Scenario(world, target_frame, target, tuple(donors))


# ------------------------------------------------------------------------------------------------
# Decision log
# ------------------------------------------------------------------------------------------------


def _accept(log: DecisionLog, rule: ReasonCode, stage: PolicyStage, ref: str) -> str:
    log.add(rule, stage, ref, PolicyOutcome.ACCEPT, f"{rule} passed", observed=1.0,
            operator=">=", threshold=0.0, units="fraction")
    return log.ids[-1]


def decision_log(target: Obs, donors: Sequence[AlignedDonor], fusion: FusionResult,
                 *, run_id: str = RUN, provenance_rows: bool = True
                 ) -> tuple[tuple[PolicyDecision, ...], dict[int, list[str]]]:
    """Every material decision a completed run records, with per-donor decision ids."""
    log = DecisionLog(run_id, FixedClock(START))
    ids: dict[int, list[str]] = {d.obs.frame_number: [] for d in donors}
    _accept(log, ReasonCode.SOURCE_HASH_VERIFIED, PolicyStage.INPUT, "source")
    _accept(log, ReasonCode.TRACK_CONFIRMED, PolicyStage.TRACK, TRACK)
    _accept(log, ReasonCode.TARGET_TOO_SMALL, PolicyStage.QUALITY, target.frame_id)
    _accept(log, ReasonCode.TARGET_QUALITY_TOO_LOW, PolicyStage.QUALITY, target.frame_id)
    for d in donors:
        ref, n = d.obs.frame_id, d.obs.frame_number
        ids[n].append(_accept(log, ReasonCode.DONOR_TRACK_MISMATCH, PolicyStage.PREFLIGHT, ref))
        ids[n].append(_accept(log, ReasonCode.DONOR_NOT_CLEARER, PolicyStage.PREFLIGHT, ref))
        ids[n].append(_accept(log, d.alignment.reason, PolicyStage.ALIGN, ref))
        ids[n].append(_accept(log, ReasonCode.PHOTOMETRIC_INCOMPATIBLE, PolicyStage.COLOR, ref))
    _accept(log, ReasonCode.INSUFFICIENT_COMPATIBLE_DONORS, PolicyStage.RANK, "donors")
    for t in fusion.tile_decisions:
        if t.reason_code is ReasonCode.BORROW_TILE_ACCEPTED:
            n = donors[t.source_index - 1].obs.frame_number
            ids[n].append(_accept(log, ReasonCode.BORROW_TILE_ACCEPTED, PolicyStage.FUSE,
                                  f"tile:{t.box[0]},{t.box[1]}"))
    _accept(log, ReasonCode.KEEP_ORIGINAL_CLEAR, PolicyStage.FUSE, "tiles")
    if provenance_rows:
        _accept(log, ReasonCode.PROVENANCE_COMPLETE, PolicyStage.PROVENANCE, run_id)
        _accept(log, ReasonCode.GENERATED_SEMANTIC_PIXEL, PolicyStage.PROVENANCE, run_id)
    return log.rows, ids


def npz_uri(*, run_id: str = RUN, video_id: str = VIDEO) -> str:
    return f"derived/{video_id}/reconstruction/{run_id}/provenance.npz"


def donor_map(donors: Sequence[AlignedDonor]) -> dict[int, AlignedDonor]:
    return {k: d for k, d in enumerate(donors, start=1)}


@dataclass
class Built:
    target_frame: np.ndarray
    target: Obs
    donors: tuple[AlignedDonor, ...]
    fusion: FusionResult
    decisions: tuple[PolicyDecision, ...]
    ids: dict[int, list[str]]


def build_scene(rng: np.random.Generator, cfg: PolicyConfig, *,
                borrow: Mapping[int, int] | None = None,
                provenance_rows: bool = True) -> Built:
    sc = scenario(rng, cfg)
    fusion = make_fusion(sc.target_frame, sc.target, sc.donors,
                         DEFAULT_BORROW if borrow is None else borrow, cfg)
    decisions, ids = decision_log(sc.target, sc.donors, fusion, provenance_rows=provenance_rows)
    return Built(sc.target_frame, sc.target, sc.donors, fusion, decisions, ids)


def single_donor(rng: np.random.Generator, cfg: PolicyConfig, *, Q: float, A: float,  # noqa: N803
                 method: AlignmentMethod = AlignmentMethod.AKAZE_HOMOGRAPHY) -> Built:
    world = scene(rng)
    target_frame = view(world, 0, 0)
    target = make_obs(target_frame, SUBJECT, TARGET_N, cfg)
    frame = view(world, 3, 1)
    donor = make_donor(target, frame, 26, cfg, tx=3.0, ty=1.0, R=0.9, Q=Q, A=A, method=method)
    fusion = make_fusion(target_frame, target, (donor,), {0: 1}, cfg)
    decisions, ids = decision_log(target, (donor,), fusion)
    return Built(target_frame, target, (donor,), fusion, decisions, ids)

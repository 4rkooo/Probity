"""Lane B: donor-to-target alignment (section 9, algorithm step 6). PUBLIC SIGNATURES FROZEN.

Primary: grayscale + CLAHE (feature detection only) on the expanded crops, AKAZE descriptors,
Hamming ratio test, RANSAC homography donor->target. Fallback: exactly one ECC affine attempt,
initialized from box geometry, and only when AKAZE failed for too few keypoints or matches.
Never retry, never pick by appearance. Determinism: ``cv2.setRNGSeed`` before
``findHomography``, fixed RANSAC iterations, reprojection threshold from policy.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from probity.domain.enums import AlignmentMethod, PolicyStage, ReasonCode
from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import Alignment, BBox, Gate, Obs, Operator, Warp

LANE = "lane B"
A_WEIGHTS = {"inlier": 0.40, "reprojection": 0.30, "coverage": 0.30}
# Section 9 formula constant of A (``clip(reprojection_error / 2)``), not a policy gate.
A_REPROJECTION_NORMALIZER_PX = 2.0

ALIGN = PolicyStage.ALIGN
AKAZE_OK = ReasonCode.ALIGNMENT_AKAZE_ACCEPTED
ECC_OK = ReasonCode.ALIGNMENT_ECC_FALLBACK_ACCEPTED
FAILED = ReasonCode.ALIGNMENT_FAILED
AKAZE_METHOD = AlignmentMethod.AKAZE_HOMOGRAPHY
ECC_METHOD = AlignmentMethod.ECC_AFFINE

# Policy keys requested in NOTES-person2.md but not on this branch. getattr/defaults only.
DEFAULT_RANSAC_MAX_ITERS = 2000
DEFAULT_RANSAC_CONFIDENCE = 0.995
DEFAULT_RANSAC_RNG_SEED = 20261009
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID = 4
DEFAULT_AKAZE_DETECTOR_THRESHOLD = 0.001
DEFAULT_ECC_GAUSS_FILT_SIZE = 5
DEFAULT_SCALE_RATIO_MIN = 0.67
DEFAULT_SCALE_RATIO_MAX = 1.50
FEATURE_STARVED_KEYS = frozenset({"akaze.min_keypoints", "akaze.min_good_matches"})
# Finite stand-in when every projected inlier is non-finite (inf would break A).
UNPROJECTABLE_REPROJ_PX = 1.0e9


# ------------------------------------------------------------------------------------------------
# Confidence and geometry measures
# ------------------------------------------------------------------------------------------------


def _unit(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1], got {value!r}")
    return value


def alignment_confidence(
    inlier_ratio: float, median_reproj_px: float, valid_coverage: float
) -> float:
    """A = 0.40 * inlier_ratio + 0.30 * (1 - clip(median_reproj_px / 2)) + 0.30 * coverage."""
    if not math.isfinite(median_reproj_px) or median_reproj_px < 0.0:
        raise ValueError(f"median_reproj_px must be finite and >= 0, got {median_reproj_px!r}")
    reproj = min(1.0, median_reproj_px / A_REPROJECTION_NORMALIZER_PX)
    return (
        A_WEIGHTS["inlier"] * _unit("inlier_ratio", inlier_ratio)
        + A_WEIGHTS["reprojection"] * (1.0 - reproj)
        + A_WEIGHTS["coverage"] * _unit("valid_coverage", valid_coverage)
    )


def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError("alignment matrix must be a finite float64 3x3")
    if abs(float(np.linalg.det(m))) < 1e-12:
        raise ValueError("alignment matrix is singular")
    return m


def _inverse(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Target full-frame (x, y) -> donor full-frame (sx, sy) and an in-front-of-camera mask.

    The inverse is sign-normalized so its homogeneous ``w`` is positive at the centre of the
    sampled points; points with ``w <= 0`` lie beyond the horizon and are never valid.
    """
    inv = np.linalg.inv(_check_matrix(matrix))
    w = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2]
    cx, cy = (float(x.min()) + float(x.max())) / 2.0, (float(y.min()) + float(y.max())) / 2.0
    if inv[2, 0] * cx + inv[2, 1] * cy + inv[2, 2] < 0.0:
        inv, w = -inv, -w
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sx = (inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]) / w
        sy = (inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]) / w
    front = (w > 0.0) & np.isfinite(sx) & np.isfinite(sy)
    return sx, sy, front


def _inside(sx: np.ndarray, sy: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    return (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)


def source_maps(
    matrix: np.ndarray, target_box: BBox, donor_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(source_x, source_y, valid_mask)`` for every pixel of ``target_box`` (target-crop frame).

    ``source_x/source_y`` are float32 donor full-frame coordinates of the inverse transform;
    ``valid_mask`` is True where those float32 values lie in ``[0, W-1] x [0, H-1]`` of the donor
    frame ``donor_size = (W, H)``. Points beyond the horizon or non-finite get ``-1`` and are
    invalid.
    """
    x1, y1, x2, y2 = target_box
    xs, ys = np.meshgrid(
        np.arange(x1, x2, dtype=np.float64), np.arange(y1, y2, dtype=np.float64)
    )
    sx64, sy64, front = _inverse(matrix, xs, ys)
    sx = np.where(front, sx64, -1.0).astype(np.float32)
    sy = np.where(front, sy64, -1.0).astype(np.float32)
    valid = front & _inside(sx, sy, donor_size)
    return np.ascontiguousarray(sx), np.ascontiguousarray(sy), valid


def valid_coverage(matrix: np.ndarray, target: Obs, donor_size: tuple[int, int]) -> float:
    """Fraction of the target expanded box whose source coordinates fall inside the donor frame."""
    return float(source_maps(matrix, target.expanded_box, donor_size)[2].mean())


def corner_outside_fraction(
    matrix: np.ndarray, target: Obs, donor_size: tuple[int, int]
) -> float:
    """Fraction of the four target-expanded-box corner pixels that map outside the donor frame."""
    x1, y1, x2, y2 = target.expanded_box
    xs = np.array([x1, x2 - 1, x2 - 1, x1], dtype=np.float64)
    ys = np.array([y1, y1, y2 - 1, y2 - 1], dtype=np.float64)
    sx, sy, front = _inverse(matrix, xs, ys)
    inside = front & _inside(sx, sy, donor_size)
    return float((~inside).sum()) / 4.0


def transform_scale(matrix: np.ndarray, donor_box: BBox) -> float:
    """sqrt(area of the projected donor subject box / its area); 0.0 if folded or degenerate.

    The box is the half-open pixel-edge polygon; a mirrored (negative-area) projection or a
    corner beyond the horizon measures 0.0 so it can never pass a minimum-scale gate.
    """
    m = _check_matrix(matrix)
    x1, y1, x2, y2 = donor_box
    pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]]).T
    proj = m @ pts
    w = proj[2]
    if not (np.all(w > 0.0) or np.all(w < 0.0)):
        return 0.0
    px, py = proj[0] / w, proj[1] / w
    area = 0.5 * float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
    base = float((x2 - x1) * (y2 - y1))
    if not math.isfinite(area) or area <= 0.0 or base <= 0.0:
        return 0.0
    return math.sqrt(area / base)


# ------------------------------------------------------------------------------------------------
# Gates (pure; thresholds read from the policy)
# ------------------------------------------------------------------------------------------------


def _gate(
    accept: ReasonCode,
    observed: float | int | str,
    operator: Operator,
    threshold: float | int | str,
    units: str | None,
    key: str,
    accept_reason: str,
    reject_reason: str,
) -> Gate:
    """An ALIGN-stage gate that ACCEPTs as ``accept`` and REJECTs as ``ALIGNMENT_FAILED``."""
    return Gate(
        accept, ALIGN, observed, operator, threshold, units, key, accept_reason, reject_reason,
        FAILED,
    )


def corner_gate(fraction: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, fraction, "<=", cfg.alignment.max_corner_outside_fraction, "fraction",
        "alignment.max_corner_outside_fraction",
        "Projected corners inside the donor frame",
        "Too many projected corners outside the donor frame",
    )


def coverage_gate(coverage: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, coverage, ">=", cfg.alignment.min_valid_coverage, "fraction",
        "alignment.min_valid_coverage",
        "Valid warp coverage passed", "Valid warp coverage below minimum",
    )


def keypoints_gate(count: int, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, int(count), ">=", cfg.akaze.min_keypoints, "keypoints",
        "akaze.min_keypoints",
        "Enough AKAZE keypoints on both expanded crops", "Too few AKAZE keypoints",
    )


def matches_gate(count: int, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, int(count), ">=", cfg.akaze.min_good_matches, "matches",
        "akaze.min_good_matches",
        "Enough ratio-test matches", "Too few ratio-test matches",
    )


def inlier_gate(ratio: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, ratio, ">=", cfg.alignment.min_inlier_ratio, "fraction",
        "alignment.min_inlier_ratio",
        "RANSAC inlier ratio passed", "RANSAC inlier ratio below minimum",
    )


def reproj_gate(median_px: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, median_px, "<=", cfg.alignment.max_median_reprojection_px, "px",
        "alignment.max_median_reprojection_px",
        "Median reprojection inside the limit", "Median reprojection above the limit",
    )


def scale_min_gate(scale: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, scale, ">=", _scale_min(cfg), "ratio",
        "alignment.scale_ratio_min",
        "Alignment scale above the minimum", "Alignment scale below the minimum",
    )


def scale_max_gate(scale: float, cfg: PolicyConfig, accept: ReasonCode = AKAZE_OK) -> Gate:
    return _gate(
        accept, scale, "<=", _scale_max(cfg), "ratio",
        "alignment.scale_ratio_max",
        "Alignment scale below the maximum", "Alignment scale above the maximum",
    )


def correlation_gate(cc: float, cfg: PolicyConfig, accept: ReasonCode = ECC_OK) -> Gate:
    return _gate(
        accept, cc, ">=", cfg.ecc.min_correlation, "correlation",
        "ecc.min_correlation",
        "ECC correlation passed", "ECC correlation below minimum",
    )


def _scale_min(cfg: PolicyConfig) -> float:
    return float(getattr(cfg.alignment, "scale_ratio_min", DEFAULT_SCALE_RATIO_MIN))


def _scale_max(cfg: PolicyConfig) -> float:
    return float(getattr(cfg.alignment, "scale_ratio_max", DEFAULT_SCALE_RATIO_MAX))


def _opt(section: object, name: str, default: float | int) -> float | int:
    return getattr(section, name, default)


# ------------------------------------------------------------------------------------------------
# Feature detection, matching, RANSAC, ECC
# ------------------------------------------------------------------------------------------------


def _clahe_luma(bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE for AKAZE only. Tiny crops skip CLAHE (tile grid would throw)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tiles = DEFAULT_CLAHE_TILE_GRID
    if gray.shape[0] <= tiles or gray.shape[1] <= tiles:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=DEFAULT_CLAHE_CLIP_LIMIT, tileGridSize=(tiles, tiles),
    )
    return clahe.apply(gray)


def _akaze_points(
    crop: np.ndarray, origin: tuple[int, int], cfg: PolicyConfig,
) -> tuple[int, np.ndarray, np.ndarray | None]:
    gray = _clahe_luma(crop)
    threshold = float(_opt(cfg.akaze, "detector_threshold", DEFAULT_AKAZE_DETECTOR_THRESHOLD))
    detector = cv2.AKAZE_create(threshold=threshold)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints or descriptors is None:
        return 0, np.empty((0, 2), dtype=np.float64), None
    ox, oy = origin
    pts = np.array([(kp.pt[0] + ox, kp.pt[1] + oy) for kp in keypoints], dtype=np.float64)
    return len(keypoints), pts, descriptors


def _ratio_pairs(
    desc_d: np.ndarray, desc_t: np.ndarray, pts_d: np.ndarray, pts_t: np.ndarray, ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(desc_d, desc_t, k=2)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            src.append(pts_d[first.queryIdx])
            dst.append(pts_t[first.trainIdx])
    if not src:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)


def _ransac_homography(
    src: np.ndarray, dst: np.ndarray, cfg: PolicyConfig,
) -> tuple[np.ndarray | None, np.ndarray]:
    seed = int(_opt(cfg.ransac, "rng_seed", DEFAULT_RANSAC_RNG_SEED))
    max_iters = int(_opt(cfg.ransac, "max_iters", DEFAULT_RANSAC_MAX_ITERS))
    confidence = float(_opt(cfg.ransac, "confidence", DEFAULT_RANSAC_CONFIDENCE))
    cv2.setRNGSeed(seed)
    matrix, mask = cv2.findHomography(
        src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.ransac.reprojection_threshold_px),
        maxIters=max_iters, confidence=confidence,
    )
    empty = np.zeros(len(src), dtype=bool)
    if matrix is None or mask is None:
        return None, empty
    try:
        return _check_matrix(matrix), mask.ravel().astype(bool)
    except ValueError:
        return None, empty


def _median_reproj(
    matrix: np.ndarray, src: np.ndarray, dst: np.ndarray, inliers: np.ndarray,
) -> float:
    use = inliers if bool(inliers.any()) else np.ones(len(src), dtype=bool)
    pts = np.concatenate([src[use], np.ones((int(use.sum()), 1), dtype=np.float64)], axis=1)
    proj = matrix @ pts.T
    w = proj[2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        px, py = proj[0] / w, proj[1] / w
        err = np.hypot(px - dst[use, 0], py - dst[use, 1])
    err = err[np.isfinite(err)]
    if err.size == 0:
        return UNPROJECTABLE_REPROJ_PX
    return float(np.median(err))


def box_affine(donor_box: BBox, target_box: BBox) -> np.ndarray:
    """Axis-aligned affine mapping the donor subject box onto the target subject box."""
    dx1, dy1, dx2, dy2 = donor_box
    tx1, ty1, tx2, ty2 = target_box
    dw, dh = float(dx2 - dx1), float(dy2 - dy1)
    tw, th = float(tx2 - tx1), float(ty2 - ty1)
    if dw <= 0.0 or dh <= 0.0:
        return np.eye(3, dtype=np.float64)
    sx, sy = tw / dw, th / dh
    return np.array(
        [[sx, 0.0, tx1 - sx * dx1], [0.0, sy, ty1 - sy * dy1], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _full_to_crop_affine(
    full: np.ndarray, donor_origin: tuple[float, float], target_origin: tuple[float, float],
) -> np.ndarray:
    """2x3 float32: donor-crop coords -> target-crop coords, from a full-frame 3x3 affine."""
    linear = full[:2, :2]
    translation = full[:2, 2]
    od = np.asarray(donor_origin, dtype=np.float64)
    ot = np.asarray(target_origin, dtype=np.float64)
    crop = np.zeros((2, 3), dtype=np.float32)
    crop[:, :2] = linear
    crop[:, 2] = linear @ od + translation - ot
    return crop


def _crop_to_full_affine(
    crop: np.ndarray, donor_origin: tuple[float, float], target_origin: tuple[float, float],
) -> np.ndarray:
    """Full-frame 3x3 affine from a 2x3 crop-space ECC warp."""
    linear = np.asarray(crop[:, :2], dtype=np.float64)
    t_crop = np.asarray(crop[:, 2], dtype=np.float64)
    od = np.asarray(donor_origin, dtype=np.float64)
    ot = np.asarray(target_origin, dtype=np.float64)
    full = np.eye(3, dtype=np.float64)
    full[:2, :2] = linear
    full[:2, 2] = t_crop - linear @ od + ot
    return full


def _geometry_gates(
    matrix: np.ndarray, target: Obs, donor: Obs, cfg: PolicyConfig, accept: ReasonCode,
) -> list[Gate]:
    scale = transform_scale(matrix, donor.bbox_px)
    return [
        scale_min_gate(scale, cfg, accept),
        scale_max_gate(scale, cfg, accept),
        corner_gate(corner_outside_fraction(matrix, target, donor.frame_size), cfg, accept),
        coverage_gate(valid_coverage(matrix, target, donor.frame_size), cfg, accept),
    ]


def _result(
    method: AlignmentMethod,
    matrix: np.ndarray | None,
    inlier_ratio: float,
    median_reproj_px: float,
    target: Obs,
    donor: Obs,
    gates: list[Gate],
    accept: ReasonCode,
) -> Alignment:
    if matrix is not None:
        coverage = valid_coverage(matrix, target, donor.frame_size)
        corner = corner_outside_fraction(matrix, target, donor.frame_size)
        stored = np.ascontiguousarray(matrix, dtype=np.float64)
    else:
        coverage, corner, stored = 0.0, 0.0, None
    ok = all(gate.ok for gate in gates)
    return Alignment(
        method=method, matrix=stored, inlier_ratio=float(inlier_ratio),
        median_reproj_px=float(median_reproj_px), valid_coverage=coverage,
        corner_outside_fraction=corner,
        A=alignment_confidence(inlier_ratio, median_reproj_px, coverage),
        accepted=ok, reason=accept if ok else FAILED, gates=tuple(gates),
    )


def _feature_starved(alignment: Alignment) -> bool:
    failed = next((gate for gate in alignment.gates if not gate.ok), None)
    return failed is not None and failed.policy_key in FEATURE_STARVED_KEYS


# ------------------------------------------------------------------------------------------------
# Alignment
# ------------------------------------------------------------------------------------------------


def akaze_homography(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """AKAZE + ratio test + RANSAC homography. ``method`` AKAZE_HOMOGRAPHY.

    Gates (in order, all in ``Alignment.gates``): keypoints, good matches, inlier ratio, median
    reprojection, scale, corner-outside fraction, valid coverage.
    """
    cv2.setNumThreads(1)
    n_t, pts_t, desc_t = _akaze_points(
        target.expanded_crop, (target.expanded_box[0], target.expanded_box[1]), cfg,
    )
    n_d, pts_d, desc_d = _akaze_points(
        donor.expanded_crop, (donor.expanded_box[0], donor.expanded_box[1]), cfg,
    )
    kp_gate = keypoints_gate(min(n_t, n_d), cfg)
    if not kp_gate.ok or desc_t is None or desc_d is None:
        return _result(AKAZE_METHOD, None, 0.0, 0.0, target, donor, [kp_gate], AKAZE_OK)
    src, dst = _ratio_pairs(desc_d, desc_t, pts_d, pts_t, cfg.akaze.ratio_test)
    match_gate = matches_gate(len(src), cfg)
    if not match_gate.ok:
        return _result(
            AKAZE_METHOD, None, 0.0, 0.0, target, donor, [kp_gate, match_gate], AKAZE_OK,
        )
    matrix, inliers = _ransac_homography(src, dst, cfg)
    ratio = float(inliers.mean()) if matrix is not None else 0.0
    gates = [kp_gate, match_gate, inlier_gate(ratio, cfg)]
    if matrix is None:
        return _result(AKAZE_METHOD, None, ratio, 0.0, target, donor, gates, AKAZE_OK)
    median_px = _median_reproj(matrix, src, dst, inliers)
    gates.append(reproj_gate(median_px, cfg))
    gates.extend(_geometry_gates(matrix, target, donor, cfg, AKAZE_OK))
    return _result(AKAZE_METHOD, matrix, ratio, median_px, target, donor, gates, AKAZE_OK)


def ecc_affine_once(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """One bounded ECC affine attempt (max iterations / epsilon / min correlation from policy)."""
    cv2.setNumThreads(1)
    template = cv2.cvtColor(target.expanded_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    incoming = cv2.cvtColor(donor.expanded_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    initial = box_affine(donor.bbox_px, target.bbox_px)
    donor_origin = (float(donor.expanded_box[0]), float(donor.expanded_box[1]))
    target_origin = (float(target.expanded_box[0]), float(target.expanded_box[1]))
    warp = _full_to_crop_affine(initial, donor_origin, target_origin)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(cfg.ecc.max_iterations), float(cfg.ecc.epsilon),
    )
    gauss = int(_opt(cfg.ecc, "gauss_filt_size", DEFAULT_ECC_GAUSS_FILT_SIZE))
    cc = 0.0
    matrix: np.ndarray | None = None
    try:
        cc_raw, warp = cv2.findTransformECC(
            template, incoming, warp, cv2.MOTION_AFFINE, criteria, None, gauss,
        )
        cc = float(cc_raw)
        if math.isfinite(cc):
            matrix = _check_matrix(_crop_to_full_affine(warp, donor_origin, target_origin))
        else:
            cc = 0.0
    except (cv2.error, ValueError):
        cc = 0.0
        matrix = None
    gates = [correlation_gate(cc, cfg)]
    if not gates[0].ok or matrix is None:
        return _result(ECC_METHOD, matrix, 0.0, 0.0, target, donor, gates, ECC_OK)
    gates.extend(_geometry_gates(matrix, target, donor, cfg, ECC_OK))
    return _result(ECC_METHOD, matrix, 0.0, 0.0, target, donor, gates, ECC_OK)


def align(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """AKAZE first; one ECC attempt only if AKAZE failed for keypoints or matches.

    The returned ``gates`` hold both attempts when the fallback ran.
    """
    primary = akaze_homography(target, donor, cfg)
    if primary.accepted or not _feature_starved(primary):
        return primary
    fallback = ecc_affine_once(target, donor, cfg)
    return Alignment(
        method=fallback.method, matrix=fallback.matrix, inlier_ratio=fallback.inlier_ratio,
        median_reproj_px=fallback.median_reproj_px, valid_coverage=fallback.valid_coverage,
        corner_outside_fraction=fallback.corner_outside_fraction, A=fallback.A,
        accepted=fallback.accepted, reason=fallback.reason,
        gates=primary.gates + fallback.gates,
    )


def warp_donor(donor_frame: np.ndarray, alignment: Alignment, target: Obs) -> Warp:
    """Resample ``donor_frame`` (full BGR frame) into the target-crop frame.

    ``source_x/source_y`` = inverse of ``alignment.matrix`` at each target-crop pixel (full-frame
    coordinates, float32); ``crop`` = cv2.remap(LANCZOS4, BORDER_REFLECT_101) at those maps.
    """
    if not alignment.accepted or alignment.matrix is None:
        raise ValueError("only an accepted alignment with a matrix can be warped")
    if donor_frame.dtype != np.uint8 or donor_frame.ndim != 3 or donor_frame.shape[2] != 3:
        raise ValueError("donor_frame must be a BGR uint8 (H, W, 3) image")
    height, width = donor_frame.shape[:2]
    sx, sy, valid = source_maps(alignment.matrix, target.expanded_box, (width, height))
    crop = cv2.remap(
        donor_frame, sx, sy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
    )
    return Warp(crop=np.ascontiguousarray(crop), valid_mask=valid, source_x=sx, source_y=sy)

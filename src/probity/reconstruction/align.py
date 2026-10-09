"""Lane B: donor-to-target alignment (section 9, algorithm step 6). PUBLIC SIGNATURES FROZEN.

Primary: grayscale + CLAHE (feature detection only) on the expanded crops, AKAZE descriptors,
Hamming ratio test, RANSAC homography donor->target. Fallback: exactly one ECC affine attempt,
initialized from box geometry, and only when AKAZE failed for too few keypoints or matches.
Never retry, never pick by appearance. Determinism: ``cv2.setRNGSeed`` before
``findHomography``, fixed RANSAC iterations, reprojection threshold from policy.
"""

from __future__ import annotations

import numpy as np

from probity.domain.policy import PolicyConfig
from probity.reconstruction.types import Alignment, Obs, Warp

LANE = "lane B"
A_WEIGHTS = {"inlier": 0.40, "reprojection": 0.30, "coverage": 0.30}


def alignment_confidence(inlier_ratio: float, median_reproj_px: float, valid_coverage: float
                         ) -> float:
    """A = 0.40 * inlier_ratio + 0.30 * (1 - clip(median_reproj_px / 2)) + 0.30 * coverage."""
    raise NotImplementedError(LANE)


def akaze_homography(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """AKAZE + ratio test + RANSAC homography. ``method`` AKAZE_HOMOGRAPHY.

    Gates (in order, all in ``Alignment.gates``): keypoints, good matches, inlier ratio, median
    reprojection, scale, corner-outside fraction, valid coverage.
    """
    raise NotImplementedError(LANE)


def ecc_affine_once(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """One bounded ECC affine attempt (max iterations / epsilon / min correlation from policy)."""
    raise NotImplementedError(LANE)


def align(target: Obs, donor: Obs, cfg: PolicyConfig) -> Alignment:
    """AKAZE first; one ECC attempt only if AKAZE failed for keypoints or matches.

    The returned ``gates`` hold both attempts when the fallback ran.
    """
    raise NotImplementedError(LANE)


def warp_donor(donor_frame: np.ndarray, alignment: Alignment, target: Obs) -> Warp:
    """Resample ``donor_frame`` (full BGR frame) into the target-crop frame.

    ``source_x/source_y`` = inverse of ``alignment.matrix`` at each target-crop pixel (full-frame
    coordinates, float32); ``crop`` = cv2.remap(LANCZOS4, BORDER_REFLECT_101) at those maps.
    """
    raise NotImplementedError(LANE)

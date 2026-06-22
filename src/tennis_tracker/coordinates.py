"""Pixel-to-court coordinate projection via planar homography.

Given a set of matched 2D points between image pixels and known court-meter
keypoints, estimate the homography matrix and project arbitrary pixel points
into the court coordinate system defined in ``court.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tennis_tracker.types import CourtPoint, PixelPoint


@dataclass(frozen=True)
class HomographyResult:
    """Result of a homography estimation attempt.

    ``valid`` is True only when the homography was successfully computed
    from a sufficient, non-degenerate set of correspondences.
    """

    matrix: Optional[np.ndarray]  # 3x3 homography matrix, or None
    valid: bool
    confidence: float
    num_matches: int
    reprojection_error_px: Optional[float] = None


def estimate_homography(
    pixel_pts: list[PixelPoint],
    court_pts: list[tuple[float, float]],
    ransac_threshold: float = 3.0,
    confidence_level: float = 0.99,
) -> HomographyResult:
    """Estimate a homography from matched pixel → court-meter correspondences.

    Parameters
    ----------
    pixel_pts :
        Detected keypoint positions in image pixel coordinates.
    court_pts :
        Corresponding court-meter positions as ``(x_m, y_m)`` tuples.
    ransac_threshold :
        Maximum reprojection error (pixels) for a point to be considered
        an inlier during RANSAC.
    confidence_level :
        Desired confidence that the estimated matrix is correct (used by
        OpenCV's RANSAC).

    Returns
    -------
    HomographyResult
        The estimated homography matrix (or None) together with validity
        and diagnostic metadata.
    """
    if len(pixel_pts) < 4:
        return HomographyResult(
            matrix=None,
            valid=False,
            confidence=0.0,
            num_matches=len(pixel_pts),
            reprojection_error_px=None,
        )

    src_pts = np.array([(p.x, p.y) for p in pixel_pts], dtype=np.float64).reshape(-1, 1, 2)
    dst_pts = np.array(court_pts, dtype=np.float64).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
        confidence=confidence_level,
    )

    if matrix is None:
        return HomographyResult(
            matrix=None,
            valid=False,
            confidence=0.0,
            num_matches=len(pixel_pts),
            reprojection_error_px=None,
        )

    inliers = int(mask.sum()) if mask is not None else 0
    num_matches = len(pixel_pts)
    confidence_ratio = inliers / num_matches if num_matches > 0 else 0.0

    # Compute mean reprojection error from inliers
    errors: list[float] = []
    for i in range(num_matches):
        if mask is not None and mask[i] == 0:
            continue
        pt = np.array([pixel_pts[i].x, pixel_pts[i].y, 1.0], dtype=np.float64)
        projected = matrix @ pt
        if projected[2] == 0:
            continue
        projected /= projected[2]
        dx = projected[0] - court_pts[i][0]
        dy = projected[1] - court_pts[i][1]
        errors.append(float(np.sqrt(dx * dx + dy * dy)))

    mean_error = float(np.mean(errors)) if errors else None

    return HomographyResult(
        matrix=matrix,
        valid=True,
        confidence=confidence_ratio,
        num_matches=num_matches,
        reprojection_error_px=mean_error,
    )


def project_pixel_to_court(
    pixel: PixelPoint,
    homography: np.ndarray,
    confidence: float,
) -> CourtPoint:
    """Project a single image pixel into the court coordinate system.

    Parameters
    ----------
    pixel :
        The pixel coordinates to project.
    homography :
        A valid 3×3 homography matrix.
    confidence :
        Confidence value to attach to the resulting CourtPoint.

    Returns
    -------
    CourtPoint
        The projected court-meter position.

    Raises
    ------
    ValueError
        If the homography matrix produces a zero scale factor.
    """
    pt = np.array([pixel.x, pixel.y, 1.0], dtype=np.float64)
    projected = homography @ pt

    if projected[2] == 0:
        raise ValueError("Invalid homography projection with zero scale factor")

    projected /= projected[2]
    return CourtPoint(
        x_m=float(projected[0]),
        y_m=float(projected[1]),
        confidence=confidence,
    )

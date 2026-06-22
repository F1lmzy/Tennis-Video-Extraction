"""Tests for pixel-to-court homography estimation and projection."""

import numpy as np
import pytest

from tennis_tracker.coordinates import (
    estimate_homography,
    project_pixel_to_court,
)
from tennis_tracker.types import CourtPoint, PixelPoint


# ── Synthetic square fixture ────────────────────────────────────────────
# A 200×200 pixel square maps to a 2.0 m × 2.0 m court square centred at (0, 0).
_SQUARE_PIXEL: list[PixelPoint] = [
    PixelPoint(0.0, 0.0),
    PixelPoint(200.0, 0.0),
    PixelPoint(200.0, 200.0),
    PixelPoint(0.0, 200.0),
]
_SQUARE_COURT: list[tuple[float, float]] = [
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
]

# A point at the centre of the square in pixel space → (0, 0) in court space.
_CENTRE_PIXEL = PixelPoint(100.0, 100.0)


class TestHomographyEstimation:
    def test_square_centre_projects_to_origin(self) -> None:
        """Centre of the 200×200 square → (0, 0) in court space."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid
        assert result.matrix is not None

        court = project_pixel_to_court(_CENTRE_PIXEL, result.matrix, result.confidence)
        assert court.x_m == pytest.approx(0.0, abs=1e-9)
        assert court.y_m == pytest.approx(0.0, abs=1e-9)

    def test_square_corner_projects_correctly(self) -> None:
        """Top-left pixel (0, 0) → (-1, -1) in court space."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid
        assert result.matrix is not None

        court = project_pixel_to_court(_SQUARE_PIXEL[0], result.matrix, result.confidence)
        assert court.x_m == pytest.approx(-1.0, abs=1e-9)
        assert court.y_m == pytest.approx(-1.0, abs=1e-9)

    def test_square_midpoint_projects_correctly(self) -> None:
        """Midpoint of right edge (200, 100) → (1, 0) in court space."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid
        assert result.matrix is not None

        mid = PixelPoint(200.0, 100.0)
        court = project_pixel_to_court(mid, result.matrix, result.confidence)
        assert court.x_m == pytest.approx(1.0, abs=1e-6)
        assert court.y_m == pytest.approx(0.0, abs=1e-6)

    def test_insufficient_keypoints_are_invalid(self) -> None:
        """Fewer than 4 correspondences must produce an invalid result."""
        pts = [PixelPoint(0.0, 0.0), PixelPoint(100.0, 0.0), PixelPoint(100.0, 100.0)]
        court = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        result = estimate_homography(pts, court)
        assert not result.valid
        assert result.matrix is None
        assert result.confidence == 0.0

    def test_no_keypoints_are_invalid(self) -> None:
        """Zero correspondences must produce an invalid result."""
        result = estimate_homography([], [])
        assert not result.valid
        assert result.matrix is None

    def test_collinear_keypoints_produce_invalid_or_low_confidence(self) -> None:
        """Collinear points produce an invalid or low-quality homography."""
        pts = [
            PixelPoint(0.0, 0.0),
            PixelPoint(100.0, 0.0),
            PixelPoint(200.0, 0.0),
            PixelPoint(300.0, 0.0),
        ]
        court = [(-1.5, 0.0), (-0.5, 0.0), (0.5, 0.0), (1.5, 0.0)]
        result = estimate_homography(pts, court)
        # RANSAC with 4 collinear points is degenerate — the result should be
        # invalid or have near-zero inlier ratio.
        assert not result.valid or result.confidence < 0.5

    def test_homography_inverts_correctly(self) -> None:
        """Pixel → court then court → pixel should recover the original within tolerance."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid
        assert result.matrix is not None

        # Invert the homography and project court → pixel
        inv_matrix = np.linalg.inv(result.matrix)
        court_pt = CourtPoint(0.5, -0.5, 1.0)
        pt = np.array([court_pt.x_m, court_pt.y_m, 1.0], dtype=np.float64)
        projected = inv_matrix @ pt
        projected /= projected[2]

        # The original pixel for (0.5, -0.5) should be near (150, 50):
        #   court_x = pixel_x / 100 - 1  →  pixel_x = (0.5 + 1) * 100 = 150
        #   court_y = pixel_y / 100 - 1  →  pixel_y = (-0.5 + 1) * 100 = 50
        assert projected[0] == pytest.approx(150.0, abs=1e-6)
        assert projected[1] == pytest.approx(50.0, abs=1e-6)


class TestProjection:
    def test_projection_raises_on_zero_scale(self) -> None:
        """A 3×3 zero matrix should raise ValueError."""
        zero_h = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="zero scale factor"):
            project_pixel_to_court(PixelPoint(10.0, 10.0), zero_h, 1.0)

    def test_projection_preserves_confidence(self) -> None:
        """The returned CourtPoint carries the given confidence."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid and result.matrix is not None

        court = project_pixel_to_court(_CENTRE_PIXEL, result.matrix, 0.85)
        assert court.confidence == 0.85

    def test_projection_with_rectangle_aspect_ratio(self) -> None:
        """A 400×200 pixel rectangle mapping to a 4 m × 2 m court rectangle."""
        pixel_pts = [
            PixelPoint(0.0, 0.0),
            PixelPoint(400.0, 0.0),
            PixelPoint(400.0, 200.0),
            PixelPoint(0.0, 200.0),
        ]
        court_pts = [(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)]

        result = estimate_homography(pixel_pts, court_pts)
        assert result.valid
        assert result.matrix is not None

        # Centre of rectangle → (0, 0)
        centre = project_pixel_to_court(PixelPoint(200.0, 100.0), result.matrix, 1.0)
        assert centre.x_m == pytest.approx(0.0, abs=1e-6)
        assert centre.y_m == pytest.approx(0.0, abs=1e-6)

        # Bottom-right pixel → (2, 1)
        br = project_pixel_to_court(PixelPoint(400.0, 200.0), result.matrix, 1.0)
        assert br.x_m == pytest.approx(2.0, abs=1e-6)
        assert br.y_m == pytest.approx(1.0, abs=1e-6)

    def test_result_metadata(self) -> None:
        """HomographyResult provides useful diagnostic metadata."""
        result = estimate_homography(_SQUARE_PIXEL, _SQUARE_COURT)
        assert result.valid
        assert result.num_matches == 4
        assert result.confidence > 0.0
        assert result.reprojection_error_px is not None
        assert result.reprojection_error_px < 1.0  # sub-pixel for perfect data

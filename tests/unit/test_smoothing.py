"""Tests for temporal smoothing and interpolation of output rows."""

from typing import Optional

import pytest

from tennis_tracker.types import OutputRow
from tennis_tracker.smoothing import smooth_output_rows


def _row(
    frame_index: int,
    time_s: float,
    *,
    ax: Optional[float],
    ay: Optional[float],
    bx: Optional[float],
    by: Optional[float],
    ballx: Optional[float],
    bally: Optional[float],
    conf: float = 0.95,
) -> OutputRow:
    return OutputRow(
        frame_index=frame_index,
        time_s=time_s,
        player_a_x_m=ax,
        player_a_y_m=ay,
        player_a_pixel_x=100.0 if ax is not None else None,
        player_a_pixel_y=200.0 if ay is not None else None,
        player_a_confidence=conf if ax is not None else None,
        player_b_x_m=bx,
        player_b_y_m=by,
        player_b_pixel_x=300.0 if bx is not None else None,
        player_b_pixel_y=400.0 if by is not None else None,
        player_b_confidence=conf if bx is not None else None,
        ball_x_m=ballx,
        ball_y_m=bally,
        ball_pixel_x=500.0 if ballx is not None else None,
        ball_pixel_y=600.0 if bally is not None else None,
        ball_confidence=conf if ballx is not None else None,
        court_confidence=0.9,
        homography_valid=True,
        diagnostics="",
    )


class TestInterpolation:
    def test_no_missing_values_unchanged(self) -> None:
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=5.0, bally=6.0),
            _row(1, 1.0, ax=1.1, ay=2.1, bx=3.1, by=4.1, ballx=5.1, bally=6.1),
            _row(2, 2.0, ax=1.2, ay=2.2, bx=3.2, by=4.2, ballx=5.2, bally=6.2),
        ]
        smoothed = smooth_output_rows(rows)
        for r, s in zip(rows, smoothed):
            assert r.player_a_x_m == s.player_a_x_m
            assert r.player_a_y_m == s.player_a_y_m
            assert r.player_b_x_m == s.player_b_x_m
            assert r.player_b_y_m == s.player_b_y_m
            assert r.ball_x_m == s.ball_x_m
            assert r.ball_y_m == s.ball_y_m

    def test_short_gap_in_ball_is_interpolated(self) -> None:
        """A single missing ball frame should be interpolated."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(2, 2.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=2.0, bally=2.0),
        ]
        smoothed = smooth_output_rows(rows)
        assert smoothed[1].ball_x_m is not None
        assert smoothed[1].ball_y_m is not None
        # Linear interpolation: midpoint of (0,0) and (2,2) → (1,1)
        assert smoothed[1].ball_x_m == pytest.approx(1.0, abs=1e-6)
        assert smoothed[1].ball_y_m == pytest.approx(1.0, abs=1e-6)

    def test_long_gap_remains_missing(self) -> None:
        """A gap longer than max_gap should not be interpolated."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(2, 2.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(3, 3.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(4, 4.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(5, 5.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(6, 6.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(7, 7.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=10.0, bally=10.0),
        ]
        # Default max_gap is 5; the gap is 6 frames → remains missing
        smoothed = smooth_output_rows(rows)
        for i in range(1, 7):
            assert smoothed[i].ball_x_m is None
        assert smoothed[7].ball_x_m == 10.0

    def test_gap_at_end_remains_missing(self) -> None:
        """Trailing missing values have no right bound → stay missing."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(2, 2.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
        ]
        smoothed = smooth_output_rows(rows)
        assert smoothed[1].ball_x_m is None
        assert smoothed[2].ball_x_m is None

    def test_gap_at_start_remains_missing(self) -> None:
        """Leading missing values have no left bound → stay missing."""
        rows = [
            _row(0, 0.0, ax=None, ay=None, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=None, ay=None, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(2, 2.0, ax=5.0, ay=6.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
        ]
        smoothed = smooth_output_rows(rows)
        assert smoothed[0].player_a_x_m is None
        assert smoothed[1].player_a_x_m is None
        assert smoothed[2].player_a_x_m == 5.0

    def test_empty_rows_returns_empty(self) -> None:
        assert smooth_output_rows([]) == []


class TestRawRowsNotMutated:
    def test_raw_rows_unchanged(self) -> None:
        """Original rows must not be modified by smoothing."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(2, 2.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=2.0, bally=2.0),
        ]
        _ = smooth_output_rows(rows)
        assert rows[1].ball_x_m is None  # still None
        assert rows[0].ball_x_m == 0.0  # still original


class TestConfidencePropagation:
    def test_confidence_forward_fills_short_gap(self) -> None:
        """Confidence should forward-fill from the last known observation."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=0.0, bally=0.0, conf=0.9),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(2, 2.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=2.0, bally=2.0, conf=0.8),
        ]
        smoothed = smooth_output_rows(rows)
        # Frame 1 has no ball observation; confidence forward-fills from frame 0 (0.9)
        assert smoothed[1].ball_confidence is not None
        assert smoothed[1].ball_confidence == pytest.approx(0.9)
        # Frame 2 has its original confidence preserved
        assert smoothed[2].ball_confidence == pytest.approx(0.8)

    def test_confidence_remains_none_when_no_prior_observation(self) -> None:
        """When no prior observation exists, confidence stays None."""
        rows = [
            _row(0, 0.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
            _row(1, 1.0, ax=1.0, ay=2.0, bx=3.0, by=4.0, ballx=None, bally=None),
        ]
        smoothed = smooth_output_rows(rows)
        assert smoothed[0].ball_confidence is None
        assert smoothed[1].ball_confidence is None


class TestMultipleFields:
    def test_player_a_and_b_independently_smoothed(self) -> None:
        rows = [
            _row(0, 0.0, ax=0.0, ay=0.0, bx=10.0, by=10.0, ballx=0.0, bally=0.0),
            _row(1, 1.0, ax=None, ay=None, bx=None, by=None, ballx=0.0, bally=0.0),
            _row(2, 2.0, ax=4.0, ay=4.0, bx=14.0, by=14.0, ballx=0.0, bally=0.0),
        ]
        smoothed = smooth_output_rows(rows)
        # Player A interpolated
        assert smoothed[1].player_a_x_m == pytest.approx(2.0, abs=1e-6)
        assert smoothed[1].player_a_y_m == pytest.approx(2.0, abs=1e-6)
        # Player B interpolated
        assert smoothed[1].player_b_x_m == pytest.approx(12.0, abs=1e-6)
        assert smoothed[1].player_b_y_m == pytest.approx(12.0, abs=1e-6)

    def test_diagnostics_preserved(self) -> None:
        """Diagnostics field should not be modified by smoothing."""
        rows = [
            OutputRow(
                frame_index=0,
                time_s=0.0,
                player_a_x_m=1.0,
                player_a_y_m=2.0,
                player_a_pixel_x=100.0,
                player_a_pixel_y=200.0,
                player_a_confidence=0.95,
                player_b_x_m=3.0,
                player_b_y_m=4.0,
                player_b_pixel_x=300.0,
                player_b_pixel_y=400.0,
                player_b_confidence=0.95,
                ball_x_m=5.0,
                ball_y_m=6.0,
                ball_pixel_x=500.0,
                ball_pixel_y=600.0,
                ball_confidence=0.95,
                court_confidence=0.9,
                homography_valid=True,
                diagnostics="missing_ball;low_confidence",
            ),
        ]
        smoothed = smooth_output_rows(rows)
        assert smoothed[0].diagnostics == "missing_ball;low_confidence"

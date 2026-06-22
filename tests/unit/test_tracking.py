"""Unit tests for player and ball tracking assignment (``tracking.py``).

All tests use fake ``PlayerDetection`` and ``BallDetection`` objects
(no model weights, no network, no GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis_tracker.detection import BallDetection, BoundingBox, PlayerDetection
from tennis_tracker.tracking import PlayerTracker, has_local_ball_motion, select_best_ball
from tennis_tracker.types import PixelPoint


# ── Test helpers ────────────────────────────────────────────────────────


def _player(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
    track_id: int | None = None,
) -> PlayerDetection:
    return PlayerDetection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=confidence,
        track_id=track_id,
    )


def _ball(
    x: float,
    y: float,
    confidence: float = 0.9,
) -> BallDetection:
    bbox = BoundingBox(x1=x - 2, y1=y - 2, x2=x + 2, y2=y + 2)
    return BallDetection(
        bbox=bbox,
        center=PixelPoint(x=x, y=y),
        confidence=confidence,
    )


def _ball_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> BallDetection:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return BallDetection(
        bbox=bbox,
        center=bbox.center,
        confidence=confidence,
    )


# ── PlayerTracker tests ────────────────────────────────────────────────


class TestPlayerTrackerInitialAssignment:
    """Tests for the first frame (no prior tracking state)."""

    def test_two_players_selected_by_confidence(self) -> None:
        """Top 2 by confidence should be selected as A and B."""
        tracker = PlayerTracker()
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.5),
            _player(10, 0, 20, 20, confidence=0.9),
            _player(20, 0, 30, 20, confidence=0.3),
        ])
        assert result.player_a is not None
        assert result.player_b is not None
        assert result.player_a.confidence == pytest.approx(0.9)
        assert result.player_b.confidence == pytest.approx(0.5)

    def test_one_player_missing(self) -> None:
        """When only one player detected, the other is None."""
        tracker = PlayerTracker()
        result = tracker.update([_player(0, 0, 10, 20, confidence=0.9)])
        assert result.player_a is not None
        assert result.player_b is None
        assert result.diagnostics.has_flag("missing_player_b")

    def test_no_players_both_missing(self) -> None:
        """When no players detected, both are None."""
        tracker = PlayerTracker()
        result = tracker.update([])
        assert result.player_a is None
        assert result.player_b is None
        assert result.diagnostics.has_flag("missing_player_a")
        assert result.diagnostics.has_flag("missing_player_b")

    def test_tracked_false_when_no_players(self) -> None:
        tracker = PlayerTracker()
        result = tracker.update([])
        assert result.tracked is False

    def test_tracked_true_when_one_player(self) -> None:
        tracker = PlayerTracker()
        result = tracker.update([_player(0, 0, 10, 20)])
        assert result.tracked is True

    def test_tracked_true_when_two_players(self) -> None:
        tracker = PlayerTracker()
        result = tracker.update([
            _player(0, 0, 10, 20),
            _player(20, 0, 30, 20),
        ])
        assert result.tracked is True

    def test_more_than_two_players_emits_ambiguous_diagnostic(self) -> None:
        """More than 2 persons detected should flag ambiguous ID."""
        tracker = PlayerTracker()
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(10, 0, 20, 20, confidence=0.8),
            _player(20, 0, 30, 20, confidence=0.7),
            _player(30, 0, 40, 20, confidence=0.6),
        ])
        assert result.player_a is not None
        assert result.player_b is not None
        assert result.diagnostics.has_flag("ambigous_player_id")


class TestPlayerTrackerContinuity:
    """Tests for label continuity across frames."""

    def test_labels_persist_across_two_frames(self) -> None:
        """Players swapped between frames should maintain identity."""
        # Frame 1: A on left, B on right
        left = _player(0, 0, 10, 20, confidence=0.9)
        right = _player(100, 0, 110, 20, confidence=0.8)

        tracker = PlayerTracker()
        frame1 = tracker.update([left, right])
        assert frame1.player_a is not None
        assert frame1.player_b is not None

        # Frame 2: same positions
        left2 = _player(0, 0, 10, 20, confidence=0.9)
        right2 = _player(100, 0, 110, 20, confidence=0.8)
        frame2 = tracker.update([left2, right2])

        # A should still be on the left (bottom_center.x ~5)
        assert frame2.player_a is not None
        assert frame2.player_b is not None
        assert frame2.player_a.bbox.bottom_center.x == pytest.approx(5.0)
        assert frame2.player_b.bbox.bottom_center.x == pytest.approx(105.0)

    def test_players_swap_sides_still_tracked(self) -> None:
        """Even if players move, identity is maintained by proximity."""
        # Frame 1: A at left, B at right
        tracker = PlayerTracker(max_match_distance=500.0)
        tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(200, 0, 210, 20, confidence=0.8),
        ])

        # Frame 2: A moves right, B stays right — B is closer to where B was
        left = _player(0, 0, 10, 20, confidence=0.9)
        center = _player(100, 0, 110, 20, confidence=0.9)
        frame2 = tracker.update([left, center])

        # The previous A was at ~5, previous B at ~205
        # Now we have ~5 and ~105.  105 is closer to 205 (100px) than 5 is (200px).
        # So the ~105 detection should become player B.
        assert frame2.player_a is not None
        assert frame2.player_b is not None
        # A should be ~5, B should be ~105
        assert frame2.player_a.bbox.bottom_center.x == pytest.approx(5.0)
        assert frame2.player_b.bbox.bottom_center.x == pytest.approx(105.0)

    def test_track_survives_brief_occlusion(self) -> None:
        """When a player is briefly missing, track is maintained for the other."""
        tracker = PlayerTracker()
        # Frame 1: both players
        tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.8),
        ])
        # Frame 2: one player missing
        result = tracker.update([_player(0, 0, 10, 20, confidence=0.9)])
        assert result.player_a is not None
        assert result.player_b is None
        assert result.diagnostics.has_flag("missing_player_b")

    def test_multiple_detections_after_initialization(self) -> None:
        """Ambiguous frames after a clean start still flag diagnostic."""
        tracker = PlayerTracker()
        # Frame 1: clean
        tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.8),
        ])
        # Frame 2: 3 persons detected
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.8),
            _player(200, 0, 210, 20, confidence=0.7),
        ])
        assert result.diagnostics.has_flag("ambigous_player_id")


class TestPlayerTrackerDiagnostics:
    """Tests for confidence-based diagnostics."""

    def test_low_confidence_player_a_diagnostic(self) -> None:
        tracker = PlayerTracker(low_confidence_threshold=0.5)
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.3),
        ])
        # Player A is the higher-confidence detection (0.9), player B is 0.3
        assert not result.diagnostics.has_flag("low_player_a_confidence")
        assert result.diagnostics.has_flag("low_player_b_confidence")

    def test_low_confidence_player_b_diagnostic(self) -> None:
        tracker = PlayerTracker(low_confidence_threshold=0.5)
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.2),
        ])
        assert not result.diagnostics.has_flag("low_player_a_confidence")
        assert result.diagnostics.has_flag("low_player_b_confidence")

    def test_both_low_confidence_diagnostics(self) -> None:
        tracker = PlayerTracker(low_confidence_threshold=0.5)
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.1),
            _player(100, 0, 110, 20, confidence=0.2),
        ])
        assert result.diagnostics.has_flag("low_player_a_confidence")
        assert result.diagnostics.has_flag("low_player_b_confidence")


class TestPlayerTrackerWithBall:
    """Tests that ball detection is passed through correctly."""

    def test_ball_is_returned(self) -> None:
        tracker = PlayerTracker()
        ball_det = _ball(50, 60, confidence=0.9)
        result = tracker.update(
            detections=[
                _player(0, 0, 10, 20),
                _player(100, 0, 110, 20),
            ],
            ball=ball_det,
        )
        assert result.ball is not None
        assert result.ball.center.x == pytest.approx(50.0)
        assert result.ball.center.y == pytest.approx(60.0)
        assert result.ball.confidence == pytest.approx(0.9)

    def test_ball_can_be_none(self) -> None:
        tracker = PlayerTracker()
        result = tracker.update(
            detections=[
                _player(0, 0, 10, 20),
                _player(100, 0, 110, 20),
            ],
            ball=None,
        )
        assert result.ball is None


class TestPlayerTrackerEdgeCases:
    """Tests for edge cases and configuration."""

    def test_default_max_match_distance(self) -> None:
        tracker = PlayerTracker()
        assert tracker._max_match_distance == 200.0

    def test_custom_max_match_distance(self) -> None:
        tracker = PlayerTracker(max_match_distance=50.0)
        assert tracker._max_match_distance == 50.0

    def test_custom_low_confidence_threshold(self) -> None:
        tracker = PlayerTracker(low_confidence_threshold=0.8)
        result = tracker.update([
            _player(0, 0, 10, 20, confidence=0.81),
            _player(100, 0, 110, 20, confidence=0.79),
        ])
        # Player A is the higher-confidence detection (0.81), player B is 0.79
        assert not result.diagnostics.has_flag("low_player_a_confidence")
        assert result.diagnostics.has_flag("low_player_b_confidence")


# ── select_best_ball tests ─────────────────────────────────────────────


class TestSelectBestBall:
    def test_selects_highest_confidence(self) -> None:
        balls = [
            _ball(10, 10, confidence=0.5),
            _ball(20, 20, confidence=0.9),
            _ball(30, 30, confidence=0.7),
        ]
        best = select_best_ball(balls)
        assert best is not None
        assert best.confidence == 0.9
        assert best.center.x == 20.0

    def test_empty_list_returns_none(self) -> None:
        assert select_best_ball([]) is None

    def test_single_detection(self) -> None:
        best = select_best_ball([_ball(10, 10, confidence=0.8)])
        assert best is not None
        assert best.confidence == 0.8

    def test_motion_score_prefers_moving_ball_over_static_false_positive(self) -> None:
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = previous.copy()

        # Static white court artifact appears in both frames.
        previous[18:23, 18:23] = 255
        current[18:23, 18:23] = 255

        # Real moving ball appears only in the current frame.
        current[58:63, 58:63] = 255

        static_false_positive = _ball(20, 20, confidence=0.95)
        moving_ball = _ball(60, 60, confidence=0.35)

        best = select_best_ball(
            [static_false_positive, moving_ball],
            previous_frame=previous,
            current_frame=current,
        )

        assert best is moving_ball

    def test_previous_ball_proximity_breaks_motion_ties(self) -> None:
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = previous.copy()
        current[28:33, 28:33] = 255
        current[78:83, 78:83] = 255

        near_previous_track = _ball(30, 30, confidence=0.5)
        far_from_track = _ball(80, 80, confidence=0.5)
        previous_ball = _ball(28, 28, confidence=0.7)

        best = select_best_ball(
            [far_from_track, near_previous_track],
            previous_frame=previous,
            current_frame=current,
            previous_ball=previous_ball,
        )

        assert best is near_previous_track

    def test_static_court_line_near_previous_ball_does_not_keep_track(self) -> None:
        previous = np.zeros((120, 120, 3), dtype=np.uint8)
        current = previous.copy()

        # A static line-like false positive is present in both frames and is
        # close to the previous selected location.
        previous[48:52, 35:65] = 255
        current[48:52, 35:65] = 255

        # The actual ball has lower confidence, but it changes the frame.
        current[78:83, 78:83] = 255

        static_line = _ball_box(35, 48, 65, 52, confidence=0.92)
        moving_ball = _ball(80, 80, confidence=0.28)
        previous_ball = _ball(50, 50, confidence=0.9)

        best = select_best_ball(
            [static_line, moving_ball],
            previous_frame=previous,
            current_frame=current,
            previous_ball=previous_ball,
        )

        assert best is moving_ball

    def test_elongated_box_is_penalized_without_motion_context(self) -> None:
        line_fragment = _ball_box(10, 20, 50, 24, confidence=0.91)
        rounder_ball = _ball(80, 80, confidence=0.75)

        best = select_best_ball([line_fragment, rounder_ball])

        assert best is rounder_ball

    def test_prefers_near_previous_moving_ball_over_far_moving_false_positive(self) -> None:
        previous = np.zeros((160, 160, 3), dtype=np.uint8)
        current = previous.copy()
        current[58:63, 58:63] = 255
        current[138:143, 138:143] = 255

        near_previous_track = _ball(60, 60, confidence=0.45)
        far_false_positive = _ball(140, 140, confidence=0.9)
        previous_ball = _ball(55, 58, confidence=0.8)

        best = select_best_ball(
            [far_false_positive, near_previous_track],
            previous_frame=previous,
            current_frame=current,
            previous_ball=previous_ball,
            max_proximity_px=40,
        )

        assert best is near_previous_track

    def test_has_local_ball_motion_rejects_static_candidate(self) -> None:
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = previous.copy()
        previous[18:23, 18:23] = 255
        current[18:23, 18:23] = 255
        current[58:63, 58:63] = 255

        static_line = _ball(20, 20, confidence=0.9)
        moving_ball = _ball(60, 60, confidence=0.4)

        assert not has_local_ball_motion(static_line, previous, current)
        assert has_local_ball_motion(moving_ball, previous, current)

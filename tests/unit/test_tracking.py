"""Unit tests for player and ball tracking assignment (``tracking.py``).

All tests use fake ``PlayerDetection`` and ``BallDetection`` objects
(no model weights, no network, no GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis_tracker.detection import BallDetection, BoundingBox, PlayerDetection
from tennis_tracker.tracking import (
    PlayerTracker,
    filter_static_ball_hotspots,
    find_static_ball_hotspots,
    has_local_ball_motion,
    refine_ball_sequence,
    select_best_ball,
)
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

    def test_high_confidence_bystander_does_not_steal_track(self) -> None:
        """After initialization, proximity beats confidence for extra people."""
        tracker = PlayerTracker(max_match_distance=80.0)
        tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.8),
        ])

        result = tracker.update([
            _player(2, 0, 12, 20, confidence=0.7),
            _player(98, 0, 108, 20, confidence=0.7),
            _player(300, 0, 310, 20, confidence=0.99),
        ])

        assert result.player_a is not None
        assert result.player_b is not None
        assert result.player_a.bbox.bottom_center.x == pytest.approx(7.0)
        assert result.player_b.bbox.bottom_center.x == pytest.approx(103.0)
        assert result.diagnostics.has_flag("ambigous_player_id")

    def test_far_detections_do_not_swap_existing_tracks(self) -> None:
        """Detections beyond max distance should not replace the tracked players."""
        tracker = PlayerTracker(max_match_distance=40.0)
        tracker.update([
            _player(0, 0, 10, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.8),
        ])

        result = tracker.update([
            _player(250, 0, 260, 20, confidence=0.99),
            _player(320, 0, 330, 20, confidence=0.98),
        ])

        assert result.player_a is None
        assert result.player_b is None

    def test_reacquires_player_b_after_single_player_start(self) -> None:
        """Starting with 1 player must still acquire player B when 2 appear."""
        tracker = PlayerTracker()
        f1 = tracker.update([_player(10, 0, 20, 20, confidence=0.9)])
        assert f1.player_a is not None
        assert f1.player_b is None

        f2 = tracker.update([
            _player(12, 0, 22, 20, confidence=0.9),
            _player(200, 0, 210, 20, confidence=0.85),
        ])
        assert f2.player_a is not None
        assert f2.player_b is not None
        assert f2.player_a.bbox.bottom_center.x == pytest.approx(17.0)
        assert f2.player_b.bbox.bottom_center.x == pytest.approx(205.0)

    def test_reacquires_player_b_after_long_absence(self) -> None:
        """Player B should be re-acquired even after expiring from state."""
        tracker = PlayerTracker(max_missing_frames=3)
        tracker.update([
            _player(10, 0, 20, 20, confidence=0.9),
            _player(200, 0, 210, 20, confidence=0.85),
        ])
        for _ in range(4):
            f_miss = tracker.update([_player(12, 0, 22, 20, confidence=0.9)])
            assert f_miss.player_a is not None
            assert f_miss.player_b is None

        f_back = tracker.update([
            _player(14, 0, 24, 20, confidence=0.9),
            _player(500, 0, 510, 20, confidence=0.85),
        ])
        assert f_back.player_a is not None
        assert f_back.player_b is not None
        assert f_back.player_a.bbox.bottom_center.x == pytest.approx(19.0)
        assert f_back.player_b.bbox.bottom_center.x == pytest.approx(505.0)

    def test_single_detection_matches_closest_track_b(self) -> None:
        """When only player B remains visible, keep label B instead of swapping to A."""
        tracker = PlayerTracker()
        tracker.update([
            _player(10, 0, 20, 20, confidence=0.9),
            _player(300, 0, 310, 20, confidence=0.85),
        ])
        f2 = tracker.update([_player(304, 0, 314, 20, confidence=0.88)])
        assert f2.player_a is None
        assert f2.player_b is not None
        assert f2.player_b.bbox.bottom_center.x == pytest.approx(309.0)
        assert f2.diagnostics.has_flag("missing_player_a")

    def test_one_missing_player_does_not_drop_both_tracks(self) -> None:
        """If only one track has a nearby match among >=2 detections, keep that track."""
        tracker = PlayerTracker(max_match_distance=50.0)
        tracker.update([
            _player(10, 0, 20, 20, confidence=0.9),
            _player(100, 0, 110, 20, confidence=0.85),
        ])
        f2 = tracker.update([
            _player(14, 0, 24, 20, confidence=0.9),
            _player(400, 0, 410, 20, confidence=0.95),
        ])
        assert f2.player_a is not None
        assert f2.player_a.bbox.bottom_center.x == pytest.approx(19.0)
        assert f2.player_b is None


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
        previous_ball = _ball(76, 76, confidence=0.9)

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

    def test_camera_transform_cancels_panned_static_artifact(self) -> None:
        """A static logo shifted by camera pan should be rejected with camera_transform."""
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = np.zeros((100, 100, 3), dtype=np.uint8)
        # Static white dot shifts +4px in X due to camera pan
        previous[20:25, 20:25] = 255
        current[20:25, 24:29] = 255

        panned_logo = _ball(26, 22, confidence=0.8)
        # Without camera compensation, pixel diff sees apparent motion
        assert has_local_ball_motion(panned_logo, previous, current, min_motion_score=5.0)
        # Transform mapping current -> previous shifts X by -4
        cam_tx = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert not has_local_ball_motion(
            panned_logo,
            previous,
            current,
            min_motion_score=5.0,
            camera_transform=cam_tx,
        )

    def test_returns_none_instead_of_static_line_when_track_exists(self) -> None:
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = previous.copy()
        previous[18:23, 18:23] = 255
        current[18:23, 18:23] = 255

        static_line = _ball(20, 20, confidence=0.95)
        previous_ball = _ball(22, 22, confidence=0.8)

        best = select_best_ball(
            [static_line],
            previous_frame=previous,
            current_frame=current,
            previous_ball=previous_ball,
        )

        assert best is None

    def test_initial_acquisition_requires_motion_when_frames_available(self) -> None:
        previous = np.zeros((100, 100, 3), dtype=np.uint8)
        current = previous.copy()
        previous[18:23, 18:23] = 255
        current[18:23, 18:23] = 255

        static_line = _ball(20, 20, confidence=0.95)

        best = select_best_ball(
            [static_line],
            previous_frame=previous,
            current_frame=current,
        )

        assert best is None

    def test_velocity_prediction_and_racket_penalty_prefer_departing_ball(self) -> None:
        """A departing open-air ball beats a lingering racket detection."""
        previous = np.zeros((700, 800, 3), dtype=np.uint8)
        current = previous.copy()
        current[547:553, 637:643] = 255  # real ball
        current[582:588, 671:677] = 255  # follow-through racket

        real_ball = _ball(640, 549, confidence=0.55)
        racket_fp = _ball(674, 585, confidence=0.55)
        prev_ball = _ball(625, 595, confidence=0.48)
        predicted = PixelPoint(x=644.0, y=539.0)
        racket_box = BoundingBox(x1=660.0, y1=570.0, x2=690.0, y2=605.0)

        best = select_best_ball(
            [racket_fp, real_ball],
            previous_frame=previous,
            current_frame=current,
            previous_ball=prev_ball,
            predicted_ball_center=predicted,
            racket_boxes=[racket_box],
        )

        assert best is real_ball

    def test_refine_ball_sequence_swaps_racket_detour_for_collinear_ball(self) -> None:
        """Bidirectional refinement fixes both mid-flight and start-of-run racket swaps."""
        # Scenario matching f=238..240 (racket detour at t=1)
        b0 = _ball(625, 595, confidence=0.48)
        racket_1 = _ball(674, 585, confidence=0.53)
        real_1 = _ball(640, 549, confidence=0.55)
        b2 = _ball(654, 507, confidence=0.54)

        refined = refine_ball_sequence(
            [b0, racket_1, b2],
            [[b0], [racket_1, real_1], [b2]],
        )
        assert refined[1] is real_1

        # Scenario matching f=151..154 (start-of-run racket at t=1, real ball at t=1)
        racket_start = _ball(1171, 656, confidence=0.20)
        real_start = _ball(1178, 564, confidence=0.63)
        b_next1 = _ball(1149, 524, confidence=0.67)
        b_next2 = _ball(1120, 484, confidence=0.70)

        refined2 = refine_ball_sequence(
            [None, racket_start, b_next1, b_next2],
            [[], [racket_start, real_start], [b_next1], [b_next2]],
        )
        assert refined2[1] is real_start

    def test_refine_ball_sequence_prunes_isolated_kink_but_keeps_bounce(self) -> None:
        """Isolated 1-frame sideways spike is dropped; real bounce reversal is preserved."""
        # Smooth flight with 1-frame 80px sideways racket spike at t=2
        s0 = _ball(100, 100)
        s1 = _ball(120, 120)
        spike = _ball(210, 110)  # 70px off midpoint (140, 140)
        s3 = _ball(160, 160)
        s4 = _ball(180, 180)
        refined = refine_ball_sequence(
            [s0, s1, spike, s3, s4], [[x] for x in (s0, s1, spike, s3, s4)]
        )
        assert refined[2] is None

        # Real bounce cusp at t=2 (falling into y=320 then rising sharply)
        f0 = _ball(100, 280)
        f1 = _ball(110, 300)
        bounce = _ball(120, 320)
        f3 = _ball(130, 290)
        f4 = _ball(140, 260)
        refined_b = refine_ball_sequence(
            [f0, f1, bounce, f3, f4], [[x] for x in (f0, f1, bounce, f3, f4)]
        )
        assert refined_b[2] is bounce

    def test_refinement_repairs_short_detached_track_across_missing_frames(self) -> None:
        before = [_ball(100, 300), _ball(112, 290)]
        after = [_ball(184, 230), _ball(196, 220)]
        wrong = [_ball(240, 145, confidence=0.9), _ball(242, 145, confidence=0.9)]
        racket = _ball(148, 260, confidence=0.5)
        actual = _ball(148, 268, confidence=0.8)
        selected = [*before, None, *wrong, None, None, *after]
        candidates = [[d] if d is not None else [] for d in selected]
        candidates[4].extend((racket, actual))
        racket_boxes = [[] for _ in selected]
        racket_boxes[4].append(BoundingBox(147, 259, 149, 261))

        refined = refine_ball_sequence(selected, candidates, racket_boxes_per_frame=racket_boxes)

        assert refined == [*before, None, None, actual, None, None, *after]

    def test_static_hotspot_mask_removes_persistent_wall_logo_keeps_moving_ball(self) -> None:
        """Repeated detections on a static logo are masked while moving balls survive."""
        frames_dets = []
        for f in range(30):
            # Static wall logo drifting slightly with camera pan
            logo = _ball(910.0 + 0.1 * f, 120.0 + 0.05 * f, confidence=0.55)
            # Real moving ball crossing the court
            moving = _ball(400.0 + 20.0 * f, 600.0 - 12.0 * f, confidence=0.65)
            frames_dets.append([logo, moving])

        hotspots = find_static_ball_hotspots(frames_dets, min_frame_hits=15)
        assert len(hotspots) == 1
        assert hotspots[0][0] == pytest.approx(911.5, abs=5.0)
        assert hotspots[0][1] == pytest.approx(120.7, abs=5.0)

        cleaned = filter_static_ball_hotspots(frames_dets[10], hotspots)
        assert len(cleaned) == 1
        assert cleaned[0].center.x == pytest.approx(600.0)

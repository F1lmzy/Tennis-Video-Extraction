"""Unit tests for hybrid 3D ball elevation / parallax correction (``ballistics.py``)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tennis_tracker.ballistics import (
    _detect_anchors,
    correct_ball_elevation,
    correct_ball_elevation_with_heights,
    estimate_camera_position,
)
from tennis_tracker.types import OutputRow


def _make_row(
    frame_index: int,
    *,
    near_xy: tuple[float, float] = (0.0, -13.0),
    far_xy: tuple[float, float] = (0.0, 12.0),
    ball_xy: tuple[float, float] | None = None,
    ball_px: tuple[float, float] | None = None,
) -> OutputRow:
    return OutputRow(
        frame_index=frame_index,
        time_s=frame_index / 30.0,
        player_a_x_m=near_xy[0],
        player_a_y_m=near_xy[1],
        player_a_pixel_x=960.0,
        player_a_pixel_y=850.0,
        player_a_confidence=0.9,
        player_b_x_m=far_xy[0],
        player_b_y_m=far_xy[1],
        player_b_pixel_x=960.0,
        player_b_pixel_y=250.0,
        player_b_confidence=0.9,
        ball_x_m=ball_xy[0] if ball_xy else None,
        ball_y_m=ball_xy[1] if ball_xy else None,
        ball_pixel_x=ball_px[0] if ball_px else None,
        ball_pixel_y=ball_px[1] if ball_px else None,
        ball_confidence=0.8 if ball_xy else None,
        court_confidence=1.0,
        homography_valid=True,
        diagnostics="",
    )


def _project_airborne(
    x: float, y: float, z: float, cam: tuple[float, float, float]
) -> tuple[float, float]:
    xc, yc, zc = cam
    scale = zc / (zc - z)
    return xc + scale * (x - xc), yc + scale * (y - yc)


class TestEstimateCameraPosition:
    def test_recovers_synthetic_pinhole_camera_pose(self) -> None:
        w, h = 1920, 1080
        f_px = (w / 2.0) / np.tan(np.deg2rad(44.0 / 2.0))
        K = np.array([[f_px, 0.0, w / 2.0], [0.0, f_px, h / 2.0], [0.0, 0.0, 1.0]])

        # Camera at (0, -32, 10) looking at court center (0, 0, 0)
        cam_true = np.array([0.0, -32.0, 10.0])
        z_axis = (np.array([0.0, 0.0, 0.0]) - cam_true)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.vstack([x_axis, y_axis, z_axis])
        tvec = -R @ cam_true
        rvec, _ = cv2.Rodrigues(R)

        court_pts = [
            (-5.485, -11.885),
            (5.485, -11.885),
            (-5.485, 11.885),
            (5.485, 11.885),
            (-4.115, -6.4),
            (4.115, -6.4),
            (-4.115, 6.4),
            (4.115, 6.4),
        ]
        obj = np.array([[x, y, 0.0] for x, y in court_pts], dtype=np.float64)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        pixel_pts = [(float(p[0]), float(p[1])) for p in proj.reshape(-1, 2)]

        est = estimate_camera_position(pixel_pts, court_pts, w, h)
        assert est is not None
        assert est[0] == pytest.approx(0.0, abs=0.5)
        assert est[1] == pytest.approx(-32.0, abs=1.5)
        assert est[2] == pytest.approx(10.0, abs=0.8)


class TestCorrectBallElevation:
    @pytest.mark.parametrize(
        ("player_depth", "second_exit_vx", "missing_after_contact", "kind"),
        [
            (-13.0, -12.0, False, "near_bounce"),
            (-13.0, 24.0, False, "near_hit"),
            (-8.0, -12.0, False, "near_hit"),
            (-13.0, -12.0, True, "near_hit"),
        ],
        ids=["distant-bounce", "racket-reversal", "near-player", "uncertain-exit"],
    )
    def test_near_court_contact_with_both_hit_and_bounce_evidence(
        self,
        player_depth: float,
        second_exit_vx: float,
        missing_after_contact: bool,
        kind: str,
    ) -> None:
        # A depth minimum can be claimed as a hit even when the visible ball
        # bounces well in front of the player. Inspect its motion on both sides.
        contact = 20
        rows = []
        for f in range(37):
            t = f - contact
            if t <= 0:
                depth, x, y = -5.0 - 0.8 * t, 776.0 - 14.0 * t, 551.0 + 14.0 * t
            else:
                depth, x, y = -5.0 + 0.2 * t, 764.0 + second_exit_vx * (t - 1), 551.0 - 6.0 * t
            visible = not (missing_after_contact and t in (1, 2))
            rows.append(
                _make_row(
                    f,
                    near_xy=(-2.5, player_depth),
                    ball_xy=(-2.0, depth) if visible else None,
                    ball_px=(x, y) if visible else None,
                )
            )

        anchor = _detect_anchors(rows, (0.0, -33.0, 10.0))[contact]
        assert anchor[0] == kind
        if kind == "near_bounce":
            assert anchor[1:] == (-5.0, 0.0)
        else:
            assert anchor[1] == player_depth

    @pytest.mark.parametrize("approach", [-2.0, 0.2])
    def test_detects_far_bounce_while_ball_image_is_still_rising(self, approach: float) -> None:
        contact = 10
        rows = [
            _make_row(
                i,
                ball_xy=(1.0, 7.0 + (0.1 if i <= contact else 0.6) * (i - contact)),
                ball_px=(
                    900.0 + 2.0 * i,
                    335.0 + (approach if i <= contact else -12.0) * (i - contact),
                ),
            )
            for i in range(19)
        ]
        assert _detect_anchors(rows, (0.0, -33.0, 10.0))[contact] == ("far_bounce", 7.0, 0.0)

    def test_smooth_far_apex_does_not_become_a_racket_hit(self) -> None:
        rows = [
            _make_row(i, ball_xy=(1.0, 17.0 - 0.03 * (i - 15) ** 2),
                      ball_px=(900.0 + 5.0 * i, 210.0 + 0.35 * (i - 15) ** 2))
            for i in range(31)
        ]
        assert _detect_anchors(rows, (0.0, -33.0, 10.0)) == {}

    def test_near_racket_contact_does_not_require_a_depth_minimum(self) -> None:
        rows = []
        for i in range(30):
            t = i - 10
            depth = -8.5 + (0.05 if t <= 0 else 1.1) * t
            x = 750.0 + (-10 if t <= 0 else 15) * t
            y = 690.0 + (1 if t <= 0 else -32) * t
            rows.append(_make_row(i, ball_xy=(0.0, depth), ball_px=(x, y)))
        assert _detect_anchors(rows, (0.0, -33.0, 10.0))[10][0] == "near_hit"

    def test_far_bounce_and_hit_can_be_four_frames_apart(self) -> None:
        rows = []
        for i in range(22):
            if i <= 8:
                depth, x, y = 10.0 - 0.2 * (i - 8), 900.0 + 4 * i, 320.0 + 2 * (i - 8)
            elif i <= 12:
                depth = 10.0 + 1.6 * (i - 8)
                x, y = 932.0 + 2 * (i - 8), 320.0 - 10 * (i - 8)
            else:
                depth, x, y = 16.4 - 0.8 * (i - 12), 940.0 - 6 * (i - 12), 280.0 + 4 * (i - 12)
            rows.append(_make_row(i, ball_xy=(1.0, depth), ball_px=(x, y)))

        corrected, heights = correct_ball_elevation_with_heights(rows, (0.0, -33.0, 10.0))

        assert heights[8] == 0.0
        assert 0.0 < heights[10] < heights[12]
        assert corrected[12].ball_y_m == pytest.approx(12.0)

    def test_reconstructs_short_far_return_to_bounce_on_same_half(self) -> None:
        rows = []
        for i in range(32):
            if i <= 8:
                depth, x, y = 20.0 + 0.2 * i, 910.0 + 2 * i, 230.0 - 5 * i
            elif i <= 25:
                depth, x, y = 21.6 - 1.1 * (i - 8), 926.0 - 5 * (i - 8), 190.0 + 13 * (i - 8)
            else:
                depth, x, y = 2.9 + 0.1 * (i - 25), 841.0 - 5 * (i - 25), 411.0 - 7 * (i - 25)
            rows.append(_make_row(i, ball_xy=(1.0, depth), ball_px=(x, y)))

        corrected, heights = correct_ball_elevation_with_heights(rows, (0.0, -33.0, 10.0))

        assert corrected[8].ball_y_m == pytest.approx(12.0)
        assert 0.0 < heights[16] < heights[8]
        assert heights[25] == 0.0
        assert corrected[25].ball_y_m == pytest.approx(2.9)

    def test_removes_behind_far_player_parallax_while_preserving_bounce(self) -> None:
        cam = (0.0, -33.0, 10.0)
        rows: list[OutputRow] = []

        # Build a realistic rally:
        # f=0..10: approaching far bounce at f=10 (y=7.0, z=0)
        # f=10..20: rising from bounce (y=7.0, z=0) to far hit at f=20 (y=12.0, z=1.2)
        # f=20..30: returning toward net (y=12.0 -> 2.0, z=1.2 -> 0.5) to near bounce at f=30
        for f in range(31):
            if f <= 10:
                t = f / 10.0
                y_true = -3.0 + 10.0 * t
                z_true = 1.5 * (1.0 - t)
                px_y = 260.0 + 60.0 * t  # falling toward bounce max=320 at f=10
            elif f <= 20:
                t = (f - 10) / 10.0
                y_true = 7.0 + 5.0 * t
                z_true = 1.2 * t
                px_y = 320.0 - 120.0 * t  # sharply rising from bounce to hit at 200
            else:
                t = (f - 20) / 10.0
                y_true = 12.0 - 19.0 * t
                z_true = 1.2 * (1.0 - t)
                px_y = 200.0 + 250.0 * t  # returning toward near bounce at f=30
            bx_proj, by_proj = _project_airborne(0.5, y_true, z_true, cam)
            rows.append(_make_row(f, ball_xy=(bx_proj, by_proj), ball_px=(960.0, px_y)))

        # Add 2 post-bounce frames after f=30 so f=30 is detected as a bounce cusp
        for k in (1, 2):
            bx_proj, by_proj = _project_airborne(0.5, -7.0 - k, 0.3 * k, cam)
            rows.append(
                _make_row(30 + k, ball_xy=(bx_proj, by_proj), ball_px=(960.0, 450.0 - 15 * k))
            )

        # Sanity check: raw projection at f=20 is >5m behind far player (y=12.0)
        assert rows[20].ball_y_m is not None
        assert rows[20].ball_y_m > 17.0

        corrected, heights = correct_ball_elevation_with_heights(rows, camera_pos=cam)

        # Raw input must not be mutated
        assert rows[20].ball_y_m > 17.0
        # Bounce at f=10 (z=0) stays exact
        assert corrected[10].ball_y_m == pytest.approx(7.0, abs=0.05)
        assert heights[10] == pytest.approx(0.0, abs=0.05)
        # Far hit at f=20 is pulled right onto the far player's court depth (12.0m)
        assert corrected[20].ball_y_m == pytest.approx(12.0, abs=0.1)
        assert heights[20] is not None and 0.8 <= heights[20] <= 1.6
        # No frame in the rally overshoots behind the far player!
        for f in range(10, 30):
            assert corrected[f].ball_y_m is not None
            assert corrected[f].ball_y_m <= 12.05

    def test_unreturned_winner_flying_past_player_is_not_clamped(self) -> None:
        cam = (0.0, -33.0, 10.0)
        # Ball flies continuously past far player (y=12.0) to y=18.0 with NO reversal
        rows = [
            _make_row(
                f,
                ball_xy=(2.0, 5.0 + 0.8 * f),
                ball_px=(980.0, 320.0 - 8.0 * f),
            )
            for f in range(20)
        ]

        corrected = correct_ball_elevation(rows, camera_pos=cam)

        # No hit reversal occurred -> trajectory passes behind player unmodified
        assert corrected[-1].ball_y_m == pytest.approx(rows[-1].ball_y_m)

    def test_does_not_bridge_incompatible_far_bounce_to_near_bounce(self) -> None:
        """A missed far hit must not linearly pull ball from far_bounce to near_bounce."""
        cam = (0.0, -33.0, 10.0)
        rows: list[OutputRow] = []
        # f=0..5: falling into far_bounce at f=5 (y=9.0)
        # f=6..15: rising toward far baseline (y=10..15) with NO reversal (far_hit missed)
        # f=16..25: falling into near_bounce at f=25 (y=-7.0)
        for f in range(28):
            if f <= 5:
                y = 5.0 + 0.8 * f
                px_y = 300.0 + 4.0 * f
            elif f <= 15:
                y = 9.0 + 0.6 * (f - 5)
                px_y = 320.0 - 10.0 * (f - 5)
            elif f <= 25:
                y = 15.0 - 2.2 * (f - 15)
                px_y = 220.0 + 25.0 * (f - 15)
            else:
                y = -7.0 - 0.5 * (f - 25)
                px_y = 470.0 - 12.0 * (f - 25)
            rows.append(_make_row(f, ball_xy=(0.0, y), ball_px=(960.0, px_y)))

        # Move far player away so no far_hit anchor is triggered at f=15
        rows = [
            _make_row(
                r.frame_index,
                near_xy=(0.0, -14.0),
                far_xy=(8.0, 12.0),
                ball_xy=(r.ball_x_m or 0.0, r.ball_y_m or 0.0),
                ball_px=(r.ball_pixel_x or 960.0, r.ball_pixel_y or 300.0),
            )
            for r in rows
        ]
        corrected = correct_ball_elevation(rows, camera_pos=cam)
        # f=10 should NOT be pulled backwards toward the net by a far_bounce->near_bounce bridge
        assert corrected[10].ball_y_m == pytest.approx(rows[10].ball_y_m)

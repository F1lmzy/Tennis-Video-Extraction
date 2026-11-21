"""Annotated video renderer with source-frame overlays and top-view court panel.

Produces annotated frames containing:
- Resized source frame with player boxes, ball trail, court keypoints,
  diagnostics text, and missing-detection warning/omission behavior.
- A side top-view court panel showing real-time player/ball positions
  on the court plane in meter coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from tennis_tracker.court import (
    DOUBLES_LENGTH,
    DOUBLES_WIDTH,
    SERVICE_LINE_DISTANCE,
    SINGLES_WIDTH,
)
from tennis_tracker.types import OutputRow
from tennis_tracker.video import iter_frames, read_video_metadata, write_video

# ── Colours (BGR) ──────────────────────────────────────────────────────

_COLOR_PLAYER_A: tuple[int, int, int] = (255, 0, 0)       # Blue
_COLOR_PLAYER_B: tuple[int, int, int] = (0, 0, 255)       # Red
_COLOR_BALL: tuple[int, int, int] = (0, 230, 255)         # Yellow
_COLOR_BALL_TRAIL: tuple[int, int, int] = (0, 180, 200)   # Darker yellow for trail
_COLOR_COURT_KEYPOINT: tuple[int, int, int] = (0, 255, 0) # Green
_COLOR_DIAGNOSTICS_TEXT: tuple[int, int, int] = (0, 0, 200)  # Red text
_COLOR_FRAME_NUM: tuple[int, int, int] = (200, 200, 200)   # Light grey
_COLOR_COURT_LINE: tuple[int, int, int] = (255, 255, 255) # White
_COLOR_COURT_FILL: tuple[int, int, int] = (60, 120, 60)   # Dark green court
_COLOR_PLAYER_MARKER: tuple[int, int, int] = (0, 165, 255)  # Orange for panel markers
_COLOR_BALL_MARKER: tuple[int, int, int] = (0, 255, 255)  # Cyan for ball marker
_COLOR_MISSING_TEXT: tuple[int, int, int] = (0, 0, 255)   # Red
_COLOR_NET: tuple[int, int, int] = (200, 200, 200)        # Light grey net line

# ── Defaults ───────────────────────────────────────────────────────────

_DEFAULT_TARGET_HEIGHT: int = 540
"""Resized source frame height in pixels (legacy; layout now uses output width)."""

_DEFAULT_OUTPUT_WIDTH: int = 1280
"""Output video width in pixels; height is derived to keep a 16:9 aspect ratio."""

_COLOR_CANVAS: tuple[int, int, int] = (24, 24, 24)
"""Dark background fill for letterbox margins and panel gutters."""

_DEFAULT_PANEL_WIDTH: int = 300
"""Width of the top-view court panel in pixels."""

_DEFAULT_PANEL_3D_WIDTH: int = 360
"""Width of the 3D corner-view court panel in pixels."""

_DEFAULT_BALL_TRAIL_LENGTH: int = 15
"""Number of recent ball positions to draw as a trail."""

_NET_HEIGHT_POST_M: float = 1.07
_NET_HEIGHT_CENTER_M: float = 0.914
_NET_POST_OFFSET_M: float = 0.914
_PLAYER_HEIGHT_M: float = 1.80
_COLOR_3D_SURROUND: tuple[int, int, int] = (45, 85, 45)
_COLOR_3D_COURT: tuple[int, int, int] = (145, 95, 55)     # Hard-court blue (BGR)
_COLOR_3D_SHADOW: tuple[int, int, int] = (25, 25, 25)
_COLOR_3D_DROP_LINE: tuple[int, int, int] = (120, 180, 180)


_DEFAULT_PANEL_RUNOFF_X_M: float = 2.0
"""Extra horizontal run-off margin (meters per side) shown on the 2D court panel."""

_DEFAULT_PANEL_RUNOFF_Y_M: float = 5.5
"""Extra baseline run-off margin (meters per end) shown on the 2D court panel."""

_DEFAULT_TRAIL_MAX_MISSING_FRAMES: int = 5
"""Consecutive missing frames before the ball trail is cleared completely."""


# ── Panel helper: court-meter → panel-pixel ────────────────────────────


def _court_to_panel(
    x_m: float,
    y_m: float,
    panel_width: int,
    panel_height: int,
    court_width_m: float = DOUBLES_WIDTH,
    padding_px: int = 20,
    runoff_x_m: float = _DEFAULT_PANEL_RUNOFF_X_M,
    runoff_y_m: float = _DEFAULT_PANEL_RUNOFF_Y_M,
) -> tuple[int, int]:
    """Convert court-meter coordinates to panel pixel coordinates.

    The panel Y-axis is inverted so that +Y (far baseline) goes upward,
    matching typical top-down orientation. Includes baseline and sideline
    run-off margins so players standing behind the baseline remain visible.
    """
    drawable_width = max(1, panel_width - 2 * padding_px)
    drawable_height = max(1, panel_height - 2 * padding_px)
    scale_x = drawable_width / (court_width_m + 2.0 * runoff_x_m)
    scale_y = drawable_height / (DOUBLES_LENGTH + 2.0 * runoff_y_m)
    # Use a uniform scale to preserve aspect ratio
    scale = min(scale_x, scale_y)
    cx = panel_width / 2.0
    cy = panel_height / 2.0
    px = cx + x_m * scale
    py = cy - y_m * scale  # Inverted Y: far baseline → top
    return int(round(px)), int(round(py))


# ── Panel drawing ──────────────────────────────────────────────────────


def _draw_court_panel(
    panel: np.ndarray,
    output_row: Optional[OutputRow],
    *,
    ball_trail_meters: list[tuple[float, float]],
    panel_width: int,
    panel_height: int,
    court_view: str = "doubles",
) -> None:
    """Draw court lines and player/ball markers on the top-view panel.

    Mutates *panel* in place.
    """
    # ── Helper to convert court meter → panel pixel ──
    active_width = SINGLES_WIDTH if court_view == "singles" else DOUBLES_WIDTH
    active_half = active_width / 2.0
    singles_half = SINGLES_WIDTH / 2.0

    def _p(x_m: float, y_m: float) -> tuple[int, int]:
        return _court_to_panel(
            x_m,
            y_m,
            panel_width,
            panel_height,
            court_width_m=active_width,
            padding_px=20,
        )

    # ── Active court area fill ──
    cv2.rectangle(
        panel,
        _p(-active_half, -DOUBLES_LENGTH / 2),
        _p(active_half, DOUBLES_LENGTH / 2),
        _COLOR_COURT_FILL,
        -1,
    )

    # ── Active outline ──
    near_left = _p(-active_half, -DOUBLES_LENGTH / 2)
    far_right = _p(active_half, DOUBLES_LENGTH / 2)
    cv2.rectangle(panel, near_left, far_right, _COLOR_COURT_LINE, 1)

    # ── Singles sidelines when the full doubles court is displayed ──
    if court_view != "singles":
        cv2.line(
            panel,
            _p(-singles_half, -DOUBLES_LENGTH / 2),
            _p(-singles_half, DOUBLES_LENGTH / 2),
            _COLOR_COURT_LINE,
            1,
        )
        cv2.line(
            panel,
            _p(singles_half, -DOUBLES_LENGTH / 2),
            _p(singles_half, DOUBLES_LENGTH / 2),
            _COLOR_COURT_LINE,
            1,
        )

    # ── Net ──
    cv2.line(
        panel,
        _p(-active_half, 0),
        _p(active_half, 0),
        _COLOR_NET,
        2,
    )

    # ── Service lines ──
    cv2.line(
        panel,
        _p(-singles_half, -SERVICE_LINE_DISTANCE),
        _p(singles_half, -SERVICE_LINE_DISTANCE),
        _COLOR_COURT_LINE,
        1,
    )
    cv2.line(
        panel,
        _p(-singles_half, SERVICE_LINE_DISTANCE),
        _p(singles_half, SERVICE_LINE_DISTANCE),
        _COLOR_COURT_LINE,
        1,
    )

    # ── Centre service line ──
    cv2.line(
        panel,
        _p(0, -SERVICE_LINE_DISTANCE),
        _p(0, SERVICE_LINE_DISTANCE),
        _COLOR_COURT_LINE,
        1,
    )

    # ── Centre marks ──
    cv2.line(
        panel,
        _p(0, -DOUBLES_LENGTH / 2),
        _p(0, -DOUBLES_LENGTH / 2 + 0.3),
        _COLOR_COURT_LINE,
        1,
    )
    cv2.line(
        panel,
        _p(0, DOUBLES_LENGTH / 2 - 0.3),
        _p(0, DOUBLES_LENGTH / 2),
        _COLOR_COURT_LINE,
        1,
    )

    # ── Ball trail ──
    for i, (bx_m, by_m) in enumerate(ball_trail_meters):
        alpha = (i + 1) / len(ball_trail_meters)
        radius = max(2, int(4 * alpha))
        trail_bgr = tuple(int(c * alpha) for c in _COLOR_BALL_TRAIL)
        cv2.circle(panel, _p(bx_m, by_m), radius, trail_bgr, -1)

    if output_row is None:
        return

    # ── Player A (blue/metre) ──
    if output_row.player_a_x_m is not None and output_row.player_a_y_m is not None:
        pa_px, pa_py = _p(output_row.player_a_x_m, output_row.player_a_y_m)
        cv2.circle(panel, (pa_px, pa_py), 6, _COLOR_PLAYER_A, -1)
        cv2.putText(
            panel, "A", (pa_px + 8, pa_py + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_PLAYER_A, 1,
        )

    # ── Player B (red) ──
    if output_row.player_b_x_m is not None and output_row.player_b_y_m is not None:
        pb_px, pb_py = _p(output_row.player_b_x_m, output_row.player_b_y_m)
        cv2.circle(panel, (pb_px, pb_py), 6, _COLOR_PLAYER_B, -1)
        cv2.putText(
            panel, "B", (pb_px + 8, pb_py + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_PLAYER_B, 1,
        )

    # ── Ball marker ──
    if output_row.ball_x_m is not None and output_row.ball_y_m is not None:
        b_px, b_py = _p(output_row.ball_x_m, output_row.ball_y_m)
        cv2.circle(panel, (b_px, b_py), 4, _COLOR_BALL_MARKER, -1)


# ── 3D corner-view projection & drawing ────────────────────────────────


def _build_3d_corner_camera(
    panel_width: int,
    panel_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(camera_pos, R_world_to_cam, K)`` for an elevated 3D corner view."""
    cam_pos = np.array([21.0, -27.0, 15.5], dtype=np.float64)
    target = np.array([0.0, -1.0, 0.5], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R = np.vstack([right, down, forward])

    # Auto-fit the focal length and principal point so the court surround (with
    # headroom for the net and airborne ball) fills the panel at any aspect ratio.
    half_w = DOUBLES_WIDTH / 2.0 + 2.5
    half_l = DOUBLES_LENGTH / 2.0 + 5.0
    corners = np.array(
        [
            [-half_w, -half_l, 0.0],
            [half_w, -half_l, 0.0],
            [half_w, half_l, 0.0],
            [-half_w, half_l, 0.0],
            [-half_w, -half_l, 3.0],
            [half_w, half_l, 3.0],
        ],
        dtype=np.float64,
    )
    pc = (R @ (corners - cam_pos).T).T
    depth = np.maximum(pc[:, 2], 1e-3)
    uv = pc[:, :2] / depth[:, None]
    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)
    pad = 14.0
    span_u = max(u_max - u_min, 1e-6)
    span_v = max(v_max - v_min, 1e-6)
    f_px = min((panel_width - 2 * pad) / span_u, (panel_height - 2 * pad) / span_v)
    cx = panel_width * 0.5 - f_px * (u_min + u_max) * 0.5
    cy = panel_height * 0.5 - f_px * (v_min + v_max) * 0.5
    K = np.array(
        [
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return cam_pos, R, K


def _project_3d_to_panel(
    x_m: float,
    y_m: float,
    z_m: float,
    panel_width: int,
    panel_height: int,
    *,
    camera: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> tuple[int, int]:
    """Project a 3D court point ``(x_m, y_m, z_m)`` into 3D corner-panel pixels."""
    cam_pos, R, K = camera if camera is not None else _build_3d_corner_camera(
        panel_width, panel_height
    )
    pt_cam = R @ (np.array([x_m, y_m, z_m], dtype=np.float64) - cam_pos)
    depth = max(float(pt_cam[2]), 1e-3)
    uvw = K @ (pt_cam / depth)
    return int(round(float(uvw[0]))), int(round(float(uvw[1])))


def _draw_court_3d_panel(
    panel: np.ndarray,
    output_row: Optional[OutputRow],
    *,
    ball_z_m: Optional[float] = None,
    ball_trail_3d: Sequence[tuple[float, float, float]] = (),
    panel_width: int,
    panel_height: int,
    court_view: str = "doubles",
) -> None:
    """Draw a 3D corner perspective court view with net, players, and 3D ball arc.

    Mutates *panel* in place.
    """
    camera = _build_3d_corner_camera(panel_width, panel_height)

    def _p3(x_m: float, y_m: float, z_m: float = 0.0) -> tuple[int, int]:
        return _project_3d_to_panel(
            x_m, y_m, z_m, panel_width, panel_height, camera=camera
        )

    active_width = SINGLES_WIDTH if court_view == "singles" else DOUBLES_WIDTH
    active_half = active_width / 2.0
    singles_half = SINGLES_WIDTH / 2.0
    half_len = DOUBLES_LENGTH / 2.0

    # ── Surround apron (run-off area) ──
    surround_poly = np.array(
        [
            _p3(-active_half - 2.5, -half_len - 5.0),
            _p3(active_half + 2.5, -half_len - 5.0),
            _p3(active_half + 2.5, half_len + 5.0),
            _p3(-active_half - 2.5, half_len + 5.0),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(panel, surround_poly, _COLOR_3D_SURROUND)

    # ── Playable court surface ──
    court_poly = np.array(
        [
            _p3(-active_half, -half_len),
            _p3(active_half, -half_len),
            _p3(active_half, half_len),
            _p3(-active_half, half_len),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(panel, court_poly, _COLOR_3D_COURT)

    # ── Court lines (z = 0) ──
    court_segments = [
        ((-active_half, -half_len), (active_half, -half_len)),
        ((-active_half, half_len), (active_half, half_len)),
        ((-active_half, -half_len), (-active_half, half_len)),
        ((active_half, -half_len), (active_half, half_len)),
        ((-singles_half, -SERVICE_LINE_DISTANCE), (singles_half, -SERVICE_LINE_DISTANCE)),
        ((-singles_half, SERVICE_LINE_DISTANCE), (singles_half, SERVICE_LINE_DISTANCE)),
        ((0.0, -SERVICE_LINE_DISTANCE), (0.0, SERVICE_LINE_DISTANCE)),
        ((0.0, -half_len), (0.0, -half_len + 0.35)),
        ((0.0, half_len - 0.35), (0.0, half_len)),
    ]
    if court_view != "singles":
        court_segments.extend([
            ((-singles_half, -half_len), (-singles_half, half_len)),
            ((singles_half, -half_len), (singles_half, half_len)),
        ])
    for (x1, y1), (x2, y2) in court_segments:
        cv2.line(panel, _p3(x1, y1, 0.0), _p3(x2, y2, 0.0), _COLOR_COURT_LINE, 1, cv2.LINE_AA)

    # ── 3D Net (posts + sagging top tape + mesh lines) ──
    net_half = active_half + _NET_POST_OFFSET_M
    # Ground shadow line under net
    cv2.line(panel, _p3(-net_half, 0.0, 0.0), _p3(net_half, 0.0, 0.0), _COLOR_3D_SHADOW, 1)
    # Net mesh verticals
    for frac in np.linspace(-1.0, 1.0, 13):
        nx = float(frac * net_half)
        nz = _NET_HEIGHT_CENTER_M + (_NET_HEIGHT_POST_M - _NET_HEIGHT_CENTER_M) * (frac**2)
        cv2.line(panel, _p3(nx, 0.0, 0.0), _p3(nx, 0.0, nz), (130, 130, 130), 1)
    # Net mid-horizontal cable + white top tape
    net_xs = np.linspace(-net_half, net_half, 17)
    top_pts = [
        _p3(
            float(nx),
            0.0,
            _NET_HEIGHT_CENTER_M
            + (_NET_HEIGHT_POST_M - _NET_HEIGHT_CENTER_M) * ((nx / net_half) ** 2),
        )
        for nx in net_xs
    ]
    mid_pts = [_p3(float(nx), 0.0, 0.45) for nx in net_xs]
    cv2.polylines(panel, [np.array(mid_pts, dtype=np.int32)], False, (140, 140, 140), 1)
    cv2.polylines(
        panel, [np.array(top_pts, dtype=np.int32)], False, _COLOR_COURT_LINE, 2, cv2.LINE_AA
    )
    # Net posts
    for side in (-1.0, 1.0):
        cv2.line(
            panel,
            _p3(side * net_half, 0.0, 0.0),
            _p3(side * net_half, 0.0, _NET_HEIGHT_POST_M),
            (220, 220, 220),
            2,
        )

    # ── 3D Ball trail (ground shadows + airborne arc) ──
    if ball_trail_3d:
        prev_air: Optional[tuple[int, int]] = None
        for i, (bx, by, bz) in enumerate(ball_trail_3d):
            alpha = (i + 1) / len(ball_trail_3d)
            g_pt = _p3(bx, by, 0.0)
            a_pt = _p3(bx, by, max(0.0, bz))
            cv2.circle(panel, g_pt, 2, _COLOR_3D_SHADOW, -1)
            trail_bgr = tuple(int(c * alpha) for c in _COLOR_BALL_TRAIL)
            if prev_air is not None:
                cv2.line(panel, prev_air, a_pt, trail_bgr, max(1, int(2 * alpha)), cv2.LINE_AA)
            cv2.circle(panel, a_pt, max(2, int(4 * alpha)), trail_bgr, -1)
            prev_air = a_pt

    # ── Players as 3D vertical pins (foot shadow -> head) ──
    if output_row is not None:
        for label, px_m, py_m, color in (
            ("A", output_row.player_a_x_m, output_row.player_a_y_m, _COLOR_PLAYER_A),
            ("B", output_row.player_b_x_m, output_row.player_b_y_m, _COLOR_PLAYER_B),
        ):
            if px_m is not None and py_m is not None:
                foot = _p3(px_m, py_m, 0.0)
                head = _p3(px_m, py_m, _PLAYER_HEIGHT_M)
                cv2.circle(panel, foot, 4, _COLOR_3D_SHADOW, -1)
                cv2.line(panel, foot, head, color, 2, cv2.LINE_AA)
                cv2.circle(panel, head, 5, color, -1)
                cv2.putText(
                    panel, label, (head[0] + 6, head[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

        # ── Current 3D ball + vertical height drop-line ──
        if output_row.ball_x_m is not None and output_row.ball_y_m is not None:
            bz = max(0.0, ball_z_m) if ball_z_m is not None else 0.0
            g_pt = _p3(output_row.ball_x_m, output_row.ball_y_m, 0.0)
            a_pt = _p3(output_row.ball_x_m, output_row.ball_y_m, bz)
            cv2.circle(panel, g_pt, 3, _COLOR_3D_SHADOW, -1)
            if bz > 0.05:
                cv2.line(panel, g_pt, a_pt, _COLOR_3D_DROP_LINE, 1, cv2.LINE_AA)
            cv2.circle(panel, a_pt, 5, _COLOR_BALL_MARKER, -1)
            cv2.putText(
                panel,
                f"z={bz:.1f}m",
                (10, panel_height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                _COLOR_BALL_MARKER,
                1,
                cv2.LINE_AA,
            )

    # ── Panel header ──
    cv2.putText(
        panel,
        "3D Corner View",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _COLOR_FRAME_NUM,
        1,
        cv2.LINE_AA,
    )


# ── Source frame overlay ──────────────────────────────────────────────


def _draw_source_overlay(
    frame: np.ndarray,
    output_row: Optional[OutputRow],
    *,
    ball_trail_pixels: list[tuple[float, float]],
    frame_index: int,
    frame_count: int,
    diagnostics_text: Optional[str],
) -> None:
    """Draw player/ball/ keypoint markers and diagnostics on the source frame.

    Mutates *frame* in place.
    """
    h, w = frame.shape[:2]

    # ── Frame counter ──
    cv2.putText(
        frame,
        f"{frame_index + 1}/{frame_count}",
        (w - 130, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _COLOR_FRAME_NUM,
        1,
    )

    if output_row is None:
        # No tracking data for this frame
        cv2.putText(
            frame,
            "No tracking data",
            (w // 2 - 80, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _COLOR_MISSING_TEXT,
            2,
        )
        return

    # ── Ball pixel trail ──
    for i, (bx, by) in enumerate(ball_trail_pixels):
        alpha = (i + 1) / len(ball_trail_pixels)
        radius = max(2, int(5 * alpha))
        trail_bgr = tuple(int(c * alpha) for c in _COLOR_BALL_TRAIL)
        cv2.circle(frame, (int(bx), int(by)), radius, trail_bgr, -1)

    # ── Player A ──
    if output_row.player_a_pixel_x is not None and output_row.player_a_pixel_y is not None:
        pa_px = int(output_row.player_a_pixel_x)
        pa_py = int(output_row.player_a_pixel_y)
        # Bounding box indicator (box around approximate area)
        cv2.circle(frame, (pa_px, pa_py), 10, _COLOR_PLAYER_A, 2)
        cv2.putText(
            frame, "A", (pa_px + 12, pa_py + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_PLAYER_A, 2,
        )
    else:
        cv2.putText(
            frame, "A: missing", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_MISSING_TEXT, 1,
        )

    # ── Player B ──
    if output_row.player_b_pixel_x is not None and output_row.player_b_pixel_y is not None:
        pb_px = int(output_row.player_b_pixel_x)
        pb_py = int(output_row.player_b_pixel_y)
        cv2.circle(frame, (pb_px, pb_py), 10, _COLOR_PLAYER_B, 2)
        cv2.putText(
            frame, "B", (pb_px + 12, pb_py + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_PLAYER_B, 2,
        )
    else:
        cv2.putText(
            frame, "B: missing", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_MISSING_TEXT, 1,
        )

    # ── Ball marker ──
    if output_row.ball_pixel_x is not None and output_row.ball_pixel_y is not None:
        b_px = int(output_row.ball_pixel_x)
        b_py = int(output_row.ball_pixel_y)
        cv2.circle(frame, (b_px, b_py), 4, _COLOR_BALL, -1)

    # ── Diagnostics text ──
    if diagnostics_text:
        y_offset = h - 30
        for line in diagnostics_text.split(";"):
            cv2.putText(
                frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_DIAGNOSTICS_TEXT, 1,
            )
            y_offset -= 18


def _scale_row_pixels(row: OutputRow, scale: float) -> OutputRow:
    """Return a copy of *row* with source-frame pixel coordinates scaled.

    CSV rows store coordinates in the original video frame.  The renderer
    resizes the source frame before drawing, so source overlay coordinates
    must be scaled to stay aligned with the resized image.
    """
    def scaled(value: Optional[float]) -> Optional[float]:
        return value * scale if value is not None else None

    return replace(
        row,
        player_a_pixel_x=scaled(row.player_a_pixel_x),
        player_a_pixel_y=scaled(row.player_a_pixel_y),
        player_b_pixel_x=scaled(row.player_b_pixel_x),
        player_b_pixel_y=scaled(row.player_b_pixel_y),
        ball_pixel_x=scaled(row.ball_pixel_x),
        ball_pixel_y=scaled(row.ball_pixel_y),
    )


# ── Main rendering function ───────────────────────────────────────────


@dataclass
class AnnotatedVideoResult:
    """Summary of the annotated video rendering process."""

    frame_count: int
    output_path: str
    output_width: int
    output_height: int


def _even(value: float) -> int:
    """Round to the nearest even integer (codec-friendly dimensions)."""
    return int(round(value / 2.0)) * 2


def _compute_layout(output_width: int, show_3d_view: bool) -> dict[str, int]:
    """Compute a 16:9 top/bottom panel layout for the given output width.

    With the 3D view enabled the frame has two rows: the source video beside the
    2D top-view court on top, and the full-width 3D corner view on the bottom.
    Without it, the source video and 2D court share a single 16:9 row.
    """
    width = _even(max(320, output_width))
    height = _even(width * 9.0 / 16.0)
    if show_3d_view:
        top_h = _even(height * 2.0 / 3.0)
        bot_h = height - top_h
        src_slot_w = _even(top_h * 16.0 / 9.0)
        src_slot_w = min(src_slot_w, width - 160)
        return {
            "width": width,
            "height": height,
            "top_h": top_h,
            "bot_h": bot_h,
            "src_slot_w": src_slot_w,
            "panel2d_w": width - src_slot_w,
            "panel2d_h": top_h,
            "panel3d_w": width,
            "panel3d_h": bot_h,
        }
    panel2d_w = _even(width * 0.24)
    return {
        "width": width,
        "height": height,
        "top_h": height,
        "bot_h": 0,
        "src_slot_w": width - panel2d_w,
        "panel2d_w": panel2d_w,
        "panel2d_h": height,
        "panel3d_w": 0,
        "panel3d_h": 0,
    }


def render_annotated_frames(
    video_path: str | Path,
    output_rows: list[OutputRow],
    *,
    output_width: int = _DEFAULT_OUTPUT_WIDTH,
    ball_trail_length: int = _DEFAULT_BALL_TRAIL_LENGTH,
    court_view: str = "doubles",
    show_3d_view: bool = False,
    ball_heights_m: Optional[Sequence[Optional[float]]] = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield annotated frames in a 16:9 top/bottom layout.

    Each frame stacks the source video (with overlays) and 2D top-view court in
    a top row, and — when *show_3d_view* is set — the 3D corner-view court in a
    bottom row. The source is letterboxed to preserve its aspect ratio.
    """
    layout = _compute_layout(output_width, show_3d_view)
    width = layout["width"]
    top_h = layout["top_h"]
    bot_h = layout["bot_h"]
    src_slot_w = layout["src_slot_w"]
    panel2d_w = layout["panel2d_w"]
    panel2d_h = layout["panel2d_h"]
    panel3d_w = layout["panel3d_w"]
    panel3d_h = layout["panel3d_h"]

    # ── Ball trail buffer ──
    ball_trail_pixels: list[tuple[float, float]] = []
    ball_trail_meters: list[tuple[float, float]] = []
    ball_trail_3d: list[tuple[float, float, float]] = []
    ball_missing_streak = 0

    total_rows = len(output_rows)

    for idx, (frame_index, frame_bgr) in enumerate(
        iter_frames(video_path, max_frames=total_rows)
    ):
        # ── Fit source frame into its slot, preserving aspect ratio ──
        h, w = frame_bgr.shape[:2]
        scale = min(src_slot_w / w, top_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        # ── Get the output row for this frame ──
        raw_output_row: Optional[OutputRow] = (
            output_rows[idx] if idx < total_rows else None
        )
        output_row = (
            _scale_row_pixels(raw_output_row, scale)
            if raw_output_row is not None
            else None
        )
        ball_z = (
            ball_heights_m[idx]
            if ball_heights_m is not None and idx < len(ball_heights_m)
            else None
        )

        # ── Update ball trail (decay and clear during missing streaks) ──
        has_ball_px = (
            output_row is not None
            and output_row.ball_pixel_x is not None
            and output_row.ball_pixel_y is not None
        )
        has_ball_m = (
            output_row is not None
            and output_row.ball_x_m is not None
            and output_row.ball_y_m is not None
        )
        if has_ball_px or has_ball_m:
            ball_missing_streak = 0
            if has_ball_px and output_row is not None:
                assert output_row.ball_pixel_x is not None and output_row.ball_pixel_y is not None
                ball_trail_pixels.append((output_row.ball_pixel_x, output_row.ball_pixel_y))
                if len(ball_trail_pixels) > ball_trail_length:
                    ball_trail_pixels.pop(0)
            if has_ball_m and output_row is not None:
                assert output_row.ball_x_m is not None and output_row.ball_y_m is not None
                ball_trail_meters.append((output_row.ball_x_m, output_row.ball_y_m))
                if len(ball_trail_meters) > ball_trail_length:
                    ball_trail_meters.pop(0)
                z_val = ball_z if ball_z is not None else 0.0
                ball_trail_3d.append((output_row.ball_x_m, output_row.ball_y_m, z_val))
                if len(ball_trail_3d) > ball_trail_length:
                    ball_trail_3d.pop(0)
        else:
            ball_missing_streak += 1
            if ball_missing_streak > _DEFAULT_TRAIL_MAX_MISSING_FRAMES:
                ball_trail_pixels.clear()
                ball_trail_meters.clear()
                ball_trail_3d.clear()
            else:
                if ball_trail_pixels:
                    ball_trail_pixels.pop(0)
                if ball_trail_meters:
                    ball_trail_meters.pop(0)
                if ball_trail_3d:
                    ball_trail_3d.pop(0)

        # ── Draw source overlay ──
        diag_text = output_row.diagnostics if output_row and output_row.diagnostics else None
        _draw_source_overlay(
            resized,
            output_row,
            ball_trail_pixels=ball_trail_pixels,
            frame_index=idx,
            frame_count=total_rows or 1,
            diagnostics_text=diag_text,
        )

        # ── Top row: letterboxed source (left) + 2D top-view court (right) ──
        top_row = np.full((top_h, width, 3), _COLOR_CANVAS, dtype=np.uint8)
        x_off = (src_slot_w - new_w) // 2
        y_off = (top_h - new_h) // 2
        top_row[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        panel2d = np.zeros((panel2d_h, panel2d_w, 3), dtype=np.uint8)
        _draw_court_panel(
            panel2d,
            output_row,
            ball_trail_meters=ball_trail_meters,
            panel_width=panel2d_w,
            panel_height=panel2d_h,
            court_view=court_view,
        )
        top_row[0:panel2d_h, src_slot_w:src_slot_w + panel2d_w] = panel2d

        # ── Bottom row: full-width 3D corner view (when enabled) ──
        if show_3d_view and bot_h > 0:
            panel3d = np.zeros((panel3d_h, panel3d_w, 3), dtype=np.uint8)
            _draw_court_3d_panel(
                panel3d,
                output_row,
                ball_z_m=ball_z,
                ball_trail_3d=ball_trail_3d,
                panel_width=panel3d_w,
                panel_height=panel3d_h,
                court_view=court_view,
            )
            annotated = np.concatenate([top_row, panel3d], axis=0)
        else:
            annotated = top_row

        yield (frame_index, annotated)


def render_annotated_video(
    video_path: str | Path,
    output_rows: list[OutputRow],
    output_video_path: str | Path,
    *,
    output_width: int = _DEFAULT_OUTPUT_WIDTH,
    ball_trail_length: int = _DEFAULT_BALL_TRAIL_LENGTH,
    fps: Optional[float] = None,
    court_view: str = "doubles",
    show_3d_view: bool = False,
    ball_heights_m: Optional[Sequence[Optional[float]]] = None,
) -> AnnotatedVideoResult:
    """Render an annotated video file from source + tracking data.

    Writes a 16:9 MP4 stacking the source video and 2D top-view court in a top
    row, with an optional full-width 3D corner-view court in a bottom row.
    """
    # ── Determine FPS ──
    if fps is None:
        try:
            meta = read_video_metadata(video_path)
            fps = meta.fps
        except (FileNotFoundError, ValueError):
            fps = 30.0

    # ── Stream annotated frames directly to video writer (constant RAM) ──
    frame_count = 0
    oh = 0
    ow = 0

    def _frame_stream() -> Iterator[np.ndarray]:
        nonlocal frame_count, oh, ow
        for _, annotated in render_annotated_frames(
            video_path,
            output_rows,
            output_width=output_width,
            ball_trail_length=ball_trail_length,
            court_view=court_view,
            show_3d_view=show_3d_view,
            ball_heights_m=ball_heights_m,
        ):
            if frame_count == 0:
                oh, ow = annotated.shape[:2]
            frame_count += 1
            yield annotated

    write_video(_frame_stream(), output_video_path, fps=fps)

    return AnnotatedVideoResult(
        frame_count=frame_count,
        output_path=str(output_video_path),
        output_width=ow,
        output_height=oh,
    )

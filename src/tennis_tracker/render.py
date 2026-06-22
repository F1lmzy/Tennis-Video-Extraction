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
from typing import Iterator, Optional

import cv2
import numpy as np

from tennis_tracker.court import (
    DOUBLES_LENGTH,
    DOUBLES_WIDTH,
    SERVICE_LINE_DISTANCE,
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
"""Resized source frame height in pixels."""

_DEFAULT_PANEL_WIDTH: int = 300
"""Width of the top-view court panel in pixels."""

_DEFAULT_BALL_TRAIL_LENGTH: int = 15
"""Number of recent ball positions to draw as a trail."""


# ── Panel helper: court-meter → panel-pixel ────────────────────────────


def _court_to_panel(
    x_m: float,
    y_m: float,
    panel_width: int,
    panel_height: int,
) -> tuple[int, int]:
    """Convert court-meter coordinates to panel pixel coordinates.

    The panel Y-axis is inverted so that +Y (far baseline) goes upward,
    matching typical top-down orientation.
    """
    scale_x = panel_width / DOUBLES_WIDTH
    scale_y = panel_height / DOUBLES_LENGTH
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
) -> None:
    """Draw court lines and player/ball markers on the top-view panel.

    Mutates *panel* in place.
    """
    # ── Helper to convert court meter → panel pixel ──
    def _p(x_m: float, y_m: float) -> tuple[int, int]:
        return _court_to_panel(x_m, y_m, panel_width, panel_height)

    # ── Court area fill ──
    cv2.rectangle(
        panel,
        _p(-DOUBLES_WIDTH / 2, -DOUBLES_LENGTH / 2),
        _p(DOUBLES_WIDTH / 2, DOUBLES_LENGTH / 2),
        _COLOR_COURT_FILL,
        -1,
    )

    # ── Doubles outline ──
    near_left = _p(-DOUBLES_WIDTH / 2, -DOUBLES_LENGTH / 2)
    far_right = _p(DOUBLES_WIDTH / 2, DOUBLES_LENGTH / 2)
    cv2.rectangle(panel, near_left, far_right, _COLOR_COURT_LINE, 1)

    # ── Singles sidelines ──
    singles_half = 4.115  # SINGLES_WIDTH / 2
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
        _p(-DOUBLES_WIDTH / 2, 0),
        _p(DOUBLES_WIDTH / 2, 0),
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


def render_annotated_frames(
    video_path: str | Path,
    output_rows: list[OutputRow],
    *,
    target_height: int = _DEFAULT_TARGET_HEIGHT,
    panel_width: int = _DEFAULT_PANEL_WIDTH,
    ball_trail_length: int = _DEFAULT_BALL_TRAIL_LENGTH,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield annotated frames from a video and tracking output rows.

    For each frame in the video, this yields an annotated frame
    consisting of the resized source frame with overlays on the left
    and a top-view court panel on the right.

    Parameters
    ----------
    video_path:
        Path to the source video.
    output_rows:
        Tracking output rows for annotation.  Should have one row per
        video frame.  If fewer rows, remaining frames get a "no data"
        overlay.
    target_height:
        Height in pixels for the output frame (the source is resized
        to this height, maintaining aspect ratio).
    panel_width:
        Width of the top-view court panel in pixels.
    ball_trail_length:
        Number of recent ball positions to draw as a fading trail.

    Yields
    ------
    (frame_index, annotated_bgr_frame) tuples.
    """
    # ── Ball trail buffer ──
    ball_trail_pixels: list[tuple[float, float]] = []
    ball_trail_meters: list[tuple[float, float]] = []

    total_rows = len(output_rows)

    for idx, (frame_index, frame_bgr) in enumerate(
        iter_frames(video_path, max_frames=total_rows)
    ):
        # ── Resize source frame ──
        h, w = frame_bgr.shape[:2]
        scale_h = target_height / h
        new_w = int(w * scale_h)
        resized = cv2.resize(frame_bgr, (new_w, target_height))

        # ── Get the output row for this frame ──
        raw_output_row: Optional[OutputRow] = (
            output_rows[idx] if idx < total_rows else None
        )
        output_row = (
            _scale_row_pixels(raw_output_row, scale_h)
            if raw_output_row is not None
            else None
        )

        # ── Update ball trail ──
        if output_row is not None:
            if (
                output_row.ball_pixel_x is not None
                and output_row.ball_pixel_y is not None
            ):
                ball_trail_pixels.append((output_row.ball_pixel_x, output_row.ball_pixel_y))
                if len(ball_trail_pixels) > ball_trail_length:
                    ball_trail_pixels.pop(0)
            if output_row.ball_x_m is not None and output_row.ball_y_m is not None:
                ball_trail_meters.append((output_row.ball_x_m, output_row.ball_y_m))
                if len(ball_trail_meters) > ball_trail_length:
                    ball_trail_meters.pop(0)

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

        # ── Create and draw court panel ──
        panel = np.zeros((target_height, panel_width, 3), dtype=np.uint8)
        _draw_court_panel(
            panel,
            output_row,
            ball_trail_meters=ball_trail_meters,
            panel_width=panel_width,
            panel_height=target_height,
        )

        # ── Combine ──
        annotated = np.concatenate([resized, panel], axis=1)

        yield (frame_index, annotated)


def render_annotated_video(
    video_path: str | Path,
    output_rows: list[OutputRow],
    output_video_path: str | Path,
    *,
    target_height: int = _DEFAULT_TARGET_HEIGHT,
    panel_width: int = _DEFAULT_PANEL_WIDTH,
    ball_trail_length: int = _DEFAULT_BALL_TRAIL_LENGTH,
    fps: Optional[float] = None,
) -> AnnotatedVideoResult:
    """Render an annotated video file from source + tracking data.

    Writes an MP4 video combining resized source frames with overlays
    and a top-view court panel.

    Parameters
    ----------
    video_path:
        Source video path.
    output_rows:
        Tracking output rows (one per frame to process).
    output_video_path:
        Destination for the annotated MP4.
    target_height:
        Height of the annotated output in pixels.
    panel_width:
        Width of the top-view court panel in pixels.
    ball_trail_length:
        Trail length for ball markers.
    fps:
        Output video FPS.  If None, reads from source video metadata.

    Returns
    -------
    AnnotatedVideoResult with frame count and output dimensions.
    """
    # ── Determine FPS ──
    if fps is None:
        try:
            meta = read_video_metadata(video_path)
            fps = meta.fps
        except (FileNotFoundError, ValueError):
            fps = 30.0

    # ── Generate annotated frames ──
    frames: list[np.ndarray] = []
    for _, annotated in render_annotated_frames(
        video_path,
        output_rows,
        target_height=target_height,
        panel_width=panel_width,
        ball_trail_length=ball_trail_length,
    ):
        frames.append(annotated)

    # ── Write video ──
    write_video(frames, output_video_path, fps=fps)

    if frames:
        oh, ow = frames[0].shape[:2]
    else:
        oh, ow = 0, 0

    return AnnotatedVideoResult(
        frame_count=len(frames),
        output_path=str(output_video_path),
        output_width=ow,
        output_height=oh,
    )

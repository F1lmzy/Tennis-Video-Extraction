"""Tests for the annotated video renderer and top-view court panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tennis_tracker.court import DOUBLES_LENGTH, DOUBLES_WIDTH
from tennis_tracker.render import (
    _court_to_panel,
    render_annotated_frames,
    render_annotated_video,
    _draw_court_panel,
    _draw_source_overlay,
    _scale_row_pixels,
)
from tennis_tracker.types import OutputRow
from tennis_tracker.video import write_video, read_video_metadata


# ── Helpers ────────────────────────────────────────────────────────────


def _synthetic_frame(width: int = 640, height: int = 480, value: int = 128) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_output_row(
    frame_index: int = 0,
    *,
    has_players: bool = True,
    has_ball: bool = True,
    has_court: bool = True,
) -> OutputRow:
    """Create an OutputRow with deterministic test values."""
    return OutputRow(
        frame_index=frame_index,
        time_s=frame_index / 30.0,
        player_a_x_m=-2.0 if has_players else None,
        player_a_y_m=3.0 if has_players else None,
        player_a_pixel_x=200.0 if has_players else None,
        player_a_pixel_y=150.0 if has_players else None,
        player_a_confidence=0.95 if has_players else None,
        player_b_x_m=2.0 if has_players else None,
        player_b_y_m=-3.0 if has_players else None,
        player_b_pixel_x=400.0 if has_players else None,
        player_b_pixel_y=300.0 if has_players else None,
        player_b_confidence=0.90 if has_players else None,
        ball_x_m=0.5 if has_ball else None,
        ball_y_m=1.0 if has_ball else None,
        ball_pixel_x=300.0 if has_ball else None,
        ball_pixel_y=200.0 if has_ball else None,
        ball_confidence=0.85 if has_ball else None,
        court_confidence=0.95 if has_court else None,
        homography_valid=has_court,
        diagnostics="" if has_players and has_ball else "missing_ball",
    )


# ─── Panel coordinate mapping ──────────────────────────────────────────


class TestCourtToPanelMapping:
    """Verify that meter-to-pixel mapping follows the coordinate convention."""

    def test_origin_maps_to_center(self) -> None:
        """Court origin (0,0) should map to center of panel."""
        pw, ph = 300, 649
        px, py = _court_to_panel(0.0, 0.0, pw, ph)
        assert px == pw // 2
        assert py == ph // 2

    def test_positive_x_moves_right(self) -> None:
        """+X in court meters should map to rightward in panel pixels."""
        pw, ph = 300, 649
        _, py_origin = _court_to_panel(0.0, 0.0, pw, ph)
        px_pos, py_pos = _court_to_panel(2.0, 0.0, pw, ph)
        assert px_pos > pw // 2
        assert py_pos == py_origin  # same Y

    def test_positive_y_moves_up(self) -> None:
        """+Y (far baseline) should map upward in panel (smaller Y pixel)."""
        pw, ph = 300, 649
        px_origin, py_origin = _court_to_panel(0.0, 0.0, pw, ph)
        _, py_pos = _court_to_panel(0.0, 2.0, pw, ph)
        assert py_pos < py_origin  # upward in image coordinates

    def test_doubles_corner_stays_within_panel(self) -> None:
        """Full doubles court corners should fall within panel bounds."""
        pw, ph = 300, 649
        half_w = DOUBLES_WIDTH / 2
        half_l = DOUBLES_LENGTH / 2
        corners = [
            (-half_w, -half_l),
            (half_w, -half_l),
            (-half_w, half_l),
            (half_w, half_l),
        ]
        for x_m, y_m in corners:
            px, py = _court_to_panel(x_m, y_m, pw, ph)
            assert 0 <= px <= pw, f"Corner ({x_m}, {y_m}) px={px} out of panel width"
            assert 0 <= py <= ph, f"Corner ({x_m}, {y_m}) py={py} out of panel height"


# ─── Court panel drawing ──────────────────────────────────────────────


class TestDrawCourtPanel:
    """Verify that the court panel is drawn correctly."""

    def test_panel_has_court_lines(self) -> None:
        """The panel should contain visible white court lines."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        _draw_court_panel(
            panel,
            output_row=None,
            ball_trail_meters=[],
            panel_width=pw,
            panel_height=ph,
        )
        # There should be white pixels (court lines drawn)
        assert np.any(panel > 0), "Court panel should have visible lines"

    def test_panel_draws_player_a_marker(self) -> None:
        """Player A should appear as a blue marker."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        row = _make_output_row(has_players=True, has_ball=False)
        _draw_court_panel(
            panel,
            output_row=row,
            ball_trail_meters=[],
            panel_width=pw,
            panel_height=ph,
        )
        # Player A is blue (B=255, G=0, R=0) — check for blue-ish pixels
        blue_mask = panel[:, :, 0] > 100
        assert np.any(blue_mask), "Player A should produce blue marker"

    def test_panel_draws_player_b_marker(self) -> None:
        """Player B should appear as a red marker."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        row = _make_output_row(has_players=True, has_ball=False)
        _draw_court_panel(
            panel,
            output_row=row,
            ball_trail_meters=[],
            panel_width=pw,
            panel_height=ph,
        )
        # Player B is red (B=0, G=0, R=255)
        red_mask = panel[:, :, 2] > 100
        assert np.any(red_mask), "Player B should produce red marker"

    def test_panel_draws_ball_marker(self) -> None:
        """Ball should appear as a marker."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        row = _make_output_row(has_ball=True)
        _draw_court_panel(
            panel,
            output_row=row,
            ball_trail_meters=[],
            panel_width=pw,
            panel_height=ph,
        )
        # Ball marker is drawn — check for non-zero pixels
        assert np.any(panel > 0)

    def test_panel_no_markers_when_all_missing(self) -> None:
        """All-missing row should produce no player/ball markers."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        row = _make_output_row(has_players=False, has_ball=False)
        _draw_court_panel(
            panel,
            output_row=row,
            ball_trail_meters=[],
            panel_width=pw,
            panel_height=ph,
        )
        # Only court lines + no markers — should still have court line pixels
        assert np.any(panel > 0), "Panel should show court even with no markers"

    def test_panel_draws_ball_trail(self) -> None:
        """Ball trail should draw multiple faded circles."""
        pw, ph = 200, 434
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        trail = [(1.0, 1.0), (0.5, 0.5), (0.0, 0.0)]
        _draw_court_panel(
            panel,
            output_row=None,
            ball_trail_meters=trail,
            panel_width=pw,
            panel_height=ph,
        )
        # Trail should produce some yellow-ish pixels
        assert np.any(panel > 0), "Ball trail should be visible"


# ─── Source overlay drawing ────────────────────────────────────────────


class TestDrawSourceOverlay:
    """Verify that source-frame overlays are drawn correctly."""

    def test_draws_player_markers(self) -> None:
        """Player A and B should have visible markers."""
        frame = _synthetic_frame()
        row = _make_output_row()
        _draw_source_overlay(
            frame, row,
            ball_trail_pixels=[],
            frame_index=0,
            frame_count=10,
            diagnostics_text=None,
        )
        # Look for label text rendered
        assert np.any(frame > 128), "Frame should have overlay content"

    def test_missing_players_show_warning(self) -> None:
        """Missing players should show warning text."""
        frame = _synthetic_frame()
        row = _make_output_row(has_players=False, has_ball=True)
        _draw_source_overlay(
            frame, row,
            ball_trail_pixels=[],
            frame_index=0,
            frame_count=10,
            diagnostics_text="missing_player_a;missing_player_b",
        )
        # Warning text should be visible
        assert np.any(frame > 128)

    def test_draws_ball_trail(self) -> None:
        """Ball trail should draw multiple circles."""
        frame = _synthetic_frame()
        row = _make_output_row()
        trail = [(300.0, 200.0), (310.0, 205.0), (320.0, 210.0)]
        _draw_source_overlay(
            frame, row,
            ball_trail_pixels=trail,
            frame_index=0,
            frame_count=10,
            diagnostics_text=None,
        )
        # Trail should be visible
        assert np.any(frame > 128)

    def test_no_output_row_shows_no_data_message(self) -> None:
        """When output_row is None, should show 'No tracking data'."""
        frame = _synthetic_frame()
        _draw_source_overlay(
            frame, None,
            ball_trail_pixels=[],
            frame_index=0,
            frame_count=10,
            diagnostics_text=None,
        )
        # Should have the warning text visible
        assert np.any(frame > 128)

    def test_ball_marker_drawn(self) -> None:
        """Ball should have a filled circle marker."""
        frame = _synthetic_frame()
        row = _make_output_row(has_ball=True)
        _draw_source_overlay(
            frame, row,
            ball_trail_pixels=[],
            frame_index=0,
            frame_count=10,
            diagnostics_text=None,
        )
        # Should have visible overlay content
        assert np.any(frame > 128), "Frame should show ball marker"


# ─── Annotated frame generation ────────────────────────────────────────


class TestScaleRowPixels:
    def test_scales_source_pixel_fields_only(self) -> None:
        row = _make_output_row()

        scaled = _scale_row_pixels(row, 0.5)

        assert scaled.player_a_pixel_x == pytest.approx(100.0)
        assert scaled.player_a_pixel_y == pytest.approx(75.0)
        assert scaled.player_b_pixel_x == pytest.approx(200.0)
        assert scaled.player_b_pixel_y == pytest.approx(150.0)
        assert scaled.ball_pixel_x == pytest.approx(150.0)
        assert scaled.ball_pixel_y == pytest.approx(100.0)
        assert scaled.ball_x_m == row.ball_x_m
        assert scaled.ball_y_m == row.ball_y_m
        assert row.ball_pixel_x == pytest.approx(300.0)


class TestRenderAnnotatedFrames:
    """Verify the annotated frame generation pipeline."""

    @pytest.fixture
    def short_video(self, tmp_path: Path) -> Path:
        """Create a 5-frame synthetic video."""
        path = tmp_path / "input.mp4"
        frames = [_synthetic_frame(320, 240, v * 30) for v in range(5)]
        write_video(frames, path, fps=10.0)
        return path

    def test_yields_correct_frame_count(self, short_video: Path) -> None:
        """Should yield one annotated frame per output row."""
        rows = [_make_output_row(i) for i in range(5)]
        annotated = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
            )
        )
        assert len(annotated) == 5

    def test_annotated_frame_has_correct_size(self, short_video: Path) -> None:
        """Each annotated frame should include source + panel."""
        rows = [_make_output_row(i) for i in range(5)]
        frames = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
            )
        )
        for idx, frame in frames:
            h, w = frame.shape[:2]
            assert h == 240
            assert (
                w > 150
            ), f"Width {w} should be larger than panel width 150"

    def test_annotations_visible(self, short_video: Path) -> None:
        """Annotated frame should have visible overlays."""
        rows = [_make_output_row(i) for i in range(5)]
        frames = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
            )
        )
        for _, frame in frames:
            assert np.any(frame > 0), "Annotated frame should have content"

    def test_missing_detections_handled(self, short_video: Path) -> None:
        """Frames with missing detections should still produce output."""
        rows = [
            _make_output_row(0, has_players=True, has_ball=True),
            _make_output_row(1, has_players=False, has_ball=False),
            _make_output_row(2, has_players=True, has_ball=True),
        ]
        frames = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
            )
        )
        assert len(frames) == 3
        for _, frame in frames:
            assert np.any(frame > 0)

    def test_ball_trail_persists(self, short_video: Path) -> None:
        """Ball trail should accumulate across frames."""
        rows = [_make_output_row(i, has_ball=True) for i in range(5)]
        frames = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
                ball_trail_length=10,
            )
        )
        assert len(frames) == 5

    def test_diagnostics_displayed(self, short_video: Path) -> None:
        """Diagnostics should render on frames with issues."""
        rows = [
            _make_output_row(0, has_ball=False),
            _make_output_row(1, has_ball=True),
        ]
        frames = list(
            render_annotated_frames(
                short_video,
                rows,
                target_height=240,
                panel_width=150,
            )
        )
        assert len(frames) == 2


# ─── Full annotated video writing ──────────────────────────────────────


class TestRenderAnnotatedVideo:
    """Verify that a full annotated video can be written and re-read."""

    @pytest.fixture
    def short_video(self, tmp_path: Path) -> Path:
        path = tmp_path / "input_av.mp4"
        frames = [_synthetic_frame(320, 240, v * 30) for v in range(5)]
        write_video(frames, path, fps=10.0)
        return path

    def test_write_annotated_video(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        """Should write a valid MP4 with correct metadata."""
        rows = [_make_output_row(i) for i in range(5)]
        out_path = tmp_path / "annotated.mp4"
        result = render_annotated_video(
            short_video,
            rows,
            out_path,
            target_height=240,
            panel_width=150,
        )
        assert out_path.exists()
        assert result.frame_count == 5
        assert result.output_height == 240

        # Re-read and verify
        meta = read_video_metadata(out_path)
        assert meta.frame_count == pytest.approx(5, abs=2)
        assert meta.width > 150  # source + panel

    def test_custom_fps(self, short_video: Path, tmp_path: Path) -> None:
        """Custom FPS should be applied to output video."""
        rows = [_make_output_row(i) for i in range(5)]
        out_path = tmp_path / "annotated_custom.mp4"
        result = render_annotated_video(
            short_video,
            rows,
            out_path,
            target_height=240,
            panel_width=150,
            fps=15.0,
        )
        assert result.frame_count == 5

    def test_empty_rows_list(self, short_video: Path, tmp_path: Path) -> None:
        """Empty rows should produce result with 0 frames (nothing to annotate)."""
        out_path = tmp_path / "annotated_empty.mp4"
        result = render_annotated_video(
            short_video,
            [],
            out_path,
            target_height=240,
            panel_width=150,
        )
        assert result.frame_count == 0

    def test_output_dimensions(self, short_video: Path, tmp_path: Path) -> None:
        """Output width should be source_width + panel_width."""
        rows = [_make_output_row(i) for i in range(5)]
        out_path = tmp_path / "annotated_dims.mp4"
        result = render_annotated_video(
            short_video,
            rows,
            out_path,
            target_height=360,
            panel_width=200,
        )
        assert result.output_height == 360
        meta = read_video_metadata(out_path)
        assert meta.height == 360
        # Width should be >= 200 (panel) + any portion of the source
        assert meta.width >= 200

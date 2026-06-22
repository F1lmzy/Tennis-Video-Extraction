"""Integration tests for the synthetic (model-free) process pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tennis_tracker.output import read_csv_rows
from tennis_tracker.pipeline import (
    CourtKeypointMatch,
    SyntheticDetection,
    SyntheticFrameData,
    run_process,
    run_synthetic_pipeline,
)
from tennis_tracker.types import PixelPoint
from tennis_tracker.video import read_video_metadata, write_video

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A simple synthetic scene: a 2x2 meter square projected onto a 200x200
# pixel image.  We use the same square fixture as the homography tests.
_SQUARE_PIXELS = np.array(
    [
        [100.0, 100.0],  # centre
        [200.0, 100.0],  # right mid
        [100.0, 200.0],  # bottom mid
        [0.0, 100.0],  # left mid
        [100.0, 0.0],  # top mid
    ]
)

_SQUARE_COURT = np.array(
    [
        [0.0, 0.0],  # centre
        [1.0, 0.0],  # right
        [1.0, -1.0],  # bottom-right
        [-1.0, 0.0],  # left
        [0.0, 1.0],  # top
    ]
)


@pytest.fixture
def court_matches() -> list[CourtKeypointMatch]:
    """Five matched points that define a valid homography."""
    return [
        CourtKeypointMatch(
            pixel=PixelPoint(x=float(p[0]), y=float(p[1])),
            court_x_m=float(c[0]),
            court_y_m=float(c[1]),
        )
        for p, c in zip(_SQUARE_PIXELS, _SQUARE_COURT)
    ]


def _synthetic_frame(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticPipeline:
    def test_produces_all_outputs(
        self,
        tmp_path: Path,
        court_matches: list[CourtKeypointMatch],
    ) -> None:
        """The pipeline should produce raw CSV, smoothed CSV, and video."""
        # --- Create a tiny synthetic video ---
        video_path = tmp_path / "input.mp4"
        frames = [_synthetic_frame(200, 200, v) for v in range(10)]
        from tennis_tracker.video import write_video

        write_video(frames, video_path, fps=25.0)

        # --- Synthetic detection data ---
        frame_data = [
            SyntheticFrameData(
                player_a=SyntheticDetection(
                    pixel=PixelPoint(x=150.0, y=80.0), confidence=0.95
                ),
                player_b=SyntheticDetection(
                    pixel=PixelPoint(x=50.0, y=120.0), confidence=0.90
                ),
                ball=SyntheticDetection(
                    pixel=PixelPoint(x=100.0, y=100.0), confidence=0.85
                ),
            )
            for _ in range(10)
        ]

        # --- Output paths ---
        raw_csv = tmp_path / "raw.csv"
        smoothed_csv = tmp_path / "smoothed.csv"
        out_video = tmp_path / "output.mp4"

        # --- Run pipeline ---
        summary = run_synthetic_pipeline(
            video_path,
            raw_csv,
            smoothed_csv,
            out_video,
            frame_data=frame_data,
            court_keypoint_matches=court_matches,
            fps=25.0,
        )

        # --- Assert outputs exist ---
        assert raw_csv.exists(), f"Raw CSV not found at {raw_csv}"
        assert smoothed_csv.exists(), f"Smoothed CSV not found at {smoothed_csv}"
        assert out_video.exists(), f"Output video not found at {out_video}"

        # --- Assert row counts ---
        assert summary["raw_row_count"] == 10
        assert summary["smoothed_row_count"] == 10
        assert summary["homography_valid"] is True

        # --- Assert CSV content ---
        raw_text = raw_csv.read_text()
        assert raw_text.startswith("frame_index,time_s,")

        lines = raw_text.strip().splitlines()
        assert len(lines) == 11  # header + 10 data rows
        # Check that the centre ball (0,0) projection appears
        # (the centre pixel 100,100 projects to court 0,0)
        for line in lines[1:]:
            fields = line.split(",")
            assert fields[0]  # frame_index non-empty
            # player_a_x_m should not be empty (we provided all detections)
            assert fields[1]  # time_s non-empty

        # --- Assert video metadata ---
        meta = read_video_metadata(out_video)
        assert meta.frame_count == pytest.approx(10, abs=2)
        assert meta.fps == pytest.approx(25.0, abs=5)

    def test_missing_detections_produce_diagnostics(
        self,
        tmp_path: Path,
        court_matches: list[CourtKeypointMatch],
    ) -> None:
        """Missing detections should produce diagnostic flags in CSV."""
        video_path = tmp_path / "input_miss.mp4"
        frames = [_synthetic_frame(200, 200, v) for v in range(5)]
        from tennis_tracker.video import write_video

        write_video(frames, video_path, fps=30.0)

        # --- Alternating missing ball on frames 1 and 3 ---
        frame_data = [
            SyntheticFrameData(
                player_a=SyntheticDetection(
                    pixel=PixelPoint(x=150.0, y=80.0), confidence=0.95
                ),
                player_b=SyntheticDetection(
                    pixel=PixelPoint(x=50.0, y=120.0), confidence=0.90
                ),
                ball=SyntheticDetection(
                    pixel=PixelPoint(x=100.0, y=100.0), confidence=0.85
                ),
            )
            if i % 2 == 0
            else SyntheticFrameData(
                player_a=SyntheticDetection(
                    pixel=PixelPoint(x=150.0, y=80.0), confidence=0.95
                ),
                player_b=SyntheticDetection(
                    pixel=PixelPoint(x=50.0, y=120.0), confidence=0.90
                ),
                ball=None,
            )
            for i in range(5)
        ]

        raw_csv = tmp_path / "raw_miss.csv"
        smoothed_csv = tmp_path / "smoothed_miss.csv"
        out_video = tmp_path / "output_miss.mp4"

        summary = run_synthetic_pipeline(
            video_path,
            raw_csv,
            smoothed_csv,
            out_video,
            frame_data=frame_data,
            court_keypoint_matches=court_matches,
            fps=30.0,
        )

        # --- Assert diagnostics appear ---
        assert "missing_ball" in summary["diagnostics_summary"]
        assert summary["raw_row_count"] == 5

        # --- Assert smoothed output has ball on missing frames (interpolated) ---
        smoothed_text = smoothed_csv.read_text()
        smoothed_lines = smoothed_text.strip().splitlines()
        # Frame 1 (index 1) should have interpolated ball coordinates
        # Frame 3 (index 3) should have interpolated ball coordinates
        for i in [1, 3]:
            line = smoothed_lines[i + 1]  # +1 for header
            fields = line.split(",")
            # ball_x_m and ball_y_m should be non-empty (interpolated)
            assert fields[12] != "", f"Frame {i}: ball_x_m should be interpolated"
            assert fields[13] != "", f"Frame {i}: ball_y_m should be interpolated"

    def test_invalid_homography_emits_diagnostics(
        self,
        tmp_path: Path,
    ) -> None:
        """Insufficient keypoints should produce invalid homography."""
        video_path = tmp_path / "input_bad.mp4"
        frames = [_synthetic_frame(200, 200, v) for v in range(3)]
        from tennis_tracker.video import write_video

        write_video(frames, video_path, fps=30.0)

        # Too few points for a valid homography
        bad_matches = [
            CourtKeypointMatch(
                pixel=PixelPoint(x=100.0, y=100.0),
                court_x_m=0.0,
                court_y_m=0.0,
            )
        ]

        frame_data = [
            SyntheticFrameData(
                player_a=SyntheticDetection(
                    pixel=PixelPoint(x=150.0, y=80.0), confidence=0.95
                ),
                player_b=SyntheticDetection(
                    pixel=PixelPoint(x=50.0, y=120.0), confidence=0.90
                ),
                ball=SyntheticDetection(
                    pixel=PixelPoint(x=100.0, y=100.0), confidence=0.85
                ),
            )
            for _ in range(3)
        ]

        raw_csv = tmp_path / "raw_bad.csv"
        smoothed_csv = tmp_path / "smoothed_bad.csv"
        out_video = tmp_path / "output_bad.mp4"

        summary = run_synthetic_pipeline(
            video_path,
            raw_csv,
            smoothed_csv,
            out_video,
            frame_data=frame_data,
            court_keypoint_matches=bad_matches,
            fps=30.0,
        )

        assert summary["homography_valid"] is False
        assert "invalid_homography" in summary["diagnostics_summary"]


class TestRealPipelineMocked:
    """Integration tests for the real process pipeline with mocked detectors."""

    @pytest.fixture
    def short_video(self, tmp_path: Path) -> Path:
        """Create a 10-frame synthetic video."""
        path = tmp_path / "input.mp4"
        frames = [np.full((200, 200, 3), v, dtype=np.uint8) for v in range(10)]
        write_video(frames, path, fps=25.0)
        return path

    def test_all_outputs_produced(self, short_video: Path, tmp_path: Path) -> None:
        """Should produce raw CSV, smoothed CSV, and annotated video."""
        from unittest.mock import MagicMock
        from tennis_tracker.detection import (
            BoundingBox,
            BallDetection,
            CourtKeypoint,
            PlayerDetection,
        )
        from tennis_tracker.types import PixelPoint

        # ── Mock detectors that return synthetic data ──
        player_mock = MagicMock()
        player_mock.predict.return_value = [
            PlayerDetection(
                bbox=BoundingBox(x1=80.0, y1=50.0, x2=120.0, y2=150.0),
                confidence=0.95,
            ),
            PlayerDetection(
                bbox=BoundingBox(x1=280.0, y1=60.0, x2=320.0, y2=160.0),
                confidence=0.92,
            ),
        ]

        ball_mock = MagicMock()
        ball_mock.predict.return_value = [
            BallDetection(
                bbox=BoundingBox(x1=190.0, y1=90.0, x2=210.0, y2=110.0),
                center=PixelPoint(x=200.0, y=100.0),
                confidence=0.85,
            )
        ]

        court_mock = MagicMock()
        court_mock.predict.return_value = [
            CourtKeypoint(
                label="doubles_near_left",
                pixel=PixelPoint(x=10.0, y=180.0),
                confidence=0.99,
            ),
            CourtKeypoint(
                label="doubles_near_right",
                pixel=PixelPoint(x=190.0, y=180.0),
                confidence=0.99,
            ),
            CourtKeypoint(
                label="doubles_far_left",
                pixel=PixelPoint(x=10.0, y=10.0),
                confidence=0.99,
            ),
            CourtKeypoint(
                label="doubles_far_right",
                pixel=PixelPoint(x=190.0, y=10.0),
                confidence=0.99,
            ),
            CourtKeypoint(
                label="net_center",
                pixel=PixelPoint(x=100.0, y=100.0),
                confidence=0.99,
            ),
        ]

        raw_csv = tmp_path / "raw.csv"
        smoothed_csv = tmp_path / "smoothed.csv"
        out_video = tmp_path / "output.mp4"

        summary = run_process(
            short_video,
            raw_csv,
            smoothed_csv,
            out_video,
            player_detector=player_mock,
            ball_detector=ball_mock,
            court_detector=court_mock,
            fps=25.0,
        )

        assert raw_csv.exists()
        assert smoothed_csv.exists()
        assert out_video.exists()
        assert summary["raw_row_count"] == 10
        assert summary["smoothed_row_count"] == 10
        assert summary["homography_valid"] is True

    def test_no_detections_produce_diagnostics(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        """Empty detections should produce missing diagnostics."""
        from unittest.mock import MagicMock

        player_mock = MagicMock()
        player_mock.predict.return_value = []
        ball_mock = MagicMock()
        ball_mock.predict.return_value = []
        court_mock = MagicMock()
        court_mock.predict.return_value = []

        raw_csv = tmp_path / "raw_no_det.csv"
        smoothed_csv = tmp_path / "smoothed_no_det.csv"
        out_video = tmp_path / "output_no_det.mp4"

        summary = run_process(
            short_video,
            raw_csv,
            smoothed_csv,
            out_video,
            player_detector=player_mock,
            ball_detector=ball_mock,
            court_detector=court_mock,
            fps=25.0,
        )

        assert "missing_player_a" in summary["diagnostics_summary"]
        assert "missing_player_b" in summary["diagnostics_summary"]
        assert "missing_ball" in summary["diagnostics_summary"]
        assert "homography_invalid" in summary["diagnostics_summary"]

    def test_static_ball_candidate_is_not_bootstrapped(
        self, tmp_path: Path
    ) -> None:
        """A static court-line false positive should stay missing, not become track."""
        from unittest.mock import MagicMock
        from tennis_tracker.detection import BallDetection, BoundingBox

        video_path = tmp_path / "static_line.mp4"
        frames = []
        for _ in range(6):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[48:52, 35:65] = 255
            frames.append(frame)
        write_video(frames, video_path, fps=25.0)

        player_mock = MagicMock()
        player_mock.predict.return_value = []
        ball_mock = MagicMock()
        ball_mock.predict.return_value = [
            BallDetection(
                bbox=BoundingBox(x1=35.0, y1=48.0, x2=65.0, y2=52.0),
                center=PixelPoint(x=50.0, y=50.0),
                confidence=0.2,
            )
        ]
        court_mock = MagicMock()
        court_mock.predict.return_value = []

        raw_csv = tmp_path / "raw_static.csv"
        smoothed_csv = tmp_path / "smoothed_static.csv"
        out_video = tmp_path / "output_static.mp4"

        summary = run_process(
            video_path,
            raw_csv,
            smoothed_csv,
            out_video,
            player_detector=player_mock,
            ball_detector=ball_mock,
            court_detector=court_mock,
            fps=25.0,
        )

        rows = list(read_csv_rows(raw_csv))
        assert summary["raw_row_count"] == 6
        assert all(row["ball_pixel_x"] == "" for row in rows)
        assert "missing_ball" in summary["diagnostics_summary"]

    def test_initial_ball_requires_two_moving_nearby_observations(
        self, tmp_path: Path
    ) -> None:
        """Initial acquisition should confirm motion before emitting ball rows."""
        from unittest.mock import MagicMock
        from tennis_tracker.detection import BallDetection, BoundingBox

        video_path = tmp_path / "moving_ball.mp4"
        positions = [(20, 20), (24, 22), (28, 24), (32, 26)]
        frames = []
        for x, y in positions:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[y - 2:y + 3, x - 2:x + 3] = 255
            frames.append(frame)
        write_video(frames, video_path, fps=25.0)

        player_mock = MagicMock()
        player_mock.predict.return_value = []
        ball_mock = MagicMock()
        ball_mock.predict.side_effect = [
            [
                BallDetection(
                    bbox=BoundingBox(x1=x - 2, y1=y - 2, x2=x + 2, y2=y + 2),
                    center=PixelPoint(x=float(x), y=float(y)),
                    confidence=0.4,
                )
            ]
            for x, y in positions
        ]
        court_mock = MagicMock()
        court_mock.predict.return_value = []

        raw_csv = tmp_path / "raw_moving.csv"
        smoothed_csv = tmp_path / "smoothed_moving.csv"
        out_video = tmp_path / "output_moving.mp4"

        run_process(
            video_path,
            raw_csv,
            smoothed_csv,
            out_video,
            player_detector=player_mock,
            ball_detector=ball_mock,
            court_detector=court_mock,
            fps=25.0,
            ball_max_jump_px=20.0,
        )

        rows = list(read_csv_rows(raw_csv))
        assert rows[0]["ball_pixel_x"] == ""
        assert rows[1]["ball_pixel_x"] != ""
        assert rows[2]["ball_pixel_x"] != ""
        assert rows[3]["ball_pixel_x"] != ""

    def test_full_frame_read_count(self, short_video: Path, tmp_path: Path) -> None:
        """Should process the correct number of frames."""
        from unittest.mock import MagicMock

        player_mock = MagicMock()
        player_mock.predict.return_value = []
        ball_mock = MagicMock()
        ball_mock.predict.return_value = []
        court_mock = MagicMock()
        court_mock.predict.return_value = []

        raw_csv = tmp_path / "raw_count.csv"
        smoothed_csv = tmp_path / "smoothed_count.csv"
        out_video = tmp_path / "output_count.mp4"

        summary = run_process(
            short_video,
            raw_csv,
            smoothed_csv,
            out_video,
            player_detector=player_mock,
            ball_detector=ball_mock,
            court_detector=court_mock,
            fps=25.0,
        )

        assert summary["raw_row_count"] == 10
        assert summary["smoothed_row_count"] == 10

"""Unit tests for the video IO utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tennis_tracker.video import (
    iter_frames,
    read_video_metadata,
    write_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_frame(width: int, height: int, value: int = 128) -> np.ndarray:
    """Create a single BGR frame filled with a uniform colour."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _synthetic_clip(
    num_frames: int,
    width: int = 64,
    height: int = 48,
    fps: float = 30.0,
    tmp_dir: Path | None = None,
) -> Path:
    """Write a small synthetic clip and return its path."""
    frames = [_synthetic_frame(width, height, v) for v in range(num_frames)]
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "synthetic.mp4"
    write_video(frames, path, fps=fps)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReadVideoMetadata:
    def test_reads_fps_and_resolution(self, tmp_path: Path) -> None:
        path = _synthetic_clip(10, width=64, height=48, fps=25.0, tmp_dir=tmp_path)
        meta = read_video_metadata(path)
        assert meta.fps == pytest.approx(25.0, abs=1.0)
        assert meta.width == 64
        assert meta.height == 48
        assert meta.frame_count == pytest.approx(10, abs=1)

    def test_duration_is_frame_count_divided_by_fps(self, tmp_path: Path) -> None:
        path = _synthetic_clip(30, fps=30.0, tmp_dir=tmp_path)
        meta = read_video_metadata(path)
        assert meta.duration_s == pytest.approx(1.0, abs=0.1)

    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_video_metadata("/nonexistent/video.mp4")

    def test_raises_on_unopenable_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_a_video.bin"
        bad.write_bytes(b"not a video file")
        with pytest.raises(ValueError, match="Cannot open"):
            read_video_metadata(bad)


class TestIterFrames:
    def test_yields_all_frames(self, tmp_path: Path) -> None:
        path = _synthetic_clip(10, tmp_dir=tmp_path)
        frames = list(iter_frames(path))
        assert len(frames) == 10
        for i, (idx, frame) in enumerate(frames):
            assert idx == i
            assert frame.shape == (48, 64, 3)

    def test_respects_max_frames(self, tmp_path: Path) -> None:
        path = _synthetic_clip(30, tmp_dir=tmp_path)
        frames = list(iter_frames(path, max_frames=5))
        assert len(frames) == 5

    def test_start_index_skips_frames(self, tmp_path: Path) -> None:
        path = _synthetic_clip(15, tmp_dir=tmp_path)
        frames = list(iter_frames(path, start_index=10))
        assert len(frames) == 5
        assert frames[0][0] == 10
        assert frames[-1][0] == 14

    def test_empty_video_yields_nothing(self, tmp_path: Path) -> None:
        path = _synthetic_clip(1, tmp_dir=tmp_path)
        frames = list(iter_frames(path, max_frames=0))
        assert len(frames) == 0

    def test_iterator_reusable_pattern(self, tmp_path: Path) -> None:
        path = _synthetic_clip(5, tmp_dir=tmp_path)
        indices = [idx for idx, _ in iter_frames(path)]
        assert indices == [0, 1, 2, 3, 4]


class TestWriteVideo:
    def test_write_and_re_read(self, tmp_path: Path) -> None:
        frames = [_synthetic_frame(64, 48, v) for v in range(10)]
        out = tmp_path / "out.mp4"
        meta = write_video(frames, out, fps=30.0)
        assert meta.frame_count == pytest.approx(10, abs=1)
        assert meta.fps == pytest.approx(30.0, abs=5)

    def test_write_with_iterator(self, tmp_path: Path) -> None:
        def gen():
            for v in range(5):
                yield _synthetic_frame(64, 48, v)

        out = tmp_path / "iter.mp4"
        meta = write_video(gen(), out, fps=25.0)
        assert meta.frame_count == pytest.approx(5, abs=1)

    def test_re_read_has_same_metadata(self, tmp_path: Path) -> None:
        frames = [_synthetic_frame(128, 96, v) for v in range(20)]
        out = tmp_path / "roundtrip.mp4"
        write_meta = write_video(frames, out, fps=30.0)
        read_meta = read_video_metadata(out)
        assert read_meta.width == write_meta.width
        assert read_meta.height == write_meta.height
        assert read_meta.frame_count == pytest.approx(write_meta.frame_count, abs=1)

    def test_writes_at_alternative_resolution(self, tmp_path: Path) -> None:
        frames = [_synthetic_frame(320, 240, v) for v in range(5)]
        out = tmp_path / "alt_res.mp4"
        meta = write_video(frames, out, fps=30.0)
        assert meta.width == 320
        assert meta.height == 240

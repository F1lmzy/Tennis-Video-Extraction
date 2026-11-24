"""Unit and end-to-end tests for court-keypoint video preprocessing."""

from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tennis_tracker.court import named_point_by_label
from tennis_tracker.detection import (
    ROBOFLOW_TENNIS_COURT_15_KEYPOINT_LABELS,
    CourtKeypoint,
    CourtKeypointDetector,
)
from tennis_tracker.preprocess import (
    CourtSegment,
    PreprocessConfig,
    binary_search_transition,
    compute_scan_dimensions,
    intervals_from_samples,
    is_full_court_visible,
    merge_and_filter_segments,
    preprocess_video,
    probe_video_ffprobe,
    resolve_output_path,
)
from tennis_tracker.types import PixelPoint


def _load_preprocess_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "preprocess_video.py"
    spec = importlib.util.spec_from_file_location("preprocess_video", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_full_court_keypoints(
    width: int = 640,
    height: int = 360,
    conf: float = 0.99,
) -> list[CourtKeypoint]:
    """Build 15 geometrically valid court keypoints projected into a 640x360 image."""
    # Simple perspective-like mapping from court meters to image pixels (scaled to image size)
    kps: list[CourtKeypoint] = []
    for idx in sorted(ROBOFLOW_TENNIS_COURT_15_KEYPOINT_LABELS):
        label = ROBOFLOW_TENNIS_COURT_15_KEYPOINT_LABELS[idx]
        pt = named_point_by_label(label)
        assert pt is not None
        # Map x_m in [-5.485, 5.485], y_m in [-11.885, 11.885] into image center
        depth = 1.0 - 0.025 * pt.y_m  # perspective foreshortening toward far baseline
        px = width * (0.5 + 0.04 * pt.x_m * depth)
        py = height * (0.55 - 0.025 * pt.y_m)
        kps.append(CourtKeypoint(label=label, pixel=PixelPoint(x=px, y=py), confidence=conf))
    return kps


@dataclass
class _FakeKeypoints:
    xy: torch.Tensor
    conf: Optional[torch.Tensor] = None


@dataclass
class _FakeResult:
    keypoints: Optional[_FakeKeypoints] = None


class TestCourtKeypointDetectorBatch:
    def test_expected_keypoint_count_from_kpt_shape(self) -> None:
        mock_model = MagicMock()
        mock_model.model.kpt_shape = [15, 3]
        det = CourtKeypointDetector(model=mock_model)
        assert det.expected_keypoint_count == 15

    def test_predict_batch_returns_per_frame_keypoints(self) -> None:
        kps1 = _FakeKeypoints(
            xy=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
            conf=torch.tensor([[0.9, 0.8]]),
        )
        kps2 = _FakeKeypoints(
            xy=torch.tensor([[[0.0, 0.0], [50.0, 60.0]]]),
            conf=torch.tensor([[0.9, 0.1]]),
        )
        mock_model = MagicMock()
        mock_model.predict.return_value = [_FakeResult(kps1), _FakeResult(kps2)]

        det = CourtKeypointDetector(model=mock_model, keypoint_labels={0: "p0", 1: "p1"})
        imgs = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        out = det.predict_batch(imgs, conf_threshold=0.5)

        assert len(out) == 2
        assert [k.label for k in out[0]] == ["p0", "p1"]
        assert out[1] == []


class TestIsFullCourtVisible:
    def test_passes_when_all_15_keypoints_valid(self) -> None:
        kps = _make_full_court_keypoints(640, 360, conf=0.95)
        assert is_full_court_visible(kps, 640, 360, expected_keypoints=15, conf_threshold=0.5)

    def test_fails_when_one_keypoint_missing(self) -> None:
        kps = _make_full_court_keypoints(640, 360, conf=0.95)[:-1]
        assert not is_full_court_visible(kps, 640, 360, expected_keypoints=15, conf_threshold=0.5)

    def test_fails_when_one_keypoint_below_conf(self) -> None:
        kps = _make_full_court_keypoints(640, 360, conf=0.95)
        kps[0] = CourtKeypoint(label=kps[0].label, pixel=kps[0].pixel, confidence=0.2)
        assert not is_full_court_visible(kps, 640, 360, expected_keypoints=15, conf_threshold=0.5)

    def test_fails_when_keypoint_out_of_bounds(self) -> None:
        kps = _make_full_court_keypoints(640, 360, conf=0.95)
        kps[0] = CourtKeypoint(
            label=kps[0].label,
            pixel=PixelPoint(x=-5.0, y=100.0),
            confidence=0.9,
        )
        assert not is_full_court_visible(
            kps, 640, 360, expected_keypoints=15, conf_threshold=0.5, require_in_bounds=True
        )

    def test_fails_on_duplicate_labels_only(self) -> None:
        kps = _make_full_court_keypoints(640, 360, conf=0.95)
        kps[-1] = kps[0]  # only 14 unique labels
        assert not is_full_court_visible(kps, 640, 360, expected_keypoints=15, conf_threshold=0.5)


class TestSegmentLogic:
    def test_resolve_output_path_mkv_to_trimmed_mp4(self) -> None:
        out = resolve_output_path("data/input/game-full.mkv", "data/input")
        assert out == Path("data/input/game-full-trimmed.mp4")

    def test_resolve_output_path_idempotent_suffix(self) -> None:
        out = resolve_output_path("data/input/game-full-trimmed.mkv", "data/input")
        assert out == Path("data/input/game-full-trimmed.mp4")

    def test_compute_scan_dimensions_preserves_even_aspect(self) -> None:
        assert compute_scan_dimensions(1280, 720, 640) == (640, 360)
        assert compute_scan_dimensions(640, 360, 640) == (640, 360)
        assert compute_scan_dimensions(1920, 1080, 640) == (640, 360)

    def test_intervals_from_samples(self) -> None:
        samples = [
            (0.0, False),
            (1.0, True),
            (2.0, True),
            (3.0, False),
            (4.0, True),
        ]
        segs = intervals_from_samples(samples, sample_interval_s=1.0, duration_s=5.0)
        assert segs == [
            CourtSegment(start_s=1.0, end_s=3.0),
            CourtSegment(start_s=4.0, end_s=5.0),
        ]

    def test_merge_and_filter_segments(self) -> None:
        raw = [
            CourtSegment(1.0, 5.0),
            CourtSegment(5.2, 10.0),  # 0.2s gap
            CourtSegment(15.0, 16.0),  # 1.0s clip -> filtered out by min_clip_s=2.0
            CourtSegment(20.0, 26.0),
        ]
        merged = merge_and_filter_segments(raw, max_gap_s=0.3, min_clip_s=2.0, duration_s=30.0)
        assert merged == [
            CourtSegment(1.0, 10.0),
            CourtSegment(20.0, 26.0),
        ]

        strict = merge_and_filter_segments(raw, max_gap_s=0.0, min_clip_s=2.0, duration_s=30.0)
        assert strict == [
            CourtSegment(1.0, 5.0),
            CourtSegment(5.2, 10.0),
            CourtSegment(20.0, 26.0),
        ]

    @pytest.mark.parametrize("edge", [1, 2, 7, 15, 24, 25])
    def test_binary_search_rising_edge(self, edge: int) -> None:
        calls = 0

        def pred(i: int) -> bool:
            nonlocal calls
            calls += 1
            return i >= edge

        found = binary_search_transition(pred, 0, 25, rising=True)
        assert found == edge
        assert calls <= 6

    @pytest.mark.parametrize("edge", [0, 1, 8, 17, 23, 24])
    def test_binary_search_falling_edge(self, edge: int) -> None:
        calls = 0

        def pred(i: int) -> bool:
            nonlocal calls
            calls += 1
            return i <= edge

        found = binary_search_transition(pred, 0, 25, rising=False)
        assert found == edge
        assert calls <= 6


class _ColorCodedCourtDetector:
    """Mock detector that reports all 15 court keypoints iff frame is green (court)."""

    expected_keypoint_count = 15

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[CourtKeypoint]:
        h, w = image.shape[:2]
        # BGR format: green channel dominant means "full court visible"
        mean_bgr = image.mean(axis=(0, 1))
        if mean_bgr[1] > mean_bgr[0] + 40 and mean_bgr[1] > mean_bgr[2] + 40:
            return _make_full_court_keypoints(w, h, conf=0.99)
        return []

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[list[CourtKeypoint]]:
        return [
            self.predict(img, conf_threshold=conf_threshold, imgsz=imgsz, device=device)
            for img in images
        ]


def _make_synthetic_mkv(path: Path) -> None:
    """Create a 10s MKV (25fps, 320x180) with two green 'court' segments: [2..5s] and [7..9.5s]."""
    # Segment timeline:
    # 0.0 - 2.0s: black (non-court)
    # 2.0 - 5.0s: green (court, 3.0s)
    # 5.0 - 7.0s: red (non-court)
    # 7.0 - 9.5s: green (court, 2.5s)
    # 9.5 - 10.0s: black (non-court)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=25:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=320x180:r=25:d=3",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x180:r=25:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=320x180:r=25:d=2.5",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=25:d=0.5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=10",
        "-filter_complex",
        "[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "5:a",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)


class TestPreprocessEndToEnd:
    def test_trims_mkv_to_trimmed_mp4_in_output_dir(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "data" / "input"
        in_dir.mkdir(parents=True)
        src_mkv = in_dir / "sample-game.mkv"
        _make_synthetic_mkv(src_mkv)

        cfg = PreprocessConfig(
            output_dir=in_dir,
            sample_interval_s=0.5,
            min_clip_s=1.5,
            max_gap_s=0.0,
            imgsz=320,
            video_codec="libx264",
            preset="ultrafast",
            workers=2,
        )
        result = preprocess_video(
            src_mkv,
            config=cfg,
            court_detector=_ColorCodedCourtDetector(),
        )

        expected_out = in_dir / "sample-game-trimmed.mp4"
        assert result.output_path == expected_out
        assert expected_out.exists()
        assert len(result.segments) == 2

        # Segment 1 ~ [2.0, 5.0] (3.0s), Segment 2 ~ [7.0, 9.5] (2.5s) -> total ~ 5.5s
        assert result.segments[0].start_s == pytest.approx(2.0, abs=0.15)
        assert result.segments[0].end_s == pytest.approx(5.0, abs=0.15)
        assert result.segments[1].start_s == pytest.approx(7.0, abs=0.15)
        assert result.segments[1].end_s == pytest.approx(9.5, abs=0.15)

        out_info = probe_video_ffprobe(expected_out)
        assert out_info.duration_s == pytest.approx(5.5, abs=0.3)
        assert out_info.width == 320
        assert out_info.height == 180
        assert out_info.has_audio is True

    def test_script_parser_defaults(self) -> None:
        mod = _load_preprocess_script()
        args = mod.build_parser().parse_args(["data/input/game-full.mkv"])
        assert args.inputs == [Path("data/input/game-full.mkv")]
        assert args.output_dir == Path("data/input")
        assert args.court_model == Path("models/court_keypoints_yolo26.pt")
        assert args.court_conf == pytest.approx(0.5)
        assert args.sample_interval == pytest.approx(1.0)

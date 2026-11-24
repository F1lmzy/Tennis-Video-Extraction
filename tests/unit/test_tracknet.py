"""Unit tests for TrackNetV3 multi-frame ball detection (``tracknet.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tennis_tracker.tracknet import (
    TrackNetBallDetector,
    TrackNetV3,
    compute_video_median,
    decode_heatmap_detections,
    load_tracknet_state_dict,
)
from tennis_tracker.video import write_video


class _FakeHeatmapModel(nn.Module):
    """Lightweight deterministic model returning a Gaussian blob at a configurable location."""

    def __init__(self, target_xy: tuple[float, float] | None = (256.0, 144.0)) -> None:
        super().__init__()
        self.target_xy = target_xy
        self.last_input_shape: tuple[int, ...] | None = None
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(frames.shape)
        b, _, h, w = frames.shape
        out = torch.zeros((b, 1, h, w), dtype=torch.float32, device=frames.device)
        if self.target_xy is not None:
            cx, cy = self.target_xy
            yy, xx = torch.meshgrid(
                torch.arange(h, device=frames.device, dtype=torch.float32),
                torch.arange(w, device=frames.device, dtype=torch.float32),
                indexing="ij",
            )
            blob = torch.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (3.5**2))
            out[:, 0] = blob
        return out


class TestTrackNetV3Architecture:
    def test_forward_output_shape_and_bounds(self) -> None:
        net = TrackNetV3(in_dim=15, out_dim=4, d_model=8, last_only=True)
        x = torch.rand((2, 15, 32, 64))
        with torch.no_grad():
            y = net(x)
        assert y.shape == (2, 1, 32, 64)
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0

    def test_load_state_dict_strips_module_prefix(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "fake.pth"
        torch.save(
            {
                "state_dict": {
                    "module.predictor.weight": torch.ones((4, 64, 1, 1)),
                    "module.predictor.bias": torch.zeros((4,)),
                }
            },
            ckpt_path,
        )
        sd = load_tracknet_state_dict(ckpt_path)
        assert set(sd.keys()) == {"predictor.weight", "predictor.bias"}


class TestHeatmapDecoding:
    def test_decodes_gaussian_peak_scaled_to_original_resolution(self) -> None:
        h, w = 288, 512
        yy, xx = np.mgrid[0:h, 0:w]
        hm = np.exp(-0.5 * ((xx - 128.0) ** 2 + (yy - 72.0) ** 2) / (3.5**2)).astype(np.float32)

        dets = decode_heatmap_detections(hm, orig_width=1280, orig_height=720, conf_threshold=0.5)
        assert len(dets) == 1
        assert dets[0].center.x == pytest.approx(320.0, abs=1.5)
        assert dets[0].center.y == pytest.approx(180.0, abs=1.5)
        assert dets[0].confidence > 0.7

    def test_empty_heatmap_returns_empty_list(self) -> None:
        hm = np.zeros((288, 512), dtype=np.float32)
        assert decode_heatmap_detections(hm, orig_width=1280, orig_height=720) == []

    def test_multiple_blobs_sorted_by_confidence(self) -> None:
        h, w = 288, 512
        yy, xx = np.mgrid[0:h, 0:w]
        blob1 = 0.7 * np.exp(-0.5 * ((xx - 100.0) ** 2 + (yy - 100.0) ** 2) / (3.0**2))
        blob2 = 0.95 * np.exp(-0.5 * ((xx - 350.0) ** 2 + (yy - 200.0) ** 2) / (3.0**2))
        hm = np.maximum(blob1, blob2).astype(np.float32)

        dets = decode_heatmap_detections(hm, orig_width=512, orig_height=288, conf_threshold=0.5)
        assert len(dets) == 2
        assert dets[0].center.x == pytest.approx(350.0, abs=1.0)
        assert dets[1].center.x == pytest.approx(100.0, abs=1.0)


class TestTrackNetBallDetector:
    def test_predict_stacks_median_and_sequence_frames(self) -> None:
        fake_model = _FakeHeatmapModel(target_xy=(256.0, 144.0))
        median = np.full((3, 288, 512), 0.5, dtype=np.float32)
        det = TrackNetBallDetector(model=fake_model, seq_len=4, median_chw=median, device="cpu")
        assert det.is_temporal is True

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        preds = det.predict(frame, conf_threshold=0.5)
        assert fake_model.last_input_shape == (1, 15, 288, 512)
        assert len(preds) == 1
        assert preds[0].center.x == pytest.approx(640.0, abs=2.0)
        assert preds[0].center.y == pytest.approx(360.0, abs=2.0)

        det.reset()
        preds2 = det.predict(frame, conf_threshold=0.5)
        assert len(preds2) == 1

    def test_compute_video_median_shape_and_range(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        frames = [np.full((90, 160, 3), 100 + i * 5, dtype=np.uint8) for i in range(10)]
        write_video(frames, video, fps=25.0)

        med = compute_video_median(video, width=64, height=36, max_frames=10)
        assert med.shape == (3, 36, 64)
        assert med.dtype == np.float32
        assert 0.0 <= float(med.min()) <= float(med.max()) <= 1.0

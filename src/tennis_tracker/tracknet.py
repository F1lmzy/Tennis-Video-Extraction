"""TrackNetV3 multi-frame ball detection wrapper (RacketVision BallTrack).

Implements the pure-PyTorch TrackNetV3 U-Net architecture from
``OrcustD/RacketVision`` (MIT license) without requiring ``mmengine`` at
runtime.  Given a median background frame and ``seq_len`` consecutive BGR
frames resized to ``512 x 288``, the model predicts a per-pixel Gaussian
heatmap for the latest frame in the sequence.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from tennis_tracker.detection import BallDetection, BoundingBox
from tennis_tracker.types import PixelPoint

DEFAULT_TRACKNET_WIDTH: int = 512
DEFAULT_TRACKNET_HEIGHT: int = 288
DEFAULT_TRACKNET_SEQ_LEN: int = 4
DEFAULT_TRACKNET_CONF: float = 0.5


# ── Pure-PyTorch TrackNetV3 architecture (matches RacketVision BallTrack) ─


class Conv2DBlock(nn.Module):
    """3×3 Conv + BatchNorm + ReLU block matching RacketVision TrackNetV3."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    """Two stacked ``Conv2DBlock`` layers."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    """Three stacked ``Conv2DBlock`` layers."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNetV3(nn.Module):
    """TrackNetV3 encoder-decoder network predicting per-frame ball heatmaps."""

    def __init__(
        self,
        in_dim: int = 3 * (DEFAULT_TRACKNET_SEQ_LEN + 1),
        out_dim: int = DEFAULT_TRACKNET_SEQ_LEN,
        *,
        d_model: int = 64,
        last_only: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.last_only = last_only

        self.down_block_1 = Double2DConv(in_dim, d_model)
        self.down_block_2 = Double2DConv(d_model, d_model * 2)
        self.down_block_3 = Triple2DConv(d_model * 2, d_model * 4)
        self.bottleneck = Triple2DConv(d_model * 4, d_model * 8)
        self.up_block_1 = Triple2DConv(d_model * (8 + 4), d_model * 4)
        self.up_block_2 = Double2DConv(d_model * (4 + 2), d_model * 2)
        self.up_block_3 = Double2DConv(d_model * (2 + 1), d_model)
        self.predictor = nn.Conv2d(d_model, out_dim, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

        self._pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._upsample = nn.Upsample(scale_factor=2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block_1(frames)
        x = self._pool(x1)
        x2 = self.down_block_2(x)
        x = self._pool(x2)
        x3 = self.down_block_3(x)
        x = self._pool(x3)
        x = self.bottleneck(x)
        x = torch.cat([self._upsample(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([self._upsample(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([self._upsample(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.sigmoid(self.predictor(x))
        if self.last_only:
            return x[:, -1:, :, :]
        return x


# ── Dependency-free checkpoint loader (strips mmengine metadata) ───────


class _MmengineStubMeta(type):
    def __getattr__(cls, _name: str):
        return cls


class _MmengineStub(dict, metaclass=_MmengineStubMeta):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __setstate__(self, _state) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self


class _MmengineSafeUnpickler(pickle.Unpickler):
    """Unpickler that stubs ``mmengine.*`` metadata classes while keeping tensors intact."""

    def find_class(self, module: str, name: str):
        if module.startswith("mmengine"):
            return _MmengineStub
        return super().find_class(module, name)


class _MmengineSafePickle:
    Unpickler = _MmengineSafeUnpickler

    @staticmethod
    def load(file_obj, **kwargs):
        return _MmengineSafeUnpickler(file_obj, **kwargs).load()


def load_tracknet_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a TrackNetV3 checkpoint (pure PyTorch or mmengine ``.pth``)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"TrackNet checkpoint not found: {path}")

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            pickle_module=_MmengineSafePickle,
        )

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported TrackNet checkpoint structure in {path}")

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        if isinstance(v, torch.Tensor):
            cleaned[name] = v
    return cleaned


def compute_video_median(
    video_path: str | Path,
    *,
    width: int = DEFAULT_TRACKNET_WIDTH,
    height: int = DEFAULT_TRACKNET_HEIGHT,
    max_frames: int = 80,
) -> np.ndarray:
    """Compute a normalized ``(3, height, width)`` BGR median background frame.

    Matches RacketVision's ``create_median.py`` + ``uball.py`` preprocessing:
    samples evenly across the video in OpenCV BGR order, takes the pixel-wise
    median, resizes to ``(width, height)``, and scales to ``[0.0, 1.0]``.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled: list[np.ndarray] = []

        if total > 0 and max_frames > 0:
            n_sample = min(total, max_frames)
            indices = np.linspace(0, max(0, total - 1), n_sample, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    sampled.append(cv2.resize(frame, (width, height)))
        else:
            step = 10
            i = 0
            while len(sampled) < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if i % step == 0:
                    sampled.append(cv2.resize(frame, (width, height)))
                i += 1
    finally:
        cap.release()

    if not sampled:
        raise ValueError(f"No readable frames in video for median computation: {path}")

    median_bgr = np.median(np.stack(sampled, axis=0), axis=0).astype(np.float32) / 255.0
    return np.moveaxis(median_bgr, -1, 0).copy()  # (3, H, W)


def decode_heatmap_detections(
    heatmap: np.ndarray,
    *,
    orig_width: int,
    orig_height: int,
    conf_threshold: float = DEFAULT_TRACKNET_CONF,
) -> list[BallDetection]:
    """Extract ball detections from a single ``(H, W)`` TrackNet probability map.

    Returns all connected components above ``conf_threshold`` sorted by
    decreasing confidence, using intensity-weighted centroids for sub-pixel
    precision.
    """
    if heatmap.ndim != 2 or heatmap.size == 0:
        return []
    hm_h, hm_w = heatmap.shape
    mask = heatmap > conf_threshold
    if not np.any(mask):
        return []

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    sx = orig_width / float(hm_w)
    sy = orig_height / float(hm_h)

    detections: list[BallDetection] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        roi = heatmap[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        weights = np.where(roi_mask, roi, 0.0)
        w_sum = float(weights.sum())
        if w_sum <= 0.0:
            continue

        ys, xs = np.mgrid[y : y + h, x : x + w]
        cx_hm = float((xs * weights).sum() / w_sum)
        cy_hm = float((ys * weights).sum() / w_sum)
        conf = float(roi[roi_mask].mean())

        cx = cx_hm * sx
        cy = cy_hm * sy
        half_w = max(3.0, (w * sx) / 2.0)
        half_h = max(3.0, (h * sy) / 2.0)
        bbox = BoundingBox(
            x1=max(0.0, cx - half_w),
            y1=max(0.0, cy - half_h),
            x2=min(float(orig_width), cx + half_w),
            y2=min(float(orig_height), cy + half_h),
        )
        detections.append(
            BallDetection(
                bbox=bbox,
                center=PixelPoint(x=cx, y=cy),
                confidence=conf,
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# ── Drop-in BallDetector-compatible TrackNetV3 wrapper ─────────────────


class TrackNetBallDetector:
    """Multi-frame TrackNetV3 ball detector compatible with ``pipeline.run_process``.

    Maintains a rolling buffer of ``seq_len`` BGR frames and a median background
    image so it can be called frame-by-frame via ``predict(frame_bgr, ...)``
    just like ``BallDetector``.
    """

    is_temporal: bool = True

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[str | Path] = None,
        *,
        seq_len: int = DEFAULT_TRACKNET_SEQ_LEN,
        width: int = DEFAULT_TRACKNET_WIDTH,
        height: int = DEFAULT_TRACKNET_HEIGHT,
        device: Optional[str] = None,
        median_chw: Optional[np.ndarray] = None,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError(
                "Either ``model`` or ``model_path`` must be provided for TrackNetBallDetector."
            )

        self._width = width
        self._height = height
        self._seq_len = seq_len
        self._device = self._resolve_device(device)

        if model is not None:
            self._model = model
        else:
            assert model_path is not None
            sd = load_tracknet_state_dict(model_path)
            in_dim = int(sd["down_block_1.conv_1.conv.weight"].shape[1])
            out_dim = int(sd["predictor.weight"].shape[0])
            d_model = int(sd["down_block_1.conv_1.conv.weight"].shape[0])
            self._seq_len = out_dim
            net = TrackNetV3(in_dim=in_dim, out_dim=out_dim, d_model=d_model, last_only=True)
            net.load_state_dict(sd, strict=True)
            self._model = net

        self._model.to(self._device).eval()
        self._median_chw: Optional[np.ndarray] = (
            median_chw.astype(np.float32) if median_chw is not None else None
        )
        self._buffer: deque[np.ndarray] = deque(maxlen=self._seq_len)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def set_median_from_video(self, video_path: str | Path, max_frames: int = 80) -> None:
        """Compute and cache the median background frame from ``video_path``."""
        self._median_chw = compute_video_median(
            video_path,
            width=self._width,
            height=self._height,
            max_frames=max_frames,
        )

    def reset(self) -> None:
        """Clear the temporal frame buffer (e.g. across a hard camera cut)."""
        self._buffer.clear()

    def _prepare_frame(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self._width, self._height))
        chw = np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0)
        return chw

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_TRACKNET_CONF,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[BallDetection]:
        """Push ``image`` (BGR) into the temporal window and return ball detections."""
        del imgsz  # TrackNetV3 uses fixed (512, 288) input resolution
        if device is not None:
            target = self._resolve_device(device)
            if target != self._device:
                self._device = target
                self._model.to(self._device).eval()

        orig_h, orig_w = image.shape[:2]
        frame_chw = self._prepare_frame(image)
        self._buffer.append(frame_chw)

        # Match uball.py padding for the first seq_len - 1 frames: repeat earliest frame
        seq = list(self._buffer)
        while len(seq) < self._seq_len:
            seq.insert(0, seq[0])

        median = (
            self._median_chw
            if self._median_chw is not None
            else np.median(np.stack(seq, axis=0), axis=0).astype(np.float32)
        )

        # Concatenate [median, f_{t-3}, f_{t-2}, f_{t-1}, f_t] -> (1, 15, H, W)
        stacked = np.concatenate([median, *seq], axis=0)[np.newaxis, ...]
        tensor = torch.from_numpy(stacked).to(self._device)

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = self._model(tensor)
            else:
                out = self._model(tensor)

        heatmap = out[0, -1].detach().float().cpu().numpy()
        return decode_heatmap_detections(
            heatmap,
            orig_width=orig_w,
            orig_height=orig_h,
            conf_threshold=conf_threshold,
        )

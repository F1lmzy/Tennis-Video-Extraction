"""Video IO utilities for reading and writing video frames.

Provides deterministic frame iteration, metadata extraction, and
MP4 writing suitable for both production processing and testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from an input video file."""

    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Read and return metadata from a video file.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file cannot be opened as a video.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count > 0 and fps > 0:
            duration_s = frame_count / fps
        else:
            duration_s = 0.0

        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_s=duration_s,
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path,
    *,
    max_frames: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) tuples from a video file.

    Parameters
    ----------
    path:
        Path to the video file.
    max_frames:
        If set, yield at most this many frames.  The first yielded
        frame always has ``frame_index == start_index`` regardless
        of whether zero or more frames were skipped at the start.
    start_index:
        Skip this many frames before yielding the first frame.
        The yielded indices count from this value, not from zero.

    Yields
    ------
    (frame_index, frame_bgr) tuples.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        # Skip requested starting frames
        for _ in range(start_index):
            ret = cap.grab()
            if not ret:
                return

        count = 0
        idx = start_index
        while True:
            if max_frames is not None and count >= max_frames:
                return

            ret, frame = cap.read()
            if not ret:
                return

            yield (idx, frame)
            idx += 1
            count += 1
    finally:
        cap.release()


def write_video(
    frames: list[np.ndarray] | Iterator[np.ndarray],
    output_path: str | Path,
    *,
    fps: float = 30.0,
    fourcc: str = "mp4v",
) -> VideoMetadata:
    """Write a list or iterator of BGR frames to an MP4 video file.

    Parameters
    ----------
    frames:
        BGR frames to write.  All frames should have the same
        height and width.
    output_path:
        Destination path for the video.
    fps:
        Frames per second in the output video.
    fourcc:
        FourCC code for the video codec.  Default ``"mp4v"``
        produces widely-compatible MP4 files.

    Returns
    -------
    VideoMetadata describing the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec = cv2.VideoWriter_fourcc(*fourcc)

    first = True
    writer: Optional[cv2.VideoWriter] = None
    written = 0

    try:
        for frame in frames:
            if first:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), codec, fps, (w, h))
                if not writer or not writer.isOpened():
                    raise RuntimeError(
                        f"Cannot create video writer for {output_path} "
                        f"with codec {fourcc}, fps={fps}, size={(w, h)}"
                    )
                first = False

            writer.write(frame)
            written += 1

        if first:
            # No frames provided — create an empty file placeholder
            h, w = 1, 1
            writer = cv2.VideoWriter(str(output_path), codec, fps, (w, h))
            writer.write(np.zeros((1, 1, 3), dtype=np.uint8))
            written = 1

    finally:
        if writer is not None:
            writer.release()

    duration_s = written / fps if fps > 0 else 0.0
    if writer is not None:
        # Re-read to get accurate metadata from the file
        try:
            return read_video_metadata(output_path)
        except (FileNotFoundError, ValueError):
            pass

    return VideoMetadata(
        fps=fps,
        width=w if not first else 1,
        height=h if not first else 1,
        frame_count=written,
        duration_s=duration_s,
    )

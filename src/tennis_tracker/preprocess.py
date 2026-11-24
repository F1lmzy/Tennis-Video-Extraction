"""Efficient video preprocessing using court keypoint detection.

Scans a full-length tennis match video to keep only segments where all
court keypoints are detected, trims out all non-court footage (replays,
closeups, changeovers, commercials), and converts/exports the kept clips
to MP4 using ``ffmpeg``.

Efficiency strategy (coarse-to-fine):
1. **Coarse scan**: Stream downscaled frames from ``ffmpeg`` at a low
   sampling rate (default 1 fps) and run batched court-keypoint inference
   concurrently with decoding.
2. **Boundary refinement**: For each rising/falling transition detected in
   the coarse pass, fast-seek once with ``ffmpeg`` to extract only the
   narrow boundary window and locate the exact transition frame via binary
   search (O(log N) inferences per cut).
3. **Fast-seek MP4 export**: Extract each kept clip independently using
   fast ``ffmpeg`` input seeking (skipping all cut-out footage), encode
   with consistent H.264/AAC parameters in parallel, and concatenate
   losslessly via the ``ffmpeg`` concat demuxer into ``*-trimmed.mp4``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tennis_tracker.coordinates import estimate_homography
from tennis_tracker.court import named_point_by_label
from tennis_tracker.detection import CourtKeypoint, CourtKeypointDetector

DEFAULT_OUTPUT_DIR = Path("data/input")
DEFAULT_COURT_MODEL = Path("models/court_keypoints_yolo26.pt")


@dataclass(frozen=True)
class VideoProbeInfo:
    """Video stream metadata obtained via ``ffprobe``."""

    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool


@dataclass(frozen=True)
class CourtSegment:
    """A half-open time interval ``[start_s, end_s)`` where the full court is visible."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for court-visibility preprocessing."""

    court_model_path: Path = DEFAULT_COURT_MODEL
    output_dir: Path = DEFAULT_OUTPUT_DIR
    conf_threshold: float = 0.5
    expected_keypoints: Optional[int] = None
    require_in_bounds: bool = True
    require_valid_homography: bool = True
    min_homography_inlier_ratio: float = 0.8
    sample_interval_s: float = 1.0
    batch_size: int = 8
    refine_boundaries: bool = True
    min_clip_s: float = 2.0
    max_gap_s: float = 0.0
    imgsz: int = 640
    device: str = "cpu"
    video_codec: str = "auto"  # "auto", "libx264", "h264_videotoolbox"
    crf: int = 20
    preset: str = "fast"
    workers: int = 2
    keep_audio: bool = True
    split_clips: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    """Summary returned after preprocessing a video."""

    input_path: Path
    output_path: Path
    clip_paths: tuple[Path, ...]
    segments: tuple[CourtSegment, ...]
    source_duration_s: float
    kept_duration_s: float
    removed_duration_s: float
    coarse_frames_evaluated: int
    refined_frames_evaluated: int
    elapsed_s: float


def resolve_output_path(
    input_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Build the ``<output_dir>/<stem>-trimmed.mp4`` path for an input video."""
    src = Path(input_path)
    out_dir = Path(output_dir)
    stem = src.stem
    if stem.endswith("-trimmed"):
        filename = f"{stem}.mp4"
    else:
        filename = f"{stem}-trimmed.mp4"
    return out_dir / filename


def _ensure_ffmpeg_tools() -> None:
    """Verify ``ffmpeg`` and ``ffprobe`` are available on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"Required binary '{tool}' was not found on PATH. "
                "Please install ffmpeg (e.g. `brew install ffmpeg`)."
            )


def _parse_fps(rate_str: str) -> float:
    """Parse an ffprobe rational frame-rate string such as ``'25/1'`` or ``'30000/1001'``."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            return num / den if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def probe_video_ffprobe(video_path: str | Path) -> VideoProbeInfo:
    """Inspect video duration, FPS, resolution, and audio presence via ``ffprobe``."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")

    _ensure_ffmpeg_tools()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    fps = _parse_fps(video_stream.get("avg_frame_rate", ""))
    if fps <= 0.0:
        fps = _parse_fps(video_stream.get("r_frame_rate", ""))
    if fps <= 0.0:
        fps = 25.0

    duration_s = 0.0
    for candidate in (fmt.get("duration"), video_stream.get("duration")):
        if candidate is not None:
            try:
                val = float(candidate)
                if val > 0:
                    duration_s = val
                    break
            except (TypeError, ValueError):
                pass
    if duration_s <= 0.0 and video_stream.get("nb_frames"):
        try:
            duration_s = int(video_stream["nb_frames"]) / fps
        except (TypeError, ValueError):
            pass

    if duration_s <= 0.0:
        raise ValueError(f"Could not determine duration for {path}")

    return VideoProbeInfo(
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )


def compute_scan_dimensions(width: int, height: int, target_max_dim: int) -> tuple[int, int]:
    """Compute even-dimensioned downscaled ``(scan_w, scan_h)`` preserving aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame dimensions: {width}x{height}")
    max_dim = max(width, height)
    if target_max_dim <= 0 or max_dim <= target_max_dim:
        scan_w = max(2, width - (width % 2))
        scan_h = max(2, height - (height % 2))
        return (scan_w, scan_h)

    scale = target_max_dim / float(max_dim)
    scan_w = max(2, int(round(width * scale / 2.0)) * 2)
    scan_h = max(2, int(round(height * scale / 2.0)) * 2)
    return (scan_w, scan_h)


def is_full_court_visible(
    keypoints: list[CourtKeypoint],
    frame_width: int,
    frame_height: int,
    *,
    expected_keypoints: int = 15,
    conf_threshold: float = 0.5,
    require_in_bounds: bool = True,
    require_valid_homography: bool = True,
    min_homography_inlier_ratio: float = 0.8,
) -> bool:
    """Return True only when ALL expected court keypoints are confidently available.

    A frame passes when:
    1. At least ``expected_keypoints`` uniquely labeled keypoints meet
       ``conf_threshold`` and have non-zero coordinates.
    2. When ``require_in_bounds`` is True, every qualifying keypoint lies
       within ``[0, frame_width] x [0, frame_height]``.
    3. When ``require_valid_homography`` is True and at least 4 keypoints
       match named court geometry points, ``estimate_homography`` succeeds
       with inlier ratio >= ``min_homography_inlier_ratio``.
    """
    if expected_keypoints <= 0:
        raise ValueError("expected_keypoints must be positive")

    qualifying: dict[str, CourtKeypoint] = {}
    for kp in keypoints:
        if kp.confidence < conf_threshold:
            continue
        x, y = kp.pixel.x, kp.pixel.y
        if x == 0.0 and y == 0.0:
            continue
        if require_in_bounds and not (0.0 <= x <= frame_width and 0.0 <= y <= frame_height):
            continue
        prev = qualifying.get(kp.label)
        if prev is None or kp.confidence > prev.confidence:
            qualifying[kp.label] = kp

    if len(qualifying) < expected_keypoints:
        return False

    if require_valid_homography:
        pixel_pts = []
        court_pts = []
        for kp in qualifying.values():
            named = named_point_by_label(kp.label)
            if named is not None:
                pixel_pts.append(kp.pixel)
                court_pts.append((named.x_m, named.y_m))
        if len(pixel_pts) >= 4:
            hres = estimate_homography(pixel_pts, court_pts)
            if not hres.valid or hres.confidence < min_homography_inlier_ratio:
                return False

    return True


def intervals_from_samples(
    samples: list[tuple[float, bool]],
    sample_interval_s: float,
    duration_s: float,
) -> list[CourtSegment]:
    """Convert coarse ``(time_s, is_visible)`` samples into rough segments."""
    if not samples:
        return []

    segments: list[CourtSegment] = []
    in_seg = False
    seg_start = 0.0

    for idx, (t_s, visible) in enumerate(samples):
        if visible and not in_seg:
            seg_start = t_s
            in_seg = True
        elif not visible and in_seg:
            # Transition occurred between previous sample and current sample
            seg_end = min(duration_s, t_s)
            segments.append(CourtSegment(start_s=seg_start, end_s=seg_end))
            in_seg = False

    if in_seg:
        last_t = samples[-1][0]
        seg_end = min(duration_s, max(last_t + sample_interval_s, duration_s))
        segments.append(CourtSegment(start_s=seg_start, end_s=seg_end))

    return segments


def merge_and_filter_segments(
    segments: list[CourtSegment],
    *,
    max_gap_s: float = 0.0,
    min_clip_s: float = 2.0,
    duration_s: Optional[float] = None,
) -> list[CourtSegment]:
    """Merge segments separated by <= ``max_gap_s`` and drop clips shorter than ``min_clip_s``."""
    if not segments:
        return []

    cleaned: list[CourtSegment] = []
    for seg in sorted(segments, key=lambda s: (s.start_s, s.end_s)):
        start = max(0.0, seg.start_s)
        end = min(duration_s, seg.end_s) if duration_s is not None else seg.end_s
        if end <= start:
            continue
        if cleaned and (start - cleaned[-1].end_s) <= max_gap_s:
            prev = cleaned[-1]
            cleaned[-1] = CourtSegment(start_s=prev.start_s, end_s=max(prev.end_s, end))
        else:
            cleaned.append(CourtSegment(start_s=start, end_s=end))

    return [seg for seg in cleaned if seg.duration_s >= min_clip_s]


def binary_search_transition(
    eval_fn: Callable[[int], bool],
    low: int,
    high: int,
    *,
    rising: bool,
) -> int:
    """Locate the transition index inside ``[low, high]`` in O(log N) evaluations.

    Parameters
    ----------
    eval_fn:
        Predicate returning True when the full court is visible at ``index``.
    low, high:
        Inclusive frame index bounds with ``low <= high``.
    rising:
        If True (False → True transition), assumes ``eval_fn(low)`` is False
        and ``eval_fn(high)`` is True, returning the **smallest** index ``i``
        such that ``eval_fn(i)`` is True.
        If False (True → False transition), assumes ``eval_fn(low)`` is True
        and ``eval_fn(high)`` is False, returning the **largest** index ``i``
        such that ``eval_fn(i)`` is True.
    """
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    if low == high:
        return low

    lo, hi = low, high
    if rising:
        while lo < hi:
            mid = (lo + hi) // 2
            if eval_fn(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if eval_fn(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _predict_batch_compat(
    detector: CourtKeypointDetector,
    frames: list[np.ndarray],
    *,
    conf_threshold: float,
    imgsz: int,
    device: Optional[str],
) -> list[list[CourtKeypoint]]:
    """Run batched prediction when supported, falling back to per-frame predict."""
    if hasattr(detector, "predict_batch"):
        return detector.predict_batch(
            frames,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
        )
    return [
        detector.predict(
            frame,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
        )
        for frame in frames
    ]


def _extract_window_frames(
    video_path: Path,
    start_s: float,
    duration_s: float,
    scan_w: int,
    scan_h: int,
) -> list[np.ndarray]:
    """Fast-seek with ``ffmpeg`` and decode a short frame window into BGR numpy arrays."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_s):.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{max(0.01, duration_s):.6f}",
        "-vf",
        f"scale={scan_w}:{scan_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg window extraction failed: {err}")

    frame_size = scan_w * scan_h * 3
    buf = proc.stdout
    n_frames = len(buf) // frame_size
    if n_frames == 0:
        return []
    arr = np.frombuffer(buf[: n_frames * frame_size], dtype=np.uint8).reshape(
        n_frames, scan_h, scan_w, 3
    )
    return [arr[i] for i in range(n_frames)]


def detect_court_segments(
    video_path: str | Path,
    *,
    config: PreprocessConfig,
    probe_info: Optional[VideoProbeInfo] = None,
    court_detector: Optional[CourtKeypointDetector] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[CourtSegment], dict[str, int]]:
    """Scan a video coarse-to-fine and return exact full-court-visible segments.

    Parameters
    ----------
    video_path:
        Path to the input video file (e.g. ``.mkv`` or ``.mp4``).
    config:
        Preprocessing configuration.
    probe_info:
        Optional pre-probed video metadata.
    court_detector:
        Optional injected ``CourtKeypointDetector`` (useful for testing).
    progress_callback:
        Optional callable receiving human-readable progress messages.

    Returns
    -------
    (segments, stats)
        ``segments`` is the final merged/filtered list of ``CourtSegment``.
        ``stats`` contains ``coarse_frames_evaluated`` and ``refined_frames_evaluated``.
    """
    path = Path(video_path)
    info = probe_info or probe_video_ffprobe(path)

    if court_detector is None:
        if not Path(config.court_model_path).exists():
            raise FileNotFoundError(f"Court keypoint model not found: {config.court_model_path}")
        court_detector = CourtKeypointDetector(model_path=config.court_model_path)

    expected_kps = (
        config.expected_keypoints
        if config.expected_keypoints is not None
        else getattr(court_detector, "expected_keypoint_count", 15)
    )

    scan_w, scan_h = compute_scan_dimensions(info.width, info.height, config.imgsz)
    sample_interval_s = max(1.0 / info.fps, config.sample_interval_s)
    sample_fps = 1.0 / sample_interval_s

    def _frame_visible(kps: list[CourtKeypoint]) -> bool:
        return is_full_court_visible(
            kps,
            scan_w,
            scan_h,
            expected_keypoints=expected_kps,
            conf_threshold=config.conf_threshold,
            require_in_bounds=config.require_in_bounds,
            require_valid_homography=config.require_valid_homography,
            min_homography_inlier_ratio=config.min_homography_inlier_ratio,
        )

    # ── Stage 1: Coarse scan streamed from ffmpeg ──────────────────────
    frame_bytes = scan_w * scan_h * 3
    vf = f"fps={sample_fps:.6f}:start_time=0,scale={scan_w}:{scan_h}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    samples: list[tuple[float, bool]] = []
    batch_frames: list[np.ndarray] = []
    batch_times: list[float] = []
    sample_idx = 0
    expected_total_samples = max(1, int(info.duration_s / sample_interval_s))
    last_log_t = time.monotonic()
    stage1_t0 = last_log_t

    def _flush_batch() -> None:
        if not batch_frames:
            return
        preds = _predict_batch_compat(
            court_detector,
            batch_frames,
            conf_threshold=config.conf_threshold,
            imgsz=config.imgsz,
            device=config.device,
        )
        for t_s, kps in zip(batch_times, preds):
            samples.append((t_s, _frame_visible(kps)))
        batch_frames.clear()
        batch_times.clear()

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(scan_h, scan_w, 3)
            batch_frames.append(frame)
            batch_times.append(sample_idx * sample_interval_s)
            sample_idx += 1

            if len(batch_frames) >= max(1, config.batch_size):
                _flush_batch()
                now = time.monotonic()
                if progress_callback and (now - last_log_t >= 5.0):
                    pct = min(100.0, 100.0 * sample_idx / expected_total_samples)
                    rate = sample_idx / max(1e-3, now - stage1_t0)
                    eta_s = max(0.0, (expected_total_samples - sample_idx) / max(1e-3, rate))
                    progress_callback(
                        f"[Stage 1/3] Coarse scan: {sample_idx}/{expected_total_samples} "
                        f"samples ({pct:.1f}%, {rate:.1f} samples/s, ETA {eta_s:.0f}s)"
                    )
                    last_log_t = now

        _flush_batch()
    finally:
        proc.stdout.close()
        stderr_data = proc.stderr.read() if proc.stderr else b""
        if proc.stderr:
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg coarse scan failed (exit {rc}): {stderr_data.decode(errors='replace')}"
            )

    coarse_count = len(samples)
    if progress_callback:
        pos_count = sum(1 for _, vis in samples if vis)
        progress_callback(
            f"[Stage 1/3] Coarse scan complete: {coarse_count} samples evaluated, "
            f"{pos_count} full-court ({100.0 * pos_count / max(1, coarse_count):.1f}%)"
        )

    # Identify coarse positive runs as (first_true_idx, last_true_idx)
    runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for idx, (_, vis) in enumerate(samples):
        if vis and run_start is None:
            run_start = idx
        elif not vis and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(samples) - 1))

    # ── Stage 2: Binary-search boundary refinement ─────────────────────
    refined_count = 0
    refined_segments: list[CourtSegment] = []

    if not config.refine_boundaries or sample_interval_s <= (1.5 / info.fps):
        refined_segments = intervals_from_samples(samples, sample_interval_s, info.duration_s)
    else:
        pad_s = min(0.25, 0.5 * sample_interval_s)

        for run_idx, (first_true, last_true) in enumerate(runs):
            # Refine start boundary (rising edge: False -> True)
            if first_true == 0:
                seg_start_s = 0.0
            else:
                t_false = samples[first_true - 1][0]
                t_true = samples[first_true][0]
                win_start = max(0.0, t_false - pad_s)
                win_end = min(info.duration_s, t_true + pad_s)
                win_frames = _extract_window_frames(
                    path, win_start, win_end - win_start, scan_w, scan_h
                )
                if not win_frames:
                    seg_start_s = t_true
                else:
                    cache: dict[int, bool] = {}

                    def _eval_start(i: int) -> bool:
                        nonlocal refined_count
                        if i not in cache:
                            kps = court_detector.predict(
                                win_frames[i],
                                conf_threshold=config.conf_threshold,
                                imgsz=config.imgsz,
                                device=config.device,
                            )
                            refined_count += 1
                            cache[i] = _frame_visible(kps)
                        return cache[i]

                    last_i = len(win_frames) - 1
                    if _eval_start(0):
                        edge_idx = 0
                    elif not _eval_start(last_i):
                        edge_idx = last_i
                    else:
                        edge_idx = binary_search_transition(_eval_start, 0, last_i, rising=True)
                    seg_start_s = min(info.duration_s, win_start + edge_idx / info.fps)

            # Refine end boundary (falling edge: True -> False)
            if last_true == len(samples) - 1:
                seg_end_s = info.duration_s
            else:
                t_true = samples[last_true][0]
                t_false = samples[last_true + 1][0]
                win_start = max(0.0, t_true - pad_s)
                win_end = min(info.duration_s, t_false + pad_s)
                win_frames = _extract_window_frames(
                    path, win_start, win_end - win_start, scan_w, scan_h
                )
                if not win_frames:
                    seg_end_s = t_false
                else:
                    cache_end: dict[int, bool] = {}

                    def _eval_end(i: int) -> bool:
                        nonlocal refined_count
                        if i not in cache_end:
                            kps = court_detector.predict(
                                win_frames[i],
                                conf_threshold=config.conf_threshold,
                                imgsz=config.imgsz,
                                device=config.device,
                            )
                            refined_count += 1
                            cache_end[i] = _frame_visible(kps)
                        return cache_end[i]

                    last_i = len(win_frames) - 1
                    if _eval_end(last_i):
                        edge_idx = last_i
                    elif not _eval_end(0):
                        edge_idx = -1
                    else:
                        edge_idx = binary_search_transition(_eval_end, 0, last_i, rising=False)
                    seg_end_s = min(info.duration_s, win_start + (edge_idx + 1) / info.fps)

            if seg_end_s > seg_start_s:
                refined_segments.append(CourtSegment(start_s=seg_start_s, end_s=seg_end_s))

            if progress_callback and ((run_idx + 1) % 25 == 0 or run_idx + 1 == len(runs)):
                progress_callback(
                    f"[Stage 2/3] Refined {run_idx + 1}/{len(runs)} candidate segments "
                    f"({refined_count} boundary frames evaluated)"
                )

    final_segments = merge_and_filter_segments(
        refined_segments,
        max_gap_s=config.max_gap_s,
        min_clip_s=config.min_clip_s,
        duration_s=info.duration_s,
    )

    stats = {
        "coarse_frames_evaluated": coarse_count,
        "refined_frames_evaluated": refined_count,
    }
    return final_segments, stats


def _probe_encoder_works(codec: str) -> bool:
    """Return True if ``ffmpeg`` can encode a synthetic frame using ``codec``."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.05",
        "-frames:v",
        "1",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    return proc.returncode == 0


def _select_video_encoder_args(config: PreprocessConfig) -> list[str]:
    """Choose ffmpeg video encoder flags, with hardware fallback for ``auto``."""
    codec = config.video_codec
    if codec == "auto":
        if _probe_encoder_works("h264_videotoolbox"):
            codec = "h264_videotoolbox"
        else:
            codec = "libx264"

    if codec == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt", "yuv420p"]
    if codec == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", codec, "-pix_fmt", "yuv420p"]


def _encode_single_segment(
    input_path: Path,
    segment_path: Path,
    segment: CourtSegment,
    *,
    video_args: list[str],
    include_audio: bool,
    fps: float,
) -> Path:
    """Fast-seek and encode one kept segment to an MP4 file."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment.start_s:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{segment.duration_s:.6f}",
        "-map",
        "0:v:0",
    ]
    if include_audio:
        cmd.extend(["-map", "0:a:0"])

    cmd.extend(video_args)
    cmd.extend(["-r", f"{fps:.6f}"])

    if include_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"])
    else:
        cmd.append("-an")

    cmd.extend(["-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(segment_path)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed encoding segment [{segment.start_s:.2f}, {segment.end_s:.2f}]: "
            f"{proc.stderr.strip()}"
        )
    return segment_path


def export_trimmed_mp4(
    input_path: str | Path,
    output_path: str | Path,
    segments: list[CourtSegment],
    *,
    config: PreprocessConfig,
    probe_info: Optional[VideoProbeInfo] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, list[Path]]:
    """Cut the kept segments and convert/concatenate into an MP4 using ``ffmpeg``.

    Returns
    -------
    (output_path, clip_paths)
        When ``config.split_clips`` is False (default), returns ``(output_path, [])``.
        When ``config.split_clips`` is True, also saves individual clip files alongside
        the combined trimmed MP4 and returns their paths.
    """
    if not segments:
        raise ValueError("Cannot export trimmed video: no court-visible segments were found.")

    src = Path(input_path)
    dst = Path(output_path)
    if src.resolve() == dst.resolve():
        raise ValueError(f"Output path must differ from input path: {dst}")

    _ensure_ffmpeg_tools()
    info = probe_info or probe_video_ffprobe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    video_args = _select_video_encoder_args(config)
    include_audio = bool(config.keep_audio and info.has_audio)
    workers = max(1, min(config.workers, len(segments)))

    if progress_callback:
        progress_callback(
            f"[Stage 3/3] Exporting {len(segments)} clips with ffmpeg "
            f"(codec={video_args[1]}, workers={workers}) -> {dst}"
        )

    # Single-segment fast path when split_clips is False
    if len(segments) == 1 and not config.split_clips:
        _encode_single_segment(
            src,
            dst,
            segments[0],
            video_args=video_args,
            include_audio=include_audio,
            fps=info.fps,
        )
        return dst, []

    clip_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="tennis_trim_", dir=str(dst.parent)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        seg_files: list[Path] = [tmp_path / f"seg_{i:05d}.mp4" for i in range(len(segments))]

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _encode_single_segment,
                    src,
                    seg_files[i],
                    seg,
                    video_args=video_args,
                    include_audio=include_audio,
                    fps=info.fps,
                ): i
                for i, seg in enumerate(segments)
            }
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if progress_callback and (completed % 20 == 0 or completed == len(segments)):
                    progress_callback(
                        f"[Stage 3/3] Encoded {completed}/{len(segments)} clips..."
                    )

        # Write concat manifest and stream-copy stitch into final MP4
        concat_file = tmp_path / "concat.txt"
        lines = []
        for sf in seg_files:
            escaped = str(sf.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        concat_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        proc = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.strip()}")

        if config.split_clips:
            base_stem = dst.stem
            for i, sf in enumerate(seg_files, start=1):
                clip_dst = dst.parent / f"{base_stem}_clip_{i:03d}.mp4"
                shutil.move(str(sf), str(clip_dst))
                clip_paths.append(clip_dst)

    return dst, clip_paths


def preprocess_video(
    input_path: str | Path,
    *,
    config: Optional[PreprocessConfig] = None,
    output_path: Optional[str | Path] = None,
    court_detector: Optional[CourtKeypointDetector] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> PreprocessResult:
    """Run full court-keypoint trimming and MP4 conversion on an input video."""
    t0 = time.perf_counter()
    cfg = config or PreprocessConfig()
    src = Path(input_path)
    dst = (
        Path(output_path)
        if output_path is not None
        else resolve_output_path(src, cfg.output_dir)
    )

    info = probe_video_ffprobe(src)
    segments, stats = detect_court_segments(
        src,
        config=cfg,
        probe_info=info,
        court_detector=court_detector,
        progress_callback=progress_callback,
    )

    clip_paths: list[Path] = []
    if not cfg.dry_run:
        if not segments:
            raise RuntimeError(
                f"No full-court segments found in {src} "
                f"(conf_threshold={cfg.conf_threshold}, min_clip_s={cfg.min_clip_s})."
            )
        dst, clip_paths = export_trimmed_mp4(
            src,
            dst,
            segments,
            config=cfg,
            probe_info=info,
            progress_callback=progress_callback,
        )

    kept_s = sum(seg.duration_s for seg in segments)
    removed_s = max(0.0, info.duration_s - kept_s)
    elapsed_s = time.perf_counter() - t0

    return PreprocessResult(
        input_path=src,
        output_path=dst,
        clip_paths=tuple(clip_paths),
        segments=tuple(segments),
        source_duration_s=info.duration_s,
        kept_duration_s=kept_s,
        removed_duration_s=removed_s,
        coarse_frames_evaluated=stats["coarse_frames_evaluated"],
        refined_frames_evaluated=stats["refined_frames_evaluated"],
        elapsed_s=elapsed_s,
    )

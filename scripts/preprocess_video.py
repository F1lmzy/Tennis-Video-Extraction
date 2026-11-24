#!/usr/bin/env python3
"""Preprocess full tennis match videos to keep only full-court clips.

Uses the court keypoint detection model to find segments where ALL court
keypoints are visible, cuts out all non-court footage (closeups, replays,
crowd shots, changeovers), converts the kept clips to MP4 via ``ffmpeg``,
and writes ``<stem>-trimmed.mp4`` into ``data/input/`` by default.

Usage:
    # Trim the full game MKV into data/input/game-full-trimmed.mp4
    uv run python scripts/preprocess_video.py data/input/game-full.mkv

    # Dry-run to inspect detected segments without exporting
    uv run python scripts/preprocess_video.py data/input/game-full.mkv --dry-run

    # Custom sampling rate, confidence threshold, or output directory
    uv run python scripts/preprocess_video.py \
        --input data/input/game-full.mkv \
        --output-dir data/input \
        --sample-interval 1.0 \
        --court-conf 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tennis_tracker.preprocess import (
    DEFAULT_COURT_MODEL,
    DEFAULT_OUTPUT_DIR,
    PreprocessConfig,
    PreprocessResult,
    preprocess_video,
)


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS.s for readable terminal output."""
    total = max(0.0, seconds)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60.0
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for match preprocessing."""
    parser = argparse.ArgumentParser(
        prog="preprocess-video",
        description=(
            "Trim a tennis match video using court keypoint detection, keeping "
            "only segments where all court keypoints are visible, and export "
            "as *-trimmed.mp4 in data/input via ffmpeg."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input video file(s) to preprocess (e.g. data/input/game-full.mkv).",
    )
    parser.add_argument(
        "--input",
        dest="input_flag",
        type=Path,
        action="append",
        default=[],
        help="Input video path (may be passed multiple times).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for *-trimmed.mp4 files (default: data/input).",
    )
    parser.add_argument(
        "--court-model",
        type=Path,
        default=DEFAULT_COURT_MODEL,
        help="Path to YOLO court keypoint model (default: models/court_keypoints_yolo26.pt).",
    )
    parser.add_argument(
        "--court-conf",
        type=float,
        default=0.5,
        help="Minimum confidence required for every court keypoint (default: 0.5).",
    )
    parser.add_argument(
        "--expected-keypoints",
        type=int,
        default=None,
        help="Required number of court keypoints (default: auto-detected from model, e.g. 15).",
    )
    parser.add_argument(
        "--no-homography-check",
        action="store_true",
        help="Disable planar homography geometric consistency check on detected keypoints.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Coarse scan sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for coarse court keypoint inference (default: 8).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip Stage 2 binary-search boundary refinement.",
    )
    parser.add_argument(
        "--min-clip-s",
        type=float,
        default=2.0,
        help="Discard kept clips shorter than this duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.0,
        help="Merge adjacent court segments separated by <= this gap in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu, mps, or cuda (default: cpu).",
    )
    parser.add_argument(
        "--video-codec",
        choices=["auto", "libx264", "h264_videotoolbox"],
        default="auto",
        help="ffmpeg H.264 encoder (default: auto, prefers h264_videotoolbox then libx264).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="libx264 CRF quality value; lower is higher quality (default: 20).",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="libx264 encoding speed preset (default: fast).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel ffmpeg segment encoding workers (default: 2).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop audio tracks when exporting trimmed MP4.",
    )
    parser.add_argument(
        "--split-clips",
        action="store_true",
        help="Also save each detected rally/court clip as an individual MP4 file.",
    )
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=None,
        help="Optional path to save detected court segments metadata as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect court segments and print summary without exporting video.",
    )
    return parser


def _print_summary(result: PreprocessResult) -> None:
    kept_pct = (
        100.0 * result.kept_duration_s / result.source_duration_s
        if result.source_duration_s > 0
        else 0.0
    )
    removed_pct = 100.0 - kept_pct
    print(f"\nSummary for {result.input_path}:")
    print(f"  Source duration : {_format_duration(result.source_duration_s)}")
    print(f"  Kept duration   : {_format_duration(result.kept_duration_s)} ({kept_pct:.1f}%)")
    print(f"  Removed         : {_format_duration(result.removed_duration_s)} ({removed_pct:.1f}%)")
    print(f"  Kept clips      : {len(result.segments)}")
    print(
        f"  Frames scanned  : {result.coarse_frames_evaluated} coarse + "
        f"{result.refined_frames_evaluated} boundary-refinement"
    )
    print(f"  Elapsed         : {result.elapsed_s:.1f}s")
    print(f"  Output MP4      : {result.output_path}")
    if result.clip_paths:
        print(f"  Split clips     : {len(result.clip_paths)} files in {result.output_path.parent}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_paths: list[Path] = [*args.inputs, *args.input_flag]
    if not input_paths:
        parser.error("Provide at least one input video (e.g. data/input/game-full.mkv).")

    config = PreprocessConfig(
        court_model_path=args.court_model,
        output_dir=args.output_dir,
        conf_threshold=args.court_conf,
        expected_keypoints=args.expected_keypoints,
        require_valid_homography=not args.no_homography_check,
        sample_interval_s=args.sample_interval,
        batch_size=args.batch_size,
        refine_boundaries=not args.no_refine,
        min_clip_s=args.min_clip_s,
        max_gap_s=args.max_gap_s,
        imgsz=args.imgsz,
        device=args.device,
        video_codec=args.video_codec,
        crf=args.crf,
        preset=args.preset,
        workers=args.workers,
        keep_audio=not args.no_audio,
        split_clips=args.split_clips,
        dry_run=args.dry_run,
    )

    all_manifests: list[dict] = []
    for src in input_paths:
        print(f"Preprocessing {src} ...")
        result = preprocess_video(
            src,
            config=config,
            progress_callback=lambda msg: print(msg, flush=True),
        )
        _print_summary(result)
        all_manifests.append(
            {
                "input": str(result.input_path),
                "output": str(result.output_path),
                "source_duration_s": round(result.source_duration_s, 3),
                "kept_duration_s": round(result.kept_duration_s, 3),
                "removed_duration_s": round(result.removed_duration_s, 3),
                "segments": [
                    {
                        "start_s": round(s.start_s, 3),
                        "end_s": round(s.end_s, 3),
                        "duration_s": round(s.duration_s, 3),
                    }
                    for s in result.segments
                ],
            }
        )

    if args.segments_json is not None:
        args.segments_json.parent.mkdir(parents=True, exist_ok=True)
        payload = all_manifests[0] if len(all_manifests) == 1 else all_manifests
        args.segments_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  Segments JSON   : {args.segments_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

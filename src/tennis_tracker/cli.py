"""CLI entry point for tennis-tracker."""

import argparse
from pathlib import Path


def main() -> None:
    """Parse and dispatch CLI commands."""
    parser = argparse.ArgumentParser(
        prog="tennis-tracker",
        description=(
            "Process tennis match videos to extract player and ball "
            "court-plane coordinates."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- process ---
    process_parser = subparsers.add_parser("process", help="Process a tennis video")
    process_parser.add_argument("--input", required=True, help="Path to input video")
    process_parser.add_argument("--raw-output", help="Raw CSV output path")
    process_parser.add_argument("--smoothed-output", help="Smoothed CSV output path")
    process_parser.add_argument("--video-output", help="Annotated video output path")
    process_parser.add_argument(
        "--player-model", default="yolo26n.pt", help="Player detection model path"
    )
    process_parser.add_argument(
        "--ball-model", default="yolo26n.pt", help="Ball detection model path"
    )
    process_parser.add_argument(
        "--court-model", default="yolo26n-pose.pt", help="Court keypoint model path"
    )
    process_parser.add_argument(
        "--device", default="cpu", help="Inference device (cpu, mps, cuda)"
    )
    process_parser.add_argument(
        "--target-fps", type=int, default=30, help="Target processing FPS"
    )
    process_parser.add_argument(
        "--input-resolution", default="1080p", help="Input video resolution"
    )
    process_parser.add_argument(
        "--auto-optimize-models",
        action="store_true",
        default=True,
        help="Auto-export optimized model artifacts",
    )
    process_parser.add_argument(
        "--resize-annotated-output",
        action="store_true",
        default=True,
        help="Resize source frame to make room for 2D court panel",
    )

    # --- train-ball ---
    train_ball_parser = subparsers.add_parser(
        "train-ball", help="Train/fine-tune ball detection model"
    )
    train_ball_parser.add_argument("--data", required=True, help="Dataset YAML path")
    train_ball_parser.add_argument(
        "--base-model", default="yolo26n.pt", help="Base YOLO26 model"
    )
    train_ball_parser.add_argument("--output", required=True, help="Output model path")
    train_ball_parser.add_argument(
        "--device", default="cpu", help="Training device"
    )

    # --- train-court ---
    train_court_parser = subparsers.add_parser(
        "train-court", help="Train/fine-tune court keypoint model"
    )
    train_court_parser.add_argument("--data", required=True, help="Dataset YAML path")
    train_court_parser.add_argument(
        "--base-model", default="yolo26n-pose.pt", help="Base YOLO26 pose model"
    )
    train_court_parser.add_argument(
        "--output", required=True, help="Output model path"
    )
    train_court_parser.add_argument(
        "--device", default="cpu", help="Training device"
    )

    # --- benchmark ---
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark CPU throughput"
    )
    benchmark_parser.add_argument("--input", required=True, help="Path to input video")
    benchmark_parser.add_argument("--player-model", help="Player detection model path")
    benchmark_parser.add_argument("--ball-model", help="Ball detection model path")
    benchmark_parser.add_argument("--court-model", help="Court keypoint model path")
    benchmark_parser.add_argument(
        "--device", default="cpu", help="Inference device"
    )
    benchmark_parser.add_argument(
        "--max-frames", type=int, default=100, help="Maximum frames to process"
    )

    # --- optimize-model ---
    optimize_parser = subparsers.add_parser(
        "optimize-model", help="Export/optimize a model for CPU inference"
    )
    optimize_parser.add_argument("--input", required=True, help="Input model path")
    optimize_parser.add_argument("--output", required=True, help="Output path")
    optimize_parser.add_argument(
        "--format", choices=["onnx", "openvino"], default="openvino"
    )
    optimize_parser.add_argument(
        "--quantize", choices=["int8", "fp16"], help="Quantization type"
    )

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate model accuracy against ground truth"
    )
    validate_parser.add_argument("--data", help="Validation data path")
    validate_parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Run synthetic validation fixture (no real data needed)",
    )
    validate_parser.add_argument("--player-model", help="Player model path")
    validate_parser.add_argument("--ball-model", help="Ball model path")
    validate_parser.add_argument("--court-model", help="Court model path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "process":
        _handle_process(args)
    elif args.command == "train-ball":
        _handle_train_ball(args)
    elif args.command == "train-court":
        _handle_train_court(args)
    elif args.command == "benchmark":
        _handle_benchmark(args)
    elif args.command == "optimize-model":
        _handle_optimize(args)
    elif args.command == "validate":
        _handle_validate(args)


def _handle_process(args: argparse.Namespace) -> None:
    """Run the real model-based process pipeline."""
    from tennis_tracker.pipeline import run_process

    # Resolve output paths with defaults
    base = Path(args.input).stem
    out_dir = Path("data/output")
    raw_csv = Path(args.raw_output) if args.raw_output else out_dir / f"{base}_raw.csv"
    smoothed_csv = (
        Path(args.smoothed_output) if args.smoothed_output else out_dir / f"{base}_smoothed.csv"
    )
    video_out = (
        Path(args.video_output) if args.video_output else out_dir / f"{base}_annotated.mp4"
    )

    summary = run_process(
        video_path=args.input,
        raw_csv_path=raw_csv,
        smoothed_csv_path=smoothed_csv,
        video_output_path=video_out,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        court_model_path=args.court_model,
        fps=args.target_fps,
    )

    print(f"  Processed {summary['raw_row_count']} frames")
    print(f"  Raw CSV: {raw_csv}")
    print(f"  Smoothed CSV: {smoothed_csv}")
    print(f"  Annotated video: {video_out}")
    print(f"  Homography valid: {summary['homography_valid']}")
    print(f"  Diagnostics: {summary['diagnostics_summary']}")


def _handle_train_ball(args: argparse.Namespace) -> None:
    """Placeholder: train ball model."""
    print(f"train-ball: data={args.data}, base={args.base_model}, output={args.output}")


def _handle_train_court(args: argparse.Namespace) -> None:
    """Placeholder: train court model."""
    print(
        f"train-court: data={args.data}, base={args.base_model}, output={args.output}"
    )


def _handle_benchmark(args: argparse.Namespace) -> None:
    """Run the pipeline benchmark."""
    from tennis_tracker.benchmark import run_benchmark

    report = run_benchmark(frame_count=args.max_frames)
    for line in report.summary_lines():
        print(line)


def _handle_optimize(args: argparse.Namespace) -> None:
    """Run the model optimisation workflow."""
    from tennis_tracker.optimization import optimize_model

    result = optimize_model(
        input_path=args.input,
        output_dir=args.output,
        format=args.format,
        quantize=args.quantize,
    )
    selected = result.selected
    if selected:
        print(f"  Selected artifact: {selected.artifact_path}")
        print(f"  Format: {selected.export_format}")
        if selected.benchmark_fps is not None:
            print(f"  Benchmark FPS: {selected.benchmark_fps:.1f}")
    else:
        print("  No valid artifact selected.")
    print(f"  Diagnostics: {result.diagnostics}")
    for art in result.artifacts:
        print(f"  - [{art.status}] {art.export_format}: {art.diagnostics}")


def _handle_validate(args: argparse.Namespace) -> None:
    """Run validation against ground truth."""

    # In CI/test mode, run synthetic validation to prove the CLI path works.
    # Real validation data requires user-provided annotations.
    if args.synthetic_fixture:
        report = _synthetic_validation_fixture()
        for line in report.summary_lines():
            print(line)
        return

    print(f"validate: data={args.data}")
    print("  (Provide --data with a path to validation annotations, or")
    print("   use --data --synthetic-fixture for a synthetic test run.)")


def _synthetic_validation_fixture():
    """Generate a synthetic validation report for CLI smoke-testing."""
    from tennis_tracker.validation import (
        DetectionPrediction,
        DetectionGroundTruth,
        CourtCoordinatePrediction,
        CourtCoordinateGroundTruth,
        run_validation,
    )

    # 10 perfect player detections
    player_preds = [
        DetectionPrediction(class_id=0, confidence=0.95, x1=10, y1=20, x2=40, y2=80)
        for _ in range(10)
    ]
    player_gts = [
        DetectionGroundTruth(class_id=0, x1=10, y1=20, x2=40, y2=80)
        for _ in range(10)
    ]

    # 10 perfect ball detections
    ball_preds = [
        DetectionPrediction(class_id=1, confidence=0.85, x1=50, y1=60, x2=56, y2=66)
        for _ in range(10)
    ]
    ball_gts = [
        DetectionGroundTruth(class_id=1, x1=50, y1=60, x2=56, y2=66)
        for _ in range(10)
    ]

    # 5 court coordinates within tolerance
    coord_preds = [
        CourtCoordinatePrediction(label="player_a", x_m=1.0, y_m=2.0, confidence=0.9),
        CourtCoordinatePrediction(label="player_b", x_m=-1.5, y_m=-3.0, confidence=0.8),
        CourtCoordinatePrediction(label="ball", x_m=0.05, y_m=0.1, confidence=0.7),
    ]
    coord_gts = [
        CourtCoordinateGroundTruth(label="player_a", x_m=1.05, y_m=2.05),
        CourtCoordinateGroundTruth(label="player_b", x_m=-1.45, y_m=-2.95),
        CourtCoordinateGroundTruth(label="ball", x_m=0.04, y_m=0.09),
    ]

    return run_validation(
        detections=[
            (player_preds, player_gts, "player"),
            (ball_preds, ball_gts, "ball"),
        ],
        coordinates=(coord_preds, coord_gts),
    )

"""Court keypoint detection training entry point.

Usage:
    uv run python -m tennis_tracker.training.court --help
    uv run python -m tennis_tracker.training.court \\
        --data data/datasets/abiya-thesis-tennis-court-suuzy/data.yaml \\
        --base-model yolo26n-pose.pt \\
        --output models/court_keypoints_yolo26.pt \\
        --epochs 100

This module wraps Ultralytics YOLO pose training for the court keypoint model.
It does not start training at import time.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for court keypoint training."""
    parser = argparse.ArgumentParser(
        prog="tennis-tracker-train-court",
        description="Train/fine-tune a YOLO26 court keypoint (pose) model.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (e.g., data.yaml from Roboflow export)",
    )
    parser.add_argument(
        "--base-model",
        default="yolo26n-pose.pt",
        help="Base YOLO26 pose model path or Ultralytics short name",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for trained model weights",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda", "0", "1"],
        help="Training device",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for training",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disable)",
    )
    parser.add_argument(
        "--kobj",
        type=float,
        default=None,
        help="Keypoint objectness loss weight (pose-specific)",
    )
    return parser


def run_training(args: argparse.Namespace) -> None:
    """Execute court keypoint training.

    This function is separated from main() so it can be called from
    the top-level CLI or tested with mocked Ultralytics.
    """
    from ultralytics import YOLO

    model = YOLO(args.base_model)
    output_path = Path(args.output)
    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_path.parent),
        name=output_path.stem,
        exist_ok=True,
        patience=args.patience,
    )
    if args.kobj is not None:
        kwargs["kobj"] = args.kobj

    model.train(**kwargs)

    import shutil

    src = output_path.parent / output_path.stem / "weights" / "best.pt"
    if src.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(output_path))
        print(f"Trained model saved to: {output_path}")


def main() -> None:
    """Parse args and run training."""
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

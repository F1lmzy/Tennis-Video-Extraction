#!/usr/bin/env python3
"""Download the pretrained RacketVision TrackNetV3 BallTrack checkpoint.

Downloads ``checkpoints/balltrack_best.pth`` from
``linfeng302/RacketVision-Models`` on Hugging Face into
``models/balltrack_best.pth`` by default.

Usage:
    uv run python scripts/download_tracknet_checkpoint.py
    uv run python scripts/download_tracknet_checkpoint.py --output models/balltrack_best.pth
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HF_REPO_ID = "linfeng302/RacketVision-Models"
HF_FILENAME = "checkpoints/balltrack_best.pth"
DEFAULT_OUTPUT = Path("models/balltrack_best.pth")


def download_tracknet_checkpoint(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> Path:
    """Download ``balltrack_best.pth`` from Hugging Face if missing."""
    output_path = Path(output_path)
    if output_path.is_file() and not force:
        print(f"Checkpoint already exists: {output_path}")
        return output_path

    from huggingface_hub import hf_hub_download

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO_ID}/{HF_FILENAME} -> {output_path} ...")
    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        filename=HF_FILENAME,
    )
    shutil.copy2(cached, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.1f} MiB)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download RacketVision TrackNetV3 BallTrack checkpoint from Hugging Face."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination checkpoint path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if destination file already exists.",
    )
    args = parser.parse_args()
    download_tracknet_checkpoint(args.output, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

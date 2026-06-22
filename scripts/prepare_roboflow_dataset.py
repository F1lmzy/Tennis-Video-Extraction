#!/usr/bin/env python3
"""Download and prepare Roboflow datasets for tennis model training.

Usage:
    cp .env.example .env
    # Edit .env and set ROBOFLOW_API_KEY=your_key_here
    uv run python scripts/prepare_roboflow_dataset.py --dataset court
    uv run python scripts/prepare_roboflow_dataset.py --dataset ball --version 1
    uv run python scripts/prepare_roboflow_dataset.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known workspace/project identifiers (approved in docs/spec.md)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "court": {
        "workspace": "abiya-thesis",
        "project": "tennis-court-suuzy",
        "description": "Court keypoint detection (pose/keypoint model)",
        "default_version": "8",
    },
    "ball": {
        "workspace": "viren-dhanwani",
        "project": "tennis-ball-detection",
        "description": "Ball detection (detection model)",
        "default_version": "6",
    },
}

# ---------------------------------------------------------------------------
# Default output root (matches .gitignore patterns)
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_ROOT = Path("data/datasets")

# ---------------------------------------------------------------------------
# Supported export formats (Roboflow Python library)
# ---------------------------------------------------------------------------
_SUPPORTED_FORMATS = [
    "yolo26",        # YOLO detection format (default for ball)
    "yolo26",   # YOLO pose/keypoint format (default for court)
    "coco",
    "voc",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare-roboflow-dataset",
        description=(
            "Download a Roboflow dataset for tennis model training using "
            "the roboflow Python library.  Requires a valid API key from "
            "--api-key, ROBOFLOW_API_KEY, or a .env file."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY.keys()),
        required=True,
        help="Which dataset to download",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "Roboflow workspace slug.  Overrides the default for the "
            "chosen --dataset."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Roboflow project slug.  Overrides the default for the "
            "chosen --dataset."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Dataset version number.  Defaults to the latest available "
            "version for the project."
        ),
    )
    parser.add_argument(
        "--format",
        choices=_SUPPORTED_FORMATS,
        default=None,
        help=(
            "Export format.  Default: yolov8-pose for court, yolov8 for ball."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory.  Defaults to "
            "data/datasets/{workspace}-{project}/"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Roboflow API key. Falls back to ROBOFLOW_API_KEY env var, then "
            "ROBOFLOW_API_KEY from --env-file."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file containing ROBOFLOW_API_KEY. Default: .env",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without downloading anything.",
    )

    return parser


def _load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env file.

    This intentionally supports only the subset needed here so the dataset
    helper does not require an extra runtime dependency such as python-dotenv.
    Missing files return an empty mapping.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _resolve_args(args: argparse.Namespace) -> dict:
    """Resolve defaults, validate, and return a flat config dict."""
    info = DATASET_REGISTRY[args.dataset]
    workspace = args.workspace or info["workspace"]
    project = args.project or info["project"]
    version = args.version or info["default_version"]

    # Resolve output format
    fmt = args.format
    if fmt is None:
        fmt = "yolo26"

    # Resolve output path
    output = args.output
    if output is None:
        output = _DEFAULT_OUTPUT_ROOT / f"{workspace}-{project}"

    # Resolve API key. Precedence: explicit CLI > process env > .env file.
    env_file_values = _load_env_file(args.env_file)
    api_key = (
        args.api_key
        or os.environ.get("ROBOFLOW_API_KEY")
        or env_file_values.get("ROBOFLOW_API_KEY")
    )

    return {
        "dataset_key": args.dataset,
        "workspace": workspace,
        "project": project,
        "version": version,
        "format": fmt,
        "output": output.resolve(),
        "api_key": api_key,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
        "env_file": args.env_file,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _resolve_args(args)

    # ── Help-only is safe without API key ──────────────────────────────
    # (Already handled by argparse --help)

    # ── Report configuration ───────────────────────────────────────────
    info = DATASET_REGISTRY[args.dataset]
    print(f"Dataset:          {args.dataset} ({info['description']})")
    print(f"Workspace/project: {config['workspace']}/{config['project']}")
    print(f"Version:          {config['version']}")
    print(f"Format:           {config['format']}")
    print(f"Output:           {config['output']}")
    print(f"Env file:         {config['env_file']}")
    print(f"API key set:      {'Yes' if config['api_key'] else 'No'}")
    print(f"Overwrite:        {'Yes' if config['overwrite'] else 'No'}")

    if config["dry_run"]:
        print("\n[Dry-run mode] No files were downloaded.")
        sys.exit(0)

    # ── Validate API key ───────────────────────────────────────────────
    if not config["api_key"]:
        print(
            "\nError: No API key found. Set ROBOFLOW_API_KEY in .env, "
            "export ROBOFLOW_API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Check output path ──────────────────────────────────────────────
    output_path = Path(config["output"])
    if output_path.exists() and not config["overwrite"]:
        print(
            f"\nError: Output path already exists: {output_path}\n"
            "Use --overwrite to replace existing files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Download via roboflow library ──────────────────────────────────
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=config["api_key"])
        project = rf.workspace(config["workspace"]).project(config["project"])
        project.version(config["version"]).download(
            model_format=config["format"],
            location=str(output_path),
            overwrite=config["overwrite"],
        )

        print(
            f"\nDownload complete.\n"
            f"  Location:  {output_path}\n"
            f"  Samples:   {len(list(output_path.rglob('*')))} files "
            "(approx)\n"
            f"  Training YAML expected at: "
            f"{output_path / 'data.yaml'}"
        )

    except ImportError:
        print(
            "\nError: The 'roboflow' Python package is not installed.\n"
            "Install it with:  uv add roboflow  (or pip install roboflow)",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"\nError downloading dataset: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Process a tennis match video on a Google Colab GPU VM via the `colab` CLI.
#
# Usage:
#   ./scripts/process_video.sh [INPUT_VIDEO] [OPTIONS...]
#
# Flags:
#   --3d, --3d-court-view   Include a side-by-side 3D corner court panel next to the 2D overview
#   --gpu <T4|L4|A100|...>  Colab GPU accelerator type (default: T4)
#   --dry-run               Print planned commands without provisioning a VM
#   (any other tennis_tracker process flags are forwarded verbatim)
#
# Examples:
#   ./scripts/process_video.sh
#   ./scripts/process_video.sh data/input/match.mp4 --3d-court-view
#   ./scripts/process_video.sh data/input/match.mp4 --3d --gpu L4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_VIDEO="data/input/match.mp4"
if [[ $# -gt 0 && "$1" != --* ]]; then
  INPUT_VIDEO="$1"
  shift
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --3d|--3d-court-view)
      EXTRA_ARGS+=("--3d-court-view")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "Error: Input video not found at '${INPUT_VIDEO}'" >&2
  echo "Place a video at data/input/match.mp4 or pass a path: $0 path/to/video.mp4" >&2
  exit 1
fi

exec uv run python scripts/process_on_colab.py \
  --input "${INPUT_VIDEO}" \
  --player-model yolo26n.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt \
  --device cuda \
  --target-fps 30 \
  --imgsz 1280 \
  "${EXTRA_ARGS[@]}"

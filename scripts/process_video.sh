#!/usr/bin/env bash
set -euo pipefail

# Process a tennis match video on a Google Colab GPU VM via the `colab` CLI.
#
# Usage:
#   ./scripts/process_video.sh [INPUT_VIDEO] [OPTIONS...]
#
# Flags:
#   --3d, --3d-court-view   Include a side-by-side 3D corner court panel (enabled by default)
#   --no-3d                 Disable the 3D corner court panel
#   --gpu <T4|L4|A100|...>  Colab GPU accelerator type (default: T4)
#   --dry-run               Print planned commands without provisioning a VM
#   (any other tennis_tracker process flags are forwarded verbatim)
#
# Examples:
#   ./scripts/process_video.sh
#   ./scripts/process_video.sh data/input/match.mp4
#   ./scripts/process_video.sh data/input/match.mp4 --no-3d
#   ./scripts/process_video.sh data/input/match.mp4 --gpu L4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_VIDEO="data/input/match.mp4"
if [[ $# -gt 0 && "$1" != --* ]]; then
  INPUT_VIDEO="$1"
  shift
fi

ENABLE_3D=1
BALL_MODEL="models/balltrack_best.pth"
if [[ ! -f "${BALL_MODEL}" ]]; then
  BALL_MODEL="models/ball_yolo26.pt"
fi
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --3d|--3d-court-view)
      ENABLE_3D=1
      shift
      ;;
    --no-3d)
      ENABLE_3D=0
      shift
      ;;
    --yolo-ball)
      BALL_MODEL="models/ball_yolo26.pt"
      shift
      ;;
    --ball-model)
      BALL_MODEL="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${ENABLE_3D}" -eq 1 ]]; then
  EXTRA_ARGS+=("--3d-court-view")
fi

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "Error: Input video not found at '${INPUT_VIDEO}'" >&2
  echo "Place a video at data/input/match.mp4 or pass a path: $0 path/to/video.mp4" >&2
  exit 1
fi

exec uv run python scripts/process_on_colab.py \
  --input "${INPUT_VIDEO}" \
  --player-model yolo26n.pt \
  --ball-model "${BALL_MODEL}" \
  --court-model models/court_keypoints_yolo26.pt \
  --device cuda \
  --imgsz 1280 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

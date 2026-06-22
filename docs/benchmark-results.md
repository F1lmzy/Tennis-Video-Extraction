# Benchmark Results

> **⚠️ MANUAL STEP** — This file requires a real M1 benchmark run with trained models and representative 1080p input.
> Complete below once you have the required artifacts.

## Prerequisites Checklist

- [ ] Trained ball detection model (`models/ball_yolo26.pt`)
- [ ] Trained court keypoint model (`models/court_keypoints_yolo26.pt`)
- [ ] Player model (YOLO26 COCO `person`, e.g. `yolo26n.pt`)
- [ ] Representative 1080p fixed-camera singles match video (`data/input/benchmark.mp4`)
- [ ] Standard Apple M1 hardware
- [ ] Optimized artifacts generated (optional, improves FPS)

## Commands

```bash
# Basic benchmark (no model optimization)
uv run python -m tennis_tracker benchmark \
  --input data/input/benchmark.mp4 \
  --player-model yolo26n.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt \
  --device cpu \
  --max-frames 500

# Benchmark with optimized artifacts
uv run python -m tennis_tracker optimize-model \
  --input models/ball_yolo26.pt \
  --output models/optimized/ --format openvino --quantize int8

uv run python -m tennis_tracker benchmark \
  --input data/input/benchmark.mp4 \
  --player-model yolo26n.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt \
  --device cpu \
  --max-frames 500
```

## Results

| Metric | Value | Target | Pass? |
|---|---|---|---|
| Total FPS | `___` | ≥ 30 | ☐ |
| Decode FPS | `___` | — | — |
| Court FPS | `___` | — | — |
| Player FPS | `___` | — | — |
| Ball FPS | `___` | — | — |
| Tracking/Projection FPS | `___` | — | — |
| Rendering FPS | `___` | — | — |
| Total time (500 frames) | `___` s | — | — |
| Input resolution | 1080p | — | — |
| Device | Apple M1 CPU | — | — |

## Mitigation Notes (if 30 FPS not met)

- Reduce inference resolution (`--input-resolution 720p`)
- Enable ONNX or OpenVINO export + INT8 quantization
- Use smaller YOLO variant (e.g. `yolo26n` → `yolo26t`)
- Benchmark rendering separately; disable `--resize-annotated-output`
- Reduce frame processing (run benchmark with `--max-frames 100` for quick iteration)

## Hardware

| Field | Value |
|---|---|
| Chip | Apple M1 |
| RAM | `___` GB |
| macOS version | `___` |
| Python version | 3.11+ |
| OpenCV version | `___` |
| Ultralytics version | `___` |

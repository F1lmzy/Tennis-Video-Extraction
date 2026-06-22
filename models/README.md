# Model Artifacts

This directory holds trained and optimized model artifacts for the tennis tracking pipeline.

**All model files are gitignored** and should not be committed to the repository.
Only this README is versioned.

## Model Types

| File Pattern | Model Type | Task |
|---|---|---|
| `player_*.pt` | YOLO26 detection | Player (person) detection |
| `ball_*.pt` | YOLO26 detection | Ball detection |
| `court_keypoints_*.pt` | YOLO26 pose | Court keypoint detection |
| `*.onnx` | ONNX export | CPU-optimized inference |
| `*_openvino/` | OpenVINO IR | CPU-optimized inference |

## Source Datasets

### Ball Detection

- **Dataset:** `viren-dhanwani/tennis-ball-detection`
- **Roboflow workspace:** `viren-dhanwani`
- **Roboflow project:** `tennis-ball-detection`
- **License:** As listed on the Roboflow dataset page (accepted for this project)
- **Training command:** `uv run python -m tennis_tracker.train_ball`

### Court Keypoint Detection

- **Dataset:** `abiya-thesis/tennis-court-suuzy`
- **Roboflow workspace:** `abiya-thesis`
- **Roboflow project:** `tennis-court-suuzy`
- **License:** As listed on the Roboflow dataset page (accepted for this project)
- **Training command:** `uv run python -m tennis_tracker.train_court`

### Player Detection

- First version uses YOLO26 pretrained COCO `person` class (no training required)
- A dedicated tennis-player model may be added later if validation shows COCO is insufficient

## Dataset Preparation

Create a local `.env` file at the repository root:

```bash
cp .env.example .env
# Edit .env and set:
# ROBOFLOW_API_KEY=your_key_here
```

`.env` is gitignored and must not be committed.

Then download datasets; the helper reads `.env` automatically:

```bash
# Download court dataset
uv run python scripts/prepare_roboflow_dataset.py --dataset court

# Download ball dataset
uv run python scripts/prepare_roboflow_dataset.py --dataset ball

# Optional: use another env file
uv run python scripts/prepare_roboflow_dataset.py --dataset ball --env-file .env.local
```

## Model Training

```bash
uv run python -m tennis_tracker.train_ball \
  --data data/datasets/viren-dhanwani-tennis-ball-detection/data.yaml \
  --base-model yolo26n.pt \
  --output models/ball_yolo26.pt

uv run python -m tennis_tracker.train_court \
  --data data/datasets/abiya-thesis-tennis-court-suuzy/data.yaml \
  --base-model yolo26n-pose.pt \
  --output models/court_keypoints_yolo26.pt
```

## Model Optimization

```bash
uv run python -m tennis_tracker optimize-model \
  --input models/ball_yolo26.pt \
  --output models/optimized/ \
  --format openvino \
  --quantize int8
```

## Gitignore Rules

The `.gitignore` file covers these patterns for this directory:

```gitignore
models/*.pt
models/*.onnx
models/*.xml
models/*.bin
models/*openvino*/
```

This ensures model weights and optimized artifacts are never accidentally committed.

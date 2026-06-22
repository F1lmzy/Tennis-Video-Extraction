# Validation Results

> **⚠️ MANUAL STEP** — This file requires ~100 manually annotated/reviewed validation images,
> trained models, and running the validate CLI.
> Complete below once you have the required artifacts.

## Prerequisites Checklist

- [ ] ~100 annotated validation images with:
  - Detection bounding boxes for players (person) and ball
  - Court keypoint annotations (if available)
  - Court-plane coordinate ground truth for players (≤ 0.2 m tolerance) and ball (≤ 0.1 m tolerance)
- [ ] Validation annotations in a readable format (JSON or CSV)
- [ ] Trained ball detection model (`models/ball_yolo26.pt`)
- [ ] Trained court keypoint model (`models/court_keypoints_yolo26.pt`)
- [ ] Player model (`yolo26n.pt` or fine-tuned variant)

## Commands

```bash
# Run validation with real annotations
uv run python -m tennis_tracker validate \
  --data data/validation/annotations.json \
  --player-model yolo26n.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt

# Run synthetic fixture to verify CLI path works (no real data needed)
uv run python -m tennis_tracker validate --synthetic-fixture
```

## Results

### Detection Precision/Recall (target: >99%)

| Task | TP | FP | FN | Precision | Recall | Pass? |
|---|---|---|---|---|---|---|
| Player (person) | `___` | `___` | `___` | `___` | `___` | ☐ |
| Ball | `___` | `___` | `___` | `___` | `___` | ☐ |
| Court keypoints | `___` | `___` | `___` | `___` | `___` | ☐ |

### Coordinate Tolerances

| Label | Within Tolerance | Total | Tolerance (m) | Pass Rate | Pass? |
|---|---|---|---|---|---|
| Player A | `___` | `___` | 0.2 | `___` | ☐ |
| Player B | `___` | `___` | 0.2 | `___` | ☐ |
| Ball | `___` | `___` | 0.1 | `___` | ☐ |

### Overall Verdict

| Criterion | Pass? |
|---|---|
| All detection tasks >99% precision/recall | ☐ |
| All coordinate tolerances met | ☐ |
| **Overall** | ☐ |

## Mitigation Notes (if targets not met)

### Detection < 99%

- Player detector: fine-tune a dedicated tennis-player model on tennis-specific data
- Ball detector: increase training epochs, add more ball-specific augmentation, improve dataset quality
- Court keypoints: verify keypoint mapping table, increase training data, adjust `--kobj` weight

### Coordinate Tolerance Exceeded

- Verify court keypoint → homography mapping table
- Check court keypoint detection quality per frame
- Increase temporal smoothing window
- Validate reprojection error before projecting coordinates

### Dataset / Annotation Notes

| Field | Value |
|---|---|
| Validation image count | `___` |
| Annotation format | `___` (JSON / CSV / YOLO) |
| Source dataset(s) | `___` |
| Annotation tool | `___` |
| Data split | `___` (train/val/test) |

# Spec: Tennis Video Player/Ball Tracking and Court-Plane Coordinate Extraction

## Objective
Build a CLI-based computer vision pipeline that processes a fixed-camera tennis match recording from behind the baseline and outputs research/production-grade 2D court-plane coordinates for the players and tennis ball.

The system will:

- Detect and track both tennis players across video frames.
- Detect and track the tennis ball across video frames.
- Automatically detect tennis court keypoints/lines using a trained court keypoint model.
- Estimate a homography from image pixels to real-world court coordinates using known tennis court dimensions.
- Convert player and ball positions from image coordinates into real-world court coordinates measured in meters.
- Export two frame-by-frame CSV files: raw coordinates and smoothed coordinates.
- Generate an annotated output video showing ball tracking, court keypoints, and a side-by-side/top-view 2D court animation with player and ball positions.

Primary user: a researcher, analyst, or engineer who wants structured spatial tracking data from tennis match footage.

Success means the user can run one CLI command on a fixed-camera tennis video and receive raw and smoothed CSV outputs plus an annotated video containing stable real-world court coordinates for players and ball, with quality diagnostics sufficient to trust or reject each frame's output. The target runtime mode is 30 FPS real-time processing on Apple M-series CPU hardware, which must be benchmarked and treated as a first-class product requirement.

## Tech Stack

- Language: Python 3.11+
- CLI: `argparse` or `typer`
- Detection/tracking: Ultralytics YOLO26 via `ultralytics>=8.4.0`
- Model tasks:
  - Player detection/tracking: YOLO26 pretrained COCO `person` class for the first version; a dedicated tennis-player model can be added later if validation shows COCO is insufficient
  - Ball detection/tracking: YOLO26 detection model fine-tuned on the Roboflow tennis ball dataset: https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection
  - Court keypoint detection: YOLO26 pose/keypoint model or compatible keypoint detector trained on the Roboflow tennis court dataset: https://universe.roboflow.com/abiya-thesis/tennis-court-suuzy
- Video processing/rendering: OpenCV
- 2D court animation overlay: OpenCV drawing primitives initially; optionally a dedicated renderer later if needed
- Numeric computation: NumPy
- Data output: pandas or Python CSV module
- Tracking/smoothing:
  - Ultralytics tracking modes where useful
  - Additional temporal smoothing/interpolation for ball trajectory and court homography stability
- CPU acceleration/export: ONNX and/or OpenVINO export, quantization, and model-size selection are allowed to meet 30 FPS on Apple M-series CPU.
- Optimized model artifacts must be generated automatically by the setup/training/optimization workflow when missing or stale.
- Optimization workflow benchmarks ONNX and OpenVINO artifacts on the target machine and selects the faster valid artifact for runtime.
- Target hardware: standard Apple M1 chip, CPU-first execution.
- Testing: pytest
- Optional experiment tooling: notebooks under `notebooks/`, but production entry point remains CLI

## Commands

Initial expected commands:

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run lint/format checks, if configured
ruff check .
ruff format --check .

# Format code
ruff format .

# Process a video and generate CSV + annotated video outputs
python -m tennis_tracker process \
  --input data/input/match.mp4 \
  --raw-output data/output/match_raw_coordinates.csv \
  --smoothed-output data/output/match_smoothed_coordinates.csv \
  --video-output data/output/match_annotated.mp4 \
  --player-model models/player_yolo26.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt \
  --device cpu \
  --target-fps 30 \
  --input-resolution 1080p \
  --auto-optimize-models \
  --resize-annotated-output

# Train/fine-tune the ball model from a dataset config
python -m tennis_tracker.train_ball \
  --data data/datasets/tennis_ball/data.yaml \
  --base-model yolo26n.pt \
  --output models/ball_yolo26.pt \
  --device cpu

# Train/fine-tune the court keypoint model from a dataset config
python -m tennis_tracker.train_court \
  --data data/datasets/tennis_court_keypoints/data.yaml \
  --base-model yolo26n-pose.pt \
  --output models/court_keypoints_yolo26.pt \
  --device cpu

# Benchmark CPU throughput on a representative video
python -m tennis_tracker benchmark \
  --input data/input/match.mp4 \
  --player-model models/player_yolo26.pt \
  --ball-model models/ball_yolo26.pt \
  --court-model models/court_keypoints_yolo26.pt \
  --device cpu

# Export/optimize a trained model for CPU inference manually if automatic optimization needs to be rerun
python -m tennis_tracker optimize_model \
  --input models/ball_yolo26.pt \
  --output models/ball_yolo26_openvino/ \
  --format openvino \
  --quantize int8
```

## Project Structure

```text
tennis-video-extraction/
  docs/
    spec.md                         # This specification
  data/
    input/                          # Local input videos, gitignored
    output/                         # Generated CSV/debug outputs, gitignored
    datasets/                       # Downloaded/exported datasets, gitignored or DVC-managed
  models/
    README.md                       # Model download/training notes
    *.pt                            # Trained YOLO26 weights, gitignored unless intentionally versioned
  notebooks/
    exploration/                    # Optional research notebooks, not production path
  scripts/
    prepare_roboflow_dataset.py     # Dataset download/export helpers, no secrets committed
  src/
    tennis_tracker/
      __init__.py
      __main__.py                   # CLI entry point
      cli.py                        # Argument parsing and command dispatch
      video.py                      # Video frame reading/writing utilities
      detection.py                  # YOLO26 model loading and inference wrappers
      tracking.py                   # Player/ball tracking and ID stabilization
      court.py                      # Court keypoint detection and homography estimation
      coordinates.py                # Pixel-to-court coordinate conversion
      smoothing.py                  # Temporal filtering/interpolation
      render.py                     # Annotated video and top-view court animation rendering
      output.py                     # Raw/smoothed CSV writing and schema validation
      diagnostics.py                # Confidence scores, warnings, quality metrics
      benchmark.py                  # CPU performance measurement
      optimization.py               # ONNX/OpenVINO export and quantization helpers
      training/
        ball.py                     # Ball model training/fine-tuning entry point
        court.py                    # Court keypoint model training/fine-tuning entry point
        players.py                  # Optional player model fine-tuning entry point
  tests/
    unit/
      test_coordinates.py
      test_court.py
      test_output.py
      test_smoothing.py
    integration/
      test_process_short_video.py
  requirements.txt
  pyproject.toml
  README.md
```

## Code Style

Use typed, small, testable Python functions. Keep model inference wrappers separate from geometry and CSV logic so geometry can be tested without video/model dependencies.

Example style:

```python
from dataclasses import dataclass
from typing import Literal

import numpy as np

TrackKind = Literal["player_a", "player_b", "ball"]


@dataclass(frozen=True)
class CourtPoint:
    """A point on the tennis court plane, measured in meters."""

    x_m: float
    y_m: float
    confidence: float


def pixel_to_court_point(
    pixel_xy: tuple[float, float],
    homography: np.ndarray,
    confidence: float,
) -> CourtPoint:
    """Project an image-space pixel into the court coordinate system."""
    px = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    projected = homography @ px

    if projected[2] == 0:
        raise ValueError("Invalid homography projection with zero scale")

    projected /= projected[2]
    return CourtPoint(
        x_m=float(projected[0]),
        y_m=float(projected[1]),
        confidence=confidence,
    )
```

Conventions:

- Use `snake_case` for functions, variables, and module names.
- Use `PascalCase` for dataclasses and classes.
- Use explicit units in names where possible, e.g. `x_m`, `fps`, `frame_index`.
- Avoid mixing pixel coordinates and court coordinates in the same structure unless clearly named.
- Keep CLI behavior deterministic and reproducible through explicit config/model paths.
- Never hide low-confidence detections silently; mark them in diagnostics/output.

## Testing Strategy

Testing must cover geometry and data correctness first, then model-pipeline integration.

### Unit tests

Located under `tests/unit/`.

Required coverage:

- Homography estimation from known court keypoints.
- Pixel-to-court coordinate conversion.
- CSV schema and missing-value handling.
- Track smoothing/interpolation behavior.
- Ball/player coordinate selection rules.
- Failure handling when court keypoints are missing or low confidence.

### Integration tests

Located under `tests/integration/`.

Required coverage:

- Process a short sample video or synthetic clip through the CLI.
- Confirm CSV is produced with expected columns.
- Confirm frame count/time columns are valid.
- Confirm diagnostics are emitted for low-confidence or missing detections.

### Manual/benchmark validation

For research/production-grade output, the project needs a validation set with manually annotated player/ball/court positions.

Metrics should include:

- Court keypoint reprojection error in pixels.
- Player coordinate error in meters against annotated reference points.
- Ball coordinate error in meters where ground-truth ball position is available.
- Track continuity: percentage of frames with valid player IDs and ball coordinates.
- Runtime on standard Apple M1 CPU: frames per second, total processing time, and whether 30 FPS real-time throughput is reached on 1080p input.
- Output video correctness: annotated ball trail, court keypoints, and synchronized top-view 2D court animation are visually aligned with the source video.

## CSV Output Schema

The pipeline writes two coordinate CSV files with the same schema:

1. Raw output: direct per-frame detections projected to court coordinates.
2. Smoothed output: temporally filtered/interpolated coordinates suitable for analysis and visualization.

Initial CSV schema:

```csv
frame_index,time_s,player_a_x_m,player_a_y_m,player_a_pixel_x,player_a_pixel_y,player_a_confidence,player_b_x_m,player_b_y_m,player_b_pixel_x,player_b_pixel_y,player_b_confidence,ball_x_m,ball_y_m,ball_pixel_x,ball_pixel_y,ball_confidence,court_confidence,homography_valid,diagnostics
```

Rules:

- One row per video frame.
- Court coordinates are in meters on the court plane.
- Debug pixel coordinates are included for player foot/contact points and ball center points.
- Missing detections are represented as empty CSV fields, not zeroes.
- `homography_valid` is `true` only when court calibration for that frame or nearby stabilized frame is valid.
- `diagnostics` contains compact semicolon-separated flags, e.g. `missing_ball;low_court_confidence`.

## Coordinate System

Use a real-world tennis court coordinate system measured in meters.

Confirmed convention:

- Origin `(0, 0)` is the center of the full doubles court.
- X-axis is the short axis of the court, across the court width.
- Y-axis is the long axis of the court, along the court length.
- Orientation is court-rule-fixed rather than camera-relative.
- Positive Y points toward the far baseline from the camera-facing baseline perspective.
- Positive X points to the right from the court center in the top-view court coordinate system.
- Full doubles court dimensions:
  - Width: 10.97 m
  - Length: 23.77 m
- Singles sidelines and service boxes can be represented as known internal keypoints for calibration.

This origin and axis convention is approved: origin at full doubles court center, X across court width with positive X to the right, Y along court length with positive Y toward the far baseline.

## Processing Pipeline

1. Load video and metadata.
2. Load YOLO26 player, ball, and court keypoint models.
3. For each frame or sampled calibration frame:
   - Detect court keypoints/lines.
   - Match detected keypoints to known court geometry.
   - Estimate homography from image pixels to court meters.
   - Stabilize homography over time.
4. For each frame:
   - Detect and track players.
   - Detect and track ball.
   - Choose foot/contact point for each player bounding box, usually bottom-center of box.
   - Choose ball center point from ball detection.
   - Project these points onto the court plane using the current stabilized homography.
   - Apply temporal smoothing/interpolation where appropriate.
   - Emit confidence and diagnostics.
5. Write raw CSV output before smoothing.
6. Apply smoothing/interpolation and write smoothed CSV output.
7. Render annotated output video containing:
   - Resized source frame with ball tracking overlay.
   - Court keypoint overlay.
   - Player tracking overlay.
   - A side panel with top-down 2D court animation showing player and ball positions in court coordinates.
8. Emit benchmark and diagnostic summary.

## Boundaries

### Always

- Use explicit model paths and dataset/model version notes.
- Keep raw videos, datasets, generated outputs, and large model files out of git unless explicitly approved.
- Preserve confidence scores and missing-detection markers in the CSV.
- Validate homography before projecting coordinates.
- Include diagnostics for low-confidence court, player, or ball detections.
- Run tests before claiming a change is complete.
- Optimize for CPU inference because CPU is the primary deployment target.
- Treat 30 FPS real-time CPU throughput on standard Apple M1 hardware with 1080p input as a core requirement, and measure FPS on every representative benchmark run.
- Keep training scripts reproducible with explicit dataset config paths, model names, epochs, image sizes, and output directories.
- Keep the first production scope limited to singles matches.
- Assume no player side changes within the first-version input video.

### Ask first

- Adding nontrivial dependencies beyond the planned stack.
- Changing the coordinate-system convention.
- Switching from YOLO26 to another model family.
- Requiring GPU for the main processing path.
- Relaxing the 30 FPS real-time processing target.
- Expanding scope to handle side changes or player identity swaps across sides.
- Changing the CSV schema after downstream users depend on it.
- Introducing cloud services or hosted APIs.
- Using paid datasets, paid labeling tools, or paid Roboflow plans.

### Never

- Commit private videos, dataset exports, secrets, API keys, or large generated files.
- Silently replace missing detections with zero coordinates.
- Report coordinates without calibration validity/confidence.
- Edit third-party/vendor files directly.
- Remove failing tests to make the build pass.
- Assume broadcast camera cuts/zooms are supported unless explicitly added to scope.
- Claim research/production-grade quality from model confidence alone without ground-truth validation.

## Success Criteria

The project is successful when:

1. A user can process a fixed behind-baseline singles tennis video from the CLI using CPU.
2. The CLI produces two CSV files, raw and smoothed, with one row per frame and the agreed schema.
3. Player and ball coordinates are expressed in real-world meters on the court plane, centered at the full doubles court center.
4. Court calibration is automatic using a court keypoint/line model and known tennis court dimensions.
5. The pipeline emits confidence and diagnostic information for every frame.
6. The system supports YOLO26 model weights for player, ball, and court keypoint detection.
7. Training scripts exist for ball and court keypoint models using Roboflow-exported datasets.
8. The CLI produces an annotated video with source-frame overlays and a synchronized top-view 2D court animation panel.
9. Unit tests validate coordinate conversion, homography behavior, CSV output, rendering coordinate mapping, and smoothing logic.
10. An integration test confirms the CLI can process a short video and produce raw CSV, smoothed CSV, and annotated video output.
11. CPU performance is measured and reported on a standard Apple M1 chip using 1080p input.
12. The main processing target is 30 FPS real-time CPU throughput on standard Apple M1 hardware.
13. Model optimization artifacts are generated automatically when missing or stale.
14. Accuracy is evaluated against a manually annotated validation sample before calling the output research/production-grade.
15. Detection precision and recall targets are both above 99% for the validated player, ball, and court-keypoint detection tasks.
16. Coordinate accuracy is separately evaluated using tolerance thresholds of 0.2 m for player positions and 0.1 m for ball positions.
17. Validation uses approximately 100 manually reviewed/annotated images before claiming >99% precision/recall performance.
18. Selected Roboflow dataset licenses/usage terms are considered acceptable for this project.

## Open Questions

None for the current MVP spec. Future decisions may be added if validation shows the COCO `person` detector, selected datasets, or 30 FPS M1 target need revision.

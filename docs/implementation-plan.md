# Implementation Plan: Tennis Video Player/Ball Tracking and Court-Plane Coordinate Extraction

## Overview

Build the project in vertical, verifiable slices: first establish the Python package, data contracts, court geometry, CSV output, and synthetic integration path; then add model wrappers, training, optimization, real video processing, rendering, benchmarking, and validation. The highest-risk areas are court homography correctness, 30 FPS CPU performance on a standard M1 with 1080p input, and the >99% detection precision/recall target, so those are introduced and measured early rather than deferred.

## Architecture Decisions

- **CLI-first Python package:** production use is through `python -m tennis_tracker ...`, with notebooks limited to exploration.
- **Typed internal data contracts:** detections, court keypoints, homographies, track rows, diagnostics, and output rows should have explicit dataclasses/types before model integration.
- **Geometry is independent of models:** court coordinate conversion and homography logic must be unit-testable without YOLO/OpenCV video dependencies.
- **Two-stage output:** write raw projected coordinates first, then produce smoothed coordinates from the raw track stream.
- **CPU performance is part of the architecture:** ONNX and OpenVINO artifacts are generated automatically when missing/stale, benchmarked, and the fastest valid artifact is selected.
- **MVP player model:** use YOLO26 COCO `person` class for player detection; ball and court models are trained/fine-tuned from the selected Roboflow datasets.
- **MVP input scope:** fixed behind-baseline, singles, 1080p, no side changes, positive Y toward far baseline, positive X to the right in top-view coordinates.

## Dependency Graph

```text
Project/package foundation
  ├── Shared data contracts + diagnostics
  │     ├── CSV schema/output
  │     ├── smoothing
  │     ├── rendering contracts
  │     └── integration pipeline contracts
  ├── Court geometry + homography
  │     ├── coordinate projection
  │     ├── top-view rendering mapping
  │     └── validation metrics
  ├── Video IO
  │     ├── process CLI
  │     ├── annotated video rendering
  │     └── benchmark CLI
  ├── YOLO model wrappers
  │     ├── player detection/tracking
  │     ├── ball detection/tracking
  │     └── court keypoint detection
  ├── Training scripts
  │     ├── ball model
  │     └── court keypoint model
  ├── Optimization workflow
  │     ├── ONNX export/benchmark
  │     ├── OpenVINO export/benchmark
  │     └── runtime artifact selection
  └── Full process pipeline
        ├── raw CSV
        ├── smoothed CSV
        ├── annotated video + top-view animation
        ├── benchmark summary
        └── validation report
```

---

## Phase 1: Project Foundation and Contracts

### Task 1: Create Python package skeleton and uv project configuration

**Description:** Establish the source layout, package entry points, uv-based dependency management, lint/test configuration, and gitignored local artifact directories.

**Acceptance criteria:**
- [ ] `src/tennis_tracker/` package exists with `__init__.py`, `__main__.py`, and CLI module stub.
- [ ] `pyproject.toml` defines Python 3.11+ project metadata and dependencies for pytest, ruff, OpenCV, NumPy, Ultralytics, and the Roboflow Python library.
- [ ] `uv.lock` is generated and committed for reproducible installs.
- [ ] No `requirements.txt` is required for the primary workflow unless later added as an export artifact.
- [ ] Local large/generated paths are ignored: `data/input/`, `data/output/`, `data/datasets/`, model artifacts.

**Verification:**
- [ ] `uv sync` completes successfully.
- [ ] `uv run python -m tennis_tracker --help` runs.
- [ ] `uv run pytest tests/ -v` runs with at least one smoke test.
- [ ] `uv run ruff check .` runs.

**Dependencies:** None

**Files likely touched:**
- `pyproject.toml`
- `uv.lock`
- `.gitignore`
- `src/tennis_tracker/__init__.py`
- `src/tennis_tracker/__main__.py`
- `src/tennis_tracker/cli.py`
- `tests/unit/test_smoke.py`

**Estimated scope:** Medium: 5 files plus config

---

### Task 2: Define core data contracts and diagnostics

**Description:** Add typed dataclasses/enums for frame metadata, detections, court keypoints, court points, track rows, diagnostics, and model artifact metadata.

**Acceptance criteria:**
- [ ] Shared types represent pixel coordinates, court-meter coordinates, confidence, object kind, frame index, and diagnostics.
- [ ] Missing detections can be represented without using zero coordinates.
- [ ] Diagnostics support compact CSV serialization such as `missing_ball;low_court_confidence`.

**Verification:**
- [ ] `pytest tests/unit/test_diagnostics.py -v`
- [ ] `ruff check src/tennis_tracker tests/unit/test_diagnostics.py`

**Dependencies:** Task 1

**Files likely touched:**
- `src/tennis_tracker/types.py`
- `src/tennis_tracker/diagnostics.py`
- `tests/unit/test_diagnostics.py`

**Estimated scope:** Small: 3 files

---

### Task 3: Implement CSV schema writer for raw/smoothed outputs

**Description:** Create CSV output logic using the approved schema, including meter coordinates, debug pixel coordinates, confidence fields, homography validity, and diagnostics.

**Acceptance criteria:**
- [ ] Writer produces the exact approved column order.
- [ ] Missing values are empty fields, not zeroes.
- [ ] Same writer supports both raw and smoothed output streams.

**Verification:**
- [ ] `pytest tests/unit/test_output.py -v`
- [ ] Manual check of a generated fixture CSV header and missing-value row.

**Dependencies:** Task 2

**Files likely touched:**
- `src/tennis_tracker/output.py`
- `tests/unit/test_output.py`

**Estimated scope:** Small: 2 files

---

## Checkpoint: Foundation Contracts

- [ ] `python -m tennis_tracker --help` works.
- [ ] `pytest tests/unit -v` passes.
- [ ] `ruff check .` passes.
- [ ] CSV schema matches `docs/spec.md`.
- [ ] Human review before moving into geometry/model code.

---

## Phase 2: Court Geometry, Projection, and Smoothing

### Task 4: Implement court geometry constants and coordinate convention

**Description:** Encode full doubles court dimensions, singles lines, service box references, origin at court center, positive X to the right, and positive Y toward the far baseline.

**Acceptance criteria:**
- [ ] Court geometry module exposes named real-world keypoints in meters.
- [ ] Dimensions match 10.97 m width and 23.77 m length.
- [ ] Tests prove axis orientation and major court-line coordinates.

**Verification:**
- [ ] `pytest tests/unit/test_court_geometry.py -v`

**Dependencies:** Task 2

**Files likely touched:**
- `src/tennis_tracker/court.py`
- `tests/unit/test_court_geometry.py`

**Estimated scope:** Small: 2 files

---

### Task 5: Implement homography estimation and pixel-to-court projection

**Description:** Estimate image-to-court homography from detected court keypoints and project player foot points / ball centers to court-meter coordinates.

**Acceptance criteria:**
- [ ] Homography estimation accepts matched pixel keypoints and known court-meter points.
- [ ] Projection returns meter coordinates plus confidence.
- [ ] Invalid/insufficient keypoints produce invalid homography diagnostics instead of fake coordinates.

**Verification:**
- [ ] `pytest tests/unit/test_coordinates.py tests/unit/test_court.py -v`
- [ ] Synthetic square/rectangle projection fixture gives expected coordinates within tolerance.

**Dependencies:** Task 4

**Files likely touched:**
- `src/tennis_tracker/coordinates.py`
- `src/tennis_tracker/court.py`
- `tests/unit/test_coordinates.py`
- `tests/unit/test_court.py`

**Estimated scope:** Medium: 4 files

---

### Task 6: Implement temporal smoothing and interpolation

**Description:** Convert raw per-frame track rows into smoothed track rows while preserving diagnostics and distinguishing observed vs interpolated values.

**Acceptance criteria:**
- [ ] Smoothing handles short gaps without filling long missing intervals blindly.
- [ ] Raw coordinates remain unchanged in the raw CSV path.
- [ ] Smoothed output preserves confidence/diagnostic context.

**Verification:**
- [ ] `pytest tests/unit/test_smoothing.py -v`
- [ ] Fixture with missing ball frames produces expected interpolation behavior.

**Dependencies:** Task 3, Task 5

**Files likely touched:**
- `src/tennis_tracker/smoothing.py`
- `tests/unit/test_smoothing.py`

**Estimated scope:** Small: 2 files

---

## Checkpoint: Geometry and Data Pipeline

- [ ] Geometry and smoothing unit tests pass.
- [ ] Synthetic raw and smoothed CSV files can be generated without YOLO models.
- [ ] Coordinate convention is verified by tests and matches the spec.
- [ ] Human review before integrating models.

---

## Phase 3: Video IO and Synthetic End-to-End Pipeline

### Task 7: Implement video reader/writer utilities

**Description:** Add OpenCV utilities for reading 1080p input video metadata/frames and writing resized annotated video output.

**Acceptance criteria:**
- [ ] Video metadata includes FPS, resolution, frame count, and duration where available.
- [ ] Frame iterator supports bounded short-clip processing for tests.
- [ ] Video writer can produce an MP4 from rendered frames.

**Verification:**
- [ ] `pytest tests/unit/test_video.py -v`
- [ ] Synthetic generated clip can be read and rewritten.

**Dependencies:** Task 1, Task 2

**Files likely touched:**
- `src/tennis_tracker/video.py`
- `tests/unit/test_video.py`

**Estimated scope:** Small: 2 files

---

### Task 8: Build synthetic process pipeline without YOLO

**Description:** Create a process pipeline that accepts synthetic detections/keypoints and produces raw CSV, smoothed CSV, and a minimal annotated video. This proves end-to-end contracts before adding real models.

**Acceptance criteria:**
- [ ] Pipeline processes a short synthetic clip.
- [ ] Raw CSV, smoothed CSV, and video output are produced.
- [ ] Diagnostics are emitted for missing detections in fixture frames.

**Verification:**
- [ ] `pytest tests/integration/test_process_short_video.py -v`

**Dependencies:** Task 3, Task 5, Task 6, Task 7

**Files likely touched:**
- `src/tennis_tracker/pipeline.py`
- `src/tennis_tracker/cli.py`
- `tests/integration/test_process_short_video.py`
- `tests/fixtures/` helpers if needed

**Estimated scope:** Medium: 3-5 files

---

## Checkpoint: Synthetic End-to-End

- [ ] `python -m tennis_tracker process` can run in synthetic/test mode.
- [ ] Integration test produces all three required outputs.
- [ ] No YOLO/model dependency is required for geometry/output integration tests.

---

## Phase 4: Model Wrappers, Training, and Dataset Preparation

### Task 9: Implement YOLO26 inference wrappers

**Description:** Add model loading and inference wrappers for player/person detection, ball detection, and court keypoint detection, with a common output contract.

**Acceptance criteria:**
- [ ] Player wrapper filters YOLO COCO results to `person` class.
- [ ] Ball wrapper returns ball center, bounding box, confidence.
- [ ] Court wrapper returns named/matched keypoints with confidence.
- [ ] Wrappers can be mocked in tests without loading real weights.

**Verification:**
- [ ] `pytest tests/unit/test_detection.py -v`
- [ ] `ruff check src/tennis_tracker/detection.py tests/unit/test_detection.py`

**Dependencies:** Task 2, Task 5

**Files likely touched:**
- `src/tennis_tracker/detection.py`
- `tests/unit/test_detection.py`

**Estimated scope:** Small: 2 files

---

### Task 10: Implement player and ball tracking assignment logic

**Description:** Stabilize player A/B and ball tracks across frames for singles without side changes. Use bottom-center of player boxes and ball center as projection points.

**Acceptance criteria:**
- [ ] Selects two singles players from person detections when possible.
- [ ] Maintains player A/B labels across adjacent frames under the no-side-change assumption.
- [ ] Emits missing/ambiguous diagnostics instead of guessing when detections are unreliable.

**Verification:**
- [ ] `pytest tests/unit/test_tracking.py -v`

**Dependencies:** Task 9

**Files likely touched:**
- `src/tennis_tracker/tracking.py`
- `tests/unit/test_tracking.py`

**Estimated scope:** Small: 2 files

---

### Task 11: Add Roboflow dataset preparation helper

**Description:** Provide a script that uses the Roboflow Python library to authenticate, load/download/export the specified Roboflow datasets, and place them at the training command defaults without committing API keys or dataset contents.

**Acceptance criteria:**
- [ ] Script uses the `roboflow` Python package rather than manual URL downloads.
- [ ] Script documents required environment variables, especially `ROBOFLOW_API_KEY`.
- [ ] Script supports the approved datasets:
  - Court keypoints: `abiya-thesis/tennis-court-suuzy`
  - Ball detection: `viren-dhanwani/tennis-ball-detection`
- [ ] Dataset paths match training command defaults.
- [ ] Dataset contents and credentials remain gitignored.

**Verification:**
- [ ] `uv run python scripts/prepare_roboflow_dataset.py --help`
- [ ] `uv run ruff check scripts/prepare_roboflow_dataset.py`

**Dependencies:** Task 1

**Files likely touched:**
- `scripts/prepare_roboflow_dataset.py`
- `README.md` or `models/README.md`

**Estimated scope:** Small: 2 files

---

### Task 12: Implement ball and court training entry points

**Description:** Add CLI modules for YOLO26 ball detection training and court keypoint training using dataset YAML paths and explicit model/epoch/imgsz/output options.

**Acceptance criteria:**
- [ ] `python -m tennis_tracker.train_ball --help` works.
- [ ] `python -m tennis_tracker.train_court --help` works.
- [ ] Training commands are reproducible and write outputs under `models/`.
- [ ] No training starts at import time.

**Verification:**
- [ ] `pytest tests/unit/test_training_cli.py -v`
- [ ] Manual `--help` checks for both modules.

**Dependencies:** Task 9, Task 11

**Files likely touched:**
- `src/tennis_tracker/training/ball.py`
- `src/tennis_tracker/training/court.py`
- `src/tennis_tracker/training/__init__.py`
- `tests/unit/test_training_cli.py`

**Estimated scope:** Medium: 4 files

---

## Checkpoint: Model Interfaces and Training

- [ ] Detection/tracking unit tests pass with mocked model outputs.
- [ ] Dataset helper and training CLIs expose safe `--help` commands.
- [ ] Human confirms dataset preparation flow before real training runs.

---

## Phase 5: Optimization and Benchmarking

### Task 13: Implement automatic model optimization workflow

**Description:** Detect missing/stale optimized artifacts, export to ONNX and OpenVINO, optionally quantize, benchmark both, and select the fastest valid artifact for runtime.

**Acceptance criteria:**
- [ ] Optimization metadata records source model, export format, quantization, timestamp, and benchmark FPS.
- [ ] ONNX and OpenVINO paths can be generated or skipped with clear diagnostics if unavailable.
- [ ] Runtime selection chooses the fastest valid artifact and falls back safely to `.pt` if needed.

**Verification:**
- [ ] `pytest tests/unit/test_optimization.py -v`
- [ ] `python -m tennis_tracker optimize_model --help`

**Dependencies:** Task 9

**Files likely touched:**
- `src/tennis_tracker/optimization.py`
- `src/tennis_tracker/cli.py`
- `tests/unit/test_optimization.py`

**Estimated scope:** Medium: 3 files

---

### Task 14: Implement benchmark CLI and performance report

**Description:** Measure pipeline FPS, per-stage timings, total runtime, and whether the 30 FPS target is met on 1080p input with selected model artifacts.

**Acceptance criteria:**
- [ ] Benchmark reports total FPS and stage timings for decode, court, player, ball, tracking/projection, rendering.
- [ ] Report clearly states pass/fail for 30 FPS target.
- [ ] Benchmark can run on a frame limit for quick iteration.

**Verification:**
- [ ] `pytest tests/unit/test_benchmark.py -v`
- [ ] `python -m tennis_tracker benchmark --help`

**Dependencies:** Task 7, Task 9, Task 13

**Files likely touched:**
- `src/tennis_tracker/benchmark.py`
- `src/tennis_tracker/cli.py`
- `tests/unit/test_benchmark.py`

**Estimated scope:** Medium: 3 files

---

## Checkpoint: Performance Infrastructure

- [ ] Optimization tests pass.
- [ ] Benchmark CLI can run with mocked/fixture inference.
- [ ] Plan is ready for real M1 performance measurement once models are trained.

---

## Phase 6: Rendering and Full Real-Model Pipeline

### Task 15: Implement annotated video renderer and top-view court panel

**Description:** Render resized source frames with player boxes, ball trail, court keypoints, diagnostics, and a side panel showing top-view court positions.

**Acceptance criteria:**
- [ ] Top-view panel maps court meters to panel pixels using the approved coordinate convention.
- [ ] Ball and player markers are synchronized with current frame rows.
- [ ] Missing detections are visually distinguishable or omitted with warning text.

**Verification:**
- [ ] `pytest tests/unit/test_render.py -v`
- [ ] Manual check generated synthetic annotated video.

**Dependencies:** Task 4, Task 7, Task 8

**Files likely touched:**
- `src/tennis_tracker/render.py`
- `tests/unit/test_render.py`

**Estimated scope:** Small: 2 files

---

### Task 16: Connect real model inference into process CLI

**Description:** Replace synthetic detections with real YOLO wrappers in the processing pipeline while preserving testability through dependency injection/mocks.

**Acceptance criteria:**
- [ ] `process` command loads player, ball, and court models or optimized artifacts.
- [ ] Per-frame detections are tracked, projected, written to raw CSV, smoothed, and rendered to video.
- [ ] Low-confidence/missing model outputs produce diagnostics rather than crashes or fake coordinates.

**Verification:**
- [ ] `pytest tests/integration/test_process_short_video.py -v`
- [ ] Manual run on a short local clip when model files are available.

**Dependencies:** Task 10, Task 13, Task 15

**Files likely touched:**
- `src/tennis_tracker/pipeline.py`
- `src/tennis_tracker/cli.py`
- `src/tennis_tracker/detection.py`
- `tests/integration/test_process_short_video.py`

**Estimated scope:** Medium: 4 files

---

## Checkpoint: Full MVP Flow

- [ ] CLI produces raw CSV, smoothed CSV, and annotated video for a short clip.
- [ ] Pipeline runs with mocked models in CI and with real models locally.
- [ ] Diagnostics are visible in CSV and video output.
- [ ] Human review before validation/performance tuning.

---

## Phase 7: Validation, Documentation, and Release Readiness

### Task 17: Implement validation metrics for >99% precision/recall and coordinate tolerances

**Description:** Add tools to compare model outputs against approximately 100 manually annotated/reviewed validation images and report detection precision/recall plus coordinate tolerance results.

**Acceptance criteria:**
- [ ] Validation reports precision and recall for player/person, ball, and court-keypoint tasks.
- [ ] Validation reports percentage of player positions within 0.2 m and ball positions within 0.1 m where ground truth exists.
- [ ] Report states pass/fail against the spec thresholds.

**Verification:**
- [ ] `pytest tests/unit/test_validation.py -v`
- [ ] `python -m tennis_tracker validate --help`

**Dependencies:** Task 5, Task 9, Task 16

**Files likely touched:**
- `src/tennis_tracker/validation.py`
- `src/tennis_tracker/cli.py`
- `tests/unit/test_validation.py`

**Estimated scope:** Medium: 3 files

---

### Task 18: Document setup, training, processing, optimization, benchmarking, and validation

**Description:** Write user-facing instructions for environment setup, dataset preparation, training, automatic optimization, processing videos, benchmarking on M1, and interpreting outputs.

**Acceptance criteria:**
- [ ] README includes complete command sequence from setup to output generation.
- [ ] `models/README.md` documents model artifacts, source datasets, licenses accepted, and gitignore rules.
- [ ] Documentation explains coordinate system and CSV columns.

**Verification:**
- [ ] Follow README commands in a clean environment up to `--help`/test commands.
- [ ] Links to `docs/spec.md` and this plan are present.

**Dependencies:** Task 16, Task 17

**Files likely touched:**
- `README.md`
- `models/README.md`
- `docs/spec.md` if decisions changed
- `docs/implementation-plan.md`

**Estimated scope:** Medium: 3-4 files

---

### Task 19: Final M1 benchmark and acceptance run

**Description:** Run the full pipeline on representative 1080p fixed-camera singles footage on a standard M1 CPU and produce final benchmark/validation evidence.

**Acceptance criteria:**
- [ ] Benchmark report shows whether 30 FPS is achieved.
- [ ] Validation report covers approximately 100 annotated/reviewed images.
- [ ] Raw CSV, smoothed CSV, annotated video, benchmark report, and validation report are produced locally.
- [ ] Any failure to meet 30 FPS or >99% precision/recall is documented with mitigation recommendations.

**Verification:**
- [ ] `pytest tests/ -v`
- [ ] `ruff check .`
- [ ] Manual M1 benchmark command from README.
- [ ] Manual validation command from README.

**Dependencies:** Task 18

**Files likely touched:**
- `docs/benchmark-results.md` or generated report path
- `docs/validation-results.md` or generated report path
- README updates if needed

**Estimated scope:** Medium: 2-4 files plus local generated artifacts

---

## Checkpoint: Complete

- [ ] All tests pass: `pytest tests/ -v`.
- [ ] Lint passes: `ruff check .`.
- [ ] CLI help works for process, training, optimization, benchmark, and validation commands.
- [ ] Short-video integration test produces all outputs.
- [ ] Real M1 benchmark and validation results are documented.
- [ ] Human reviews whether unmet performance/accuracy targets require scope or model changes.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| 30 FPS on standard M1 CPU with three YOLO models may not be achievable at 1080p | High | Benchmark early; use YOLO26n-scale models, frame sampling for court homography, ONNX/OpenVINO, quantization, reduced inference resolution, and staged rendering optimizations. |
| Tennis ball is small and fast, causing missed detections | High | Fine-tune ball model, use temporal smoothing, use track continuity heuristics, validate on tennis-specific clips, and keep missing detections explicit. |
| Court keypoint model may not map cleanly to required named court geometry | High | Define a keypoint mapping table early; unit-test homography with partial/low-confidence keypoints; add diagnostics for mismatches. |
| >99% precision/recall may be unrealistic with 100 images or COCO player detector | High | Treat validation as a gate; if unmet, add dedicated player dataset/model or revise target with human approval. |
| Roboflow dataset format may differ from expected YOLO pose/detection format | Medium | Add dataset preparation/inspection step before training; document exact expected `data.yaml` structure. |
| Annotated video rendering may reduce throughput | Medium | Benchmark rendering separately; allow benchmark mode with/without rendering; resize annotated output as approved. |
| Player identity may swap during occlusions even without side changes | Medium | Use court-side constraints and nearest-neighbor continuity; emit ambiguity diagnostics when uncertain. |
| Homography drift or bad court detections can corrupt all coordinates | High | Stabilize homography over time; validate reprojection error; mark frames invalid instead of projecting with bad calibration. |

## Parallelization Opportunities

Safe to parallelize after Phase 1:

- Court geometry/projection tests and CSV output tests.
- Video IO utilities and dataset preparation helper.
- Rendering tests and smoothing tests once contracts exist.
- Documentation can evolve alongside implementation after command interfaces stabilize.

Must be sequential:

- Data contracts before CSV, geometry, model wrappers, and pipeline.
- Court geometry before homography and top-view rendering.
- Model wrappers before real process pipeline.
- Optimization before final M1 benchmark.
- Full pipeline before validation/reporting.

## Current Open Questions

None blocking implementation. If validation or benchmarks fail, revisit:

- Whether COCO `person` is sufficient for players.
- Whether the selected Roboflow datasets can support >99% precision/recall.
- Whether 30 FPS on standard M1 requires reduced inference resolution, model size changes, or revised output/rendering strategy.

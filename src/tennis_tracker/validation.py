"""Validation metrics for detection tasks and coordinate tolerances.

Computes per-task precision/recall (player, ball, court keypoint)
and coordinate tolerance pass rates (players within 0.2 m, ball
within 0.1 m).  All comparisons use synthetic/testable records so
the module can be fully tested without real model inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Thresholds (from docs/spec.md) ─────────────────────────────────────

PLAYER_COORD_TOLERANCE_M: float = 0.2
"""Maximum allowed error (meters) for a player court position."""

BALL_COORD_TOLERANCE_M: float = 0.1
"""Maximum allowed error (meters) for a ball court position."""

TARGET_PRECISION_RECALL: float = 0.99
""">99 % target for detection precision and recall."""

IOU_THRESHOLD: float = 0.5
"""IoU threshold for counting a detection as a true positive."""


# ── Validation record types ────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionPrediction:
    """A single prediction from a detection model.

    ``class_id`` is the COCO/yolo class ID.  For matched
    comparisons the same ID space is used for ground truth.
    """

    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DetectionGroundTruth:
    """Ground-truth annotation for a single detection."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class KeypointPrediction:
    """A predicted court keypoint with its geometry label."""

    label: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class KeypointGroundTruth:
    """Ground-truth court keypoint annotation."""

    label: str
    x: float
    y: float


@dataclass(frozen=True)
class CourtCoordinatePrediction:
    """Predicted court-plane coordinate (meters) with a label."""

    label: str  # e.g. "player_a", "player_b", "ball"
    x_m: float
    y_m: float
    confidence: float


@dataclass(frozen=True)
class CourtCoordinateGroundTruth:
    """Ground-truth court-plane coordinate (meters)."""

    label: str
    x_m: float
    y_m: float


# ── IoU helper ─────────────────────────────────────────────────────────


def _iou(
    x1: float, y1: float, x2: float, y2: float,
    gx1: float, gy1: float, gx2: float, gy2: float,
) -> float:
    """Intersection-over-union for two axis-aligned boxes."""
    ix1 = max(x1, gx1)
    iy1 = max(y1, gy1)
    ix2 = min(x2, gx2)
    iy2 = min(y2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    box_area = (x2 - x1) * (y2 - y1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = box_area + gt_area - inter
    if union <= 0:
        return 0.0
    return inter / union


# ── Per-task metrics ───────────────────────────────────────────────────


@dataclass
class TaskMetrics:
    """Precision/recall results for one detection task."""

    task_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP).  0 if no predictions exist."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN).  0 if no ground truth exists."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def precision_passed(self) -> bool:
        """True when precision >= 0.99."""
        return self.precision >= TARGET_PRECISION_RECALL

    @property
    def recall_passed(self) -> bool:
        """True when recall >= 0.99."""
        return self.recall >= TARGET_PRECISION_RECALL


def compute_detection_metrics(
    predictions: list[DetectionPrediction],
    ground_truths: list[DetectionGroundTruth],
    task_name: str = "detection",
) -> TaskMetrics:
    """Compute precision/recall for a detection task.

    A prediction matches a ground truth when their class IDs match
    and IoU >= *IOU_THRESHOLD*.  Each ground truth can match at
    most one prediction.
    """
    metrics = TaskMetrics(task_name=task_name)

    # Sort predictions descending by confidence for a PR-like match
    matched_gt: set[int] = set()
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx: Optional[int] = None
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if gt.class_id != pred.class_id:
                continue
            i = _iou(
                pred.x1, pred.y1, pred.x2, pred.y2,
                gt.x1, gt.y1, gt.x2, gt.y2,
            )
            if i > best_iou:
                best_iou = i
                best_idx = gt_idx

        if best_iou >= IOU_THRESHOLD and best_idx is not None:
            metrics.true_positives += 1
            matched_gt.add(best_idx)
        else:
            metrics.false_positives += 1

    metrics.false_negatives = len(ground_truths) - len(matched_gt)
    return metrics


def compute_keypoint_metrics(
    predictions: list[KeypointPrediction],
    ground_truths: list[KeypointGroundTruth],
    match_distance_px: float = 10.0,
    task_name: str = "court_keypoints",
) -> TaskMetrics:
    """Compute precision/recall for keypoint predictions.

    A prediction matches a ground truth when the label matches and
    Euclidean pixel distance <= *match_distance_px*.
    """
    metrics = TaskMetrics(task_name=task_name)
    matched_gt: set[int] = set()
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    for pred in sorted_preds:
        best_dist = float("inf")
        best_idx: Optional[int] = None
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if gt.label != pred.label:
                continue
            dist = ((pred.x - gt.x) ** 2 + (pred.y - gt.y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = gt_idx

        if best_dist <= match_distance_px and best_idx is not None:
            metrics.true_positives += 1
            matched_gt.add(best_idx)
        else:
            metrics.false_positives += 1

    metrics.false_negatives = len(ground_truths) - len(matched_gt)
    return metrics


# ── Coordinate tolerance metrics ───────────────────────────────────────


@dataclass
class CoordinateToleranceMetrics:
    """Results of coordinate tolerance checking for one label group."""

    label: str
    within_tolerance: int = 0
    total: int = 0
    tolerance_m: float = 0.2

    @property
    def pass_rate(self) -> float:
        """Fraction of coordinates within tolerance."""
        return self.within_tolerance / self.total if self.total > 0 else 1.0

    @property
    def passed(self) -> bool:
        """True when pass_rate == 1.0 (all within tolerance)."""
        return self.pass_rate >= 1.0  # strict: all must pass


def compute_coordinate_tolerance(
    predictions: list[CourtCoordinatePrediction],
    ground_truths: list[CourtCoordinateGroundTruth],
) -> list[CoordinateToleranceMetrics]:
    """Compute coordinate tolerance pass rates grouped by label.

    For each prediction the corresponding ground truth is found
    by matching label, and the Euclidean distance in meters is
    compared against the task-appropriate tolerance.

    Player-labeled records use *PLAYER_COORD_TOLERANCE_M* (0.2 m),
    ball-labeled records use *BALL_COORD_TOLERANCE_M* (0.1 m).
    """
    # Build lookup: label -> ground truth
    gt_by_label: dict[str, CourtCoordinateGroundTruth] = {}
    for gt in ground_truths:
        gt_by_label[gt.label] = gt

    # Group results by label
    results: dict[str, CoordinateToleranceMetrics] = {}

    for pred in predictions:
        if pred.label not in results:
            tol = (
                BALL_COORD_TOLERANCE_M
                if pred.label == "ball"
                else PLAYER_COORD_TOLERANCE_M
            )
            results[pred.label] = CoordinateToleranceMetrics(
                label=pred.label,
                tolerance_m=tol,
            )

        gt = gt_by_label.get(pred.label)
        if gt is None:
            continue  # no ground truth to compare against

        dist = ((pred.x_m - gt.x_m) ** 2 + (pred.y_m - gt.y_m) ** 2) ** 0.5
        tol = BALL_COORD_TOLERANCE_M if pred.label == "ball" else PLAYER_COORD_TOLERANCE_M
        if dist <= tol:
            results[pred.label].within_tolerance += 1
        results[pred.label].total += 1

    return list(results.values())


# ── Aggregate result ───────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """Complete validation output summarising detection and coordinate results."""

    task_metrics: list[TaskMetrics] = field(default_factory=list)
    coordinate_metrics: list[CoordinateToleranceMetrics] = field(default_factory=list)

    @property
    def all_tasks_passed(self) -> bool:
        """All detection tasks meet the >99% target."""
        return all(m.precision_passed and m.recall_passed for m in self.task_metrics)

    @property
    def all_coordinates_passed(self) -> bool:
        """All coordinate groups are within tolerance."""
        return all(m.passed for m in self.coordinate_metrics)

    @property
    def overall_passed(self) -> bool:
        """Both detection and coordinate criteria met."""
        return self.all_tasks_passed and self.all_coordinates_passed

    def summary_lines(self) -> list[str]:
        """Human-readable summary for CLI output."""
        lines = ["Validation Report", "=" * 40, ""]

        lines.append("Detection Tasks")
        lines.append("-" * 20)
        for m in self.task_metrics:
            p = "✅" if m.precision_passed else "❌"
            r = "✅" if m.recall_passed else "❌"
            lines.append(
                f"  {m.task_name:25s}  "
                f"Precision: {m.precision:.4f} {p}  "
                f"Recall: {m.recall:.4f} {r}  "
                f"(TP={m.true_positives} FP={m.false_positives} FN={m.false_negatives})"
            )

        lines.append("")
        lines.append("Coordinate Tolerances")
        lines.append("-" * 20)
        for m in self.coordinate_metrics:
            s = "✅" if m.passed else "❌"
            lines.append(
                f"  {m.label:25s}  "
                f"{m.within_tolerance}/{m.total} within {m.tolerance_m:.1f} m  "
                f"({m.pass_rate:.1%}) {s}"
            )

        lines.append("")
        verdict = "✅ PASS" if self.overall_passed else "❌ FAIL"
        lines.append(f"Verdict: {verdict}")
        return lines


# Type aliases for readability of run_validation signature
_Detections = list[tuple[list[DetectionPrediction], list[DetectionGroundTruth], str]]
_Keypoints = list[tuple[list[KeypointPrediction], list[KeypointGroundTruth], str]]
_Coordinates = tuple[list[CourtCoordinatePrediction], list[CourtCoordinateGroundTruth]]


def run_validation(
    *,
    detections: Optional[_Detections] = None,
    keypoints: Optional[_Keypoints] = None,
    coordinates: Optional[_Coordinates] = None,
) -> ValidationReport:
    """Run validation over optional named tasks and produce a report.

    Parameters
    ----------
    detections:
        Zero or more ``(predictions, ground_truths, task_name)`` tuples
        for bounding-box detection tasks (player, ball, etc.).
    keypoints:
        Zero or more ``(predictions, ground_truths, task_name)`` tuples
        for keypoint detection tasks (court keypoints).
    coordinates:
        A single ``(predictions, ground_truths)`` tuple for court-plane
        coordinate tolerance checking.
    """
    report = ValidationReport()

    if detections:
        for preds, gts, name in detections:
            report.task_metrics.append(
                compute_detection_metrics(preds, gts, task_name=name)
            )

    if keypoints:
        for preds, gts, name in keypoints:
            report.task_metrics.append(
                compute_keypoint_metrics(preds, gts, task_name=name)
            )

    if coordinates:
        preds, gts = coordinates
        report.coordinate_metrics = compute_coordinate_tolerance(preds, gts)

    return report

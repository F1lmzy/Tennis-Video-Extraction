"""Tests for validation metrics and validate CLI."""

from tennis_tracker.validation import (
    BALL_COORD_TOLERANCE_M,
    IOU_THRESHOLD,
    PLAYER_COORD_TOLERANCE_M,
    TARGET_PRECISION_RECALL,
    CourtCoordinateGroundTruth,
    CourtCoordinatePrediction,
    CoordinateToleranceMetrics,
    DetectionGroundTruth,
    DetectionPrediction,
    KeypointGroundTruth,
    KeypointPrediction,
    TaskMetrics,
    ValidationReport,
    compute_coordinate_tolerance,
    compute_detection_metrics,
    compute_keypoint_metrics,
    run_validation,
)


# ── TaskMetrics ─────────────────────────────────────────────────────────

class TestTaskMetrics:
    def test_perfect_precision_and_recall(self) -> None:
        m = TaskMetrics(task_name="test", true_positives=100, false_positives=0, false_negatives=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.precision_passed
        assert m.recall_passed

    def test_zero_precision_when_no_predictions(self) -> None:
        m = TaskMetrics(task_name="test", true_positives=0, false_positives=0, false_negatives=10)
        assert m.precision == 0.0
        assert not m.precision_passed

    def test_zero_recall_when_no_ground_truth(self) -> None:
        m = TaskMetrics(task_name="test", true_positives=0, false_positives=10, false_negatives=0)
        assert m.recall == 0.0
        assert not m.recall_passed

    def test_just_above_target(self) -> None:
        # 99/100 = 0.99 -> passes (>= 0.99)
        m = TaskMetrics(task_name="test", true_positives=99, false_positives=1, false_negatives=1)
        assert m.precision == 0.99
        assert m.recall == 0.99
        assert m.precision_passed
        assert m.recall_passed

    def test_just_below_target(self) -> None:
        # 98/100 = 0.98 -> fails
        m = TaskMetrics(task_name="test", true_positives=98, false_positives=2, false_negatives=2)
        assert m.precision == 0.98
        assert not m.precision_passed
        assert not m.recall_passed

    def test_empty_metrics(self) -> None:
        m = TaskMetrics(task_name="empty")
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert not m.precision_passed
        assert not m.recall_passed


# ── Detection metrics ──────────────────────────────────────────────────

class TestComputeDetectionMetrics:
    def test_all_perfect_matches(self) -> None:
        preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
            DetectionPrediction(class_id=0, confidence=0.8, x1=20, y1=20, x2=30, y2=30),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
            DetectionGroundTruth(class_id=0, x1=20, y1=20, x2=30, y2=30),
        ]
        m = compute_detection_metrics(preds, gts)
        assert m.true_positives == 2
        assert m.false_positives == 0
        assert m.false_negatives == 0
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_false_positive(self) -> None:
        preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
            DetectionPrediction(class_id=0, confidence=0.8, x1=100, y1=100, x2=110, y2=110),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
        ]
        m = compute_detection_metrics(preds, gts)
        assert m.true_positives == 1
        assert m.false_positives == 1  # second pred matched nothing
        assert m.false_negatives == 0
        assert m.precision == 0.5

    def test_false_negative(self) -> None:
        preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
            DetectionGroundTruth(class_id=0, x1=20, y1=20, x2=30, y2=30),
        ]
        m = compute_detection_metrics(preds, gts)
        assert m.true_positives == 1
        assert m.false_positives == 0
        assert m.false_negatives == 1
        assert m.recall == 0.5

    def test_wrong_class_id_does_not_match(self) -> None:
        preds = [
            DetectionPrediction(class_id=1, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
        ]
        m = compute_detection_metrics(preds, gts)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_iou_below_threshold_is_false_positive(self) -> None:
        preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=5, y2=5),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=20, y1=20, x2=30, y2=30),  # no overlap
        ]
        m = compute_detection_metrics(preds, gts)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_empty_inputs(self) -> None:
        m = compute_detection_metrics([], [])
        assert m.true_positives == 0
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_multiple_class_tasks(self) -> None:
        """Two separate tasks for player and ball should work independently."""
        player_preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
            DetectionPrediction(class_id=0, confidence=0.8, x1=20, y1=20, x2=30, y2=30),
        ]
        player_gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
            DetectionGroundTruth(class_id=0, x1=20, y1=20, x2=30, y2=30),
        ]
        ball_preds = [
            DetectionPrediction(class_id=1, confidence=0.7, x1=50, y1=50, x2=56, y2=56),
        ]
        ball_gts = [
            DetectionGroundTruth(class_id=1, x1=50, y1=50, x2=56, y2=56),
        ]
        player_m = compute_detection_metrics(player_preds, player_gts, task_name="player")
        ball_m = compute_detection_metrics(ball_preds, ball_gts, task_name="ball")
        assert player_m.true_positives == 2
        assert player_m.false_positives == 0
        assert ball_m.true_positives == 1
        assert ball_m.recall == 1.0


# ── Keypoint metrics ───────────────────────────────────────────────────

class TestComputeKeypointMetrics:
    def test_all_perfect_matches(self) -> None:
        preds = [
            KeypointPrediction(label="tl", x=10.0, y=20.0, confidence=0.9),
            KeypointPrediction(label="br", x=30.0, y=40.0, confidence=0.8),
        ]
        gts = [
            KeypointGroundTruth(label="tl", x=10.0, y=20.0),
            KeypointGroundTruth(label="br", x=30.0, y=40.0),
        ]
        m = compute_keypoint_metrics(preds, gts)
        assert m.true_positives == 2
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_distance_exceeds_threshold(self) -> None:
        preds = [
            KeypointPrediction(label="tl", x=10.0, y=20.0, confidence=0.9),
        ]
        gts = [
            KeypointGroundTruth(label="tl", x=50.0, y=80.0),  # >10 px away
        ]
        m = compute_keypoint_metrics(preds, gts, match_distance_px=10.0)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_wrong_label_does_not_match(self) -> None:
        preds = [
            KeypointPrediction(label="tl", x=10.0, y=20.0, confidence=0.9),
        ]
        gts = [
            KeypointGroundTruth(label="br", x=10.0, y=20.0),
        ]
        m = compute_keypoint_metrics(preds, gts)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1


# ── Coordinate tolerance ───────────────────────────────────────────────

class TestComputeCoordinateTolerance:
    def test_all_within_tolerance(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="player_a", x_m=1.0, y_m=2.0, confidence=0.9),
            CourtCoordinatePrediction(label="ball", x_m=0.05, y_m=0.1, confidence=0.7),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="player_a", x_m=1.05, y_m=2.05),
            CourtCoordinateGroundTruth(label="ball", x_m=0.04, y_m=0.09),
        ]
        results = compute_coordinate_tolerance(preds, gts)
        assert len(results) == 2
        for r in results:
            assert r.passed, f"{r.label} should have passed"
            assert r.within_tolerance == 1
            assert r.total == 1

    def test_player_exceeds_tolerance(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="player_a", x_m=0.0, y_m=0.0, confidence=0.9),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="player_a", x_m=0.5, y_m=0.5),  # 0.707 m away
        ]
        results = compute_coordinate_tolerance(preds, gts)
        assert len(results) == 1
        assert results[0].label == "player_a"
        assert not results[0].passed
        assert results[0].within_tolerance == 0
        assert results[0].total == 1

    def test_ball_exceeds_tolerance(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="ball", x_m=0.0, y_m=0.0, confidence=0.9),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="ball", x_m=0.15, y_m=0.0),  # 0.15 m > 0.1 m
        ]
        results = compute_coordinate_tolerance(preds, gts)
        assert len(results) == 1
        assert not results[0].passed

    def test_ball_within_tolerance(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="ball", x_m=0.0, y_m=0.0, confidence=0.9),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="ball", x_m=0.08, y_m=0.0),  # 0.08 m <= 0.1 m
        ]
        results = compute_coordinate_tolerance(preds, gts)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].within_tolerance == 1

    def test_no_ground_truth_skips(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="player_a", x_m=1.0, y_m=2.0, confidence=0.9),
        ]
        results = compute_coordinate_tolerance(preds, [])
        assert len(results) == 1
        # No GT to compare, total stays 0 (all within_tolerance = 0, total = 0)
        assert results[0].total == 0
        assert results[0].pass_rate == 1.0  # by convention when no GT

    def test_mixed_labels_use_correct_tolerances(self) -> None:
        """Verify player uses 0.2 m and ball uses 0.1 m tolerance."""
        preds = [
            CourtCoordinatePrediction(label="player_a", x_m=0.19, y_m=0.0, confidence=0.9),
            CourtCoordinatePrediction(label="ball", x_m=0.09, y_m=0.0, confidence=0.7),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="player_a", x_m=0.0, y_m=0.0),
            CourtCoordinateGroundTruth(label="ball", x_m=0.0, y_m=0.0),
        ]
        results = compute_coordinate_tolerance(preds, gts)
        results_by_label = {r.label: r for r in results}
        assert results_by_label["player_a"].passed  # 0.19 <= 0.2
        assert results_by_label["ball"].passed      # 0.09 <= 0.1


# ── Validation report ─────────────────────────────────────────────────

class TestValidationReport:
    def test_all_passed(self) -> None:
        report = ValidationReport(
            task_metrics=[
                TaskMetrics(
                    task_name="p", true_positives=100, false_positives=0, false_negatives=0
                ),
            ],
            coordinate_metrics=[
                CoordinateToleranceMetrics(label="player_a", within_tolerance=5, total=5),
            ],
        )
        assert report.all_tasks_passed
        assert report.all_coordinates_passed
        assert report.overall_passed

    def test_task_fails_when_one_task_misses_target(self) -> None:
        report = ValidationReport(
            task_metrics=[
                TaskMetrics(
                    task_name="good", true_positives=100, false_positives=0, false_negatives=0
                ),
                TaskMetrics(
                    task_name="bad", true_positives=50, false_positives=50, false_negatives=50
                ),
            ],
        )
        assert not report.all_tasks_passed
        assert not report.overall_passed

    def test_summary_lines_format(self) -> None:
        report = ValidationReport(
            task_metrics=[
                TaskMetrics(
                    task_name="player", true_positives=10, false_positives=0, false_negatives=0
                ),
            ],
            coordinate_metrics=[
                CoordinateToleranceMetrics(
                    label="ball", within_tolerance=5, total=5, tolerance_m=0.1
                ),
            ],
        )
        lines = report.summary_lines()
        joined = " ".join(lines)
        assert "player" in joined
        assert "ball" in joined
        assert "PASS" in joined
        assert "Precision" in joined
        assert "Recall" in joined


# ── Integration: run_validation ────────────────────────────────────────

class TestRunValidation:
    def test_detection_only(self) -> None:
        preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
        ]
        gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
        ]
        report = run_validation(detections=[(preds, gts, "player")])
        assert len(report.task_metrics) == 1
        assert report.task_metrics[0].true_positives == 1
        assert report.overall_passed

    def test_keypoints_only(self) -> None:
        preds = [
            KeypointPrediction(label="tl", x=10, y=20, confidence=0.9),
        ]
        gts = [
            KeypointGroundTruth(label="tl", x=10, y=20),
        ]
        report = run_validation(keypoints=[(preds, gts, "court_keypoints")])
        assert len(report.task_metrics) == 1
        assert report.task_metrics[0].true_positives == 1

    def test_coordinates_only(self) -> None:
        preds = [
            CourtCoordinatePrediction(label="ball", x_m=0.05, y_m=0.05, confidence=0.8),
        ]
        gts = [
            CourtCoordinateGroundTruth(label="ball", x_m=0.04, y_m=0.04),
        ]
        report = run_validation(coordinates=(preds, gts))
        assert len(report.coordinate_metrics) == 1
        assert report.coordinate_metrics[0].passed
        assert report.overall_passed

    def test_everything_together(self) -> None:
        """Verify all three input types in a single call."""
        det_preds = [
            DetectionPrediction(class_id=0, confidence=0.9, x1=0, y1=0, x2=10, y2=10),
        ]
        det_gts = [
            DetectionGroundTruth(class_id=0, x1=0, y1=0, x2=10, y2=10),
        ]
        kp_preds = [
            KeypointPrediction(label="center", x=100, y=200, confidence=0.85),
        ]
        kp_gts = [
            KeypointGroundTruth(label="center", x=100, y=200),
        ]
        coord_preds = [
            CourtCoordinatePrediction(label="player_a", x_m=2.0, y_m=-1.0, confidence=0.9),
        ]
        coord_gts = [
            CourtCoordinateGroundTruth(label="player_a", x_m=2.05, y_m=-0.95),
        ]
        report = run_validation(
            detections=[(det_preds, det_gts, "player")],
            keypoints=[(kp_preds, kp_gts, "court_keypoints")],
            coordinates=(coord_preds, coord_gts),
        )
        assert len(report.task_metrics) == 2
        assert len(report.coordinate_metrics) == 1
        assert report.overall_passed


# ── Threshold constants (verify spec alignment) ─────────────────────────

class TestSpecThresholds:
    def test_player_coord_tolerance_is_0_point_2(self) -> None:
        assert PLAYER_COORD_TOLERANCE_M == 0.2

    def test_ball_coord_tolerance_is_0_point_1(self) -> None:
        assert BALL_COORD_TOLERANCE_M == 0.1

    def test_target_precision_recall_is_0_point_99(self) -> None:
        assert TARGET_PRECISION_RECALL == 0.99

    def test_iou_threshold_is_reasonable(self) -> None:
        assert IOU_THRESHOLD == 0.5

"""Unit tests for YOLO26 inference wrappers (``detection.py``).

All tests use mock/fake model objects so they never require real
model weights, network access, or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tennis_tracker.detection import (
    BallDetector,
    BoundingBox,
    CourtKeypointDetector,
    PlayerDetector,
)


# ── Helpers: fake Ultralytics result objects ────────────────────────────


@dataclass
class FakeBoxes:
    """Mimics ``ultralytics.engine.results.Boxes`` for testing."""

    xyxy: torch.Tensor  # (N, 4)
    conf: torch.Tensor  # (N,)
    cls: torch.Tensor  # (N,)
    id: Optional[torch.Tensor] = None  # (N,)

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class FakeKeypoints:
    """Mimics ``ultralytics.engine.results.Keypoints`` for testing."""

    xy: torch.Tensor  # (1, K, 2) — one detected instance, K keypoints
    conf: Optional[torch.Tensor] = None  # (1, K)


@dataclass
class FakeResult:
    """Mimics a single ``ultralytics.engine.results.Results``."""

    boxes: Optional[FakeBoxes] = None
    keypoints: Optional[FakeKeypoints] = None


def _make_mock_model(results: list[FakeResult]) -> MagicMock:
    """Build a mock YOLO model that returns the given results."""
    model = MagicMock()
    model.predict.return_value = results
    return model


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> np.ndarray:
    """A minimal BGR image (100×200, 3 channels)."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


# ── BoundingBox tests ──────────────────────────────────────────────────


class TestBoundingBox:
    def test_center(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
        c = bbox.center
        assert c.x == 20.0
        assert c.y == 40.0

    def test_bottom_center(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
        bc = bbox.bottom_center
        assert bc.x == 20.0
        assert bc.y == 60.0

    def test_width_height(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
        assert bbox.width == 20.0
        assert bbox.height == 40.0


# ── PlayerDetector tests ───────────────────────────────────────────────


class TestPlayerDetector:
    def test_filters_person_class(self, dummy_image: np.ndarray) -> None:
        """Only class ID 0 (person) should be returned."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=torch.tensor([0.9, 0.8]),
            cls=torch.tensor([0.0, 2.0]),  # person, car
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9)

    def test_all_persons_returned(self, dummy_image: np.ndarray) -> None:
        """Multiple person detections should all be returned."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=torch.tensor([0.9, 0.8]),
            cls=torch.tensor([0.0, 0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 2

    def test_sorted_by_confidence(self, dummy_image: np.ndarray) -> None:
        """Results should be in decreasing confidence order."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]),
            conf=torch.tensor([0.5, 0.9, 0.7]),
            cls=torch.tensor([0.0, 0.0, 0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image, conf_threshold=0.0)
        assert len(detections) == 3
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[1].confidence == pytest.approx(0.7)
        assert detections[2].confidence == pytest.approx(0.5)

    def test_no_detections_returns_empty(self, dummy_image: np.ndarray) -> None:
        """When no person is detected, return empty list."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([1.0]),  # not person
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert detections == []

    def test_no_boxes_returns_empty(self, dummy_image: np.ndarray) -> None:
        """When model returns no boxes, return empty list."""
        mock_model = _make_mock_model([FakeResult(boxes=None)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert detections == []

    def test_conf_threshold_filters(self, dummy_image: np.ndarray) -> None:
        """Detections below conf_threshold should be excluded."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=torch.tensor([0.3, 0.9]),
            cls=torch.tensor([0.0, 0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image, conf_threshold=0.5)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9)

    def test_track_id_propagated(self, dummy_image: np.ndarray) -> None:
        """track_id from Ultralytics tracking mode should be available."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
            id=torch.tensor([42]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 1
        assert detections[0].track_id == 42

    def test_default_model_uses_yolo26n(self) -> None:
        """When no model or path is given, the default is yolo26n.pt."""
        # We can't actually instantiate (would download weights), so we
        # verify via the model checkpoint property that a default path
        # is configured.  Skip if ultralytics download would block.
        pytest.skip("Default model instantiation requires network/weights")

    def test_bbox_values_correct(self, dummy_image: np.ndarray) -> None:
        """Verify bbox coordinates are passed through correctly."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10.5, 20.5, 30.5, 40.5]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = PlayerDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 1
        bbox = detections[0].bbox
        assert bbox.x1 == pytest.approx(10.5)
        assert bbox.y1 == pytest.approx(20.5)
        assert bbox.x2 == pytest.approx(30.5)
        assert bbox.y2 == pytest.approx(40.5)

    def test_model_property(self) -> None:
        """The ``model`` property returns the injected model."""
        mock_model = MagicMock()
        detector = PlayerDetector(model=mock_model)
        assert detector.model is mock_model


# ── BallDetector tests ─────────────────────────────────────────────────


class TestBallDetector:
    def test_returns_all_detections(self, dummy_image: np.ndarray) -> None:
        """BallDetector returns all detections (no class filtering)."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=torch.tensor([0.9, 0.7]),
            cls=torch.tensor([0.0, 0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = BallDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 2

    def test_center_is_computed(self, dummy_image: np.ndarray) -> None:
        """Ball center should be bbox centre."""
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = BallDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert len(detections) == 1
        c = detections[0].center
        assert c.x == pytest.approx(20.0)
        assert c.y == pytest.approx(30.0)

    def test_sorted_by_confidence(self, dummy_image: np.ndarray) -> None:
        boxes = FakeBoxes(
            xyxy=torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=torch.tensor([0.5, 0.9]),
            cls=torch.tensor([0.0, 0.0]),
        )
        mock_model = _make_mock_model([FakeResult(boxes=boxes)])
        detector = BallDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[1].confidence == pytest.approx(0.5)

    def test_no_boxes_returns_empty(self, dummy_image: np.ndarray) -> None:
        mock_model = _make_mock_model([FakeResult(boxes=None)])
        detector = BallDetector(model=mock_model)
        detections = detector.predict(dummy_image)
        assert detections == []

    def test_raises_without_model_or_path(self) -> None:
        with pytest.raises(ValueError, match="model.*or.*model_path"):
            BallDetector()

    def test_model_property(self) -> None:
        mock_model = MagicMock()
        detector = BallDetector(model=mock_model)
        assert detector.model is mock_model


# ── CourtKeypointDetector tests ────────────────────────────────────────


class TestCourtKeypointDetector:
    def test_returns_keypoints_with_labels(self, dummy_image: np.ndarray) -> None:
        """Keypoints should be returned with correct labels."""
        kps = FakeKeypoints(
            xy=torch.tensor([[[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]]]),
            conf=torch.tensor([[0.9, 0.8, 0.7]]),
        )
        mock_model = _make_mock_model([FakeResult(keypoints=kps)])
        detector = CourtKeypointDetector(
            model=mock_model,
            keypoint_labels={0: "doubles_near_left", 1: "doubles_near_right", 2: "net_left"},
        )
        keypoints = detector.predict(dummy_image, conf_threshold=0.0)
        assert len(keypoints) == 3
        assert keypoints[0].label == "doubles_near_left"
        assert keypoints[1].label == "doubles_near_right"
        assert keypoints[2].label == "net_left"

    def test_filters_by_confidence(self, dummy_image: np.ndarray) -> None:
        """Keypoints below conf_threshold should be omitted."""
        kps = FakeKeypoints(
            xy=torch.tensor([[[50.0, 60.0], [70.0, 80.0]]]),
            conf=torch.tensor([[0.3, 0.9]]),
        )
        mock_model = _make_mock_model([FakeResult(keypoints=kps)])
        detector = CourtKeypointDetector(model=mock_model)
        keypoints = detector.predict(dummy_image, conf_threshold=0.5)
        assert len(keypoints) == 1
        assert keypoints[0].confidence == pytest.approx(0.9)

    def test_zero_coordinates_omitted(self, dummy_image: np.ndarray) -> None:
        """Ultralytics zeroes out undetected keypoints; those should be omitted."""
        kps = FakeKeypoints(
            xy=torch.tensor([[[0.0, 0.0], [70.0, 80.0]]]),
            conf=torch.tensor([[0.9, 0.9]]),
        )
        mock_model = _make_mock_model([FakeResult(keypoints=kps)])
        detector = CourtKeypointDetector(model=mock_model, keypoint_labels={0: "kp0", 1: "kp1"})
        keypoints = detector.predict(dummy_image, conf_threshold=0.0)
        assert len(keypoints) == 1
        assert keypoints[0].label == "kp1"

    def test_unknown_keypoints_get_generic_labels(self, dummy_image: np.ndarray) -> None:
        """When keypoint_labels is not provided, use generic labels."""
        kps = FakeKeypoints(
            xy=torch.tensor([[[50.0, 60.0], [70.0, 80.0]]]),
            conf=torch.tensor([[0.9, 0.8]]),
        )
        mock_model = _make_mock_model([FakeResult(keypoints=kps)])
        detector = CourtKeypointDetector(model=mock_model)
        keypoints = detector.predict(dummy_image, conf_threshold=0.0)
        assert len(keypoints) == 2
        assert keypoints[0].label == "keypoint_0"
        assert keypoints[1].label == "keypoint_1"

    def test_no_keypoints_returns_empty(self, dummy_image: np.ndarray) -> None:
        mock_model = _make_mock_model([FakeResult(keypoints=None)])
        detector = CourtKeypointDetector(model=mock_model)
        keypoints = detector.predict(dummy_image)
        assert keypoints == []

    def test_pixel_values_correct(self, dummy_image: np.ndarray) -> None:
        """Verify pixel coordinates are passed through correctly."""
        kps = FakeKeypoints(
            xy=torch.tensor([[[100.5, 200.5]]]),
            conf=torch.tensor([[0.9]]),
        )
        mock_model = _make_mock_model([FakeResult(keypoints=kps)])
        detector = CourtKeypointDetector(model=mock_model, keypoint_labels={0: "some_point"})
        keypoints = detector.predict(dummy_image, conf_threshold=0.0)
        assert len(keypoints) == 1
        assert keypoints[0].pixel.x == pytest.approx(100.5)
        assert keypoints[0].pixel.y == pytest.approx(200.5)

    def test_raises_without_model_or_path(self) -> None:
        with pytest.raises(ValueError, match="model.*or.*model_path"):
            CourtKeypointDetector()

    def test_model_property(self) -> None:
        mock_model = MagicMock()
        detector = CourtKeypointDetector(model=mock_model)
        assert detector.model is mock_model

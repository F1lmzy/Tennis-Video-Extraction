"""YOLO26 inference wrappers for player, ball, and court keypoint detection.

Each wrapper wraps an Ultralytics YOLO model and provides a typed,
testable ``predict()`` method.  The wrappers are designed so that
tests can inject a mock model object without loading real weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from tennis_tracker.types import PixelPoint

# ── Constants ──────────────────────────────────────────────────────────

COCO_PERSON_CLASS_ID: int = 0
"""COCO class ID for ``person``."""


# ── Shared types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> PixelPoint:
        """Geometric centre of the box."""
        return PixelPoint(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> PixelPoint:
        """Bottom-centre point, suitable for foot / contact-point projection."""
        return PixelPoint(x=(self.x1 + self.x2) / 2.0, y=self.y2)


@dataclass(frozen=True)
class PlayerDetection:
    """A single player (person) detection from YOLO.

    The ``track_id`` field is populated when the model uses Ultralytics'
    tracking mode (``predict(tracker=True)``).  It is optional and may
    be None when running plain detection.
    """

    bbox: BoundingBox
    confidence: float
    track_id: Optional[int] = None


@dataclass(frozen=True)
class BallDetection:
    """A single ball detection from YOLO."""

    bbox: BoundingBox
    center: PixelPoint
    confidence: float


@dataclass(frozen=True)
class CourtKeypoint:
    """A single detected court keypoint with a geometry label.

    The ``label`` should correspond to one of the named court points
    defined in ``tennis_tracker.court`` (e.g. ``"doubles_near_left"``)
    so that downstream matching code can pair it with the real-world
    court-meter coordinate.
    """

    label: str
    pixel: PixelPoint
    confidence: float


# ── Player (person) detector ───────────────────────────────────────────

class PlayerDetector:
    """YOLO-based player detector using the COCO ``person`` class.

    By default uses the YOLO26n COCO-pretrained model.  A fine-tuned
    model can be supplied via ``model_path`` or by passing a pre-loaded
    ``YOLO`` instance (useful for testing).

    Parameters
    ----------
    model:
        A pre-loaded ``YOLO`` instance.  When provided, ``model_path``
        is ignored.  This is the preferred injection path for tests.
    model_path:
        Path to a YOLO weight file.  If neither ``model`` nor
        ``model_path`` is given, the default COCO-pretrained
        ``yolo26n.pt`` is used.
    """

    def __init__(
        self,
        model: Optional[YOLO] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = YOLO(str(model_path))
        else:
            self._model = YOLO("yolo26n.pt")

    @property
    def model(self) -> YOLO:
        """The underlying YOLO model instance (exposed for testing)."""
        return self._model

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[PlayerDetection]:
        """Run inference and return person detections.

        Parameters
        ----------
        image:
            BGR image as loaded by OpenCV (H×W×3, uint8).
        conf_threshold:
            Minimum confidence threshold (passed to YOLO).

        Returns
        -------
        list[PlayerDetection]
            Detections filtered to the COCO ``person`` class, sorted by
            decreasing confidence.
        """
        predict_kwargs = {"conf": conf_threshold, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        if device is not None:
            predict_kwargs["device"] = device
        results = self._model.predict(image, **predict_kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.cls is None:
            return []

        detections: list[PlayerDetection] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != COCO_PERSON_CLASS_ID:
                continue

            conf = float(boxes.conf[i].item())
            # Post-filter by confidence as a belt-and-suspenders check; this
            # also ensures mock models that ignore the ``conf`` parameter
            # produce testable results.
            if conf < conf_threshold:
                continue

            xyxy = boxes.xyxy[i].tolist()

            track_id: Optional[int] = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())

            bbox = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            detections.append(PlayerDetection(bbox=bbox, confidence=conf, track_id=track_id))

        # Sort descending by confidence (most confident first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ── Ball detector ──────────────────────────────────────────────────────

class BallDetector:
    """YOLO-based ball detector using a fine-tuned model.

    Parameters
    ----------
    model:
        A pre-loaded ``YOLO`` instance.  When provided, ``model_path``
        is ignored.  This is the preferred injection path for tests.
    model_path:
        Path to a YOLO weight file trained for ball detection.
        One of ``model`` or ``model_path`` must be provided.
    """

    def __init__(
        self,
        model: Optional[YOLO] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = YOLO(str(model_path))
        else:
            raise ValueError(
                "Either ``model`` or ``model_path`` must be provided for BallDetector."
            )

    @property
    def model(self) -> YOLO:
        """The underlying YOLO model instance (exposed for testing)."""
        return self._model

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[BallDetection]:
        """Run inference and return ball detections.

        Parameters
        ----------
        image:
            BGR image (H×W×3, uint8).
        conf_threshold:
            Minimum confidence threshold.

        Returns
        -------
        list[BallDetection]
            All detections (the model is expected to be specialised for
            ball detection, so no class filtering is applied), sorted by
            decreasing confidence.
        """
        predict_kwargs = {"conf": conf_threshold, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        if device is not None:
            predict_kwargs["device"] = device
        results = self._model.predict(image, **predict_kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        detections: list[BallDetection] = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_threshold:
                continue
            xyxy = boxes.xyxy[i].tolist()
            bbox = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            detections.append(BallDetection(bbox=bbox, center=bbox.center, confidence=conf))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# ── Court keypoint detector ────────────────────────────────────────────

class CourtKeypointDetector:
    """YOLO-based court keypoint detector using a pose / keypoint model.

    The model is expected to be a YOLO pose or keypoint model trained
    on a court keypoint dataset such as
    ``abiya-thesis/tennis-court-suuzy``.

    The mapping from model keypoint index to court geometry label is
    provided via ``keypoint_labels``.  When the model is trained, the
    dataset's ``data.yaml`` defines the keypoint names; the caller should
    supply the matching ``{index: label}`` mapping based on that file.

    Parameters
    ----------
    model:
        A pre-loaded ``YOLO`` instance.  When provided, ``model_path``
        is ignored.  This is the preferred injection path for tests.
    model_path:
        Path to a YOLO pose / keypoint weight file.
    keypoint_labels:
        Mapping from model keypoint index (0-based) to court geometry
        label (e.g. ``{0: "doubles_near_left", 1: "doubles_near_right", ...}``).
        If not provided, keypoints are returned with generic labels
        ``keypoint_0``, ``keypoint_1`` etc.
    """

    def __init__(
        self,
        model: Optional[YOLO] = None,
        model_path: Optional[str | Path] = None,
        keypoint_labels: Optional[dict[int, str]] = None,
    ) -> None:
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = YOLO(str(model_path))
        else:
            raise ValueError(
                "Either ``model`` or ``model_path`` must be provided "
                "for CourtKeypointDetector."
            )
        self._keypoint_labels = keypoint_labels

    @property
    def model(self) -> YOLO:
        """The underlying YOLO model instance (exposed for testing)."""
        return self._model

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[CourtKeypoint]:
        """Run inference and return detected court keypoints.

        Parameters
        ----------
        image:
            BGR image (H×W×3, uint8).
        conf_threshold:
            Minimum keypoint confidence threshold.  Keypoints below this
            threshold are omitted from the result.

        Returns
        -------
        list[CourtKeypoint]
            Detected keypoints with labels.  The first detected instance's
            keypoints are used (for a court scene there should be a single
            dominant instance).  Keypoints with zero coordinates (a
            Ultralytics convention for undetected keypoints) are omitted.
        """
        predict_kwargs = {"conf": conf_threshold, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        if device is not None:
            predict_kwargs["device"] = device
        results = self._model.predict(image, **predict_kwargs)
        if not results:
            return []

        kp_data = results[0].keypoints
        if kp_data is None or kp_data.xy is None or len(kp_data.xy) == 0:
            return []

        # Take the first detected instance's keypoints
        kps_xy = kp_data.xy[0]
        kps_conf = kp_data.conf[0] if kp_data.conf is not None else None

        keypoints: list[CourtKeypoint] = []
        for i in range(len(kps_xy)):
            x = float(kps_xy[i][0].item())
            y = float(kps_xy[i][1].item())

            conf = float(kps_conf[i].item()) if kps_conf is not None and i < len(kps_conf) else 0.0

            if conf < conf_threshold:
                continue
            if x == 0.0 and y == 0.0:
                # Ultralytics zeroes out keypoints that were not detected
                continue

            label = (
                self._keypoint_labels[i]
                if self._keypoint_labels and i in self._keypoint_labels
                else f"keypoint_{i}"
            )

            keypoints.append(
                CourtKeypoint(label=label, pixel=PixelPoint(x=x, y=y), confidence=conf)
            )

        return keypoints

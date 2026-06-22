"""Core data contracts for the tennis-tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrackKind(str, Enum):
    """Identifies the tracked object kind in a detection."""

    PLAYER_A = "player_a"
    PLAYER_B = "player_b"
    BALL = "ball"


@dataclass(frozen=True)
class PixelPoint:
    """A 2D point in image pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class CourtPoint:
    """A 2D point on the tennis court plane, measured in meters.

    Coordinates use the spec-approved convention:
    - Origin at the center of the full doubles court.
    - X across court width (positive to the right).
    - Y along court length (positive toward the far baseline).
    """

    x_m: float
    y_m: float
    confidence: float


@dataclass(frozen=True)
class Detection:
    """A single per-frame detection for a tracked object.

    Pixel coordinates are always present (the model saw something).
    Court coordinates are optional: they are set only after
    successful homography projection.
    """

    kind: TrackKind
    pixel: PixelPoint
    confidence: float
    court: Optional[CourtPoint] = None


@dataclass
class FrameMetadata:
    """Metadata for a single video frame."""

    frame_index: int
    time_s: float


@dataclass
class TrackRow:
    """Per-frame building block for a single tracked object (player or ball).

    This is the intermediate representation before assembling the
    full output CSV row.  Missing court coordinates are represented
    as None — never as 0.0 / 0.0.
    """

    frame_index: int
    time_s: float
    kind: TrackKind
    pixel: PixelPoint
    confidence: float
    court: Optional[CourtPoint] = None

    @property
    def has_court_position(self) -> bool:
        """Whether homography projection produced a valid court position."""
        return self.court is not None


@dataclass
class CourtCalibration:
    """Per-frame or temporally-stabilised court calibration result."""

    confidence: float
    homography_valid: bool
    reprojection_error_px: Optional[float] = None


@dataclass
class OutputRow:
    """One row of the final CSV output (raw or smoothed).

    Every field that can be missing uses Optional, and the CSV writer
    must emit empty fields for None values — never 0 or "0".
    """

    frame_index: int
    time_s: float
    player_a_x_m: Optional[float]
    player_a_y_m: Optional[float]
    player_a_pixel_x: Optional[float]
    player_a_pixel_y: Optional[float]
    player_a_confidence: Optional[float]
    player_b_x_m: Optional[float]
    player_b_y_m: Optional[float]
    player_b_pixel_x: Optional[float]
    player_b_pixel_y: Optional[float]
    player_b_confidence: Optional[float]
    ball_x_m: Optional[float]
    ball_y_m: Optional[float]
    ball_pixel_x: Optional[float]
    ball_pixel_y: Optional[float]
    ball_confidence: Optional[float]
    court_confidence: Optional[float]
    homography_valid: bool
    diagnostics: str = ""

    @staticmethod
    def csv_header() -> str:
        """Return the canonical CSV header matching docs/spec.md."""
        return (
            "frame_index,time_s,"
            "player_a_x_m,player_a_y_m,"
            "player_a_pixel_x,player_a_pixel_y,"
            "player_a_confidence,"
            "player_b_x_m,player_b_y_m,"
            "player_b_pixel_x,player_b_pixel_y,"
            "player_b_confidence,"
            "ball_x_m,ball_y_m,"
            "ball_pixel_x,ball_pixel_y,"
            "ball_confidence,"
            "court_confidence,homography_valid,diagnostics"
        )

    @staticmethod
    def field_names() -> list[str]:
        """Return field names in CSV order."""
        return [
            "frame_index",
            "time_s",
            "player_a_x_m",
            "player_a_y_m",
            "player_a_pixel_x",
            "player_a_pixel_y",
            "player_a_confidence",
            "player_b_x_m",
            "player_b_y_m",
            "player_b_pixel_x",
            "player_b_pixel_y",
            "player_b_confidence",
            "ball_x_m",
            "ball_y_m",
            "ball_pixel_x",
            "ball_pixel_y",
            "ball_confidence",
            "court_confidence",
            "homography_valid",
            "diagnostics",
        ]


@dataclass
class ModelArtifactMetadata:
    """Metadata about a trained or optimised model artifact."""

    path: str
    model_type: str  # e.g. "detection", "pose", "segment"
    source: str  # e.g. "pretrained_coco", "fine_tuned_ball"
    format: str = "pt"  # "pt", "onnx", "openvino"
    quantized: Optional[str] = None  # None, "int8", "fp16"
    benchmark_fps: Optional[float] = None

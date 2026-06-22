"""Tennis court geometry constants and coordinate convention.

All coordinates are in meters with the origin at the center of the
full doubles court.  The convention matches docs/spec.md:

    X-axis : court width  (positive to the right in top-view)
    Y-axis : court length (positive toward the far baseline)

Full doubles court dimensions:
    Width  = 10.97 m
    Length = 23.77 m

Singles court width = 8.23 m.
Service line distance from the net = 6.40 m.
Net height is *not* encoded here — this module models the court plane only.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Full court dimensions ──────────────────────────────────────────────
DOUBLES_WIDTH: float = 10.97   # meters
DOUBLES_LENGTH: float = 23.77  # meters

SINGLES_WIDTH: float = 8.23    # meters

SERVICE_LINE_DISTANCE: float = 6.40  # meters from the net

# ── Half dimensions (convenience for keypoint definitions) ─────────────
_HALF_WIDTH: float = DOUBLES_WIDTH / 2.0     # 5.485
_HALF_LENGTH: float = DOUBLES_LENGTH / 2.0   # 11.885
_HALF_SINGLES: float = SINGLES_WIDTH / 2.0   # 4.115


# ── Named court points (meters, origin at center) ──────────────────────

@dataclass(frozen=True)
class _CourtPoint:
    """A named point on the court plane, stored in meters."""

    x_m: float
    y_m: float
    label: str


# Doubles court corners
DOUBLES_NEAR_LEFT: _CourtPoint = _CourtPoint(-_HALF_WIDTH, -_HALF_LENGTH, "doubles_near_left")
DOUBLES_NEAR_RIGHT: _CourtPoint = _CourtPoint(_HALF_WIDTH, -_HALF_LENGTH, "doubles_near_right")
DOUBLES_FAR_LEFT: _CourtPoint = _CourtPoint(-_HALF_WIDTH, _HALF_LENGTH, "doubles_far_left")
DOUBLES_FAR_RIGHT: _CourtPoint = _CourtPoint(_HALF_WIDTH, _HALF_LENGTH, "doubles_far_right")

# Singles court corners
SINGLES_NEAR_LEFT: _CourtPoint = _CourtPoint(-_HALF_SINGLES, -_HALF_LENGTH, "singles_near_left")
SINGLES_NEAR_RIGHT: _CourtPoint = _CourtPoint(_HALF_SINGLES, -_HALF_LENGTH, "singles_near_right")
SINGLES_FAR_LEFT: _CourtPoint = _CourtPoint(-_HALF_SINGLES, _HALF_LENGTH, "singles_far_left")
SINGLES_FAR_RIGHT: _CourtPoint = _CourtPoint(_HALF_SINGLES, _HALF_LENGTH, "singles_far_right")

# Net (center of court, spans full doubles width)
NET_CENTER: _CourtPoint = _CourtPoint(0.0, 0.0, "net_center")
NET_LEFT: _CourtPoint = _CourtPoint(-_HALF_WIDTH, 0.0, "net_left")
NET_RIGHT: _CourtPoint = _CourtPoint(_HALF_WIDTH, 0.0, "net_right")

# Service line centres (midpoints of each service line)
SERVICE_NEAR_CENTER: _CourtPoint = _CourtPoint(0.0, -SERVICE_LINE_DISTANCE, "service_near_center")
SERVICE_FAR_CENTER: _CourtPoint = _CourtPoint(0.0, SERVICE_LINE_DISTANCE, "service_far_center")

# Service line endpoints (intersection with singles sideline)
SERVICE_NEAR_LEFT: _CourtPoint = _CourtPoint(
    -_HALF_SINGLES, -SERVICE_LINE_DISTANCE, "service_near_left"
)
SERVICE_NEAR_RIGHT: _CourtPoint = _CourtPoint(
    _HALF_SINGLES, -SERVICE_LINE_DISTANCE, "service_near_right"
)
SERVICE_FAR_LEFT: _CourtPoint = _CourtPoint(
    -_HALF_SINGLES, SERVICE_LINE_DISTANCE, "service_far_left"
)
SERVICE_FAR_RIGHT: _CourtPoint = _CourtPoint(
    _HALF_SINGLES, SERVICE_LINE_DISTANCE, "service_far_right"
)

# Center marks on each baseline
CENTER_MARK_NEAR: _CourtPoint = _CourtPoint(0.0, -_HALF_LENGTH, "center_mark_near")
CENTER_MARK_FAR: _CourtPoint = _CourtPoint(0.0, _HALF_LENGTH, "center_mark_far")

# ── Convenient groupings ───────────────────────────────────────────────

ALL_DOUBLES_CORNERS: tuple[_CourtPoint, ...] = (
    DOUBLES_NEAR_LEFT,
    DOUBLES_NEAR_RIGHT,
    DOUBLES_FAR_LEFT,
    DOUBLES_FAR_RIGHT,
)

ALL_SINGLES_CORNERS: tuple[_CourtPoint, ...] = (
    SINGLES_NEAR_LEFT,
    SINGLES_NEAR_RIGHT,
    SINGLES_FAR_LEFT,
    SINGLES_FAR_RIGHT,
)

ALL_SERVICE_POINTS: tuple[_CourtPoint, ...] = (
    SERVICE_NEAR_LEFT,
    SERVICE_NEAR_CENTER,
    SERVICE_NEAR_RIGHT,
    SERVICE_FAR_LEFT,
    SERVICE_FAR_CENTER,
    SERVICE_FAR_RIGHT,
)

ALL_CENTER_MARKS: tuple[_CourtPoint, ...] = (CENTER_MARK_NEAR, CENTER_MARK_FAR)

ALL_NAMED_POINTS: tuple[_CourtPoint, ...] = (
    *ALL_DOUBLES_CORNERS,
    *ALL_SINGLES_CORNERS,
    NET_CENTER,
    NET_LEFT,
    NET_RIGHT,
    *ALL_SERVICE_POINTS,
    *ALL_CENTER_MARKS,
)


# ── Helper functions ───────────────────────────────────────────────────

def named_point_by_label(label: str) -> _CourtPoint | None:
    """Return the named point with the given label, or None."""
    for pt in ALL_NAMED_POINTS:
        if pt.label == label:
            return pt
    return None

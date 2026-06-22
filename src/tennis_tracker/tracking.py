"""Player and ball tracking assignment for singles matches.

Maintains player A/B labels across frames under the no-side-change
assumption using nearest-neighbour matching.  Selects the two most
plausible singles players from raw person detections and emits
diagnostics for missing or ambiguous cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tennis_tracker.detection import BallDetection, PlayerDetection
from tennis_tracker.diagnostics import Diagnostics

# ── Thresholds ─────────────────────────────────────────────────────────

_DEFAULT_MAX_MATCH_DIST_PX: float = 200.0
"""Maximum pixel distance for matching a player across frames.

Detections farther than this from the expected position are treated
as new detections rather than continuing an existing track.
"""

_LOW_CONFIDENCE_THRESHOLD: float = 0.3
"""Detections below this confidence are flagged with a
``low_player_*_confidence`` diagnostic."""


# ── Tracking output ────────────────────────────────────────────────────

@dataclass
class TrackedFrame:
    """Resolved tracking output for a single frame.

    ``player_a`` and ``player_b`` are the two selected singles players.
    ``ball`` is the best ball detection for this frame.
    ``tracked`` is ``True`` when *at least one* player was assigned
    (not necessarily both — one may be missing).
    """

    player_a: Optional[PlayerDetection]
    player_b: Optional[PlayerDetection]
    ball: Optional[BallDetection]
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    @property
    def tracked(self) -> bool:
        """At least one player detection was available."""
        return self.player_a is not None or self.player_b is not None


# ── Internal state ─────────────────────────────────────────────────────

@dataclass
class _PlayerTrackState:
    """Position memory for one tracked player."""

    bottom_center_x: float
    bottom_center_y: float
    confidence: float


@dataclass
class _TrackingState:
    """State maintained across frames for label continuity."""

    player_a: Optional[_PlayerTrackState]
    player_b: Optional[_PlayerTrackState]


# ── Euclidean distance helper ──────────────────────────────────────────

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# ── Player tracker ─────────────────────────────────────────────────────

class PlayerTracker:
    """Maintains player A/B labels across consecutive frames.

    Uses a simple greedy nearest-neighbour assignment:
      1. Match each previously tracked player to the closest current
         detection within ``max_match_distance`` pixels.
      2. Assign the remaining unmatched detection (if any) to the
         unmatched track.
      3. If no prior state exists, select the two highest-confidence
         detections.
      4. If fewer than two detections are available, assign what is
         available and emit missing diagnostics.

    Parameters
    ----------
    max_match_distance:
        Maximum pixel distance for a cross-frame match.
    low_confidence_threshold:
        Detections below this confidence trigger a
        ``low_player_*_confidence`` diagnostic.
    """

    def __init__(
        self,
        max_match_distance: float = _DEFAULT_MAX_MATCH_DIST_PX,
        low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._state: Optional[_TrackingState] = None
        self._max_match_distance = max_match_distance
        self._low_confidence_threshold = low_confidence_threshold

    def update(
        self,
        detections: list[PlayerDetection],
        ball: Optional[BallDetection] = None,
    ) -> TrackedFrame:
        """Process one frame and return the resolved tracking output.

        Parameters
        ----------
        detections:
            Raw person detections for this frame, in any order.
        ball:
            Best ball detection for this frame, if any.

        Returns
        -------
        TrackedFrame
            Resolved player and ball assignments with diagnostics.
        """
        diag = Diagnostics()
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # --- Resolve player A/B ---
        if len(sorted_dets) >= 2:
            # Have ≥2 detections — use top 2
            chosen = sorted_dets[:2]
            if len(sorted_dets) > 2:
                diag.add_flag("ambigous_player_id")

            state = self._state
            if state is not None and state.player_a is not None and state.player_b is not None:
                # Match to existing tracks by nearest neighbour
                a_det, b_det = self._assign_by_proximity(chosen, state)
            else:
                # No prior state — arbitrary assignment by confidence
                a_det, b_det = chosen[0], chosen[1]

        elif len(sorted_dets) == 1:
            diag.add_flag("missing_player_b")
            # The single detection goes to the track whose last position
            # is closest, or to player A if no history.
            state = self._state
            if state is not None and state.player_b is not None and state.player_a is None:
                a_det, b_det = None, sorted_dets[0]
            else:
                a_det, b_det = sorted_dets[0], None

        else:
            # No detections
            diag.add_flag("missing_player_a")
            diag.add_flag("missing_player_b")
            a_det, b_det = None, None

        # --- Low-confidence diagnostics ---
        if a_det is not None and a_det.confidence < self._low_confidence_threshold:
            diag.add_flag("low_player_a_confidence")
        if b_det is not None and b_det.confidence < self._low_confidence_threshold:
            diag.add_flag("low_player_b_confidence")

        # --- Ball ---
        if ball is not None and ball.confidence < self._low_confidence_threshold:
            pass  # ball is returned but caller can check confidence
        # We don't emit ball diagnostics here — the pipeline layer does

        # --- Update internal state ---
        self._state = _TrackingState(
            player_a=(
                _PlayerTrackState(
                    bottom_center_x=a_det.bbox.bottom_center.x,
                    bottom_center_y=a_det.bbox.bottom_center.y,
                    confidence=a_det.confidence,
                )
                if a_det is not None
                else None
            ),
            player_b=(
                _PlayerTrackState(
                    bottom_center_x=b_det.bbox.bottom_center.x,
                    bottom_center_y=b_det.bbox.bottom_center.y,
                    confidence=b_det.confidence,
                )
                if b_det is not None
                else None
            ),
        )

        return TrackedFrame(
            player_a=a_det,
            player_b=b_det,
            ball=ball,
            diagnostics=diag,
        )

    # --- private helpers -------------------------------------------------

    @staticmethod
    def _assign_by_proximity(
        detections: list[PlayerDetection],
        state: _TrackingState,
    ) -> tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """Greedy nearest-neighbour assignment of detections to tracks.

        Returns ``(player_a_detection, player_b_detection)``.  A track
        whose nearest detection is farther than ``_max_match_distance``
        gets ``None``.
        """
        # We only call this when both players are tracked and we have 2 detections
        if len(detections) < 2:
            return detections[0] if detections else None, None

        d_a, d_b = detections[0], detections[1]

        if state.player_a is None or state.player_b is None:
            return d_a, d_b

        pa_bc = (state.player_a.bottom_center_x, state.player_a.bottom_center_y)
        pb_bc = (state.player_b.bottom_center_x, state.player_b.bottom_center_y)

        da_bc = (d_a.bbox.bottom_center.x, d_a.bbox.bottom_center.y)
        db_bc = (d_b.bbox.bottom_center.x, d_b.bbox.bottom_center.y)

        # Compute all 4 distances
        a_a = _dist(pa_bc[0], pa_bc[1], da_bc[0], da_bc[1])
        a_b = _dist(pa_bc[0], pa_bc[1], db_bc[0], db_bc[1])
        b_a = _dist(pb_bc[0], pb_bc[1], da_bc[0], da_bc[1])
        b_b = _dist(pb_bc[0], pb_bc[1], db_bc[0], db_bc[1])

        # Greedy: assign the closest pair first
        if a_a + b_b <= a_b + b_a:
            return (d_a, d_b)
        else:
            return (d_b, d_a)


# ── Frame tracking convenience ─────────────────────────────────────────

def _bbox_motion_score(
    detection: BallDetection,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    padding: int = 4,
) -> float:
    """Mean absolute pixel difference around a candidate ball box.

    Static white court artifacts tend to look nearly identical between
    consecutive fixed-camera frames.  A real tennis ball usually changes
    the local patch from the previous frame, so this score is useful for
    rejecting static false positives.
    """
    height, width = current_frame.shape[:2]
    x1 = max(0, int(detection.bbox.x1) - padding)
    y1 = max(0, int(detection.bbox.y1) - padding)
    x2 = min(width, int(detection.bbox.x2) + padding)
    y2 = min(height, int(detection.bbox.y2) + padding)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    prev_patch = previous_frame[y1:y2, x1:x2].astype(np.float32)
    curr_patch = current_frame[y1:y2, x1:x2].astype(np.float32)
    if prev_patch.shape != curr_patch.shape or prev_patch.size == 0:
        return 0.0
    return float(np.mean(np.abs(curr_patch - prev_patch)))


def _center_distance(a: BallDetection, b: BallDetection) -> float:
    return _dist(a.center.x, a.center.y, b.center.x, b.center.y)


def has_local_ball_motion(
    detection: BallDetection,
    previous_frame: Optional[np.ndarray],
    current_frame: Optional[np.ndarray],
    *,
    min_motion_score: float = 1.0,
) -> bool:
    """Return True when a ball candidate changes locally between frames."""
    if previous_frame is None or current_frame is None:
        return True
    return _bbox_motion_score(detection, previous_frame, current_frame) >= min_motion_score


def _ball_shape_penalty(detection: BallDetection) -> float:
    """Penalty for boxes that look more like line segments than balls."""
    width = max(detection.bbox.width, 1e-6)
    height = max(detection.bbox.height, 1e-6)
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio <= 2.0:
        return 0.0
    return min((aspect_ratio - 2.0) / 3.0, 1.0)


def select_best_ball(
    detections: list[BallDetection],
    *,
    previous_frame: Optional[np.ndarray] = None,
    current_frame: Optional[np.ndarray] = None,
    previous_ball: Optional[BallDetection] = None,
    motion_weight: float = 1.25,
    proximity_weight: float = 0.35,
    max_proximity_px: float = 180.0,
    min_motion_score: float = 1.0,
    static_penalty: float = 0.75,
    shape_weight: float = 0.35,
) -> Optional[BallDetection]:
    """Return the most plausible ball detection, or ``None``.

    With only detections, this preserves the original behaviour and selects
    the highest-confidence box.  When consecutive frames are supplied, the
    score also rewards local pixel motion, which suppresses fixed white court
    artifacts that a ball detector may repeatedly classify as balls.
    """
    if not detections:
        return None

    if previous_frame is None or current_frame is None:
        return max(
            detections,
            key=lambda d: d.confidence - shape_weight * _ball_shape_penalty(d),
        )

    motion_by_detection = {
        det: _bbox_motion_score(det, previous_frame, current_frame)
        for det in detections
    }
    moving_detections = [
        det for det, motion_score in motion_by_detection.items()
        if motion_score >= min_motion_score
    ]

    if previous_ball is not None:
        near_moving_detections = [
            det for det in moving_detections
            if _center_distance(det, previous_ball) <= max_proximity_px
        ]
        # Once a moving ball track exists, do not fall back to static or far-away
        # detections.  Returning None is preferable because the smoothing layer
        # can interpolate short gaps, while accepting a court line corrupts the
        # trajectory and makes following frames lock onto the wrong object.
        candidates = near_moving_detections
    else:
        # During initial acquisition, require local motion when frame context is
        # available.  This avoids bootstrapping the track from a static line.
        candidates = moving_detections

    if not candidates:
        return None

    def score(det: BallDetection) -> float:
        motion_score = motion_by_detection[det]
        # Normalize typical 8-bit frame differences into roughly [0, 1].
        motion = min(motion_score / 25.0, 1.0)
        total = det.confidence + motion_weight * motion
        if motion_score < min_motion_score:
            total -= static_penalty
        total -= shape_weight * _ball_shape_penalty(det)
        if previous_ball is not None and motion_score >= min_motion_score:
            distance = _center_distance(det, previous_ball)
            proximity = max(0.0, 1.0 - distance / max_proximity_px)
            total += proximity_weight * proximity
        return total

    return max(candidates, key=score)

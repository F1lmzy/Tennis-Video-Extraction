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

from tennis_tracker.detection import BallDetection, BoundingBox, PlayerDetection
from tennis_tracker.diagnostics import Diagnostics
from tennis_tracker.types import PixelPoint

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
    missing_frames: int = 0


@dataclass
class _TrackingState:
    """State maintained across frames for label continuity."""

    player_a: Optional[_PlayerTrackState]
    player_b: Optional[_PlayerTrackState]


_DEFAULT_MAX_MISSING_FRAMES: int = 15
"""Number of consecutive missed frames to remember a player's last position."""

_MIN_REACQUIRE_SEPARATION_PX: float = 40.0
"""Minimum pixel distance from an active track when re-acquiring the other player."""


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
        max_missing_frames: int = _DEFAULT_MAX_MISSING_FRAMES,
    ) -> None:
        self._state: Optional[_TrackingState] = None
        self._max_match_distance = max_match_distance
        self._low_confidence_threshold = low_confidence_threshold
        self._max_missing_frames = max_missing_frames

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
        state = self._state
        has_prior = state is not None and (
            state.player_a is not None or state.player_b is not None
        )

        # --- Resolve player A/B ---
        if len(sorted_dets) >= 2:
            if len(sorted_dets) > 2:
                diag.add_flag("ambigous_player_id")

            if has_prior:
                assert state is not None
                a_det, b_det = self._assign_by_proximity(
                    sorted_dets,
                    state,
                    self._max_match_distance,
                )
            else:
                # No prior state — assign top two by confidence
                chosen = sorted_dets[:2]
                a_det, b_det = chosen[0], chosen[1]

        elif len(sorted_dets) == 1:
            det = sorted_dets[0]
            if has_prior:
                assert state is not None
                a_det, b_det = self._assign_single(det, state, self._max_match_distance)
            else:
                a_det, b_det = det, None

        else:
            a_det, b_det = None, None

        if a_det is None:
            diag.add_flag("missing_player_a")
        if b_det is None:
            diag.add_flag("missing_player_b")

        # --- Low-confidence diagnostics ---
        if a_det is not None and a_det.confidence < self._low_confidence_threshold:
            diag.add_flag("low_player_a_confidence")
        if b_det is not None and b_det.confidence < self._low_confidence_threshold:
            diag.add_flag("low_player_b_confidence")

        # --- Update internal state (retain last position across brief misses) ---
        prev_a = state.player_a if state is not None else None
        prev_b = state.player_b if state is not None else None

        next_a = self._next_track_state(a_det, prev_a, self._max_missing_frames)
        next_b = self._next_track_state(b_det, prev_b, self._max_missing_frames)
        self._state = _TrackingState(player_a=next_a, player_b=next_b)

        return TrackedFrame(
            player_a=a_det,
            player_b=b_det,
            ball=ball,
            diagnostics=diag,
        )

    # --- private helpers -------------------------------------------------

    @staticmethod
    def _next_track_state(
        det: Optional[PlayerDetection],
        prev: Optional[_PlayerTrackState],
        max_missing_frames: int,
    ) -> Optional[_PlayerTrackState]:
        if det is not None:
            return _PlayerTrackState(
                bottom_center_x=det.bbox.bottom_center.x,
                bottom_center_y=det.bbox.bottom_center.y,
                confidence=det.confidence,
                missing_frames=0,
            )
        if prev is not None and prev.missing_frames + 1 < max_missing_frames:
            return _PlayerTrackState(
                bottom_center_x=prev.bottom_center_x,
                bottom_center_y=prev.bottom_center_y,
                confidence=prev.confidence,
                missing_frames=prev.missing_frames + 1,
            )
        return None

    @staticmethod
    def _assign_single(
        det: PlayerDetection,
        state: _TrackingState,
        max_match_distance: float,
    ) -> tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """Assign a single detection to whichever existing track is closest."""
        bc = det.bbox.bottom_center
        dist_a = (
            _dist(state.player_a.bottom_center_x, state.player_a.bottom_center_y, bc.x, bc.y)
            if state.player_a is not None
            else float("inf")
        )
        dist_b = (
            _dist(state.player_b.bottom_center_x, state.player_b.bottom_center_y, bc.x, bc.y)
            if state.player_b is not None
            else float("inf")
        )
        if state.player_a is not None and state.player_b is not None:
            if dist_a <= dist_b and dist_a <= max_match_distance:
                return det, None
            if dist_b < dist_a and dist_b <= max_match_distance:
                return None, det
            # Neither track within threshold — keep track continuity on A if closer
            return (det, None) if dist_a <= dist_b else (None, det)
        if state.player_b is not None and state.player_a is None:
            if dist_b <= max_match_distance:
                return None, det
            return det, None
        return det, None

    @staticmethod
    def _assign_by_proximity(
        detections: list[PlayerDetection],
        state: _TrackingState,
        max_match_distance: float,
    ) -> tuple[Optional[PlayerDetection], Optional[PlayerDetection]]:
        """Assign detections to tracks by minimum movement, re-acquiring empty tracks."""
        if not detections:
            return None, None

        def center(det: PlayerDetection) -> tuple[float, float]:
            bc = det.bbox.bottom_center
            return bc.x, bc.y

        def best_for_track(
            track: Optional[_PlayerTrackState],
            excluded: set[int],
        ) -> tuple[Optional[int], float]:
            if track is None:
                return None, float("inf")
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for idx, det in enumerate(detections):
                if idx in excluded:
                    continue
                x, y = center(det)
                distance = _dist(track.bottom_center_x, track.bottom_center_y, x, y)
                if distance < best_dist:
                    best_idx = idx
                    best_dist = distance
            if best_dist > max_match_distance:
                return None, best_dist
            return best_idx, best_dist

        def pick_reacquire(excluded: set[int], other_idx: Optional[int]) -> Optional[int]:
            """Pick highest-confidence unmatched detection separated from the other player."""
            for idx, det in enumerate(detections):
                if idx in excluded:
                    continue
                if other_idx is not None:
                    ox, oy = center(detections[other_idx])
                    dx, dy = center(det)
                    if _dist(ox, oy, dx, dy) < _MIN_REACQUIRE_SEPARATION_PX:
                        continue
                return idx
            return None

        # When both tracks have history, prefer the non-overlapping pair with
        # minimum total movement.
        if state.player_a is not None and state.player_b is not None:
            best_pair: tuple[Optional[int], Optional[int]] = (None, None)
            best_total = float("inf")
            for a_idx, a_det in enumerate(detections):
                ax, ay = center(a_det)
                dist_a = _dist(
                    state.player_a.bottom_center_x,
                    state.player_a.bottom_center_y,
                    ax,
                    ay,
                )
                if dist_a > max_match_distance:
                    continue
                for b_idx, b_det in enumerate(detections):
                    if b_idx == a_idx:
                        continue
                    bx, by = center(b_det)
                    dist_b = _dist(
                        state.player_b.bottom_center_x,
                        state.player_b.bottom_center_y,
                        bx,
                        by,
                    )
                    if dist_b > max_match_distance:
                        continue
                    total = dist_a + dist_b
                    if total < best_total:
                        best_pair = (a_idx, b_idx)
                        best_total = total

            a_idx, b_idx = best_pair
            if a_idx is not None or b_idx is not None:
                return (
                    detections[a_idx] if a_idx is not None else None,
                    detections[b_idx] if b_idx is not None else None,
                )

            # Fallback: if only one of the two tracks has a nearby match, keep it
            # rather than dropping both players.
            a_idx, dist_a = best_for_track(state.player_a, set())
            b_idx, dist_b = best_for_track(state.player_b, set())
            if a_idx is not None and b_idx is not None and a_idx == b_idx:
                if dist_a <= dist_b:
                    b_idx = None
                else:
                    a_idx = None
            return (
                detections[a_idx] if a_idx is not None else None,
                detections[b_idx] if b_idx is not None else None,
            )

        # Exactly one track has history — match it first, then re-acquire the
        # empty track from any remaining spatially distinct detection.
        a_idx, _ = best_for_track(state.player_a, set())
        b_idx, _ = best_for_track(state.player_b, {a_idx} if a_idx is not None else set())

        used = {idx for idx in (a_idx, b_idx) if idx is not None}
        if state.player_a is None and a_idx is None:
            a_idx = pick_reacquire(used, b_idx)
            if a_idx is not None:
                used.add(a_idx)
        if state.player_b is None and b_idx is None:
            b_idx = pick_reacquire(used, a_idx)

        return (
            detections[a_idx] if a_idx is not None else None,
            detections[b_idx] if b_idx is not None else None,
        )


# ── Frame tracking convenience ─────────────────────────────────────────

def _bbox_motion_score(
    detection: BallDetection,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    padding: int = 2,
    camera_transform: Optional[np.ndarray] = None,
) -> float:
    """Mean absolute pixel difference around a candidate ball box.

    Static court/wall artifacts tend to look nearly identical between
    consecutive frames.  When *camera_transform* (a 3×3 homography mapping
    current-frame pixels to previous-frame pixels) is supplied, the previous
    patch is sampled at the camera-compensated location so broadcast pan/zoom
    drift does not create false motion on static wall logos.
    """
    height, width = current_frame.shape[:2]
    x1 = max(0, int(detection.bbox.x1) - padding)
    y1 = max(0, int(detection.bbox.y1) - padding)
    x2 = min(width, int(detection.bbox.x2) + padding)
    y2 = min(height, int(detection.bbox.y2) + padding)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    curr_patch = current_frame[y1:y2, x1:x2].astype(np.float32)
    ph, pw = curr_patch.shape[:2]

    if camera_transform is not None:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        vec = camera_transform @ np.array([cx, cy, 1.0], dtype=np.float64)
        if abs(vec[2]) > 1e-8:
            prev_cx = float(vec[0] / vec[2])
            prev_cy = float(vec[1] / vec[2])
            dx = int(round(prev_cx - cx))
            dy = int(round(prev_cy - cy))
            px1 = max(0, min(width - pw, x1 + dx))
            py1 = max(0, min(height - ph, y1 + dy))
            prev_patch = previous_frame[py1 : py1 + ph, px1 : px1 + pw].astype(np.float32)
        else:
            prev_patch = previous_frame[y1:y2, x1:x2].astype(np.float32)
    else:
        prev_patch = previous_frame[y1:y2, x1:x2].astype(np.float32)

    if prev_patch.shape != curr_patch.shape or prev_patch.size == 0:
        return 0.0
    return float(np.mean(np.abs(curr_patch - prev_patch)))


def _center_distance(a: BallDetection, b: BallDetection) -> float:
    return _dist(a.center.x, a.center.y, b.center.x, b.center.y)


def ball_center_distance(a: BallDetection, b: BallDetection) -> float:
    """Return pixel distance between two ball detection centres."""
    return _center_distance(a, b)


def has_local_ball_motion(
    detection: BallDetection,
    previous_frame: Optional[np.ndarray],
    current_frame: Optional[np.ndarray],
    *,
    min_motion_score: float = 1.0,
    camera_transform: Optional[np.ndarray] = None,
) -> bool:
    """Return True when a ball candidate changes locally between frames."""
    if previous_frame is None or current_frame is None:
        return True
    return (
        _bbox_motion_score(
            detection,
            previous_frame,
            current_frame,
            camera_transform=camera_transform,
        )
        >= min_motion_score
    )


def _ball_shape_penalty(detection: BallDetection) -> float:
    """Penalty for boxes that look more like line segments than balls."""
    width = max(detection.bbox.width, 1e-6)
    height = max(detection.bbox.height, 1e-6)
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio <= 2.0:
        return 0.0
    return min((aspect_ratio - 2.0) / 3.0, 1.0)


def _point_in_box(px: float, py: float, box: BoundingBox, pad: float = 0.0) -> bool:
    return (box.x1 - pad) <= px <= (box.x2 + pad) and (box.y1 - pad) <= py <= (box.y2 + pad)


def _racket_or_player_penalty(
    detection: BallDetection,
    racket_boxes: Optional[list[BoundingBox]],
    player_boxes: Optional[list[BoundingBox]],
) -> float:
    """Soft score penalty when a ball candidate overlaps a racket or player body."""
    cx, cy = detection.center.x, detection.center.y
    if racket_boxes:
        for rb in racket_boxes:
            if _point_in_box(cx, cy, rb, pad=6.0):
                return 0.25
    if player_boxes:
        for pb in player_boxes:
            if _point_in_box(cx, cy, pb, pad=4.0):
                return 0.15
    return 0.0


def select_best_ball(
    detections: list[BallDetection],
    *,
    previous_frame: Optional[np.ndarray] = None,
    current_frame: Optional[np.ndarray] = None,
    previous_ball: Optional[BallDetection] = None,
    predicted_ball_center: Optional[PixelPoint] = None,
    camera_transform: Optional[np.ndarray] = None,
    racket_boxes: Optional[list[BoundingBox]] = None,
    player_boxes: Optional[list[BoundingBox]] = None,
    motion_weight: float = 0.35,
    proximity_weight: float = 0.45,
    max_proximity_px: float = 180.0,
    min_motion_score: float = 1.0,
    static_penalty: float = 0.75,
    shape_weight: float = 0.35,
) -> Optional[BallDetection]:
    """Return the most plausible ball detection, or ``None``.

    With only detections, this preserves the original behaviour and selects
    the highest-confidence box.  When consecutive frames are supplied, local
    motion acts as a gate against static artifacts while constant-velocity
    prediction (``predicted_ball_center``) and soft racket/player-box penalties
    prevent swinging rackets from beating a departing ball.
    """
    if not detections:
        return None

    if previous_frame is None or current_frame is None:
        return max(
            detections,
            key=lambda d: (
                d.confidence
                - shape_weight * _ball_shape_penalty(d)
                - _racket_or_player_penalty(d, racket_boxes, player_boxes)
            ),
        )

    motion_by_detection = {
        det: _bbox_motion_score(
            det,
            previous_frame,
            current_frame,
            camera_transform=camera_transform,
        )
        for det in detections
    }
    moving_detections = [
        det for det, motion_score in motion_by_detection.items()
        if motion_score >= min_motion_score
    ]

    anchor_pt = (
        predicted_ball_center
        if predicted_ball_center is not None
        else (previous_ball.center if previous_ball is not None else None)
    )

    if previous_ball is not None:
        # Accept candidates close to either the velocity-predicted position or
        # the previous position (handles sharp bounces / hit reversals).
        near_moving_detections = [
            det for det in moving_detections
            if (
                _center_distance(det, previous_ball) <= max_proximity_px
                or (
                    predicted_ball_center is not None
                    and _dist(
                        det.center.x,
                        det.center.y,
                        predicted_ball_center.x,
                        predicted_ball_center.y,
                    )
                    <= max_proximity_px
                )
            )
        ]
        candidates = near_moving_detections
    else:
        candidates = moving_detections

    if not candidates:
        return None

    def score(det: BallDetection) -> float:
        motion_score = motion_by_detection[det]
        motion = min(motion_score / 25.0, 1.0)
        total = det.confidence + motion_weight * motion
        if motion_score < min_motion_score:
            total -= static_penalty
        total -= shape_weight * _ball_shape_penalty(det)
        total -= _racket_or_player_penalty(det, racket_boxes, player_boxes)
        if anchor_pt is not None and motion_score >= min_motion_score:
            distance = _dist(det.center.x, det.center.y, anchor_pt.x, anchor_pt.y)
            proximity = max(0.0, 1.0 - distance / max_proximity_px)
            total += proximity_weight * proximity
        return total

    return max(candidates, key=score)


def find_static_ball_hotspots(
    detections_per_frame: list[list[BallDetection]],
    *,
    cell_size_px: float = 24.0,
    min_frame_hits: int = 15,
    max_median_step_px: float = 4.0,
) -> list[tuple[float, float]]:
    """Identify persistent stationary background false-positive hotspots.

    Static background objects (such as wall logos or scoreboard dots) repeatedly
    trigger ball detections in nearly the same pixel neighborhood across many
    frames with very small frame-to-frame displacement, whereas a real tennis
    ball moves rapidly across the image.

    Returns a list of ``(x, y)`` pixel centers to mask out.
    """
    n_frames = len(detections_per_frame)
    if n_frames < min_frame_hits:
        return []

    # Bucket detections by coarse pixel cell (with 3x3 neighbor clustering)
    cell_obs: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for f_idx, dets in enumerate(detections_per_frame):
        for det in dets:
            cx, cy = det.center.x, det.center.y
            key = (int(cx // cell_size_px), int(cy // cell_size_px))
            cell_obs.setdefault(key, []).append((f_idx, cx, cy))

    hotspots: list[tuple[float, float]] = []
    visited: set[tuple[int, int]] = set()
    for key in sorted(cell_obs, key=lambda k: len(cell_obs[k]), reverse=True):
        if key in visited:
            continue
        # Gather 3x3 neighborhood to handle slow camera pan around a wall logo
        cluster: list[tuple[int, float, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nk = (key[0] + dx, key[1] + dy)
                if nk in cell_obs and nk not in visited:
                    cluster.extend(cell_obs[nk])
        unique_frames = {f for f, _, _ in cluster}
        if len(unique_frames) < min_frame_hits:
            continue

        # Sort by frame and compute consecutive-frame displacements
        cluster.sort(key=lambda item: item[0])
        steps: list[float] = []
        for i in range(1, len(cluster)):
            f0, x0, y0 = cluster[i - 1]
            f1, x1, y1 = cluster[i]
            if 1 <= f1 - f0 <= 3:
                steps.append(_dist(x0, y0, x1, y1) / (f1 - f0))
        if steps and float(np.median(steps)) <= max_median_step_px:
            med_x = float(np.median([x for _, x, _ in cluster]))
            med_y = float(np.median([y for _, _, y in cluster]))
            hotspots.append((med_x, med_y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    visited.add((key[0] + dx, key[1] + dy))

    return hotspots


def filter_static_ball_hotspots(
    detections: list[BallDetection],
    hotspots: list[tuple[float, float]],
    *,
    exclusion_radius_px: float = 28.0,
) -> list[BallDetection]:
    """Drop detections falling within *exclusion_radius_px* of any static hotspot."""
    if not hotspots or not detections:
        return detections
    return [
        det
        for det in detections
        if all(
            _dist(det.center.x, det.center.y, hx, hy) > exclusion_radius_px
            for hx, hy in hotspots
        )
    ]


def refine_ball_sequence(
    selected: list[Optional[BallDetection]],
    candidates_per_frame: list[list[BallDetection]],
    *,
    racket_boxes_per_frame: Optional[list[list[BoundingBox]]] = None,
    player_boxes_per_frame: Optional[list[list[BoundingBox]]] = None,
    max_bridge_px: float = 180.0,
    kink_threshold_px: float = 40.0,
) -> list[Optional[BallDetection]]:
    """Bidirectional trajectory refinement and 1-frame racket-spike pruning.

    Runs two lightweight offline passes over the per-frame ball selections:
      1. **Bidirectional candidate swap**: When frame ``t`` is bracketed by
         ``t-1`` and ``t+1`` (or starts a run followed by ``t+1`` and ``t+2``),
         check all motion-valid candidates on frame ``t``. If an alternative
         candidate is dramatically closer to the collinear trajectory (e.g., the
         departing ball instead of a follow-through racket), swap it in.
      2. **Isolated kink pruning**: If a selected ball on frame ``t`` deviates
         sharply (``> kink_threshold_px``) from the midpoint of ``(t-1, t+1)``
         while the surrounding segment ``t-2 → t-1 → t+1`` (or ``t-1 → t+1 → t+2``)
         is collinear (not a real bounce/hit reversal), drop the 1-frame spike
         so downstream smoothing fills it cleanly.
    """
    n = len(selected)
    if n < 3:
        return list(selected)

    refined = list(selected)

    def _penalty(t: int, det: BallDetection) -> float:
        r_boxes = racket_boxes_per_frame[t] if racket_boxes_per_frame else None
        p_boxes = player_boxes_per_frame[t] if player_boxes_per_frame else None
        return _racket_or_player_penalty(det, r_boxes, p_boxes)

    # Pass 1a: Bracketed candidate swap (t-1 and t+1 both present)
    for t in range(1, n - 1):
        prev_b = refined[t - 1]
        next_b = refined[t + 1]
        cands = candidates_per_frame[t]
        if prev_b is None or next_b is None or len(cands) <= 1:
            continue
        span = _center_distance(prev_b, next_b)
        if span > max_bridge_px:
            continue
        mid_x = 0.5 * (prev_b.center.x + next_b.center.x)
        mid_y = 0.5 * (prev_b.center.y + next_b.center.y)
        cur = refined[t]
        cur_err = (
            _dist(cur.center.x, cur.center.y, mid_x, mid_y)
            if cur is not None
            else float("inf")
        )
        if cur is not None and cur_err <= 20.0:
            continue

        best_cand = cur
        best_cost = (
            cur_err - 25.0 * cur.confidence + 30.0 * _penalty(t, cur)
            if cur is not None
            else float("inf")
        )
        for cand in cands:
            err = _dist(cand.center.x, cand.center.y, mid_x, mid_y)
            if err > max_bridge_px * 0.6:
                continue
            cost = err - 25.0 * cand.confidence + 30.0 * _penalty(t, cand)
            if cost < best_cost and (cur is None or err + 15.0 < cur_err):
                best_cost = cost
                best_cand = cand
        refined[t] = best_cand

    # Pass 1b: Segment-start backward lookahead (t-1 missing, t+1 & t+2 present)
    for t in range(0, n - 2):
        if (t > 0 and refined[t - 1] is not None) or refined[t] is None:
            continue
        b1, b2 = refined[t + 1], refined[t + 2]
        cands = candidates_per_frame[t]
        if b1 is None or b2 is None or len(cands) <= 1:
            continue
        step = _center_distance(b1, b2)
        if step > max_bridge_px:
            continue
        pred_x = 2.0 * b1.center.x - b2.center.x
        pred_y = 2.0 * b1.center.y - b2.center.y
        cur = refined[t]
        assert cur is not None
        cur_err = _dist(cur.center.x, cur.center.y, pred_x, pred_y)
        if cur_err <= 25.0:
            continue
        best_cand = cur
        best_err = cur_err
        for cand in cands:
            err = _dist(cand.center.x, cand.center.y, pred_x, pred_y)
            if err + 25.0 < best_err and cand.confidence >= cur.confidence:
                best_err = err
                best_cand = cand
        refined[t] = best_cand

    # Pass 2: Prune isolated 1-frame sideways kinks during smooth flight
    to_drop: list[int] = []
    for t in range(1, n - 1):
        cur = refined[t]
        prev_b = refined[t - 1]
        next_b = refined[t + 1]
        if cur is None or prev_b is None or next_b is None:
            continue
        span = _center_distance(prev_b, next_b)
        if span > max_bridge_px or span < 8.0:
            continue
        mid_x = 0.5 * (prev_b.center.x + next_b.center.x)
        mid_y = 0.5 * (prev_b.center.y + next_b.center.y)
        dev = _dist(cur.center.x, cur.center.y, mid_x, mid_y)
        if dev <= kink_threshold_px:
            continue

        # Confirm the surrounding flight is roughly collinear (not a true bounce/hit)
        vx_span = next_b.center.x - prev_b.center.x
        vy_span = next_b.center.y - prev_b.center.y
        collinear = False
        if t >= 2 and refined[t - 2] is not None:
            b_prev2 = refined[t - 2]
            assert b_prev2 is not None
            vx_in = prev_b.center.x - b_prev2.center.x
            vy_in = prev_b.center.y - b_prev2.center.y
            n1 = (vx_in**2 + vy_in**2) ** 0.5
            n2 = (vx_span**2 + vy_span**2) ** 0.5
            if n1 > 3.0 and n2 > 3.0:
                cos_ang = (vx_in * vx_span + vy_in * vy_span) / (n1 * n2)
                if cos_ang > 0.8:
                    collinear = True
        if not collinear and t + 2 < n and refined[t + 2] is not None:
            b_next2 = refined[t + 2]
            assert b_next2 is not None
            vx_out = b_next2.center.x - next_b.center.x
            vy_out = b_next2.center.y - next_b.center.y
            n1 = (vx_span**2 + vy_span**2) ** 0.5
            n2 = (vx_out**2 + vy_out**2) ** 0.5
            if n1 > 3.0 and n2 > 3.0:
                cos_ang = (vx_span * vx_out + vy_span * vy_out) / (n1 * n2)
                if cos_ang > 0.8:
                    collinear = True
        if collinear:
            to_drop.append(t)

    for t in to_drop:
        refined[t] = None

    return refined

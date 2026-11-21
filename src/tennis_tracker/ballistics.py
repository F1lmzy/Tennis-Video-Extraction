"""Bounce- and hit-anchored ballistic 3D elevation correction for 2D court coordinates.

A planar court homography assumes every pixel lies on the ground plane
(``z = 0``). For an elevated broadcast camera at ``(Xc, Yc, Zc)``, an airborne
ball at height ``z > 0`` projects along the camera ray to:

    x_proj = Xc + (Zc / (Zc - z)) * (x - Xc)
    y_proj = Yc + (Zc / (Zc - z)) * (y - Yc)

At the far baseline even a normal 1.0-1.5 m strike height shifts ``y_proj`` by
several meters *behind* the far player.

This module recovers the 3D camera position from ground-plane court keypoints
via ``cv2.solvePnP`` and detects two kinds of physically grounded anchors:

  - **Bounces** (``z = 0``): a sharp drop in the ball's downward image velocity.
    The detector is adaptive to court depth (far-court bounces move only a few
    pixels per frame) and works across short tracking gaps, so it does not rely
    on a fixed pixel cusp threshold or on interpolated pixels that erase cusps.
  - **Player hits** (``y_true ~= y_player``): rally depth reversals within reach
    of the near or far player, with a physically plausible contact height.

Between consecutive valid anchors each flight arc is reconstructed as a
gravity-constrained parabola in ``z``. The ball is then placed on its camera ray
at that height. Because the arc is physical by construction, this avoids the
``z = 0`` plateaus, single-frame plunges, and homography-jitter amplification
produced by a linear ground-depth interpolation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import cv2
import numpy as np

from tennis_tracker.types import OutputRow

_DEFAULT_CAMERA_POS: tuple[float, float, float] = (0.0, -30.0, 10.0)


def estimate_camera_position(
    pixel_pts: Sequence[tuple[float, float]],
    court_pts: Sequence[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> Optional[tuple[float, float, float]]:
    """Estimate 3D camera position ``(Xc, Yc, Zc)`` in court meters via solvePnP.

    Sweeps plausible horizontal broadcast fields of view (22°–64°) using the
    ground-plane (``z = 0``) court keypoints and selects the lowest-reprojection
    pose where the camera is elevated above the court (``Zc > 2 m``) and behind
    one baseline (``|Yc| > 12 m``).
    """
    if len(pixel_pts) < 4 or len(pixel_pts) != len(court_pts):
        return None
    if image_width <= 0 or image_height <= 0:
        return None

    obj = np.array([[cx, cy, 0.0] for cx, cy in court_pts], dtype=np.float64)
    img = np.array(pixel_pts, dtype=np.float64)

    best: Optional[tuple[float, float, float, float]] = None
    for fov_deg in range(22, 66, 2):
        f_px = (image_width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
        camera_matrix = np.array(
            [
                [f_px, 0.0, image_width / 2.0],
                [0.0, f_px, image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        reproj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, None)
        err = float(np.mean(np.linalg.norm(reproj.reshape(-1, 2) - img, axis=1)))
        rot, _ = cv2.Rodrigues(rvec)
        cam = (-rot.T @ tvec).ravel()
        xc, yc, zc = float(cam[0]), float(cam[1]), float(cam[2])
        if zc > 2.0 and abs(yc) > 12.0:
            if best is None or err < best[0]:
                best = (err, xc, yc, zc)

    if best is None:
        return None
    return (best[1], best[2], best[3])


def _cam_at(
    camera_positions: Optional[Sequence[Optional[tuple[float, float, float]]]],
    idx: int,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the per-frame camera pose, falling back to *fallback* when absent."""
    if camera_positions is not None and 0 <= idx < len(camera_positions):
        cam = camera_positions[idx]
        if cam is not None:
            return cam
    return fallback


def _get_side_players(
    row: OutputRow,
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Return ``(near_player_xy, far_player_xy)`` classified by court Y sign."""
    near: Optional[tuple[float, float]] = None
    far: Optional[tuple[float, float]] = None
    for x, y in (
        (row.player_a_x_m, row.player_a_y_m),
        (row.player_b_x_m, row.player_b_y_m),
    ):
        if x is not None and y is not None:
            if y < 0.0:
                near = (x, y)
            else:
                far = (x, y)
    return near, far


def _nearest_side_player(
    rows: Sequence[OutputRow],
    idx: int,
    side: str,
    search_radius: int = 10,
) -> Optional[tuple[float, float]]:
    """Find the nearest frame's player position on the requested court half."""
    n = len(rows)
    for d in range(search_radius + 1):
        for j in (idx - d, idx + d) if d > 0 else (idx,):
            if 0 <= j < n:
                near, far = _get_side_players(rows[j])
                cand = far if side == "far" else near
                if cand is not None:
                    return cand
    return None


def _is_valid_flight_pair(kind0: str, y0: float, kind1: str, y1: float) -> bool:
    """Return True if ``(kind0, kind1)`` represents a single physical flight arc.

    Prevents bridging across a missed hit (such as ``far_bounce`` to
    ``near_bounce``, or ``near_hit`` straight to ``near_hit``), which would
    place the reconstructed ball in the wrong part of the court.
    """
    if kind0 == "near_hit" and kind1 in ("near_bounce", "far_bounce"):
        return y1 > y0 + (2.0 if kind1 == "near_bounce" else 0.0)
    if kind0 == "far_bounce" and kind1 == "far_hit":
        return y1 >= y0 - 1.0
    if kind0 == "far_hit" and kind1 in ("near_bounce", "far_bounce"):
        return y1 < y0 - (2.0 if kind1 == "far_bounce" else 0.0)
    if kind0 == "near_bounce" and kind1 == "near_hit":
        return y1 <= y0 + 1.0
    if kind0 == "near_hit" and kind1 == "far_hit":
        return y1 > y0 + 4.0
    if kind0 == "far_hit" and kind1 == "near_hit":
        return y1 < y0 - 4.0
    return False


def _image_contact_motion(
    rows: Sequence[OutputRow], i: int,
) -> Optional[tuple[float, float, float, float]]:
    """Estimate image velocity on either side of contact, skipping short gaps."""
    x, y = rows[i].ball_pixel_x, rows[i].ball_pixel_y
    if x is None or y is None:
        return None
    velocities = []
    for direction in (-1, 1):
        for offset in (2, 1, 3, 4):
            j = i + direction * offset
            if not 0 <= j < len(rows):
                continue
            other_x, other_y = rows[j].ball_pixel_x, rows[j].ball_pixel_y
            if other_x is not None and other_y is not None:
                velocities.extend(((other_x - x) / (j - i), (other_y - y) / (j - i)))
                break
        else:
            return None
    return velocities[0], velocities[1], velocities[2], velocities[3]


def _sustained_lateral_turn(rows: Sequence[OutputRow], i: int, vx_in: float, vx_out: float) -> bool:
    """Confirm direction on both sides without using the contact sample itself."""
    for direction, velocity in ((-1, vx_in), (1, vx_out)):
        xs = [
            rows[j].ball_pixel_x
            for offset in range(1, 9)
            if 0 <= (j := i + direction * offset) < len(rows) and rows[j].ball_pixel_x is not None
        ]
        if len(xs) < 2 or any((x - xs[0]) * direction * velocity < -0.5 for x in xs[1:3]):
            return False
    return True


def _detect_hits(
    rows: Sequence[OutputRow],
    camera_pos: tuple[float, float, float],
    camera_positions: Optional[Sequence[Optional[tuple[float, float, float]]]],
    raw_anchors: dict[int, tuple[str, float, float, float]],
    *,
    max_hit_reach_m: float,
    max_hit_height_m: float,
) -> None:
    """Detect player-hit anchors (rally depth reversals) into *raw_anchors*."""
    n = len(rows)
    bx = [r.ball_x_m for r in rows]
    by = [r.ball_y_m for r in rows]

    def _try_hit(i: int, side: str, score: float) -> None:
        xc, yc, zc = _cam_at(camera_positions, i, camera_pos)
        player = _nearest_side_player(rows, i, side)
        if player is None or by[i] is None or bx[i] is None:
            return
        if by[i] <= player[1] or abs(by[i] - yc) <= 1e-6:
            return
        s_hit = (player[1] - yc) / (by[i] - yc)
        z_hit = zc * (1.0 - s_hit)
        x_hit = xc + s_hit * (bx[i] - xc)
        if 0.0 <= z_hit <= max_hit_height_m and abs(x_hit - player[0]) <= max_hit_reach_m:
            kind = "far_hit" if side == "far" else "near_hit"
            raw_anchors[i] = (kind, player[1], z_hit, score)

    # Depth reversals within a +-12 frame window (slow lobs linger near apex).
    for i in range(3, n - 3):
        if by[i] is None or bx[i] is None:
            continue
        win_vals = [
            (k, by[k])
            for k in range(max(0, i - 12), min(n, i + 13))
            if by[k] is not None
        ]
        if len(win_vals) < 5:
            continue
        left_vals = [v for k, v in win_vals if k < i]
        right_vals = [v for k, v in win_vals if k > i]
        if not left_vals or not right_vals:
            continue
        all_vals = [v for _, v in win_vals]
        motion = _image_contact_motion(rows, i)
        if motion is not None:
            vx_in, vy_in, vx_out, vy_out = motion
            dx, dy = abs(vx_out - vx_in), vy_out - vy_in
            if by[i] > 4.0:
                depth_peak = max(all_vals) - by[i] <= 0.8
                sideways_turn = (
                    vx_in * vx_out < 0 and dx >= 2.0
                    and _sustained_lateral_turn(rows, i, vx_in, vx_out)
                )
                # A serve can be struck on its way down, well after the toss apex.
                overhead = abs(vx_in) <= 2.5 and abs(vy_in) <= 8.0 and vy_out >= 14.0 and dy >= 12.0
                # A smooth apex can reverse depth without impact; only use it
                # when the velocity also changes abruptly.
                peak_impact = depth_peak and (dy >= 5.0 or (dx >= 3.5 and vx_in * vx_out > 0))
                if sideways_turn or peak_impact or overhead:
                    _try_hit(i, "far", dx + max(0.0, dy))
            if by[i] < -4.0 and vy_out <= -18.0 and dy <= -18.0:
                if dx >= 7.0 or -vy_out >= max(28.0, 1.8 * vy_in):
                    _try_hit(i, "near", abs(dy) + dx)

        if i not in raw_anchors and by[i] < -4.0 and by[i] <= min(all_vals):
            prom_l = max(left_vals) - by[i]
            prom_r = max(right_vals) - by[i]
            if prom_l >= 0.5 and prom_r >= 0.5:
                _try_hit(i, "near", prom_l + prom_r)

    # Gap-adjacent far hits: ball rises to the far player right before a tracking
    # gap (>=4 frames) and resumes heading back toward the net after the gap.
    for i in range(4, n - 5):
        if i in raw_anchors or by[i] is None or bx[i] is None:
            continue
        if by[i + 1] is not None:
            continue
        j = i + 1
        while j < n and by[j] is None:
            j += 1
        if j - (i + 1) < 4 or j >= n:
            continue
        prev_vals = [by[k] for k in range(max(0, i - 8), i) if by[k] is not None]
        post_vals = [by[k] for k in range(j, min(n, j + 8)) if by[k] is not None]
        if not prev_vals or not post_vals:
            continue
        if by[i] > 6.0 and by[i] > min(prev_vals) + 1.0 and min(post_vals) < by[i] - 2.0:
            _try_hit(i, "far", 1.0)


def _has_lateral_continuity(
    rows: Sequence[OutputRow], samples: tuple[int, int, int, int]
) -> bool:
    """Check that observed image X keeps its direction and approximate speed."""
    speeds = []
    for a, b in zip(samples, samples[1:]):
        x0, x1 = rows[a].ball_pixel_x, rows[b].ball_pixel_x
        if b - a > 2 or x0 is None or x1 is None:
            return False
        speeds.append((x1 - x0) / (b - a))
    incoming, *outgoing = speeds
    return abs(incoming) >= 1.0 and all(0.5 <= v / incoming <= 1.5 for v in outgoing)


def _detect_bounces(
    rows: Sequence[OutputRow],
    raw_anchors: dict[int, tuple[str, float, float, float]],
    *,
    hit_indices: Sequence[int],
) -> None:
    """Detect bounce anchors (``z = 0``) via an adaptive image-velocity kink.

    A bounce reverses the ball's vertical motion, producing a sharp drop in its
    downward image velocity ``d(pixel_y)/dt``. The threshold scales with the
    local ball speed so that small far-court cusps are caught while noise is
    rejected, and velocities are measured between *observed* frames so short
    tracking gaps (where the contact frame is missing) still register.
    """
    n = len(rows)
    bx = [r.ball_x_m for r in rows]
    by = [r.ball_y_m for r in rows]
    bpy = [r.ball_pixel_y for r in rows]

    observed = [
        i for i in range(n)
        if bpy[i] is not None and by[i] is not None and bx[i] is not None
    ]
    if len(observed) < 3:
        return

    # Per-frame downward image velocity assigned to the arrival frame.
    vel = np.full(n, np.nan)
    for a, b in zip(observed, observed[1:]):
        vel[b] = (bpy[b] - bpy[a]) / (b - a)

    next_obs = {a: b for a, b in zip(observed, observed[1:])}
    hit_set = set(hit_indices)

    for a, i, s in zip(observed, observed[1:], observed[2:]):
        if np.isnan(vel[i]) or np.isnan(vel[s]):
            continue
        if not (-13.0 <= by[i] <= 13.0 and -6.5 <= bx[i] <= 6.5):
            continue
        v_in = float(vel[i])
        v_out = float(vel[s])
        kink = v_in - v_out
        # In the far court, perspective can keep the ball's image rising all
        # the way into contact. The bounce still makes it rise much faster.
        if v_in <= 0.5:
            if not (by[i] > 0.0 and v_in >= -5.0 and v_out <= -5.0 and kink >= 5.0):
                continue
            motion = _image_contact_motion(rows, i)
            if motion is None or abs(motion[2] - motion[0]) > max(2.0, 0.55 * abs(motion[0])):
                continue
        elif kink < max(1.5, 0.55 * v_in):
            continue
        # Persistence: the departure velocity must stay well below the approach,
        # rejecting single-frame noise wiggles that immediately resume.
        s2 = next_obs.get(s)
        v_out2 = float(vel[s2]) if s2 is not None and not np.isnan(vel[s2]) else v_out
        if v_out2 > v_in - 0.6 * kink:
            continue
        kind = "far_bounce" if by[i] >= 0.0 else "near_bounce"
        close_hits = hit_set.intersection(range(i - 5, i + 6))
        if close_hits:
            hit = raw_anchors.get(i)
            # A near bounce can create a depth minimum that looks like a hit.
            # Only override a same-frame hit with clear ground-contact evidence.
            replace_hit = (
                kind == "near_bounce" and hit is not None and hit[0] == "near_hit"
                and by[i] - hit[1] > 3.0 and s2 is not None
                and _has_lateral_continuity(rows, (a, i, s, s2))
            )
            # A genuine bounce and the ensuing racket contact can be only a
            # few frames apart. Require separate observations of the rebound.
            separate_bounce = kink >= 4.0 and s2 is not None and all(
                i + 3 <= h and s2 < h and _is_valid_flight_pair(kind, by[i], *raw_anchors[h][:2])
                for h in close_hits
            )
            if separate_bounce:
                motion = _image_contact_motion(rows, i)
                separate_bounce = motion is not None and (
                    abs(motion[2] - motion[0]) <= max(2.5, 0.65 * abs(motion[0]))
                )
            if not replace_hit and not separate_bounce:
                continue
        raw_anchors[i] = (kind, by[i], 0.0, kink)


def _detect_anchors(
    rows: Sequence[OutputRow],
    camera_pos: tuple[float, float, float],
    *,
    camera_positions: Optional[Sequence[Optional[tuple[float, float, float]]]] = None,
    max_hit_reach_m: float = 4.0,
    max_hit_height_m: float = 3.5,
) -> dict[int, tuple[str, float, float]]:
    """Detect bounce (``z=0``) and player-hit (``y_true=y_player``) anchors.

    Returns a mapping ``frame_index -> (kind, y_true_m, z_est_m)`` where ``kind``
    is one of ``"far_bounce"``, ``"near_bounce"``, ``"far_hit"``, ``"near_hit"``.
    """
    # idx -> (kind, y_true, z_est, score)
    raw_anchors: dict[int, tuple[str, float, float, float]] = {}

    _detect_hits(
        rows, camera_pos, camera_positions, raw_anchors,
        max_hit_reach_m=max_hit_reach_m, max_hit_height_m=max_hit_height_m,
    )
    hit_indices = [i for i, a in raw_anchors.items() if a[0].endswith("hit")]
    _detect_bounces(rows, raw_anchors, hit_indices=hit_indices)

    # Deduplicate same-kind anchors closer than 15 frames (keep highest score).
    anchors: dict[int, tuple[str, float, float]] = {}
    for idx in sorted(raw_anchors.keys()):
        kind, y_t, z_e, score = raw_anchors[idx]
        dup_prev = None
        for prev_idx in reversed(sorted(anchors.keys())):
            if idx - prev_idx > 15:
                break
            if anchors[prev_idx][0] == kind:
                dup_prev = prev_idx
                break
        if dup_prev is None:
            anchors[idx] = (kind, y_t, z_e)
        elif score > raw_anchors[dup_prev][3]:
            del anchors[dup_prev]
            anchors[idx] = (kind, y_t, z_e)

    return anchors


def _estimate_fps(rows: Sequence[OutputRow]) -> float:
    """Estimate frame rate from row timestamps (fallback 30)."""
    diffs = [
        rows[i + 1].time_s - rows[i].time_s
        for i in range(min(len(rows) - 1, 30))
        if rows[i + 1].time_s > rows[i].time_s
    ]
    if not diffs:
        return 30.0
    dt = float(np.median(diffs))
    return 1.0 / dt if dt > 1e-6 else 30.0


def correct_ball_elevation_with_heights(
    rows: Sequence[OutputRow],
    camera_pos: Optional[tuple[float, float, float]] = None,
    *,
    camera_positions: Optional[Sequence[Optional[tuple[float, float, float]]]] = None,
    anchor_rows: Optional[Sequence[OutputRow]] = None,
    max_segment_frames: int = 120,
    max_segment_gap: int = 18,
    max_ball_height_m: float = 7.5,
    gravity_m_s2: float = 9.81,
    min_segment_frames: int = 3,
) -> tuple[list[OutputRow], list[Optional[float]]]:
    """Correct 2D ball court coordinates and return per-frame heights ``z_m``.

    Each flight arc between physically valid consecutive anchors is modelled as
    a gravity-constrained parabola in ``z``; the ball is placed on its camera ray
    at that height, yielding ray-consistent ground footprints ``(x_true, y_true)``.
    Unreturned winners (no closing anchor) are left untouched.

    Parameters
    ----------
    anchor_rows :
        Optional raw (pre-interpolation) rows used for anchor detection. Pass the
        raw detections so bounce cusps erased by gap interpolation are recovered.
    camera_positions :
        Optional per-frame camera poses (e.g. to track broadcast pan/zoom). Falls
        back to *camera_pos* where absent.

    Returns ``(corrected_rows, ball_heights_m)``; does not mutate *rows*.
    """
    if not rows:
        return [], []

    fallback_cam = camera_pos if camera_pos is not None else _DEFAULT_CAMERA_POS
    det_rows = anchor_rows if anchor_rows is not None else rows
    anchors = _detect_anchors(det_rows, fallback_cam, camera_positions=camera_positions)

    present = [r.ball_y_m is not None for r in rows]
    if not anchors:
        heights = [0.0 if p else None for p in present]
        return list(rows), heights

    fps = _estimate_fps(rows)
    g = gravity_m_s2
    new_bx: list[Optional[float]] = [r.ball_x_m for r in rows]
    new_by: list[Optional[float]] = [r.ball_y_m for r in rows]
    ball_z: list[Optional[float]] = [0.0 if p else None for p in present]

    sorted_a = sorted(anchors.keys())

    def _set_height(t: int, zt: float) -> None:
        raw_x = rows[t].ball_x_m
        raw_y = rows[t].ball_y_m
        if raw_x is None or raw_y is None:
            return
        xc, yc, zc = _cam_at(camera_positions, t, fallback_cam)
        if zc <= 1.0:
            return
        zt = max(0.0, min(max_ball_height_m, zt))
        s = 1.0 - zt / zc
        new_bx[t] = xc + s * (raw_x - xc)
        new_by[t] = yc + s * (raw_y - yc)
        ball_z[t] = zt

    # 1. Reconstruct ballistic flight arcs between valid consecutive anchors.
    for idx_a in range(len(sorted_a) - 1):
        t0, t1 = sorted_a[idx_a], sorted_a[idx_a + 1]
        kind0, y0, z0 = anchors[t0]
        kind1, y1, z1 = anchors[t1]
        if not _is_valid_flight_pair(kind0, y0, kind1, y1):
            continue
        span = t1 - t0
        if span > max_segment_frames or span < min_segment_frames:
            continue
        # A hit->hit arc that crosses the whole court in a few frames is spurious.
        if kind0.endswith("hit") and kind1.endswith("hit") and span < 8:
            continue
        duration = span / fps
        if duration > 1e-6 and abs(y1 - y0) / duration > 60.0:
            continue
        max_miss, cur_miss = 0, 0
        for t in range(t0, t1 + 1):
            if rows[t].ball_y_m is None:
                cur_miss += 1
                max_miss = max(max_miss, cur_miss)
            else:
                cur_miss = 0
        if max_miss > max_segment_gap:
            continue

        # Parabola z(τ) = z0 + v0 τ - ½ g τ² satisfying z(T) = z1.
        v0 = (z1 - z0 + 0.5 * g * duration * duration) / duration
        for t in range(t0, t1 + 1):
            if rows[t].ball_y_m is None:
                continue
            tau = (t - t0) / fps
            _set_height(t, z0 + v0 * tau - 0.5 * g * tau * tau)

    # 2. Isolated serve tosses directly above the far server at segment start.
    for t_anc in sorted_a:
        kind, y_anc, z_anc = anchors[t_anc]
        if kind != "far_hit":
            continue
        _set_height(t_anc, z_anc)
        has_prev_anchor = any(
            0 < t_anc - prev <= max_segment_frames
            and _is_valid_flight_pair(anchors[prev][0], anchors[prev][1], kind, y_anc)
            for prev in sorted_a
            if prev < t_anc
        )
        if has_prev_anchor:
            continue
        far_p = _nearest_side_player(det_rows, t_anc, "far")
        if far_p is None:
            continue
        for t in range(t_anc - 1, max(-1, t_anc - 25), -1):
            raw_x, raw_y = rows[t].ball_x_m, rows[t].ball_y_m
            if raw_x is None or raw_y is None or raw_y <= far_p[1]:
                break
            xc, yc, zc = _cam_at(camera_positions, t, fallback_cam)
            if zc <= 1.0 or abs(raw_y - yc) < 1e-6:
                break
            s = max(0.0, min(1.0, (far_p[1] - yc) / (raw_y - yc)))
            x_corr = xc + s * (raw_x - xc)
            if abs(x_corr - far_p[0]) > 2.5:
                break
            new_bx[t] = x_corr
            new_by[t] = yc + s * (raw_y - yc)
            ball_z[t] = zc * (1.0 - s)

    corrected = [
        replace(row, ball_x_m=new_bx[i], ball_y_m=new_by[i])
        for i, row in enumerate(rows)
    ]
    return corrected, ball_z


def correct_ball_elevation(
    rows: Sequence[OutputRow],
    camera_pos: Optional[tuple[float, float, float]] = None,
    *,
    camera_positions: Optional[Sequence[Optional[tuple[float, float, float]]]] = None,
    anchor_rows: Optional[Sequence[OutputRow]] = None,
    max_segment_frames: int = 120,
    max_segment_gap: int = 18,
    max_ball_height_m: float = 7.5,
    gravity_m_s2: float = 9.81,
) -> list[OutputRow]:
    """Correct 2D ball court coordinates for airborne height parallax."""
    corrected, _ = correct_ball_elevation_with_heights(
        rows,
        camera_pos=camera_pos,
        camera_positions=camera_positions,
        anchor_rows=anchor_rows,
        max_segment_frames=max_segment_frames,
        max_segment_gap=max_segment_gap,
        max_ball_height_m=max_ball_height_m,
        gravity_m_s2=gravity_m_s2,
    )
    return corrected

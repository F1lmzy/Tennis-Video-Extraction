"""Hybrid bounce- and hit-anchored 3D elevation correction for 2D court coordinates.

A planar court homography assumes every pixel lies on the ground plane
(``z = 0``). For an elevated broadcast camera at ``(Xc, Yc, Zc)``, an airborne
ball at height ``z > 0`` projects along the camera ray to:

    x_proj = Xc + (Zc / (Zc - z)) * (x - Xc)
    y_proj = Yc + (Zc / (Zc - z)) * (y - Yc)

At the far baseline (where ``y - Yc`` is ~45 m), even a normal 1.0–1.5 m strike
height shifts ``y_proj`` by +5 to +8 meters *behind* the far player.

This module recovers the 3D camera position from ground-plane court keypoints
via ``cv2.solvePnP``, detects:
  - **Bounces** (sharp upward cusps in image ``ball_pixel_y`` inside the court,
    where ``z = 0`` so raw homography is exact), and
  - **Player hits** (rally Y-direction reversals within reach of the near or far
    player, where ``y_true ≈ y_player`` and ``z_hit`` is physically plausible),

and reconstructs ray-consistent ground footprints ``(x_true, y_true)`` between
consecutive anchors while leaving unreturned winners untouched.
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

    Prevents linearly bridging across a missed hit (such as ``far_bounce`` to
    ``near_bounce``, or ``near_hit`` straight to ``near_hit``), which would
    place the reconstructed ball in the wrong part of the court.
    """
    if kind0 == "near_hit" and kind1 == "far_bounce":
        return y1 > y0
    if kind0 == "far_bounce" and kind1 == "far_hit":
        return y1 >= y0 - 1.0
    if kind0 == "far_hit" and kind1 == "near_bounce":
        return y1 < y0
    if kind0 == "near_bounce" and kind1 == "near_hit":
        return y1 <= y0 + 1.0
    if kind0 == "near_hit" and kind1 == "far_hit":
        return y1 > y0 + 4.0
    if kind0 == "far_hit" and kind1 == "near_hit":
        return y1 < y0 - 4.0
    return False


def _detect_anchors(
    rows: Sequence[OutputRow],
    camera_pos: tuple[float, float, float],
    *,
    max_hit_reach_m: float = 4.0,
    max_hit_height_m: float = 6.5,
) -> dict[int, tuple[str, float, float]]:
    """Detect bounce (``z=0``) and player-hit (``y_true=y_player``) anchors.

    Returns a mapping ``frame_index -> (kind, y_true_m, z_est_m)`` where
    ``kind`` is one of ``"far_bounce"``, ``"near_bounce"``, ``"far_hit"``,
    or ``"near_hit"``.
    """
    xc, yc, zc = camera_pos
    n = len(rows)
    bx = [r.ball_x_m for r in rows]
    by = [r.ball_y_m for r in rows]
    bpy = [r.ball_pixel_y for r in rows]

    # idx -> (kind, y_true, z_est, score)
    raw_anchors: dict[int, tuple[str, float, float, float]] = {}

    # 1. Detect bounces: sharp upward reversal in image Y (ball_pixel_y local max)
    for i in range(2, n - 2):
        if bpy[i] is None or by[i] is None or bx[i] is None:
            continue
        if bpy[i - 1] is None or bpy[i + 1] is None:
            continue
        assert by[i] is not None and bx[i] is not None
        assert bpy[i] is not None and bpy[i - 1] is not None and bpy[i + 1] is not None
        if not (-13.0 <= by[i] <= 13.0 and -6.5 <= bx[i] <= 6.5):
            continue
        dv_in = bpy[i] - bpy[i - 1]
        dv_out = bpy[i + 1] - bpy[i]
        cusp = dv_in - dv_out
        if dv_in >= -0.5 and dv_out <= -2.5 and cusp >= 5.0:
            win = [bpy[k] for k in range(i - 2, i + 3) if bpy[k] is not None]
            if win and bpy[i] == max(win):
                b_kind = "far_bounce" if by[i] >= 0.0 else "near_bounce"
                raw_anchors[i] = (b_kind, by[i], 0.0, cusp)

    # 2. Detect player hits: Y-direction reversals within reach of near/far player.
    #    Use a +-12 frame window so slow lobs that linger near their apex still
    #    satisfy the reversal prominence threshold.
    for i in range(3, n - 3):
        if by[i] is None or bx[i] is None:
            continue
        assert by[i] is not None and bx[i] is not None
        win_vals = [
            (k, by[k])
            for k in range(max(0, i - 12), min(n, i + 13))
            if by[k] is not None
        ]
        if len(win_vals) < 5:
            continue
        left_vals = [v for k, v in win_vals if k < i and v is not None]
        right_vals = [v for k, v in win_vals if k > i and v is not None]
        if not left_vals or not right_vals:
            continue

        all_vals = [v for _, v in win_vals if v is not None]

        # Far-side return (local maximum in y_proj)
        if by[i] > 4.0 and by[i] >= max(all_vals):
            prom_l = by[i] - min(left_vals)
            prom_r = by[i] - min(right_vals)
            if prom_l >= 0.5 and prom_r >= 0.5:
                far_p = _nearest_side_player(rows, i, "far")
                if far_p is not None and by[i] > far_p[1] and abs(by[i] - yc) > 1e-6:
                    s_hit = (far_p[1] - yc) / (by[i] - yc)
                    z_hit = zc * (1.0 - s_hit)
                    x_hit = xc + s_hit * (bx[i] - xc)
                    if (
                        0.0 <= z_hit <= max_hit_height_m
                        and abs(x_hit - far_p[0]) <= max_hit_reach_m
                    ):
                        raw_anchors[i] = ("far_hit", far_p[1], z_hit, prom_l + prom_r)

        # Near-side return (local minimum in y_proj)
        if by[i] < -4.0 and by[i] <= min(all_vals):
            prom_l = max(left_vals) - by[i]
            prom_r = max(right_vals) - by[i]
            if prom_l >= 0.5 and prom_r >= 0.5:
                near_p = _nearest_side_player(rows, i, "near")
                if near_p is not None and by[i] > near_p[1] and abs(by[i] - yc) > 1e-6:
                    s_hit = (near_p[1] - yc) / (by[i] - yc)
                    z_hit = zc * (1.0 - s_hit)
                    x_hit = xc + s_hit * (bx[i] - xc)
                    if (
                        0.0 <= z_hit <= max_hit_height_m
                        and abs(x_hit - near_p[0]) <= max_hit_reach_m
                    ):
                        raw_anchors[i] = ("near_hit", near_p[1], z_hit, prom_l + prom_r)

    # 3. Gap-adjacent far/near hit detection: if the ball rises to the far player
    #    right before a tracking gap (>=4 frames) and resumes heading back toward
    #    the net after the gap, anchor the contact right at the player.
    for i in range(4, n - 5):
        if i in raw_anchors or by[i] is None or bx[i] is None:
            continue
        if by[i + 1] is not None:
            continue
        # Measure gap length after i
        j = i + 1
        while j < n and by[j] is None:
            j += 1
        if j - (i + 1) < 4 or j >= n:
            continue
        prev_vals = [by[k] for k in range(max(0, i - 8), i) if by[k] is not None]
        post_vals = [by[k] for k in range(j, min(n, j + 8)) if by[k] is not None]
        if not prev_vals or not post_vals:
            continue
        assert by[i] is not None and bx[i] is not None
        if by[i] > 6.0 and by[i] > min(prev_vals) + 1.0 and min(post_vals) < by[i] - 2.0:
            far_p = _nearest_side_player(rows, i, "far")
            if far_p is not None and by[i] > far_p[1] and abs(by[i] - yc) > 1e-6:
                s_hit = (far_p[1] - yc) / (by[i] - yc)
                z_hit = zc * (1.0 - s_hit)
                x_hit = xc + s_hit * (bx[i] - xc)
                if 0.0 <= z_hit <= max_hit_height_m and abs(x_hit - far_p[0]) <= max_hit_reach_m:
                    raw_anchors[i] = ("far_hit", far_p[1], z_hit, 1.0)

    # 4. Deduplicate same-kind anchors closer than 15 frames (keep highest score)
    anchors: dict[int, tuple[str, float, float]] = {}
    for idx in sorted(raw_anchors.keys()):
        kind, y_t, z_e, score = raw_anchors[idx]
        # Check if a recent anchor of the same kind already exists
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


def correct_ball_elevation_with_heights(
    rows: Sequence[OutputRow],
    camera_pos: Optional[tuple[float, float, float]] = None,
    *,
    max_segment_frames: int = 120,
    max_segment_gap: int = 18,
    max_ball_height_m: float = 7.5,
) -> tuple[list[OutputRow], list[Optional[float]]]:
    """Correct 2D ball court coordinates and return estimated per-frame heights ``z_m``.

    Projects each ball observation along its 3D camera ray onto the estimated
    ground-plane depth interpolated between physically valid consecutive anchor
    pairs (``hit → bounce``, ``bounce → hit``, or cross-court ``hit → hit``),
    clamping implied elevation ``z`` to ``[0, max_ball_height_m]``.

    Returns ``(corrected_rows, ball_heights_m)`` where ``ball_heights_m[i]`` is
    the estimated height above the court in meters (``None`` when ball is missing).
    Does not mutate *rows*.
    """
    if not rows:
        return [], []

    cam = camera_pos if camera_pos is not None else _DEFAULT_CAMERA_POS
    xc, yc, zc = cam
    if zc <= 1.0:
        heights = [0.0 if r.ball_y_m is not None else None for r in rows]
        return list(rows), heights

    anchors = _detect_anchors(rows, cam)
    if not anchors:
        heights = [0.0 if r.ball_y_m is not None else None for r in rows]
        return list(rows), heights

    new_bx: list[Optional[float]] = [r.ball_x_m for r in rows]
    new_by: list[Optional[float]] = [r.ball_y_m for r in rows]
    ball_z: list[Optional[float]] = [0.0 if r.ball_y_m is not None else None for r in rows]
    s_min = max(0.18, 1.0 - max_ball_height_m / zc)

    def _apply_scale(t: int, y_target: float) -> None:
        raw_x = rows[t].ball_x_m
        raw_y = rows[t].ball_y_m
        if raw_x is None or raw_y is None or abs(raw_y - yc) < 1e-6:
            return
        s = (y_target - yc) / (raw_y - yc)
        s = max(s_min, min(1.0, s))
        new_bx[t] = xc + s * (raw_x - xc)
        new_by[t] = yc + s * (raw_y - yc)
        ball_z[t] = zc * (1.0 - s)

    sorted_a = sorted(anchors.keys())

    # 1. Reconstruct flight arcs between physically valid consecutive anchors
    for idx_a in range(len(sorted_a) - 1):
        t0, t1 = sorted_a[idx_a], sorted_a[idx_a + 1]
        kind0, y0, _ = anchors[t0]
        kind1, y1, _ = anchors[t1]
        if not _is_valid_flight_pair(kind0, y0, kind1, y1):
            continue
        if t1 - t0 > max_segment_frames:
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

        span = max(1, t1 - t0)
        for t in range(t0, t1 + 1):
            frac = (t - t0) / span
            _apply_scale(t, y0 + frac * (y1 - y0))

    # 2. Handle isolated serve tosses directly above the far server at segment start
    for t_anc in sorted_a:
        kind, y_anc, _ = anchors[t_anc]
        if kind != "far_hit":
            continue
        _apply_scale(t_anc, y_anc)
        has_prev_anchor = any(
            0 < t_anc - prev <= max_segment_frames
            and _is_valid_flight_pair(anchors[prev][0], anchors[prev][1], kind, y_anc)
            for prev in sorted_a
            if prev < t_anc
        )
        if not has_prev_anchor:
            far_p = _nearest_side_player(rows, t_anc, "far")
            if far_p is not None:
                for t in range(t_anc - 1, max(-1, t_anc - 25), -1):
                    raw_x, raw_y = rows[t].ball_x_m, rows[t].ball_y_m
                    if raw_x is None or raw_y is None or raw_y <= far_p[1]:
                        break
                    s = max(s_min, min(1.0, (far_p[1] - yc) / (raw_y - yc)))
                    x_corr = xc + s * (raw_x - xc)
                    if abs(x_corr - far_p[0]) > 2.5:
                        break
                    new_bx[t] = x_corr
                    new_by[t] = yc + s * (raw_y - yc)
                    ball_z[t] = zc * (1.0 - s)

    # Light 3-frame smoothing on estimated height z (preserving exact 0 at bounces)
    smoothed_z = list(ball_z)
    for i in range(1, len(ball_z) - 1):
        if i in anchors and anchors[i][0].endswith("bounce"):
            smoothed_z[i] = 0.0
            continue
        if ball_z[i] is not None and ball_z[i - 1] is not None and ball_z[i + 1] is not None:
            smoothed_z[i] = (
                0.25 * ball_z[i - 1] + 0.5 * ball_z[i] + 0.25 * ball_z[i + 1]  # type: ignore[operator]
            )

    corrected = [
        replace(row, ball_x_m=new_bx[i], ball_y_m=new_by[i])
        for i, row in enumerate(rows)
    ]
    return corrected, smoothed_z


def correct_ball_elevation(
    rows: Sequence[OutputRow],
    camera_pos: Optional[tuple[float, float, float]] = None,
    *,
    max_segment_frames: int = 120,
    max_segment_gap: int = 18,
    max_ball_height_m: float = 7.5,
) -> list[OutputRow]:
    """Correct 2D ball court coordinates for airborne height parallax."""
    corrected, _ = correct_ball_elevation_with_heights(
        rows,
        camera_pos=camera_pos,
        max_segment_frames=max_segment_frames,
        max_segment_gap=max_segment_gap,
        max_ball_height_m=max_ball_height_m,
    )
    return corrected

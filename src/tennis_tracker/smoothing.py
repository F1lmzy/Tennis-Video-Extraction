"""Temporal smoothing and interpolation for per-frame track rows.

This module consumes raw per-object track rows (TrackRow or OutputRow) and
produces smoothed rows where short gaps are linearly interpolated and
long gaps remain missing.
"""

from __future__ import annotations

from typing import Optional, Sequence

from tennis_tracker.types import OutputRow


# Default interpolation gap: up to this many consecutive None values
# will be linearly interpolated.  Longer gaps stay missing.
_DEFAULT_MAX_INTERP_GAP: int = 5


def _interpolate_1d(values: list[Optional[float]], max_gap: int) -> list[Optional[float]]:
    """Linearly interpolate short gaps in a 1-D list of optional floats.

    Parameters
    ----------
    values :
        The raw float sequence where None marks missing observations.
    max_gap :
        Maximum number of consecutive None values that will be interpolated.
        Longer gaps remain None.

    Returns
    -------
    list[Optional[float]]
        Smoothed sequence with short gaps filled.
    """
    n = len(values)
    if n == 0:
        return []

    out: list[Optional[float]] = list(values)

    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue

        # Find the start and end of this None-run
        gap_start = i
        while i < n and out[i] is None:
            i += 1
        gap_end = i  # one past the last None

        gap_len = gap_end - gap_start

        # Do not interpolate gaps longer than max_gap
        if gap_len > max_gap:
            continue

        # Find the bounding known values
        left_val = None
        for j in range(gap_start - 1, -1, -1):
            if out[j] is not None:
                left_val = out[j]
                break

        right_val = None
        for j in range(gap_end, n):
            if out[j] is not None:
                right_val = out[j]
                break

        # Both sides must be known to interpolate
        if left_val is None or right_val is None:
            continue

        # Linear interpolation
        for k in range(gap_len):
            t = (k + 1) / (gap_len + 1)
            out[gap_start + k] = left_val + t * (right_val - left_val)

    return out


def smooth_output_rows(
    rows: Sequence[OutputRow],
    max_interp_gap: int = _DEFAULT_MAX_INTERP_GAP,
) -> list[OutputRow]:
    """Smooth a sequence of output rows by filling short missing-value gaps.

    The following fields are independently interpolated:
        - player_a_x_m, player_a_y_m
        - player_b_x_m, player_b_y_m
        - ball_x_m, ball_y_m

    Confidence fields are NOT interpolated — they propagate from the nearest
    known observation (backward fill) so interpolated positions retain a
    credible confidence signal.

    The ``diagnostics`` field is NOT interpolated; each row keeps its raw
    diagnostics.  Interpolated rows that were previously missing are
    **not** flagged — callers that need to distinguish observed vs.
    interpolated values should compare against the raw input.

    Parameters
    ----------
    rows :
        The raw (unsmoothed) output rows.  Not modified in place.
    max_interp_gap :
        Maximum number of consecutive missing values to interpolate.

    Returns
    -------
    list[OutputRow]
        New rows with interpolated short gaps.
    """
    if not rows:
        return []

    # Extract the coordinate fields to interpolate
    raw_a_x = [r.player_a_x_m for r in rows]
    raw_a_y = [r.player_a_y_m for r in rows]
    raw_b_x = [r.player_b_x_m for r in rows]
    raw_b_y = [r.player_b_y_m for r in rows]
    raw_ball_x = [r.ball_x_m for r in rows]
    raw_ball_y = [r.ball_y_m for r in rows]

    # Interpolate
    smooth_a_x = _interpolate_1d(raw_a_x, max_interp_gap)
    smooth_a_y = _interpolate_1d(raw_a_y, max_interp_gap)
    smooth_b_x = _interpolate_1d(raw_b_x, max_interp_gap)
    smooth_b_y = _interpolate_1d(raw_b_y, max_interp_gap)
    smooth_ball_x = _interpolate_1d(raw_ball_x, max_interp_gap)
    smooth_ball_y = _interpolate_1d(raw_ball_y, max_interp_gap)

    # Confidence propagation: backward fill from nearest known observation
    def _backward_fill(values: list[Optional[float]]) -> list[Optional[float]]:
        filled: list[Optional[float]] = list(values)
        last_known: Optional[float] = None
        for i in range(len(filled)):
            if filled[i] is not None:
                last_known = filled[i]
            elif last_known is not None:
                filled[i] = last_known
        return filled

    # Pre-compute backward-filled confidences once
    bfill_a_conf = _backward_fill([r.player_a_confidence for r in rows])
    bfill_b_conf = _backward_fill([r.player_b_confidence for r in rows])
    bfill_ball_conf = _backward_fill([r.ball_confidence for r in rows])

    # Build new rows without modifying originals
    result: list[OutputRow] = []
    for i, row in enumerate(rows):
        result.append(
            OutputRow(
                frame_index=row.frame_index,
                time_s=row.time_s,
                player_a_x_m=smooth_a_x[i],
                player_a_y_m=smooth_a_y[i],
                player_a_pixel_x=row.player_a_pixel_x,
                player_a_pixel_y=row.player_a_pixel_y,
                player_a_confidence=bfill_a_conf[i],
                player_b_x_m=smooth_b_x[i],
                player_b_y_m=smooth_b_y[i],
                player_b_pixel_x=row.player_b_pixel_x,
                player_b_pixel_y=row.player_b_pixel_y,
                player_b_confidence=bfill_b_conf[i],
                ball_x_m=smooth_ball_x[i],
                ball_y_m=smooth_ball_y[i],
                ball_pixel_x=row.ball_pixel_x,
                ball_pixel_y=row.ball_pixel_y,
                ball_confidence=bfill_ball_conf[i],
                court_confidence=row.court_confidence,
                homography_valid=row.homography_valid,
                diagnostics=row.diagnostics,
            )
        )

    return result

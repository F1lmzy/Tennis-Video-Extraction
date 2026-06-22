"""Process pipeline for end-to-end tennis tracking.

Provides two pipeline entry points:

- ``run_synthetic_pipeline``: accepts deterministic fixture data for
  test environments (no model weights required).
- ``run_process``: accepts real or injected model detectors and produces
  raw CSV, smoothed CSV, and annotated video.  Tests inject mock
  detectors; real usage provides model paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from tennis_tracker.coordinates import estimate_homography, project_pixel_to_court
from tennis_tracker.diagnostics import Diagnostics
from tennis_tracker.output import TrackingCsvWriter
from tennis_tracker.render import render_annotated_video
from tennis_tracker.smoothing import smooth_output_rows
from tennis_tracker.types import OutputRow, PixelPoint
from tennis_tracker.video import iter_frames, read_video_metadata, write_video


@dataclass
class SyntheticDetection:
    """Synthetic detection for a single object on a single frame."""

    pixel: PixelPoint
    confidence: float


@dataclass
class SyntheticFrameData:
    """Fixture data for one frame of the synthetic pipeline.

    Fields that are None represent a missing detection on that frame.
    """

    player_a: Optional[SyntheticDetection] = None
    player_b: Optional[SyntheticDetection] = None
    ball: Optional[SyntheticDetection] = None


@dataclass
class CourtKeypointMatch:
    """A matched court keypoint for homography calibration."""

    pixel: PixelPoint
    court_x_m: float
    court_y_m: float


def run_synthetic_pipeline(
    video_path: str | Path,
    raw_csv_path: str | Path,
    smoothed_csv_path: str | Path,
    video_output_path: str | Path,
    *,
    frame_data: list[SyntheticFrameData],
    court_keypoint_matches: list[CourtKeypointMatch],
    fps: float = 30.0,
    smoothing_max_gap: int = 5,
) -> dict:
    """Run the synthetic process pipeline without real YOLO models.

    This exists to prove end-to-end contracts (geometry, CSV, smoothing,
    video output) before model integration in Phase 6.

    Parameters
    ----------
    video_path:
        Path to a video file whose metadata drives timing.
    raw_csv_path:
        Destination for raw (unsmoothed) coordinate CSV.
    smoothed_csv_path:
        Destination for smoothed coordinate CSV.
    video_output_path:
        Destination for annotated MP4 video.
    frame_data:
        Per-frame synthetic detection fixtures, one element per
        frame to process.
    court_keypoint_matches:
        Matched pixel-to-court keypoints used to estimate homography
        for the entire clip (single static calibration).
    fps:
        Video framerate for CSV time calculation.
    smoothing_max_gap:
        Maximum consecutive missing frames to interpolate during
        smoothing.

    Returns
    -------
    dict with keys:
        ``raw_row_count``, ``smoothed_row_count``,
        ``homography_valid``, ``diagnostics_summary``.
    """
    video_path = Path(video_path)
    raw_csv_path = Path(raw_csv_path)
    smoothed_csv_path = Path(smoothed_csv_path)
    video_output_path = Path(video_output_path)

    # --- Estimate homography from the supplied keypoint matches ---
    homography_result = estimate_homography(
        [m.pixel for m in court_keypoint_matches],
        [(m.court_x_m, m.court_y_m) for m in court_keypoint_matches],
    )

    court_confidence: Optional[float] = (
        homography_result.confidence if homography_result.valid else None
    )

    # --- Build raw output rows ---
    raw_rows: list[OutputRow] = []

    for idx, fd in enumerate(frame_data):
        time_s = idx / fps

        ka = fd.player_a
        kb = fd.player_b
        kb2 = fd.ball

        pa_court = None
        pb_court = None
        ba_court = None

        if ka is not None and homography_result.valid and homography_result.matrix is not None:
            try:
                pa_court = project_pixel_to_court(
                    PixelPoint(x=ka.pixel.x, y=ka.pixel.y),
                    homography_result.matrix,
                    ka.confidence,
                )
            except ValueError:
                pass

        if kb is not None and homography_result.valid and homography_result.matrix is not None:
            try:
                pb_court = project_pixel_to_court(
                    PixelPoint(x=kb.pixel.x, y=kb.pixel.y),
                    homography_result.matrix,
                    kb.confidence,
                )
            except ValueError:
                pass

        if kb2 is not None and homography_result.valid and homography_result.matrix is not None:
            try:
                ba_court = project_pixel_to_court(
                    PixelPoint(x=kb2.pixel.x, y=kb2.pixel.y),
                    homography_result.matrix,
                    kb2.confidence,
                )
            except ValueError:
                pass

        # Build diagnostics
        diag = []
        if ka is None:
            diag.append("missing_player_a")
        if kb is None:
            diag.append("missing_player_b")
        if kb2 is None:
            diag.append("missing_ball")
        if not homography_result.valid:
            diag.append("invalid_homography")

        raw_rows.append(
            OutputRow(
                frame_index=idx,
                time_s=time_s,
                player_a_x_m=pa_court.x_m if pa_court else None,
                player_a_y_m=pa_court.y_m if pa_court else None,
                player_a_pixel_x=ka.pixel.x if ka else None,
                player_a_pixel_y=ka.pixel.y if ka else None,
                player_a_confidence=ka.confidence if ka else None,
                player_b_x_m=pb_court.x_m if pb_court else None,
                player_b_y_m=pb_court.y_m if pb_court else None,
                player_b_pixel_x=kb.pixel.x if kb else None,
                player_b_pixel_y=kb.pixel.y if kb else None,
                player_b_confidence=kb.confidence if kb else None,
                ball_x_m=ba_court.x_m if ba_court else None,
                ball_y_m=ba_court.y_m if ba_court else None,
                ball_pixel_x=kb2.pixel.x if kb2 else None,
                ball_pixel_y=kb2.pixel.y if kb2 else None,
                ball_confidence=kb2.confidence if kb2 else None,
                court_confidence=court_confidence,
                homography_valid=homography_result.valid,
                diagnostics=";".join(diag),
            )
        )

    # --- Write raw CSV ---
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OutputRow.csv_header().split(","))
        for row in raw_rows:
            writer.writerow(
                [
                    "" if v is None else "true" if v is True else "false" if v is False else v
                    for v in [
                        row.frame_index,
                        row.time_s,
                        row.player_a_x_m,
                        row.player_a_y_m,
                        row.player_a_pixel_x,
                        row.player_a_pixel_y,
                        row.player_a_confidence,
                        row.player_b_x_m,
                        row.player_b_y_m,
                        row.player_b_pixel_x,
                        row.player_b_pixel_y,
                        row.player_b_confidence,
                        row.ball_x_m,
                        row.ball_y_m,
                        row.ball_pixel_x,
                        row.ball_pixel_y,
                        row.ball_confidence,
                        row.court_confidence,
                        row.homography_valid,
                        row.diagnostics,
                    ]
                ]
            )

    # --- Smooth and write smoothed CSV ---
    smoothed_rows = smooth_output_rows(raw_rows, max_interp_gap=smoothing_max_gap)
    smoothed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(smoothed_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OutputRow.csv_header().split(","))
        for row in smoothed_rows:
            writer.writerow(
                [
                    "" if v is None else "true" if v is True else "false" if v is False else v
                    for v in [
                        row.frame_index,
                        row.time_s,
                        row.player_a_x_m,
                        row.player_a_y_m,
                        row.player_a_pixel_x,
                        row.player_a_pixel_y,
                        row.player_a_confidence,
                        row.player_b_x_m,
                        row.player_b_y_m,
                        row.player_b_pixel_x,
                        row.player_b_pixel_y,
                        row.player_b_confidence,
                        row.ball_x_m,
                        row.ball_y_m,
                        row.ball_pixel_x,
                        row.ball_pixel_y,
                        row.ball_confidence,
                        row.court_confidence,
                        row.homography_valid,
                        row.diagnostics,
                    ]
                ]
            )

    # --- Render minimal annotated video ---
    # For the synthetic pipeline we copy source frames, overlaying frame number
    # and detection indicators.  Full production rendering arrives in Task 15.
    annotated_frames: list[np.ndarray] = []
    for idx, frame_bgr in iter_frames(video_path, max_frames=len(frame_data)):
        overlay = frame_bgr.copy()
        # Frame number text
        cv2.putText(
            overlay,
            f"Frame {idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        fd = frame_data[idx] if idx < len(frame_data) else None
        if fd:
            # Mark player positions with circles
            if fd.player_a is not None:
                cv2.circle(
                    overlay,
                    (int(fd.player_a.pixel.x), int(fd.player_a.pixel.y)),
                    8,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    overlay,
                    "A",
                    (int(fd.player_a.pixel.x) + 10, int(fd.player_a.pixel.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                2,)
            if fd.player_b is not None:
                cv2.circle(
                    overlay,
                    (int(fd.player_b.pixel.x), int(fd.player_b.pixel.y)),
                    8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    overlay,
                    "B",
                    (int(fd.player_b.pixel.x) + 10, int(fd.player_b.pixel.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            if fd.ball is not None:
                cv2.circle(
                    overlay,
                    (int(fd.ball.pixel.x), int(fd.ball.pixel.y)),
                    4,
                    (0, 255, 255),
                    -1,
                )

        annotated_frames.append(overlay)

    write_video(annotated_frames, video_output_path, fps=fps)

    # --- Summary diagnostics ---
    raw_diags = [r.diagnostics for r in raw_rows if r.diagnostics]
    summary = {
        "raw_row_count": len(raw_rows),
        "smoothed_row_count": len(smoothed_rows),
        "homography_valid": homography_result.valid,
        "diagnostics_summary": ";".join(sorted(set(d for d in raw_diags for d in d.split(";"))))
        if raw_diags
        else "",
    }
    return summary


# ── Real model-based process pipeline ──────────────────────────────────


def run_process(
    video_path: str | Path,
    raw_csv_path: str | Path,
    smoothed_csv_path: str | Path,
    video_output_path: str | Path,
    *,
    player_detector=None,
    ball_detector=None,
    court_detector=None,
    player_model_path: Optional[str | Path] = None,
    ball_model_path: Optional[str | Path] = None,
    court_model_path: Optional[str | Path] = None,
    fps: Optional[float] = None,
    smoothing_max_gap: int = 5,
    homography_every_frame: bool = False,
    player_conf: float = 0.25,
    ball_conf: float = 0.05,
    court_conf: float = 0.25,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    ball_motion_threshold: float = 5.0,
    ball_max_jump_px: float = 180.0,
    ball_min_initial_displacement_px: float = 2.0,
) -> dict:
    """Run the real (or mocked) detection-based process pipeline.

    Accepts either pre-constructed detector objects (for testing) or model
    file paths (for real usage).  At least one of *detector / *model_path
    must be provided for each stage.

    Parameters
    ----------
    video_path:
        Path to the input video.
    raw_csv_path:
        Destination for raw coordinate CSV.
    smoothed_csv_path:
        Destination for smoothed coordinate CSV.
    video_output_path:
        Destination for annotated video.
    player_detector:
        Injected PlayerDetector instance (for tests).  If None, loads from
        *player_model_path*.
    ball_detector:
        Injected BallDetector instance (for tests).  If None, loads from
        *ball_model_path*.
    court_detector:
        Injected CourtKeypointDetector instance (for tests).  If None,
        loads from *court_model_path*.
    player_model_path:
        Path to player detection model weights.
    ball_model_path:
        Path to ball detection model weights.
    court_model_path:
        Path to court keypoint detection model weights.
    fps:
        Video FPS.  If None, reads from source video metadata.
    smoothing_max_gap:
        Max consecutive missing frames to interpolate during smoothing.
    homography_every_frame:
        If False (default), estimate homography once on the first frame
        with sufficient court keypoints.  If True, re-estimate per frame.

    Returns
    -------
    Summary dict with ``raw_row_count``, ``smoothed_row_count``,
    ``homography_valid``, and ``diagnostics_summary``.
    """
    # ── Late imports to avoid circular dependency at module level ──
    from tennis_tracker.tracking import (
        PlayerTracker,
        ball_center_distance,
        has_local_ball_motion,
        select_best_ball,
    )

    # ── Resolve video metadata ──
    if fps is None:
        try:
            meta = read_video_metadata(video_path)
            fps = meta.fps
        except (FileNotFoundError, ValueError):
            fps = 30.0

    # ── Resolve detectors ──
    if player_detector is None and player_model_path is not None:
        from tennis_tracker.detection import PlayerDetector
        player_detector = PlayerDetector(model_path=player_model_path)
    if ball_detector is None and ball_model_path is not None:
        from tennis_tracker.detection import BallDetector
        ball_detector = BallDetector(model_path=ball_model_path)
    if court_detector is None and court_model_path is not None:
        from tennis_tracker.detection import CourtKeypointDetector
        court_detector = CourtKeypointDetector(model_path=court_model_path)

    # ── State ──
    tracker = PlayerTracker()
    raw_rows: list[OutputRow] = []
    homography_matrix = None
    homography_valid = False
    court_confidence: Optional[float] = None
    first_frame_processed = False
    previous_frame_bgr: Optional[np.ndarray] = None
    previous_ball = None
    pending_ball = None

    # ── Process each frame ──
    for idx, (frame_index, frame_bgr) in enumerate(iter_frames(video_path)):
        time_s = frame_index / fps if fps and fps > 0 else 0.0

        # ── Court keypoint detection and homography ──
        if court_detector is not None and (
            not first_frame_processed or homography_every_frame
        ):
            court_keypoints = court_detector.predict(
                frame_bgr,
                conf_threshold=court_conf,
                imgsz=imgsz,
                device=device,
            )
            if len(court_keypoints) >= 4:
                from tennis_tracker.court import named_point_by_label

                pixel_pts: list[PixelPoint] = []
                court_pts: list[tuple[float, float]] = []
                for kp in court_keypoints:
                    named = named_point_by_label(kp.label)
                    if named is not None:
                        pixel_pts.append(kp.pixel)
                        court_pts.append((named.x_m, named.y_m))

                if len(pixel_pts) >= 4:
                    result = estimate_homography(pixel_pts, court_pts)
                    if result.valid:
                        homography_matrix = result.matrix
                        court_confidence = result.confidence
                        homography_valid = True

        first_frame_processed = True

        # ── Player detection and tracking ──
        player_dets: list = []
        if player_detector is not None:
            player_dets = player_detector.predict(
                frame_bgr,
                conf_threshold=player_conf,
                imgsz=imgsz,
                device=device,
            )

        # ── Ball detection ──
        ball_dets: list = []
        if ball_detector is not None:
            ball_dets = ball_detector.predict(
                frame_bgr,
                conf_threshold=ball_conf,
                imgsz=imgsz,
                device=device,
            )
        ball_candidate = select_best_ball(
            ball_dets,
            previous_frame=previous_frame_bgr,
            current_frame=frame_bgr,
            previous_ball=previous_ball,
            max_proximity_px=ball_max_jump_px,
            min_motion_score=ball_motion_threshold,
        ) if ball_dets else None

        best_ball = ball_candidate
        candidate_has_motion = ball_candidate is not None and has_local_ball_motion(
            ball_candidate,
            previous_frame_bgr,
            frame_bgr,
            min_motion_score=ball_motion_threshold,
        )
        if previous_ball is None:
            best_ball = None
            if ball_candidate is not None and candidate_has_motion:
                if pending_ball is not None:
                    initial_displacement = ball_center_distance(ball_candidate, pending_ball)
                    if (
                        ball_min_initial_displacement_px
                        <= initial_displacement
                        <= ball_max_jump_px
                    ):
                        best_ball = ball_candidate
                pending_ball = ball_candidate
            else:
                pending_ball = None

        # ── Track players ──
        tracked = tracker.update(player_dets, ball=best_ball)

        # ── Project to court coordinates ──
        def _project(det, use_bottom_center: bool = True) -> Optional[dict]:
            """Project a detection pixel to court meter coordinates."""
            if det is None or homography_matrix is None:
                return None
            pixel = det.bbox.bottom_center if use_bottom_center else det.bbox.center
            try:
                cp = project_pixel_to_court(pixel, homography_matrix, det.confidence)
                return {"x_m": cp.x_m, "y_m": cp.y_m, "conf": cp.confidence}
            except ValueError:
                return None

        pa_proj = _project(tracked.player_a)
        pb_proj = _project(tracked.player_b)
        ball_proj = _project(best_ball, use_bottom_center=False) if best_ball else None

        # ── Build diagnostics ──
        diag = Diagnostics()
        diag.merge(tracked.diagnostics)
        if best_ball is None:
            diag.add_flag("missing_ball")
        elif best_ball.confidence < 0.3:
            diag.add_flag("low_ball_confidence")
        if not homography_valid:
            diag.add_flag("homography_invalid")
        if court_confidence is not None and court_confidence < 0.5:
            diag.add_flag("low_court_confidence")

        # ── Build OutputRow ──
        pa = tracked.player_a
        pb = tracked.player_b

        row = OutputRow(
            frame_index=frame_index,
            time_s=time_s,
            player_a_x_m=pa_proj["x_m"] if pa_proj else None,
            player_a_y_m=pa_proj["y_m"] if pa_proj else None,
            player_a_pixel_x=pa.bbox.bottom_center.x if pa else None,
            player_a_pixel_y=pa.bbox.bottom_center.y if pa else None,
            player_a_confidence=pa.confidence if pa else None,
            player_b_x_m=pb_proj["x_m"] if pb_proj else None,
            player_b_y_m=pb_proj["y_m"] if pb_proj else None,
            player_b_pixel_x=pb.bbox.bottom_center.x if pb else None,
            player_b_pixel_y=pb.bbox.bottom_center.y if pb else None,
            player_b_confidence=pb.confidence if pb else None,
            ball_x_m=ball_proj["x_m"] if ball_proj else None,
            ball_y_m=ball_proj["y_m"] if ball_proj else None,
            ball_pixel_x=best_ball.center.x if best_ball else None,
            ball_pixel_y=best_ball.center.y if best_ball else None,
            ball_confidence=best_ball.confidence if best_ball else None,
            court_confidence=court_confidence,
            homography_valid=homography_valid,
            diagnostics=diag.to_string(),
        )
        raw_rows.append(row)
        if best_ball is not None and has_local_ball_motion(
            best_ball,
            previous_frame_bgr,
            frame_bgr,
            min_motion_score=ball_motion_threshold,
        ):
            previous_ball = best_ball
            pending_ball = None
        previous_frame_bgr = frame_bgr.copy()

    # ── Write raw CSV ──
    raw_csv_path = Path(raw_csv_path)
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with TrackingCsvWriter(raw_csv_path) as writer:
        writer.write_rows(raw_rows)

    # ── Smooth and write smoothed CSV ──
    smoothed_rows = smooth_output_rows(raw_rows, max_interp_gap=smoothing_max_gap)
    smoothed_csv_path = Path(smoothed_csv_path)
    smoothed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with TrackingCsvWriter(smoothed_csv_path) as writer:
        writer.write_rows(smoothed_rows)

    # ── Render annotated video using the Task 15 renderer ──
    if video_output_path:
        render_annotated_video(
            video_path,
            smoothed_rows,
            video_output_path,
            fps=fps,
        )

    # ── Summary diagnostics ──
    raw_diags = [r.diagnostics for r in raw_rows if r.diagnostics]
    summary = {
        "raw_row_count": len(raw_rows),
        "smoothed_row_count": len(smoothed_rows),
        "homography_valid": homography_valid,
        "diagnostics_summary": ";".join(sorted(set(
            d for row in raw_rows for d in row.diagnostics.split(";") if d
        )))
        if raw_diags
        else "",
    }
    return summary

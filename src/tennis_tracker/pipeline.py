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

from tennis_tracker.ballistics import (
    correct_ball_elevation_with_heights,
    estimate_camera_position,
)
from tennis_tracker.coordinates import estimate_homography, project_pixel_to_court
from tennis_tracker.court import DOUBLES_LENGTH, DOUBLES_WIDTH
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
    smoothing_max_gap: int = 10,
    homography_every_frame: bool = True,
    homography_ema_alpha: float = 0.35,
    player_conf: float = 0.25,
    ball_conf: float = 0.05,
    court_conf: float = 0.25,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    ball_motion_threshold: float = 2.0,
    ball_max_jump_px: float = 180.0,
    ball_min_initial_displacement_px: float = 2.0,
    ball_max_missing_frames: int = 5,
    flip_court_x: bool = False,
    flip_court_y: bool = False,
    court_x_scale: float = 1.0,
    court_y_scale: float = 1.0,
    player_court_margin_m: float = 1.5,
    player_baseline_margin_m: float = 5.0,
    ball_projection_anchor: str = "bottom_center",
    top_view_court: str = "doubles",
    correct_ball_parallax: bool = True,
    show_3d_view: bool = False,
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
        _ball_shape_penalty,
        _bbox_motion_score,
        _racket_or_player_penalty,
        ball_center_distance,
        filter_static_ball_hotspots,
        find_static_ball_hotspots,
        refine_ball_sequence,
    )

    # ── Resolve video metadata ──
    total_frames: Optional[int] = None
    try:
        meta = read_video_metadata(video_path)
        total_frames = meta.frame_count
        if fps is None:
            fps = meta.fps
    except (FileNotFoundError, ValueError):
        if fps is None:
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
    prev_homography_matrix = None
    camera_pos: Optional[tuple[float, float, float]] = None
    homography_valid = False
    court_confidence: Optional[float] = None
    first_frame_processed = False
    previous_frame_bgr: Optional[np.ndarray] = None
    # Per-frame buffers for hotspot-cleaned 2-pass ball tracking & refinement
    raw_ball_dets_per_frame: list[list] = []
    ball_motion_per_frame: list[dict] = []
    racket_boxes_per_frame: list[list] = []
    player_boxes_per_frame: list[list] = []
    homographies_per_frame: list[Optional[np.ndarray]] = []
    base_diagnostics_per_frame: list[Diagnostics] = []

    def _smooth_homography(
        prev_h: Optional[np.ndarray],
        new_h: np.ndarray,
        alpha: float,
    ) -> Optional[np.ndarray]:
        """Validate and EMA-smooth a per-frame court homography."""
        if abs(float(new_h[2, 2])) < 1e-12:
            return prev_h
        cand = new_h.astype(np.float64) / float(new_h[2, 2])
        if prev_h is None:
            return cand
        # Reject single-frame orientation flips (e.g. mirrored keypoints)
        prev_det = float(np.linalg.det(prev_h[:2, :2]))
        cand_det = float(np.linalg.det(cand[:2, :2]))
        if prev_det * cand_det <= 0:
            return prev_h
        # Reject implausible single-frame jumps in projected court center
        try:
            prev_inv = np.linalg.inv(prev_h)
            cand_inv = np.linalg.inv(cand)
        except np.linalg.LinAlgError:
            return prev_h
        p_prev = prev_inv @ np.array([0.0, 0.0, 1.0])
        p_cand = cand_inv @ np.array([0.0, 0.0, 1.0])
        if abs(p_prev[2]) > 1e-8 and abs(p_cand[2]) > 1e-8:
            jump_px = float(
                np.hypot(
                    p_prev[0] / p_prev[2] - p_cand[0] / p_cand[2],
                    p_prev[1] / p_prev[2] - p_cand[1] / p_cand[2],
                )
            )
            if jump_px > 120.0:
                return prev_h
        blended = alpha * cand + (1.0 - alpha) * prev_h
        if abs(float(blended[2, 2])) < 1e-12:
            return prev_h
        return blended / float(blended[2, 2])

    def _deduplicate_players(dets: list, min_foot_dist_px: float = 35.0) -> list:
        """Drop duplicate overlapping person boxes on the same player."""
        if len(dets) <= 1:
            return dets
        kept = []
        for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
            bc = det.bbox.bottom_center
            is_dup = False
            for existing in kept:
                ebc = existing.bbox.bottom_center
                dist = ((bc.x - ebc.x) ** 2 + (bc.y - ebc.y) ** 2) ** 0.5
                if dist < min_foot_dist_px:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        return kept

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
                    if result.valid and result.matrix is not None:
                        if result.reprojection_error_px <= 15.0 or homography_matrix is None:
                            updated_h = _smooth_homography(
                                homography_matrix,
                                result.matrix,
                                homography_ema_alpha,
                            )
                            if updated_h is not None:
                                homography_matrix = updated_h
                                court_confidence = result.confidence
                                homography_valid = True
                                if camera_pos is None:
                                    ground_px = [
                                        (kp.pixel.x, kp.pixel.y)
                                        for kp in court_keypoints
                                        if kp.label != "net_center"
                                        and named_point_by_label(kp.label) is not None
                                    ]
                                    ground_ct = [
                                        (named.x_m, named.y_m)
                                        for kp in court_keypoints
                                        if kp.label != "net_center"
                                        and (named := named_point_by_label(kp.label)) is not None
                                    ]
                                    fh, fw = frame_bgr.shape[:2]
                                    camera_pos = estimate_camera_position(
                                        ground_px, ground_ct, fw, fh
                                    )

        first_frame_processed = True

        # ── Camera-motion transform (current pixel -> previous pixel) ──
        camera_transform: Optional[np.ndarray] = None
        if prev_homography_matrix is not None and homography_matrix is not None:
            try:
                camera_transform = np.linalg.inv(prev_homography_matrix) @ homography_matrix
            except np.linalg.LinAlgError:
                camera_transform = None

        # ── Player detection and tracking ──
        player_dets: list = []
        racket_boxes: list = []
        if player_detector is not None:
            player_dets = player_detector.predict(
                frame_bgr,
                conf_threshold=player_conf,
                imgsz=imgsz,
                device=device,
            )
            racket_boxes = list(getattr(player_detector, "last_racket_boxes", []))
        player_boxes = [d.bbox for d in player_dets]

        # ── Ball detection (collect raw candidates + camera-compensated motion) ──
        ball_dets: list = []
        if ball_detector is not None:
            ball_dets = ball_detector.predict(
                frame_bgr,
                conf_threshold=ball_conf,
                imgsz=imgsz,
                device=device,
            )
        frame_motions = {}
        for det in ball_dets:
            if previous_frame_bgr is None:
                frame_motions[det] = float("inf")
            else:
                frame_motions[det] = _bbox_motion_score(
                    det,
                    previous_frame_bgr,
                    frame_bgr,
                    camera_transform=camera_transform,
                )

        # ── Drop person detections that project clearly outside the playable
        # court area before A/B assignment.  Use a deeper baseline margin than
        # sideline margin because players routinely return 2-5m behind the
        # baseline while ballkids/linesmen sit further back.
        deduped_player_dets = _deduplicate_players(player_dets)
        filtered_player_dets = deduped_player_dets
        if homography_matrix is not None:
            filtered_player_dets = []
            half_width = DOUBLES_WIDTH / 2.0 + player_court_margin_m
            half_length = DOUBLES_LENGTH / 2.0 + player_baseline_margin_m
            for det in deduped_player_dets:
                try:
                    court_point = project_pixel_to_court(
                        det.bbox.bottom_center,
                        homography_matrix,
                        det.confidence,
                    )
                except ValueError:
                    continue
                if (
                    -half_width <= court_point.x_m <= half_width
                    and -half_length <= court_point.y_m <= half_length
                ):
                    filtered_player_dets.append(det)

        # ── Track players ──
        tracked = tracker.update(filtered_player_dets, ball=None)

        # ── Project players to court coordinates ──
        def _project_with_h(
            det, h_mat: Optional[np.ndarray], anchor: str = "bottom_center"
        ) -> Optional[dict]:
            if det is None or h_mat is None:
                return None
            pixel = det.bbox.center if anchor == "center" else det.bbox.bottom_center
            try:
                cp = project_pixel_to_court(pixel, h_mat, det.confidence)
                x_m = -cp.x_m if flip_court_x else cp.x_m
                y_m = -cp.y_m if flip_court_y else cp.y_m
                return {
                    "x_m": x_m * court_x_scale,
                    "y_m": y_m * court_y_scale,
                    "conf": cp.confidence,
                }
            except ValueError:
                return None

        pa_proj = _project_with_h(tracked.player_a, homography_matrix)
        pb_proj = _project_with_h(tracked.player_b, homography_matrix)

        # ── Base diagnostics (ball diagnostics added after 2-pass ball tracking) ──
        diag = Diagnostics()
        diag.merge(tracked.diagnostics)
        if not homography_valid:
            diag.add_flag("homography_invalid")
        if court_confidence is not None and court_confidence < 0.5:
            diag.add_flag("low_court_confidence")

        # ── Build OutputRow (ball populated after hotspot-cleaned tracking) ──
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
            ball_x_m=None,
            ball_y_m=None,
            ball_pixel_x=None,
            ball_pixel_y=None,
            ball_confidence=None,
            court_confidence=court_confidence,
            homography_valid=homography_valid,
            diagnostics=diag.to_string(),
        )
        raw_rows.append(row)
        base_diagnostics_per_frame.append(diag)
        raw_ball_dets_per_frame.append(ball_dets)
        ball_motion_per_frame.append(frame_motions)
        racket_boxes_per_frame.append(racket_boxes)
        player_boxes_per_frame.append(player_boxes)
        homographies_per_frame.append(
            homography_matrix.copy() if homography_matrix is not None else None
        )

        prev_homography_matrix = (
            homography_matrix.copy() if homography_matrix is not None else None
        )
        previous_frame_bgr = frame_bgr.copy()
        if (idx + 1) % 100 == 0:
            suffix = f"/{total_frames}" if total_frames else ""
            print(f"[process] Processed {idx + 1}{suffix} frames...", flush=True)

    # ── Pass 2: Static-hotspot masking + velocity-predicted ball tracking ──
    static_hotspots = find_static_ball_hotspots(raw_ball_dets_per_frame)
    selected_balls_per_frame: list = []
    ball_candidates_per_frame: list[list] = []
    previous_ball = None
    previous_ball_vel: Optional[tuple[float, float]] = None
    pending_ball = None
    pending_ball_idx: Optional[int] = None
    ball_missing_frames = 0

    for i, raw_dets in enumerate(raw_ball_dets_per_frame):
        cleaned_dets = filter_static_ball_hotspots(raw_dets, static_hotspots)
        motions = ball_motion_per_frame[i]
        r_boxes = racket_boxes_per_frame[i]
        p_boxes = player_boxes_per_frame[i]

        # On active tracks, allow a softer motion floor (0.5x) for high-confidence
        # detections (preserves slow apexes on high lobs and serve tosses while
        # still rejecting static zero-motion artifacts).
        moving_dets = [
            det
            for det in cleaned_dets
            if (
                motions.get(det, 0.0) >= ball_motion_threshold
                or (
                    previous_ball is not None
                    and det.confidence >= 0.45
                    and motions.get(det, 0.0) >= min(ball_motion_threshold * 0.4, 0.8)
                )
            )
        ]
        ball_candidates_per_frame.append(moving_dets)

        effective_max_jump = ball_max_jump_px * (
            1.0 + 0.5 * min(ball_missing_frames, ball_max_missing_frames)
        )
        predicted_center: Optional[PixelPoint] = None
        if previous_ball is not None and previous_ball_vel is not None:
            dt = ball_missing_frames + 1
            predicted_center = PixelPoint(
                x=previous_ball.center.x + previous_ball_vel[0] * dt,
                y=previous_ball.center.y + previous_ball_vel[1] * dt,
            )

        if previous_ball is not None:
            cands = [
                det
                for det in moving_dets
                if (
                    ball_center_distance(det, previous_ball) <= effective_max_jump
                    or (
                        predicted_center is not None
                        and (
                            (det.center.x - predicted_center.x) ** 2
                            + (det.center.y - predicted_center.y) ** 2
                        )
                        ** 0.5
                        <= effective_max_jump
                    )
                )
            ]
        else:
            cands = moving_dets

        anchor_pt = (
            predicted_center
            if predicted_center is not None
            else (previous_ball.center if previous_ball is not None else None)
        )

        def _score_cand(
            det,
            _motions=motions,
            _r=r_boxes,
            _p=p_boxes,
            _anc=anchor_pt,
            _lim=effective_max_jump,
        ):
            m_val = _motions.get(det, 0.0)
            m_norm = 1.0 if m_val == float("inf") else min(m_val / 25.0, 1.0)
            total = det.confidence + 0.35 * m_norm
            total -= 0.35 * _ball_shape_penalty(det)
            total -= _racket_or_player_penalty(det, _r, _p)
            if _anc is not None:
                dist = ((det.center.x - _anc.x) ** 2 + (det.center.y - _anc.y) ** 2) ** 0.5
                total += 0.45 * max(0.0, 1.0 - dist / _lim)
            return total

        ball_candidate = max(cands, key=_score_cand) if cands else None
        best_ball = ball_candidate

        if previous_ball is None:
            best_ball = None
            if ball_candidate is not None:
                if pending_ball is not None:
                    initial_disp = ball_center_distance(ball_candidate, pending_ball)
                    if ball_min_initial_displacement_px <= initial_disp <= ball_max_jump_px:
                        best_ball = ball_candidate
                        # Backfill mid-video re-acquisition frame (keep frame 0 empty)
                        if pending_ball_idx is not None and pending_ball_idx > 0:
                            selected_balls_per_frame[pending_ball_idx] = pending_ball
                pending_ball = ball_candidate
                pending_ball_idx = i
            else:
                pending_ball = None
                pending_ball_idx = None

        selected_balls_per_frame.append(best_ball)

        if best_ball is not None:
            if previous_ball is not None:
                dt = max(1, ball_missing_frames + 1)
                previous_ball_vel = (
                    (best_ball.center.x - previous_ball.center.x) / dt,
                    (best_ball.center.y - previous_ball.center.y) / dt,
                )
            elif (
                pending_ball_idx is not None
                and pending_ball_idx > 0
                and selected_balls_per_frame[pending_ball_idx] is not None
            ):
                pb_prev = selected_balls_per_frame[pending_ball_idx]
                previous_ball_vel = (
                    best_ball.center.x - pb_prev.center.x,
                    best_ball.center.y - pb_prev.center.y,
                )
            else:
                previous_ball_vel = None
            previous_ball = best_ball
            pending_ball = None
            pending_ball_idx = None
            ball_missing_frames = 0
        elif previous_ball is not None:
            ball_missing_frames += 1
            if ball_missing_frames >= ball_max_missing_frames:
                previous_ball = None
                previous_ball_vel = None
                pending_ball = None
                pending_ball_idx = None
                ball_missing_frames = 0

    # ── Bidirectional ball trajectory refinement (kills racket detours) ──
    refined_balls = refine_ball_sequence(
        selected_balls_per_frame,
        ball_candidates_per_frame,
        racket_boxes_per_frame=racket_boxes_per_frame,
        player_boxes_per_frame=player_boxes_per_frame,
        max_bridge_px=ball_max_jump_px,
    )
    for i, ref_b in enumerate(refined_balls):
        row = raw_rows[i]
        diag = base_diagnostics_per_frame[i]
        if ref_b is None:
            diag.add_flag("missing_ball")
        else:
            if ref_b.confidence < 0.3:
                diag.add_flag("low_ball_confidence")
            row.ball_pixel_x = ref_b.center.x
            row.ball_pixel_y = ref_b.center.y
            row.ball_confidence = ref_b.confidence
            h_mat = homographies_per_frame[i]
            if h_mat is not None:
                pixel = (
                    ref_b.bbox.center
                    if ball_projection_anchor == "center"
                    else ref_b.bbox.bottom_center
                )
                try:
                    cp = project_pixel_to_court(pixel, h_mat, ref_b.confidence)
                    x_m = -cp.x_m if flip_court_x else cp.x_m
                    y_m = -cp.y_m if flip_court_y else cp.y_m
                    row.ball_x_m = x_m * court_x_scale
                    row.ball_y_m = y_m * court_y_scale
                except ValueError:
                    row.ball_x_m = None
                    row.ball_y_m = None
        row.diagnostics = diag.to_string()

    # ── Write raw CSV ──
    raw_csv_path = Path(raw_csv_path)
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with TrackingCsvWriter(raw_csv_path) as writer:
        writer.write_rows(raw_rows)

    # ── Smooth, apply 3D ball elevation correction, and write smoothed CSV ──
    smoothed_rows = smooth_output_rows(raw_rows, max_interp_gap=smoothing_max_gap)
    ball_heights_m: Optional[list[Optional[float]]] = None
    if correct_ball_parallax and camera_pos is not None:
        smoothed_rows, ball_heights_m = correct_ball_elevation_with_heights(
            smoothed_rows, camera_pos=camera_pos
        )
    smoothed_csv_path = Path(smoothed_csv_path)
    smoothed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with TrackingCsvWriter(smoothed_csv_path) as writer:
        writer.write_rows(smoothed_rows)

    # ── Render annotated video using the Task 15 renderer ──
    if video_output_path:
        print(f"[process] Rendering annotated video ({len(smoothed_rows)} frames)...", flush=True)
        render_annotated_video(
            video_path,
            smoothed_rows,
            video_output_path,
            fps=fps,
            court_view=top_view_court,
            show_3d_view=show_3d_view,
            ball_heights_m=ball_heights_m,
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

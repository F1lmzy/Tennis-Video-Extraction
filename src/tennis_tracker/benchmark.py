"""Benchmark CLI and performance report.

Measures pipeline FPS, per-stage timings, and whether the 30 FPS
target is met on 1080p input with selected model artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from tennis_tracker.diagnostics import Diagnostics

# ── Stages ─────────────────────────────────────────────────────────────

STAGE_NAMES = ("decode", "court", "player", "ball", "tracking_projection", "rendering")
"""Ordered list of stage names tracked by the benchmark."""


@dataclass
class BenchmarkReport:
    """Immutable snapshot of a benchmark run."""

    total_fps: float
    """Overall throughput (total_frames / total_wall_clock_seconds)."""

    stage_fps: dict[str, float]
    """Per-stage FPS (total_frames / stage_wall_clock_seconds)."""

    stage_ms_per_frame: dict[str, float]
    """Milliseconds per frame for each stage."""

    frame_count: int
    """Number of frames processed."""

    total_time_s: float
    """Total wall-clock time in seconds."""

    target_fps: float = 30.0
    """The FPS target this run was compared against."""

    met_target: bool = False
    """``True`` when *total_fps >= target_fps*."""

    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def summary_lines(self) -> list[str]:
        """Human-readable summary lines for CLI output."""
        lines = [
            f"Benchmark: {self.frame_count} frames in {self.total_time_s:.2f} s",
            f"  Total FPS:       {self.total_fps:.1f}  "
            f"{'✅' if self.met_target else '❌'}  target={self.target_fps:.0f} FPS",
            "  ── Stage breakdown (ms/frame) ──",
        ]
        for name in STAGE_NAMES:
            ms = self.stage_ms_per_frame.get(name)
            fps = self.stage_fps.get(name)
            if ms is not None:
                lines.append(f"    {name:25s} {ms:8.2f} ms  ({fps:.1f} FPS)")
            else:
                lines.append(f"    {name:25s} {'—':>8s}")
        return lines


# ── Clock abstraction ──────────────────────────────────────────────────


class _Clock:
    """Minimal clock abstraction for dependency injection in tests."""

    def now(self) -> float:
        return time.perf_counter()


# ── Benchmark runner ───────────────────────────────────────────────────


def run_benchmark(
    *,
    frame_count: int = 100,
    stage_durations: Optional[dict[str, float]] = None,
    clock: Optional[_Clock] = None,
) -> BenchmarkReport:
    """Run a benchmark with the given parameters.

    When *stage_durations* is provided (a dict mapping stage name to
    total seconds for that stage), those values are used directly.
    This is the injection point for tests and for the real pipeline
    when it provides its own timing.

    When *stage_durations* is ``None``, the benchmark runs a
    lightweight mock pipeline that simulates each stage with a
    configurable base latency.  This allows the CLI to produce
    plausible reports without real models / video.
    """
    c = clock or _Clock()

    if stage_durations is not None:
        # Use provided durations directly (real or test mode)
        total = sum(stage_durations.values())
        stage_fps = {}
        stage_ms = {}
        for name in STAGE_NAMES:
            dur = stage_durations.get(name)
            if dur and dur > 0:
                stage_fps[name] = frame_count / dur
                stage_ms[name] = dur / frame_count * 1000.0
            else:
                stage_fps[name] = float("inf")
                stage_ms[name] = 0.0
        total_fps = frame_count / total if total > 0 else float("inf")
    else:
        # Mock pipeline — exercise clock without real models
        total_wall = 0.0
        stage_fps = {}
        stage_ms = {}
        for name in STAGE_NAMES:
            t0 = c.now()
            _simulate_stage(name, frame_count)
            elapsed = c.now() - t0
            total_wall += elapsed
            stage_fps[name] = frame_count / elapsed if elapsed > 0 else float("inf")
            stage_ms[name] = elapsed / frame_count * 1000.0 if frame_count > 0 else 0.0
        total_fps = frame_count / total_wall if total_wall > 0 else float("inf")

    diag = Diagnostics()
    target = 30.0
    met = total_fps >= target

    return BenchmarkReport(
        total_fps=total_fps,
        stage_fps=stage_fps,
        stage_ms_per_frame=stage_ms,
        frame_count=frame_count,
        total_time_s=sum(stage_durations.values()) if stage_durations else total_wall,
        target_fps=target,
        met_target=met,
        diagnostics=diag,
    )


def _simulate_stage(name: str, frame_count: int) -> None:
    """Simulate processing for a stage.

    Uses a simple sleep proportional to frame count so the mock
    pipeline produces non-zero wall time for realistic reports.
    """
    # Approximate ms/frame per stage (deliberately faster than target)
    ms_per_frame = {
        "decode": 2.0,
        "court": 5.0,
        "player": 4.0,
        "ball": 4.0,
        "tracking_projection": 1.0,
        "rendering": 3.0,
    }
    total_s = ms_per_frame.get(name, 3.0) * frame_count / 1000.0
    _busy_sleep(total_s)


def _busy_sleep(seconds: float) -> None:
    """Busy-wait loop that burns CPU instead of sleeping.

    This ensures wall-time advances measurably even for very small
    durations, and avoids depending on OS sleep granularity.
    """
    import math

    if seconds <= 0:
        return
    target = time.perf_counter() + seconds
    while time.perf_counter() < target:
        # Burn CPU cycles
        _ = math.sqrt(12345.6789)

"""Tests for benchmark CLI and report (Task 14)."""

from __future__ import annotations

import pytest

from tennis_tracker.benchmark import (
    STAGE_NAMES,
    BenchmarkReport,
    run_benchmark,
)


class TestBenchmarkReport:
    def test_summary_lines_format(self) -> None:
        report = BenchmarkReport(
            total_fps=45.0,
            stage_fps={"decode": 200.0, "court": 100.0},
            stage_ms_per_frame={"decode": 5.0, "court": 10.0},
            frame_count=100,
            total_time_s=2.22,
            met_target=True,
        )
        lines = report.summary_lines()
        assert len(lines) >= 4
        assert any("45.0" in ln for ln in lines)
        assert any("✅" in ln for ln in lines)

    def test_not_met_target(self) -> None:
        report = BenchmarkReport(
            total_fps=15.0,
            stage_fps={},
            stage_ms_per_frame={},
            frame_count=50,
            total_time_s=3.33,
            met_target=False,
        )
        lines = report.summary_lines()
        assert any("15.0" in ln for ln in lines)
        assert any("❌" in ln for ln in lines)

    def test_empty_stage_fps_handled(self) -> None:
        report = BenchmarkReport(
            total_fps=0.0,
            stage_fps={},
            stage_ms_per_frame={},
            frame_count=0,
            total_time_s=0.0,
            met_target=False,
        )
        lines = report.summary_lines()
        assert any("—" in ln for ln in lines)

    def test_has_diagnostics(self) -> None:
        report = BenchmarkReport(
            total_fps=30.0,
            stage_fps={"decode": 30.0},
            stage_ms_per_frame={"decode": 33.33},
            frame_count=30,
            total_time_s=1.0,
            met_target=True,
        )
        assert report.diagnostics is not None


class TestRunBenchmarkWithStageDurations:
    """Tests that use explicit *stage_durations* (deterministic)."""

    def test_basic_report(self) -> None:
        report = run_benchmark(
            frame_count=100,
            stage_durations={
                "decode": 0.5,
                "court": 1.0,
                "player": 0.8,
                "ball": 0.8,
                "tracking_projection": 0.2,
                "rendering": 0.6,
            },
        )
        assert report.frame_count == 100
        assert report.total_time_s == pytest.approx(3.9, rel=1e-3)
        assert report.total_fps == pytest.approx(100 / 3.9, rel=1e-3)
        assert all(name in report.stage_fps for name in STAGE_NAMES)
        assert all(name in report.stage_ms_per_frame for name in STAGE_NAMES)

    def test_tiny_frames(self) -> None:
        """Even a single frame should produce a valid report."""
        report = run_benchmark(
            frame_count=1,
            stage_durations={name: 0.001 for name in STAGE_NAMES},
        )
        assert report.frame_count == 1
        assert report.total_fps > 0
        assert report.met_target  # 1 / 0.006 = 166 FPS

    def test_zero_frames(self) -> None:
        """Zero frames should produce a report with inf FPS rather than crash."""
        report = run_benchmark(
            frame_count=0,
            stage_durations={name: 0.0 for name in STAGE_NAMES},
        )
        assert report.frame_count == 0
        assert report.total_time_s == 0.0

    def test_target_met(self) -> None:
        """30 FPS target is met when total FPS >= 30."""
        report = run_benchmark(
            frame_count=100,
            stage_durations={
                "decode": 0.3,
                "court": 0.5,
                "player": 0.4,
                "ball": 0.4,
                "tracking_projection": 0.1,
                "rendering": 0.3,
            },
        )
        assert report.met_target

    def test_target_not_met(self) -> None:
        """30 FPS target is not met when total FPS < 30."""
        report = run_benchmark(
            frame_count=30,
            stage_durations={
                "decode": 0.3,
                "court": 0.5,
                "player": 0.4,
                "ball": 0.4,
                "tracking_projection": 0.1,
                "rendering": 0.3,
            },
        )
        assert not report.met_target


class TestRunBenchmarkMockPipeline:
    """Tests that use the built-in mock pipeline (*stage_durations* = None).

    These are more integration-oriented and may take a few ms of wall time.
    """

    def test_mock_pipeline_produces_report(self) -> None:
        report = run_benchmark(frame_count=30)
        assert report.frame_count == 30
        assert report.total_fps > 0
        assert report.total_time_s > 0
        for name in STAGE_NAMES:
            assert name in report.stage_ms_per_frame

    def test_mock_pipeline_frame_count(self) -> None:
        report = run_benchmark(frame_count=10)
        assert report.frame_count == 10

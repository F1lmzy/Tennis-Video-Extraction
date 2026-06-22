"""Tests for ball and court training entry points.

These tests verify CLI argument parsing and training invocation
without running actual model training.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tennis_tracker.training import ball, court


class TestBallTrainingCLI:
    """Ball detection training CLI tests."""

    def test_help_does_not_import_yolo(self) -> None:
        """Running --help should not trigger Ultralytics import."""
        with patch.object(sys, "argv", ["ball", "--help"]):
            with pytest.raises(SystemExit) as exc:
                ball.build_parser().parse_args(["--help"])
        assert exc.value.code == 0

    def test_parse_required_args(self) -> None:
        """Required args are present and parsed correctly."""
        args = ball.build_parser().parse_args(
            [
                "--data",
                "data.yaml",
                "--output",
                "models/ball.pt",
            ]
        )
        assert args.data == "data.yaml"
        assert args.output == "models/ball.pt"
        assert args.base_model == "yolo26n.pt"  # default
        assert args.device == "cpu"  # default
        assert args.epochs == 100  # default
        assert args.imgsz == 640  # default

    def test_parse_all_args(self) -> None:
        """All arguments are parsed correctly."""
        args = ball.build_parser().parse_args(
            [
                "--data",
                "data.yaml",
                "--base-model",
                "yolo26m.pt",
                "--output",
                "models/ball.pt",
                "--device",
                "mps",
                "--epochs",
                "50",
                "--imgsz",
                "1280",
                "--batch",
                "32",
                "--patience",
                "10",
            ]
        )
        assert args.data == "data.yaml"
        assert args.base_model == "yolo26m.pt"
        assert args.output == "models/ball.pt"
        assert args.device == "mps"
        assert args.epochs == 50
        assert args.imgsz == 1280
        assert args.batch == 32
        assert args.patience == 10

    def test_run_training_calls_yolo_train(self) -> None:
        """run_training calls YOLO().train() with correct arguments."""
        mock_yolo = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = ball.build_parser().parse_args(
            [
                "--data",
                "/tmp/test/data.yaml",
                "--output",
                "/tmp/test/models/ball.pt",
                "--epochs",
                "10",
            ]
        )

        with patch("ultralytics.YOLO", mock_yolo):
            with patch("shutil.copy2"):
                ball.run_training(args)

        mock_yolo.assert_called_once_with("yolo26n.pt")
        call_kwargs = mock_model_instance.train.call_args[1]
        assert call_kwargs["data"] == "/tmp/test/data.yaml"
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["device"] == "cpu"

    def test_no_training_at_import(self) -> None:
        """Importing the module should not start training."""
        import importlib

        spec = importlib.util.find_spec("tennis_tracker.training.ball")
        assert spec is not None

    def test_main_function_exists(self) -> None:
        """The module has a main() entry point."""
        assert callable(ball.main)


class TestCourtTrainingCLI:
    """Court keypoint training CLI tests."""

    def test_help_does_not_import_yolo(self) -> None:
        """Running --help should not trigger Ultralytics import."""
        with pytest.raises(SystemExit) as exc:
            court.build_parser().parse_args(["--help"])
        assert exc.value.code == 0

    def test_parse_required_args(self) -> None:
        """Required args are present and parsed correctly."""
        args = court.build_parser().parse_args(
            [
                "--data",
                "data.yaml",
                "--output",
                "models/court.pt",
            ]
        )
        assert args.data == "data.yaml"
        assert args.output == "models/court.pt"
        assert args.base_model == "yolo26n-pose.pt"  # default
        assert args.device == "cpu"  # default
        assert args.epochs == 100  # default
        assert args.imgsz == 640  # default

    def test_parse_all_args(self) -> None:
        """All arguments including kobj are parsed correctly."""
        args = court.build_parser().parse_args(
            [
                "--data",
                "data.yaml",
                "--base-model",
                "yolo26n-pose.pt",
                "--output",
                "models/court.pt",
                "--device",
                "mps",
                "--epochs",
                "50",
                "--imgsz",
                "1280",
                "--batch",
                "8",
                "--patience",
                "5",
                "--kobj",
                "0.5",
            ]
        )
        assert args.data == "data.yaml"
        assert args.base_model == "yolo26n-pose.pt"
        assert args.output == "models/court.pt"
        assert args.device == "mps"
        assert args.epochs == 50
        assert args.imgsz == 1280
        assert args.batch == 8
        assert args.patience == 5
        assert args.kobj == 0.5

    def test_run_training_calls_yolo_train_with_kobj(self) -> None:
        """run_training passes kobj to YOLO().train() when provided."""
        mock_yolo = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = court.build_parser().parse_args(
            [
                "--data",
                "/tmp/test/data.yaml",
                "--output",
                "/tmp/test/models/court.pt",
                "--epochs",
                "10",
                "--kobj",
                "0.7",
            ]
        )

        with patch("ultralytics.YOLO", mock_yolo):
            with patch("shutil.copy2"):
                court.run_training(args)

        mock_yolo.assert_called_once_with("yolo26n-pose.pt")
        call_kwargs = mock_model_instance.train.call_args[1]
        assert call_kwargs["kobj"] == 0.7

    def test_run_training_without_kobj(self) -> None:
        """run_training does not pass kobj when not provided."""
        mock_yolo = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = court.build_parser().parse_args(
            [
                "--data",
                "/tmp/test/data.yaml",
                "--output",
                "/tmp/test/models/court.pt",
                "--epochs",
                "10",
            ]
        )

        with patch("ultralytics.YOLO", mock_yolo):
            with patch("shutil.copy2"):
                court.run_training(args)

        call_kwargs = mock_model_instance.train.call_args[1]
        assert "kobj" not in call_kwargs

    def test_no_training_at_import(self) -> None:
        """Importing the module should not start training."""
        import importlib

        spec = importlib.util.find_spec("tennis_tracker.training.court")
        assert spec is not None

    def test_main_function_exists(self) -> None:
        """The module has a main() entry point."""
        assert callable(court.main)


class TestModuleEntryPoints:
    """Verify python -m module_name --help works."""

    def test_ball_module_entry(self) -> None:
        """python -m tennis_tracker.training.ball --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tennis_tracker.training.ball", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train-ball" in result.stdout or "ball" in result.stdout.lower()

    def test_court_module_entry(self) -> None:
        """python -m tennis_tracker.training.court --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tennis_tracker.training.court", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train-court" in result.stdout or "court" in result.stdout.lower()


class TestTopLevelCLIHooks:
    """Verify that the top-level CLI stubs work with training args."""

    def test_train_ball_cli_help(self) -> None:
        """Top-level train-ball --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tennis_tracker", "train-ball", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_train_court_cli_help(self) -> None:
        """Top-level train-court --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tennis_tracker", "train-court", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

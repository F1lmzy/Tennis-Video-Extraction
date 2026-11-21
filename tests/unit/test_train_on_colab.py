"""Unit tests for scripts/train_on_colab.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_on_colab.py"
    spec = importlib.util.spec_from_file_location("train_on_colab", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestColabTrainingParser:
    def test_ball_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load_script_module()
        monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")

        args = module.build_parser().parse_args(["--task", "ball"])
        cfg = module.resolve_config(args, repo_root=tmp_path)

        assert cfg["task"] == "ball"
        assert cfg["module"] == "tennis_tracker.training.ball"
        assert cfg["gpu"] == "T4"
        assert cfg["session"] == "tennis-train-ball"
        assert cfg["dataset_source"] == "roboflow"
        assert cfg["output_path"] == Path("models/ball_yolo26.pt")
        assert cfg["device"] == "0"
        assert cfg["poll_interval"] == 15.0

    def test_court_defaults_with_local_dataset_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _load_script_module()
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        dataset_dir = tmp_path / "data" / "datasets" / "abiya-thesis-tennis-court-suuzy"
        dataset_dir.mkdir(parents=True)

        args = module.build_parser().parse_args(["--task", "court", "--kobj", "2.5"])
        cfg = module.resolve_config(args, repo_root=tmp_path)

        assert cfg["task"] == "court"
        assert cfg["module"] == "tennis_tracker.training.court"
        assert cfg["dataset_source"] == "upload"
        assert cfg["kobj"] == 2.5
        assert cfg["output_path"] == Path("models/court_keypoints_yolo26.pt")

    def test_missing_dataset_and_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _load_script_module()
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)

        args = module.build_parser().parse_args(["--task", "ball"])
        with pytest.raises(ValueError, match="Cannot resolve dataset source"):
            module.resolve_config(args, repo_root=tmp_path)


class TestColabArchiveAndScript:
    def test_create_repo_archive_excludes_pycache(self, tmp_path: Path) -> None:
        module = _load_script_module()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "src" / "tennis_tracker").mkdir(parents=True)
        (tmp_path / "src" / "tennis_tracker" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "src" / "tennis_tracker" / "__pycache__").mkdir()
        (tmp_path / "src" / "tennis_tracker" / "__pycache__" / "a.pyc").write_bytes(b"x")
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "prepare_roboflow_dataset.py").write_text("", encoding="utf-8")

        archive = module.create_repo_archive(tmp_path, tmp_path / "repo.tar.gz")
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()

        assert "pyproject.toml" in names
        assert "src/tennis_tracker/__init__.py" in names
        assert "scripts/prepare_roboflow_dataset.py" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    def test_build_remote_script_for_court_with_kobj(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _load_script_module()
        monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")

        args = module.build_parser().parse_args(
            ["--task", "court", "--epochs", "50", "--imgsz", "1280", "--kobj", "1.5"]
        )
        cfg = module.resolve_config(args, repo_root=tmp_path)
        script = module.build_remote_script(cfg)

        assert "tennis_tracker.training.court" in script
        assert "'--epochs', '50'" in script
        assert "'--imgsz', '1280'" in script
        assert "'--kobj', '1.5'" in script
        assert "'--device', '0'" in script
        assert "start_new_session=True" in script
        assert module.LAUNCH_MARKER in script
        compile(script, "<remote_script>", "exec")

    def test_parse_poll_output(self) -> None:
        module = _load_script_module()
        payload = {"offset": 42, "done": False, "exitcode": None, "chunk": "Epoch 1/10\n"}
        stdout = f"some banner\n{module.POLL_MARKER}{json.dumps(payload)}\n"

        parsed = module.parse_poll_output(stdout)

        assert parsed == payload

    def test_run_colab_training_retries_transient_poll_drop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _load_script_module()
        monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

        args = module.build_parser().parse_args(
            ["--task", "ball", "--colab-bin", "colab", "--poll-interval", "0"]
        )
        cfg = module.resolve_config(args, repo_root=tmp_path)

        poll_count = 0
        recorded_cmds: list[list[str]] = []

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            nonlocal poll_count
            recorded_cmds.append(cmd)
            if "exec" in cmd and capture_output:
                script_file = Path(cmd[-1])
                if script_file.name.startswith("remote_train_"):
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout=f"[remote] OK\n{module.LAUNCH_MARKER}\n", stderr=""
                    )
                poll_count += 1
                if poll_count == 1:
                    raise subprocess.CalledProcessError(1, cmd, stderr="transient ws drop")
                if poll_count == 2:
                    payload = {"offset": 12, "done": False, "exitcode": None, "chunk": "Epoch 1\n"}
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout=f"{module.POLL_MARKER}{json.dumps(payload)}\n", stderr=""
                    )
                payload = {"offset": 20, "done": True, "exitcode": 0, "chunk": "Done!\n"}
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{module.POLL_MARKER}{json.dumps(payload)}\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            with patch.object(module.time, "sleep"):
                module.run_colab_training(cfg, repo_root=tmp_path)

        out = capsys.readouterr().out
        assert "Epoch 1\n" in out
        assert "Done!\n" in out
        assert any("download" in cmd for cmd in recorded_cmds)
        assert any("stop" in cmd for cmd in recorded_cmds)

    def test_run_colab_training_stops_session_on_exec_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _load_script_module()
        monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

        args = module.build_parser().parse_args(["--task", "ball", "--colab-bin", "colab"])
        cfg = module.resolve_config(args, repo_root=tmp_path)

        recorded_cmds: list[list[str]] = []

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            recorded_cmds.append(cmd)
            if "exec" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            with pytest.raises(subprocess.CalledProcessError):
                module.run_colab_training(cfg, repo_root=tmp_path)

        assert any("new" in cmd for cmd in recorded_cmds)
        assert any("stop" in cmd for cmd in recorded_cmds)

    def test_run_colab_training_rescues_checkpoints_on_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = _load_script_module()
        monkeypatch.setenv("ROBOFLOW_API_KEY", "rf_test_key")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

        args = module.build_parser().parse_args(
            ["--task", "court", "--colab-bin", "colab", "--timeout", "1", "--poll-interval", "0"]
        )
        cfg = module.resolve_config(args, repo_root=tmp_path)

        recorded_cmds: list[list[str]] = []

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            recorded_cmds.append(cmd)
            if "exec" in cmd and capture_output:
                script_file = Path(cmd[-1])
                if script_file.name.startswith("remote_train_"):
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout=f"{module.LAUNCH_MARKER}\n", stderr=""
                    )
                payload = {"offset": 0, "done": False, "exitcode": None, "chunk": ""}
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{module.POLL_MARKER}{json.dumps(payload)}\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monotonic_values = iter([0.0, 2.0])
        with patch.object(module.subprocess, "run", side_effect=fake_run):
            with patch.object(module.time, "monotonic", side_effect=monotonic_values):
                with pytest.raises(TimeoutError):
                    module.run_colab_training(cfg, repo_root=tmp_path)

        download_cmds = [cmd for cmd in recorded_cmds if "download" in cmd]
        assert any("weights/best.pt" in " ".join(cmd) for cmd in download_cmds)
        assert any("weights/last.pt" in " ".join(cmd) for cmd in download_cmds)
        assert any("stop" in cmd for cmd in recorded_cmds)

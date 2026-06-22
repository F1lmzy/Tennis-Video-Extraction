from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_roboflow_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_roboflow_dataset", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "dataset": "ball",
        "workspace": None,
        "project": None,
        "version": None,
        "format": None,
        "output": None,
        "api_key": None,
        "env_file": Path(".env"),
        "overwrite": False,
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


class TestRoboflowEnvFile:
    def test_load_env_file_reads_api_key(self, tmp_path: Path) -> None:
        module = _load_script_module()
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\nROBOFLOW_API_KEY='from-env-file'\nOTHER=value\n",
            encoding="utf-8",
        )

        values = module._load_env_file(env_file)

        assert values["ROBOFLOW_API_KEY"] == "from-env-file"
        assert values["OTHER"] == "value"

    def test_missing_env_file_is_empty(self, tmp_path: Path) -> None:
        module = _load_script_module()

        assert module._load_env_file(tmp_path / "missing.env") == {}

    def test_resolve_args_uses_env_file_api_key(self, tmp_path: Path, monkeypatch) -> None:
        module = _load_script_module()
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("ROBOFLOW_API_KEY=from-env-file\n", encoding="utf-8")

        config = module._resolve_args(_args(env_file=env_file))

        assert config["api_key"] == "from-env-file"
        assert config["env_file"] == env_file

    def test_process_env_overrides_env_file(self, tmp_path: Path, monkeypatch) -> None:
        module = _load_script_module()
        env_file = tmp_path / ".env"
        env_file.write_text("ROBOFLOW_API_KEY=from-env-file\n", encoding="utf-8")
        monkeypatch.setenv("ROBOFLOW_API_KEY", "from-process-env")

        config = module._resolve_args(_args(env_file=env_file))

        assert config["api_key"] == "from-process-env"

    def test_cli_api_key_overrides_everything(self, tmp_path: Path, monkeypatch) -> None:
        module = _load_script_module()
        env_file = tmp_path / ".env"
        env_file.write_text("ROBOFLOW_API_KEY=from-env-file\n", encoding="utf-8")
        monkeypatch.setenv("ROBOFLOW_API_KEY", "from-process-env")

        config = module._resolve_args(_args(api_key="from-cli", env_file=env_file))

        assert config["api_key"] == "from-cli"

    def test_parser_exposes_env_file_option(self) -> None:
        module = _load_script_module()
        help_text = module._build_parser().format_help()

        assert "--env-file" in help_text
        assert "ROBOFLOW_API_KEY" in help_text

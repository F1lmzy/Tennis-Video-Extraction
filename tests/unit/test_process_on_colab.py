"""Unit tests for scripts/process_on_colab.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "process_on_colab.py"
    spec = importlib.util.spec_from_file_location("process_on_colab", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestProcessOnColab:
    def test_resolve_config_defaults_and_model_uploads(self, tmp_path: Path) -> None:
        module = _load_script_module()
        video = tmp_path / "data" / "input" / "match.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fake-mp4")

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "ball_yolo26.pt").write_bytes(b"ball")
        (tmp_path / "models" / "court_keypoints_yolo26.pt").write_bytes(b"court")

        args, extra = module.build_parser().parse_known_args(
            ["--input", str(video), "--imgsz", "1280", "--ball-conf", "0.1"]
        )
        cfg = module.resolve_config(args, extra_args=extra, repo_root=tmp_path)

        assert cfg["gpu"] == "T4"
        assert cfg["device"] == "cuda"
        assert cfg["remote_input_path"] == "/content/match.mp4"
        assert cfg["ball_model_remote"] == "/content/ball_yolo26.pt"
        assert cfg["court_model_remote"] == "/content/court_keypoints_yolo26.pt"
        assert cfg["player_model_remote"] == "yolo26n.pt"
        assert len(cfg["model_uploads"]) == 2
        assert cfg["extra_args"] == ["--imgsz", "1280", "--ball-conf", "0.1"]
        assert cfg["raw_output"] == Path("data/output/match_raw_coordinates.csv")
        assert cfg["smoothed_output"] == Path("data/output/match_smoothed_coordinates.csv")
        assert cfg["video_output"] == Path("data/output/match_annotated.mp4")

    def test_missing_input_video_raises(self, tmp_path: Path) -> None:
        module = _load_script_module()
        args, extra = module.build_parser().parse_known_args(["--input", "missing.mp4"])
        with pytest.raises(FileNotFoundError, match="Input video not found"):
            module.resolve_config(args, extra_args=extra, repo_root=tmp_path)

    def test_build_remote_script_compiles_and_forwards_extra_args(self, tmp_path: Path) -> None:
        module = _load_script_module()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"mp4")

        args, extra = module.build_parser().parse_known_args(
            ["--input", str(video), "--imgsz", "1280", "--flip-court-y"]
        )
        cfg = module.resolve_config(args, extra_args=extra, repo_root=tmp_path)
        script = module.build_remote_script(cfg)

        assert "'process'" in script
        assert "'--device', 'cuda'" in script
        assert "'--imgsz', '1280'" in script
        assert "'--flip-court-y'" in script
        assert "start_new_session=True" in script
        assert module.LAUNCH_MARKER in script
        compile(script, "<remote_process_script>", "exec")

    def test_run_colab_processing_uploads_polls_downloads_and_stops(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _load_script_module()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        video = tmp_path / "match.mp4"
        video.write_bytes(b"mp4")
        ball_model = tmp_path / "models" / "ball_yolo26.pt"
        ball_model.parent.mkdir()
        ball_model.write_bytes(b"pt")

        args, extra = module.build_parser().parse_known_args(
            ["--input", str(video), "--colab-bin", "colab", "--poll-interval", "0"]
        )
        cfg = module.resolve_config(args, extra_args=extra, repo_root=tmp_path)

        recorded_cmds: list[list[str]] = []
        poll_count = 0

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            nonlocal poll_count
            recorded_cmds.append(cmd)
            if "exec" in cmd and capture_output:
                script_file = Path(cmd[-1])
                if script_file.name == "remote_process.py":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout=f"[remote] OK\n{module.LAUNCH_MARKER}\n", stderr=""
                    )
                poll_count += 1
                if poll_count == 1:
                    raise subprocess.CalledProcessError(1, cmd, stderr="Connection was lost.")
                if poll_count == 2:
                    raise subprocess.TimeoutExpired(cmd, timeout=timeout or 90.0)
                payload = {
                    "offset": 24,
                    "done": True,
                    "exitcode": 0,
                    "chunk": "Processed 90 frames\n",
                }
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=f"{module.POLL_MARKER}{json.dumps(payload)}\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            with patch.object(module.time, "sleep"):
                module.run_colab_processing(cfg, repo_root=tmp_path)

        out = capsys.readouterr().out
        assert "Processed 90 frames\n" in out
        upload_cmds = [cmd for cmd in recorded_cmds if "upload" in cmd]
        download_cmds = [cmd for cmd in recorded_cmds if "download" in cmd]
        assert any("/content/match.mp4" in cmd for cmd in upload_cmds)
        assert any("/content/ball_yolo26.pt" in cmd for cmd in upload_cmds)
        assert len(download_cmds) == 3
        assert any("stop" in cmd for cmd in recorded_cmds)

    def test_upload_file_chunks_large_files_and_reassembles(self, tmp_path: Path) -> None:
        module = _load_script_module()
        big_file = tmp_path / "big.mp4"
        payload = b"0123456789abcdef" * 10  # 160 bytes
        big_file.write_bytes(payload)

        uploaded: dict[str, bytes] = {}

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            if "upload" in cmd:
                local_src = Path(cmd[-2])
                remote_dst = cmd[-1]
                uploaded[remote_dst] = local_src.read_bytes()
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "exec" in cmd:
                script_src = Path(cmd[-1]).read_text(encoding="utf-8")
                compile(script_src, "<reassemble>", "exec")
                assert f"total == {len(payload)}" in script_src
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="[remote] Reassembled\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(module.subprocess, "run", side_effect=fake_run):
            module._upload_file(
                "colab",
                "sess",
                big_file,
                "/content/big.mp4",
                tmp_dir=tmp_path,
                chunk_bytes=64,
            )

        assert len(uploaded) == 3
        reassembled = b"".join(uploaded[k] for k in sorted(uploaded))
        assert reassembled == payload

    def test_download_file_chunked_reassembles_parts(self, tmp_path: Path) -> None:
        module = _load_script_module()
        expected = b"abcdefghij" * 20
        parts_data = {
            "/content/.dl_chunks/out/part_0000": expected[:80],
            "/content/.dl_chunks/out/part_0001": expected[80:160],
            "/content/.dl_chunks/out/part_0002": expected[160:],
        }

        def fake_run(cmd, check=True, capture_output=False, text=False, timeout=None):
            if "exec" in cmd:
                script_file = Path(cmd[-1])
                if script_file.name.startswith("split_"):
                    manifest = {"size": len(expected), "parts": sorted(parts_data.keys())}
                    out = f"{module.DL_MANIFEST_MARKER}{json.dumps(manifest)}\n"
                    return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "download" in cmd:
                remote_part = cmd[-2]
                local_part = Path(cmd[-1])
                local_part.write_bytes(parts_data[remote_part])
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        out_path = tmp_path / "out.mp4"
        with patch.object(module.subprocess, "run", side_effect=fake_run):
            module._download_file_chunked(
                "colab", "sess", "/content/out.mp4", out_path, tmp_dir=tmp_path, chunk_bytes=80
            )

        assert out_path.read_bytes() == expected

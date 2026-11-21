#!/usr/bin/env python3
"""Run ``tennis_tracker process`` on a Google Colab GPU VM via the ``colab`` CLI.

Packages ``src/``, uploads the input video and local model weights to a Google
Colab GPU VM, runs ``python -m tennis_tracker process`` detached on CUDA while
polling logs (resilient to transient websocket disconnects), downloads the raw
CSV, smoothed CSV, and annotated video back to ``data/output/``, and stops the
Colab session cleanly.

Usage:
    # Dry-run to inspect commands without allocating a VM
    uv run python scripts/process_on_colab.py --input data/input/match.mp4 --dry-run

    # Process a video on a Colab T4 GPU
    uv run python scripts/process_on_colab.py --input data/input/match.mp4 --gpu T4

    # Forward extra tennis_tracker process flags (e.g. --imgsz 1280, --ball-conf 0.05)
    uv run python scripts/process_on_colab.py --input data/input/match.mp4 --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_DIR = "/content/tennis-video-extraction"
REMOTE_LOG_PATH = "/content/process.log"
REMOTE_EXITCODE_PATH = "/content/process.exitcode"
LAUNCH_MARKER = "__COLAB_PROCESS_LAUNCHED__"
POLL_MARKER = "__COLAB_POLL__:"

_GPU_CHOICES = ["T4", "L4", "G4", "H100", "A100"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Colab GPU video processing."""
    parser = argparse.ArgumentParser(
        prog="process-on-colab",
        description=(
            "Run tennis_tracker video processing on a Google Colab GPU VM. "
            "Any unrecognized flags are forwarded to `python -m tennis_tracker process`."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Local path to input match video.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Local raw CSV output path (default: data/output/<stem>_raw_coordinates.csv).",
    )
    parser.add_argument(
        "--smoothed-output",
        type=Path,
        default=None,
        help=(
            "Local smoothed CSV output path "
            "(default: data/output/<stem>_smoothed_coordinates.csv)."
        ),
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Local annotated video output path (default: data/output/<stem>_annotated.mp4).",
    )
    parser.add_argument(
        "--player-model",
        default="yolo26n.pt",
        help="Player detection model path or Ultralytics name (default: yolo26n.pt).",
    )
    parser.add_argument(
        "--ball-model",
        default="models/ball_yolo26.pt",
        help="Ball detection model path (default: models/ball_yolo26.pt).",
    )
    parser.add_argument(
        "--court-model",
        default="models/court_keypoints_yolo26.pt",
        help="Court keypoint model path (default: models/court_keypoints_yolo26.pt).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Remote inference device (default: cuda).",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target processing FPS (default: 30).",
    )
    parser.add_argument(
        "--gpu",
        choices=_GPU_CHOICES,
        default="T4",
        help="Colab GPU accelerator type (default: T4).",
    )
    parser.add_argument(
        "--session",
        default="tennis-process",
        help="Colab session name (default: tennis-process).",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="Attach to an existing running session instead of creating/stopping one.",
    )
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Keep the Colab session alive after processing finishes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=14400.0,
        help="Overall timeout in seconds for remote processing (default: 14400; <=0 for no limit).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between remote log polls (default: 10).",
    )
    parser.add_argument(
        "--colab-bin",
        default="colab",
        help="Path or name of the Google Colab CLI executable (default: colab).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned colab CLI commands and remote script without allocating a VM.",
    )
    return parser


def _resolve_model_arg(model_arg: str, repo_root: Path) -> tuple[str, Path | None]:
    """Return ``(remote_arg, local_file_to_upload_or_None)`` for a model path/name."""
    cand = Path(model_arg)
    if not cand.is_absolute():
        cand = repo_root / cand
    if cand.is_file():
        return f"/content/{cand.name}", cand
    return model_arg, None


def resolve_config(
    args: argparse.Namespace,
    extra_args: list[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve local/remote paths and upload plan from parsed CLI arguments."""
    input_path = args.input if args.input.is_absolute() else (repo_root / args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    stem = input_path.stem
    raw_output = args.raw_output or Path(f"data/output/{stem}_raw_coordinates.csv")
    smoothed_output = args.smoothed_output or Path(f"data/output/{stem}_smoothed_coordinates.csv")
    video_output = args.video_output or Path(f"data/output/{stem}_annotated.mp4")

    player_remote, player_local = _resolve_model_arg(args.player_model, repo_root)
    ball_remote, ball_local = _resolve_model_arg(args.ball_model, repo_root)
    court_remote, court_local = _resolve_model_arg(args.court_model, repo_root)

    model_uploads: list[tuple[Path, str]] = []
    seen_remotes: set[str] = set()
    for local_file, remote_path in (
        (player_local, player_remote),
        (ball_local, ball_remote),
        (court_local, court_remote),
    ):
        if local_file is not None and remote_path not in seen_remotes:
            model_uploads.append((local_file, remote_path))
            seen_remotes.add(remote_path)

    return {
        "input_path": input_path,
        "remote_input_path": f"/content/{input_path.name}",
        "raw_output": raw_output,
        "smoothed_output": smoothed_output,
        "video_output": video_output,
        "remote_raw_output": f"{REMOTE_REPO_DIR}/data/output/{raw_output.name}",
        "remote_smoothed_output": f"{REMOTE_REPO_DIR}/data/output/{smoothed_output.name}",
        "remote_video_output": f"{REMOTE_REPO_DIR}/data/output/{video_output.name}",
        "player_model_remote": player_remote,
        "ball_model_remote": ball_remote,
        "court_model_remote": court_remote,
        "model_uploads": model_uploads,
        "device": args.device,
        "target_fps": args.target_fps,
        "extra_args": list(extra_args or []),
        "gpu": args.gpu,
        "session": args.session,
        "reuse_session": args.reuse_session,
        "keep_session": args.keep_session,
        "timeout": args.timeout,
        "poll_interval": args.poll_interval,
        "colab_bin": args.colab_bin,
        "dry_run": args.dry_run,
    }


def create_repo_archive(repo_root: Path, archive_path: Path) -> Path:
    """Create a lightweight tarball of pyproject.toml and src/."""
    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts
        if any(p == "__pycache__" or p.endswith(".egg-info") for p in parts):
            return None
        if tarinfo.name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        for rel in ("pyproject.toml", "src"):
            full = repo_root / rel
            if full.exists():
                tar.add(full, arcname=rel, filter=_filter)
    return archive_path


def build_remote_script(cfg: dict[str, Any]) -> str:
    """Generate the Colab setup + detached video processing launcher script."""
    remote_repo = REMOTE_REPO_DIR
    process_cmd = [
        "sys.executable",
        "-m",
        "tennis_tracker",
        "process",
        "--input",
        cfg["remote_input_path"],
        "--raw-output",
        cfg["remote_raw_output"],
        "--smoothed-output",
        cfg["remote_smoothed_output"],
        "--video-output",
        cfg["remote_video_output"],
        "--player-model",
        cfg["player_model_remote"],
        "--ball-model",
        cfg["ball_model_remote"],
        "--court-model",
        cfg["court_model_remote"],
        "--device",
        str(cfg["device"]),
        "--target-fps",
        str(cfg["target_fps"]),
        *cfg["extra_args"],
    ]
    cmd_items = [process_cmd[0]] + [repr(item) for item in process_cmd[1:]]
    cmd_list_src = "[" + ", ".join(cmd_items) + "]"
    expected_outputs = [
        cfg["remote_raw_output"],
        cfg["remote_smoothed_output"],
        cfg["remote_video_output"],
    ]

    return f'''"""Auto-generated remote video processing launcher for Google Colab."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_DIR = {remote_repo!r}
LOG_PATH = Path({REMOTE_LOG_PATH!r})
EXITCODE_PATH = Path({REMOTE_EXITCODE_PATH!r})
Path(REPO_DIR).mkdir(parents=True, exist_ok=True)
(Path(REPO_DIR) / "data" / "output").mkdir(parents=True, exist_ok=True)

print("[remote] Extracting repository archive...")
with tarfile.open("/content/tennis_repo.tar.gz", "r:gz") as tar:
    tar.extractall(REPO_DIR)

print("[remote] Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4.0"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", REPO_DIR],
    check=True,
)

import torch
print(f"[remote] PyTorch {{torch.__version__}}, CUDA available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"[remote] GPU: {{torch.cuda.get_device_name(0)}}")

process_cmd = {cmd_list_src}
print("[remote] Launching detached video processing:", " ".join(process_cmd))

LOG_PATH.unlink(missing_ok=True)
EXITCODE_PATH.unlink(missing_ok=True)

runner_code = "\\n".join([
    "import json, os, subprocess, sys",
    "from pathlib import Path",
    "env = os.environ.copy()",
    f"env['PYTHONPATH'] = {remote_repo!r} + '/src:' + env.get('PYTHONPATH', '')",
    "env['PYTHONUNBUFFERED'] = '1'",
    f"cmd = json.loads({{json.dumps(process_cmd)!r}})",
    f"expected = json.loads({{json.dumps({expected_outputs!r})!r}})",
    "rc = 1",
    "try:",
    f"    proc = subprocess.run(cmd, cwd={remote_repo!r}, env=env)",
    "    rc = proc.returncode",
    "    if rc == 0:",
    "        for out_path in expected:",
    "            if not Path(out_path).exists():",
    "                print('[remote] ERROR: Missing ' + out_path, file=sys.stderr)",
    "                rc = 1",
    "finally:",
    f"    Path({REMOTE_EXITCODE_PATH!r}).write_text(str(rc), encoding='utf-8')",
    "",
])

runner_path = Path("/content/run_process_bg.py")
runner_path.write_text(runner_code, encoding="utf-8")

with LOG_PATH.open("w", encoding="utf-8") as log_fh:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner_path)],
        cwd=REPO_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

print(f"[remote] Detached processing started (PID {{proc.pid}}). Polling logs...")
print({LAUNCH_MARKER!r})
'''


def build_poll_script(offset: int) -> str:
    """Generate a short-lived script that reads new log bytes and job status."""
    return f"""import json
from pathlib import Path

log_path = Path({REMOTE_LOG_PATH!r})
exit_path = Path({REMOTE_EXITCODE_PATH!r})
offset = {offset}
chunk = ""
new_offset = offset
if log_path.exists():
    data = log_path.read_bytes()
    if offset < len(data):
        chunk = data[offset:].decode("utf-8", errors="replace")
        new_offset = len(data)
done = exit_path.exists()
exitcode = int(exit_path.read_text(encoding="utf-8").strip()) if done else None
payload = {{"offset": new_offset, "done": done, "exitcode": exitcode, "chunk": chunk}}
print({POLL_MARKER!r} + json.dumps(payload))
"""


def parse_poll_output(stdout: str) -> dict[str, Any] | None:
    """Extract the JSON poll payload from ``colab exec`` stdout."""
    for line in stdout.splitlines():
        if line.startswith(POLL_MARKER):
            return json.loads(line[len(POLL_MARKER) :])
    return None


def _run_colab(
    colab_bin: str,
    args: list[str],
    dry_run: bool = False,
    capture_output: bool = False,
    cmd_timeout: float | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Invoke the colab CLI with the given arguments."""
    cmd = [colab_bin, *args]
    if not capture_output:
        print(f"[colab-process] $ {shlex.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture_output,
        text=capture_output,
        timeout=cmd_timeout,
    )


def _poll_remote_job(
    colab_bin: str,
    session: str,
    tmp_dir: Path,
    timeout: float,
    poll_interval: float,
    max_consecutive_failures: int = 5,
    poll_cmd_timeout: float = 90.0,
) -> None:
    """Poll the remote log until completion, retrying transient disconnects."""
    offset = 0
    consecutive_failures = 0
    deadline = (time.monotonic() + timeout) if timeout > 0 else None
    poll_script_path = tmp_dir / "poll_process.py"

    while True:
        poll_script_path.write_text(build_poll_script(offset), encoding="utf-8")
        try:
            res = _run_colab(
                colab_bin,
                ["exec", "-s", session, "--timeout", "60.0", "-f", str(poll_script_path)],
                capture_output=True,
                cmd_timeout=poll_cmd_timeout,
            )
            consecutive_failures = 0
            stdout = res.stdout if res else ""
            payload = parse_poll_output(stdout)
            if payload is None:
                raise RuntimeError(f"Unexpected poll output from Colab:\n{stdout}")

            if payload["chunk"]:
                print(payload["chunk"], end="", flush=True)
            offset = int(payload["offset"])

            if payload["done"]:
                exitcode = int(payload["exitcode"]) if payload["exitcode"] is not None else 1
                if exitcode != 0:
                    raise RuntimeError(f"Remote video processing failed with exit code {exitcode}.")
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            consecutive_failures += 1
            if isinstance(exc, subprocess.TimeoutExpired):
                detail_line = f"colab exec hung >{poll_cmd_timeout:g}s"
            else:
                detail = (exc.stderr or exc.stdout or "").strip()
                detail_line = detail.splitlines()[-1] if detail else f"exit code {exc.returncode}"
            print(
                f"[colab-process] Poll warning ({consecutive_failures}/{max_consecutive_failures}):"
                f" {detail_line}; retrying...",
                flush=True,
            )
            if consecutive_failures >= max_consecutive_failures:
                raise

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Remote processing exceeded overall timeout of {timeout:g}s.")
        time.sleep(poll_interval)


def _download_outputs(
    colab_bin: str,
    session: str,
    cfg: dict[str, Any],
    repo_root: Path,
    dry_run: bool,
    best_effort: bool = False,
) -> None:
    """Download raw CSV, smoothed CSV, and annotated video from the Colab VM."""
    downloads = [
        (cfg["remote_raw_output"], cfg["raw_output"]),
        (cfg["remote_smoothed_output"], cfg["smoothed_output"]),
        (cfg["remote_video_output"], cfg["video_output"]),
    ]
    for remote_path, local_rel in downloads:
        local_path = local_rel if local_rel.is_absolute() else (repo_root / local_rel)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_colab(
                colab_bin,
                ["download", "-s", session, remote_path, str(local_path)],
                dry_run=dry_run,
            )
        except subprocess.CalledProcessError:
            if not best_effort:
                raise
            print(f"[colab-process] Note: {remote_path} not available for rescue download.")


def run_colab_processing(cfg: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    """Execute the end-to-end Colab GPU video processing workflow."""
    colab_bin = shutil.which(cfg["colab_bin"]) or cfg["colab_bin"]
    session = cfg["session"]
    dry_run = cfg["dry_run"]

    with tempfile.TemporaryDirectory(prefix="tennis_colab_proc_") as tmp:
        tmp_dir = Path(tmp)
        repo_tar = create_repo_archive(repo_root, tmp_dir / "tennis_repo.tar.gz")
        remote_script_path = tmp_dir / "remote_process.py"
        remote_script_src = build_remote_script(cfg)
        remote_script_path.write_text(remote_script_src, encoding="utf-8")

        if dry_run:
            print("[colab-process] --- Generated remote launch script ---")
            print(remote_script_src)
            print("[colab-process] --- End remote launch script ---")

        created_session = False
        try:
            if not cfg["reuse_session"]:
                _run_colab(
                    colab_bin,
                    ["new", "-s", session, "--gpu", cfg["gpu"]],
                    dry_run=dry_run,
                )
                created_session = True

            _run_colab(
                colab_bin,
                ["upload", "-s", session, str(repo_tar), "/content/tennis_repo.tar.gz"],
                dry_run=dry_run,
            )

            _run_colab(
                colab_bin,
                ["upload", "-s", session, str(cfg["input_path"]), cfg["remote_input_path"]],
                dry_run=dry_run,
            )

            for local_model, remote_model in cfg["model_uploads"]:
                _run_colab(
                    colab_bin,
                    ["upload", "-s", session, str(local_model), remote_model],
                    dry_run=dry_run,
                )

            launch_res = _run_colab(
                colab_bin,
                [
                    "exec",
                    "-s",
                    session,
                    "--timeout",
                    "600.0",
                    "-f",
                    str(remote_script_path),
                ],
                dry_run=dry_run,
                capture_output=True,
            )

            if dry_run:
                print(
                    f"[colab-process] (dry-run) Would poll {REMOTE_LOG_PATH} every "
                    f"{cfg['poll_interval']}s up to {cfg['timeout']}s."
                )
            else:
                launch_out = launch_res.stdout if launch_res else ""
                if launch_out:
                    print(launch_out, end="", flush=True)
                if launch_res and launch_res.stderr:
                    print(launch_res.stderr, end="", flush=True)
                if LAUNCH_MARKER not in launch_out:
                    raise RuntimeError(
                        "Remote setup/launch failed before starting detached processing."
                    )
                try:
                    _poll_remote_job(
                        colab_bin=colab_bin,
                        session=session,
                        tmp_dir=tmp_dir,
                        timeout=cfg["timeout"],
                        poll_interval=cfg["poll_interval"],
                    )
                except (TimeoutError, KeyboardInterrupt):
                    print(
                        "[colab-process] Interrupted/timed out; attempting to rescue "
                        "available outputs before stopping session...",
                        flush=True,
                    )
                    _download_outputs(
                        colab_bin=colab_bin,
                        session=session,
                        cfg=cfg,
                        repo_root=repo_root,
                        dry_run=False,
                        best_effort=True,
                    )
                    raise

            _download_outputs(
                colab_bin=colab_bin,
                session=session,
                cfg=cfg,
                repo_root=repo_root,
                dry_run=dry_run,
                best_effort=False,
            )
        finally:
            if created_session and not cfg["keep_session"]:
                _run_colab(colab_bin, ["stop", "-s", session], dry_run=dry_run)


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    cfg = resolve_config(args, extra_args=extra_args)
    run_colab_processing(cfg)


if __name__ == "__main__":
    main()

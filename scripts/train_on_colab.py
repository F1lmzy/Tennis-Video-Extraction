#!/usr/bin/env python3
"""Run YOLO model training on a Google Colab GPU VM via the ``colab`` CLI.

This script packages the local ``tennis_tracker`` source code, provisions (or
reuses) a Google Colab GPU session, prepares the dataset, launches
``tennis_tracker.training.ball`` or ``tennis_tracker.training.court`` detached
on the GPU VM, polls the remote log until completion (resilient to transient
websocket disconnects), downloads the trained weights and metrics back to the
local repository, and stops the Colab session cleanly.

Usage:
    # Dry-run to inspect the generated workflow without provisioning a VM
    uv run python scripts/train_on_colab.py --task ball --dry-run

    # Train ball detection on a T4 GPU (100 epochs, imgsz 1280 for small ball)
    uv run python scripts/train_on_colab.py --task ball --gpu T4 --epochs 100 --imgsz 1280

    # Train court keypoint detection on a T4 GPU
    uv run python scripts/train_on_colab.py --task court --gpu T4 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
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
REMOTE_LOG_PATH = "/content/train.log"
REMOTE_EXITCODE_PATH = "/content/train.exitcode"
LAUNCH_MARKER = "__COLAB_LAUNCHED__"
POLL_MARKER = "__COLAB_POLL__:"

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "ball": {
        "module": "tennis_tracker.training.ball",
        "default_base_model": "yolo26n.pt",
        "default_output": Path("models/ball_yolo26.pt"),
        "dataset_name": "ball",
        "dataset_dir": "viren-dhanwani-tennis-ball-detection",
    },
    "court": {
        "module": "tennis_tracker.training.court",
        "default_base_model": "yolo26n-pose.pt",
        "default_output": Path("models/court_keypoints_yolo26.pt"),
        "dataset_name": "court",
        "dataset_dir": "abiya-thesis-tennis-court-suuzy",
    },
}

_GPU_CHOICES = ["T4", "L4", "G4", "H100", "A100"]


def _load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env file."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _has_valid_roboflow_key(env_file: Path) -> bool:
    """Return True if a non-placeholder ROBOFLOW_API_KEY is available."""
    key = os.environ.get("ROBOFLOW_API_KEY") or _load_env_file(env_file).get("ROBOFLOW_API_KEY", "")
    return bool(key and "replace_with" not in key)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Colab GPU training."""
    parser = argparse.ArgumentParser(
        prog="train-on-colab",
        description="Train or fine-tune tennis_tracker YOLO models on a Google Colab GPU.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_CONFIGS.keys()),
        required=True,
        help="Training task to run on Colab GPU.",
    )
    parser.add_argument(
        "--gpu",
        choices=_GPU_CHOICES,
        default="T4",
        help="Colab GPU accelerator type (default: T4).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Colab session name (default: tennis-train-<task>).",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="Attach to an existing running session instead of creating/stopping one.",
    )
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Keep the Colab session alive after training finishes.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base model name (e.g. yolo26n.pt, yolo26s.pt) or path to a local "
            ".pt checkpoint to upload and fine-tune further."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Local output path for downloaded trained weights.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Local directory to save results.csv and args.yaml (default: runs/colab/<task>).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Training image size (default: 640)."
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20, 0 disables).",
    )
    parser.add_argument(
        "--kobj",
        type=float,
        default=None,
        help="Keypoint objectness loss weight (court task only).",
    )
    parser.add_argument(
        "--device",
        default="0",
        choices=["0", "1", "cuda", "cpu"],
        help="Remote training device (default: 0).",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["auto", "roboflow", "upload"],
        default="auto",
        help=(
            "How to provide the dataset on Colab: 'roboflow' downloads directly on VM "
            "(fast), 'upload' tars and uploads local data/datasets/ (default: auto)."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional Roboflow dataset version override.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file containing ROBOFLOW_API_KEY (default: .env).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=28800.0,
        help="Overall timeout in seconds for remote training (default: 28800; <=0 for no limit).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="Seconds between remote log polls (default: 15).",
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


def resolve_config(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Resolve defaults, dataset strategy, and paths from parsed CLI arguments."""
    task_cfg = TASK_CONFIGS[args.task]
    session = args.session or f"tennis-train-{args.task}"
    base_model = args.base_model or task_cfg["default_base_model"]
    local_base_model = Path(base_model)
    upload_base_model = local_base_model if local_base_model.is_file() else None

    output_path = args.output or task_cfg["default_output"]
    results_dir = args.results_dir or (Path("runs/colab") / args.task)

    env_file = args.env_file if args.env_file.is_absolute() else (repo_root / args.env_file)
    local_dataset_dir = repo_root / "data" / "datasets" / task_cfg["dataset_dir"]

    if args.dataset_source == "auto":
        if _has_valid_roboflow_key(env_file):
            dataset_source = "roboflow"
        elif local_dataset_dir.is_dir():
            dataset_source = "upload"
        else:
            raise ValueError(
                f"Cannot resolve dataset source for '{args.task}': no valid ROBOFLOW_API_KEY "
                f"in environment/{env_file} and local dataset not found at {local_dataset_dir}."
            )
    else:
        dataset_source = args.dataset_source
        if dataset_source == "upload" and not local_dataset_dir.is_dir():
            raise ValueError(f"Local dataset directory not found: {local_dataset_dir}")
        if dataset_source == "roboflow" and not _has_valid_roboflow_key(env_file):
            raise ValueError(
                f"dataset-source=roboflow requested, but no valid ROBOFLOW_API_KEY found "
                f"in environment or {env_file}."
            )

    return {
        "task": args.task,
        "module": task_cfg["module"],
        "dataset_name": task_cfg["dataset_name"],
        "dataset_dir": task_cfg["dataset_dir"],
        "local_dataset_dir": local_dataset_dir,
        "dataset_source": dataset_source,
        "dataset_version": args.dataset_version,
        "env_file": env_file,
        "gpu": args.gpu,
        "session": session,
        "reuse_session": args.reuse_session,
        "keep_session": args.keep_session,
        "base_model": base_model,
        "upload_base_model": upload_base_model,
        "output_path": output_path,
        "results_dir": results_dir,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "kobj": args.kobj,
        "device": args.device,
        "timeout": args.timeout,
        "poll_interval": args.poll_interval,
        "colab_bin": args.colab_bin,
        "dry_run": args.dry_run,
    }


def create_repo_archive(repo_root: Path, archive_path: Path) -> Path:
    """Create a lightweight tarball of pyproject.toml, src/, and scripts/."""
    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts
        if any(p == "__pycache__" or p.endswith(".egg-info") for p in parts):
            return None
        if tarinfo.name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        for rel in ("pyproject.toml", "src", "scripts/prepare_roboflow_dataset.py"):
            full = repo_root / rel
            if full.exists():
                tar.add(full, arcname=rel, filter=_filter)
    return archive_path


def create_dataset_archive(dataset_dir: Path, archive_path: Path) -> Path:
    """Create a compressed tarball of a local dataset directory."""
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dataset_dir, arcname=dataset_dir.name)
    return archive_path


def build_remote_script(cfg: dict[str, Any]) -> str:
    """Generate the Colab setup + detached training launcher script."""
    remote_repo = REMOTE_REPO_DIR
    remote_dataset_yaml = f"{remote_repo}/data/datasets/{cfg['dataset_dir']}/data.yaml"
    remote_output = f"{remote_repo}/models/{cfg['output_path'].name}"
    if cfg["upload_base_model"] is not None:
        remote_base_model = f"/content/{cfg['upload_base_model'].name}"
    else:
        remote_base_model = str(cfg["base_model"])

    train_cmd = [
        "sys.executable",
        "-m",
        cfg["module"],
        "--data",
        remote_dataset_yaml,
        "--base-model",
        remote_base_model,
        "--output",
        remote_output,
        "--device",
        str(cfg["device"]),
        "--epochs",
        str(cfg["epochs"]),
        "--imgsz",
        str(cfg["imgsz"]),
        "--batch",
        str(cfg["batch"]),
        "--patience",
        str(cfg["patience"]),
    ]
    if cfg["kobj"] is not None and cfg["task"] == "court":
        train_cmd.extend(["--kobj", str(cfg["kobj"])])

    cmd_items = [train_cmd[0]] + [repr(item) for item in train_cmd[1:]]
    cmd_list_src = "[" + ", ".join(cmd_items) + "]"

    if cfg["dataset_source"] == "roboflow":
        version_args = (
            f", '--version', {cfg['dataset_version']!r}" if cfg["dataset_version"] else ""
        )
        dataset_prep_block = f"""
if not dataset_yaml.exists():
    print("[remote] Downloading dataset from Roboflow...")
    subprocess.run(
        [
            sys.executable,
            f"{{REPO_DIR}}/scripts/prepare_roboflow_dataset.py",
            "--dataset", {cfg['dataset_name']!r},
            "--env-file", "/content/tennis_colab.env",
            "--output", str(dataset_yaml.parent){version_args},
        ],
        cwd=REPO_DIR,
        check=True,
    )
else:
    print(f"[remote] Reusing existing dataset at {{dataset_yaml}}")
"""
    else:
        dataset_prep_block = """
if not dataset_yaml.exists():
    print("[remote] Extracting uploaded dataset archive...")
    datasets_root = Path(f"{REPO_DIR}/data/datasets")
    datasets_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/dataset.tar.gz", "r:gz") as tar:
        tar.extractall(datasets_root)
else:
    print(f"[remote] Reusing existing dataset at {dataset_yaml}")
"""

    pip_pkgs = ["ultralytics>=8.4.0"]
    if cfg["dataset_source"] == "roboflow":
        pip_pkgs.append("roboflow>=1.1")

    return f'''"""Auto-generated remote setup and detached training launcher for Google Colab."""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_DIR = {remote_repo!r}
LOG_PATH = Path({REMOTE_LOG_PATH!r})
EXITCODE_PATH = Path({REMOTE_EXITCODE_PATH!r})
Path(REPO_DIR).mkdir(parents=True, exist_ok=True)

print("[remote] Extracting repository archive...")
with tarfile.open("/content/tennis_repo.tar.gz", "r:gz") as tar:
    tar.extractall(REPO_DIR)

print("[remote] Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *{pip_pkgs!r}],
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

dataset_yaml = Path({remote_dataset_yaml!r})
{dataset_prep_block.strip()}

if not dataset_yaml.exists():
    raise FileNotFoundError(f"Expected dataset YAML at {{dataset_yaml}}")

train_cmd = {cmd_list_src}
print("[remote] Launching detached training:", " ".join(train_cmd))

LOG_PATH.unlink(missing_ok=True)
EXITCODE_PATH.unlink(missing_ok=True)

import json
runner_code = "\\n".join([
    "import json, os, subprocess, sys",
    "from pathlib import Path",
    "env = os.environ.copy()",
    f"env['PYTHONPATH'] = {remote_repo!r} + '/src:' + env.get('PYTHONPATH', '')",
    "env['PYTHONUNBUFFERED'] = '1'",
    f"cmd = json.loads({{json.dumps(train_cmd)!r}})",
    "rc = 1",
    "try:",
    f"    proc = subprocess.run(cmd, cwd={remote_repo!r}, env=env)",
    "    rc = proc.returncode",
    f"    output_file = Path({remote_output!r})",
    "    if rc == 0 and not output_file.exists():",
    "        print('[remote] ERROR: Missing ' + str(output_file), file=sys.stderr)",
    "        rc = 1",
    "finally:",
    f"    Path({REMOTE_EXITCODE_PATH!r}).write_text(str(rc), encoding='utf-8')",
    "",
])

runner_path = Path("/content/run_train_bg.py")
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

print(f"[remote] Detached training started (PID {{proc.pid}}). Polling logs...")
print({LAUNCH_MARKER!r})
'''


def build_poll_script(offset: int) -> str:
    """Generate a short-lived script that reads new log bytes and training status."""
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
        print(f"[colab-train] $ {shlex.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture_output,
        text=capture_output,
        timeout=cmd_timeout,
    )


def _poll_remote_training(
    colab_bin: str,
    session: str,
    tmp_dir: Path,
    timeout: float,
    poll_interval: float,
    max_consecutive_failures: int = 5,
    poll_cmd_timeout: float = 90.0,
) -> None:
    """Poll the remote training log until completion, retrying transient drops."""
    offset = 0
    consecutive_failures = 0
    deadline = (time.monotonic() + timeout) if timeout > 0 else None
    poll_script_path = tmp_dir / "poll_train.py"

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
                    raise RuntimeError(f"Remote training failed with exit code {exitcode}.")
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            consecutive_failures += 1
            if isinstance(exc, subprocess.TimeoutExpired):
                detail_line = f"colab exec hung >{poll_cmd_timeout:g}s"
            else:
                detail = (exc.stderr or exc.stdout or "").strip()
                detail_line = detail.splitlines()[-1] if detail else f"exit code {exc.returncode}"
            print(
                f"[colab-train] Poll warning ({consecutive_failures}/{max_consecutive_failures}): "
                f"{detail_line}; retrying...",
                flush=True,
            )
            if consecutive_failures >= max_consecutive_failures:
                raise

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Remote training exceeded overall timeout of {timeout:g}s.")
        time.sleep(poll_interval)


def _download_artifacts(
    colab_bin: str,
    session: str,
    cfg: dict[str, Any],
    repo_root: Path,
    dry_run: bool,
    rescue_only: bool = False,
) -> None:
    """Download trained weights and run metrics from the Colab VM."""
    local_output = (
        cfg["output_path"]
        if cfg["output_path"].is_absolute()
        else (repo_root / cfg["output_path"])
    )
    local_output.parent.mkdir(parents=True, exist_ok=True)
    remote_output = f"{REMOTE_REPO_DIR}/models/{cfg['output_path'].name}"
    remote_run_dir = f"{REMOTE_REPO_DIR}/models/{cfg['output_path'].stem}"

    if not rescue_only:
        _run_colab(
            colab_bin,
            ["download", "-s", session, remote_output, str(local_output)],
            dry_run=dry_run,
        )
    else:
        for remote_cand, local_cand in (
            (f"{remote_run_dir}/weights/best.pt", local_output),
            (
                f"{remote_run_dir}/weights/last.pt",
                local_output.with_name(f"{local_output.stem}.last{local_output.suffix}"),
            ),
        ):
            try:
                _run_colab(
                    colab_bin,
                    ["download", "-s", session, remote_cand, str(local_cand)],
                    dry_run=dry_run,
                )
            except subprocess.CalledProcessError:
                print(f"[colab-train] Note: rescue weight {remote_cand} not available.")

    local_results_dir = (
        cfg["results_dir"]
        if cfg["results_dir"].is_absolute()
        else (repo_root / cfg["results_dir"])
    )
    local_results_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ("results.csv", "args.yaml"):
        try:
            _run_colab(
                colab_bin,
                [
                    "download",
                    "-s",
                    session,
                    f"{remote_run_dir}/{artifact}",
                    str(local_results_dir / artifact),
                ],
                dry_run=dry_run,
            )
        except subprocess.CalledProcessError:
            print(f"[colab-train] Note: optional artifact {artifact} not downloaded.")


def run_colab_training(cfg: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    """Execute the end-to-end Colab GPU training workflow."""
    colab_bin = shutil.which(cfg["colab_bin"]) or cfg["colab_bin"]
    session = cfg["session"]
    dry_run = cfg["dry_run"]

    with tempfile.TemporaryDirectory(prefix="tennis_colab_") as tmp:
        tmp_dir = Path(tmp)
        repo_tar = create_repo_archive(repo_root, tmp_dir / "tennis_repo.tar.gz")
        remote_script_path = tmp_dir / f"remote_train_{cfg['task']}.py"
        remote_script_src = build_remote_script(cfg)
        remote_script_path.write_text(remote_script_src, encoding="utf-8")

        if dry_run:
            print("[colab-train] --- Generated remote launch script ---")
            print(remote_script_src)
            print("[colab-train] --- End remote launch script ---")

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

            if cfg["dataset_source"] == "roboflow":
                env_upload_path = tmp_dir / "colab.env"
                if cfg["env_file"].exists():
                    env_upload_path.write_text(
                        cfg["env_file"].read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                else:
                    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
                    env_upload_path.write_text(f"ROBOFLOW_API_KEY={api_key}\n", encoding="utf-8")
                _run_colab(
                    colab_bin,
                    ["upload", "-s", session, str(env_upload_path), "/content/tennis_colab.env"],
                    dry_run=dry_run,
                )
            else:
                dataset_tar = create_dataset_archive(
                    cfg["local_dataset_dir"], tmp_dir / "dataset.tar.gz"
                )
                _run_colab(
                    colab_bin,
                    ["upload", "-s", session, str(dataset_tar), "/content/dataset.tar.gz"],
                    dry_run=dry_run,
                )

            if cfg["upload_base_model"] is not None:
                remote_weight = f"/content/{cfg['upload_base_model'].name}"
                _run_colab(
                    colab_bin,
                    ["upload", "-s", session, str(cfg["upload_base_model"]), remote_weight],
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
                    f"[colab-train] (dry-run) Would poll {REMOTE_LOG_PATH} every "
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
                        "Remote setup/launch failed before starting detached training."
                    )
                try:
                    _poll_remote_training(
                        colab_bin=colab_bin,
                        session=session,
                        tmp_dir=tmp_dir,
                        timeout=cfg["timeout"],
                        poll_interval=cfg["poll_interval"],
                    )
                except (TimeoutError, KeyboardInterrupt):
                    print(
                        "[colab-train] Training interrupted/timed out; attempting to rescue "
                        "latest checkpoints before stopping session...",
                        flush=True,
                    )
                    _download_artifacts(
                        colab_bin=colab_bin,
                        session=session,
                        cfg=cfg,
                        repo_root=repo_root,
                        dry_run=False,
                        rescue_only=True,
                    )
                    raise

            _download_artifacts(
                colab_bin=colab_bin,
                session=session,
                cfg=cfg,
                repo_root=repo_root,
                dry_run=dry_run,
                rescue_only=False,
            )
        finally:
            if created_session and not cfg["keep_session"]:
                _run_colab(colab_bin, ["stop", "-s", session], dry_run=dry_run)


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_config(args)
    run_colab_training(cfg)


if __name__ == "__main__":
    main()

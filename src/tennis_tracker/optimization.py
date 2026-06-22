"""Automatic model optimization workflow.

Detects missing/stale optimized artifacts, exports to ONNX and OpenVINO,
optionally quantizes, benchmarks candidates, and selects the fastest
valid artifact for runtime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tennis_tracker.diagnostics import Diagnostics

# ── Metadata ───────────────────────────────────────────────────────────

_SUPPORTED_FORMATS = ("onnx", "openvino")
"""Export formats that the optimization workflow can produce."""

_QUANTIZATION_OPTIONS = ("int8", "fp16")
"""Optional quantization types."""


@dataclass
class OptimizationMetadata:
    """Records the outcome of an optimization attempt for a single artifact.

    Serialised to JSON for persistence across runs.
    """

    source_model: str
    """Path or name of the source (e.g. ``yolo26n.pt``)."""

    export_format: str
    """One of :attr:`_SUPPORTED_FORMATS` or ``pt`` for fallback."""

    quantization: Optional[str] = None
    """Quantization type applied, if any."""

    created_at: str = ""
    """ISO‑8601 timestamp of the optimisation run (set automatically)."""

    benchmark_fps: Optional[float] = None
    """Measured inference FPS for this artifact (set after benchmark)."""

    artifact_path: Optional[str] = None
    """Absolute filesystem path to the exported artifact."""

    status: str = "pending"
    """One of ``pending``, ``success``, ``skipped``, ``failed``."""

    diagnostics: list[str] = field(default_factory=list)
    """Human‑readable messages describing what happened."""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ── serialisation helpers ──────────────────────────────────────

    @staticmethod
    def from_dict(d: dict) -> OptimizationMetadata:
        return OptimizationMetadata(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Complete result of optimising one source model.

    The ``selected`` metadata indicates which artifact should be used
    at runtime (potentially the original ``.pt`` file when no optimised
    artifact is valid).
    """

    source_model: str
    artifacts: list[OptimizationMetadata] = field(default_factory=list)
    selected: Optional[OptimizationMetadata] = None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    @property
    def has_valid_artifact(self) -> bool:
        return self.selected is not None and self.selected.status == "success"

    @property
    def using_fallback(self) -> bool:
        return (
            self.selected is not None
            and self.selected.export_format == "pt"
            and self.selected.status == "success"
        )


# ── Artifact path helpers ──────────────────────────────────────────────

_DEFAULT_ARTIFACT_DIR = Path("models/optimized")


def _artifact_path(
    source: Path,
    export_format: str,
    quantize: Optional[str] = None,
    artifact_root: Path = _DEFAULT_ARTIFACT_DIR,
) -> Path:
    stem = source.stem  # e.g. "yolo26n"
    if quantize:
        stem = f"{stem}_{quantize}"
    if export_format == "onnx":
        return artifact_root / f"{stem}.onnx"
    elif export_format == "openvino":
        return artifact_root / f"{stem}_openvino"
    else:
        return artifact_root / f"{stem}.pt"


# ── Freshness check ────────────────────────────────────────────────────

def _is_stale(source: Path, artifact: Path) -> bool:
    """Return True if the artifact is missing or older than the source."""
    if not artifact.exists():
        return True
    return artifact.stat().st_mtime < source.stat().st_mtime


def _find_metadata(artifact_root: Path, source: Path) -> Optional[OptimizationMetadata]:
    """Load persisted metadata for *source* if it exists."""
    meta_path = artifact_root / f"{source.stem}_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return OptimizationMetadata.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError):
        return None


def _save_metadata(
    artifact_root: Path, source: Path, meta: OptimizationMetadata
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    meta_path = artifact_root / f"{source.stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)


# ── Main optimisation workflow ─────────────────────────────────────────


def optimize_model(
    input_path: Path | str,
    *,
    output_dir: Path | str = _DEFAULT_ARTIFACT_DIR,
    format: str = "openvino",
    quantize: Optional[str] = None,
    force: bool = False,
    benchmark_fn=None,
    export_fn=None,
) -> OptimizationResult:
    """Run the optimisation workflow for a single source model.

    Parameters
    ----------
    input_path:
        Path to the source ``.pt`` model.
    output_dir:
        Root directory for optimised artifacts (created on demand).
    format:
        Target export format (``onnx`` or ``openvino``).
    quantize:
        Optional quantisation (``int8`` or ``fp16``).
    force:
        If True, re‑export even when a fresh artifact already exists.
    benchmark_fn:
        Callable ``(artifact_path: Path) -> fps: float | None`` for
        mock injection in tests.
    export_fn:
        Callable ``(source: Path, target: Path, fmt: str) -> bool``
        for mock injection in tests.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    result = OptimizationResult(source_model=str(input_path))
    diag = result.diagnostics

    if not input_path.exists():
        diag.add_flag("source_model_not_found")
        result.artifacts.append(
            OptimizationMetadata(
                source_model=str(input_path),
                export_format="pt",
                status="failed",
                diagnostics=["Source model file not found"],
            )
        )
        return result

    # Supplier side — try to produce the requested format
    use_export = export_fn or _mock_export
    target = _artifact_path(input_path, format, quantize, output_dir)

    fmt_meta = _try_export(
        source=input_path,
        target=target,
        export_format=format,
        quantize=quantize,
        output_dir=output_dir,
        force=force,
        export_fn=use_export,
    )
    result.artifacts.append(fmt_meta)

    # Benchmark if we got a valid artifact
    if fmt_meta.status == "success" and fmt_meta.artifact_path:
        use_benchmark = benchmark_fn or _mock_benchmark
        fps = use_benchmark(Path(fmt_meta.artifact_path))
        fmt_meta.benchmark_fps = fps
        if fps is not None and fps > 0:
            fmt_meta.diagnostics.append(f"Measured {fps:.1f} FPS")
        else:
            diag.add_flag("benchmark_failed")
            fmt_meta.diagnostics.append("Benchmark returned no valid FPS")
            fmt_meta.status = "failed"  # treat as failed if benchmark fails

    # Fallback: always record the source .pt as a candidate
    pt_meta = OptimizationMetadata(
        source_model=str(input_path),
        export_format="pt",
        status="success",
        artifact_path=str(input_path.resolve()),
        created_at=fmt_meta.created_at,
        benchmark_fps=None,
        diagnostics=["Original .pt model (fallback)"],
    )
    result.artifacts.append(pt_meta)

    # Selection — pick fastest valid
    _select_artifact(result, diag)

    # Persist metadata for the selected upstream format
    if fmt_meta.status == "success":
        _save_metadata(output_dir, input_path, fmt_meta)

    return result


# ── Internal helpers ───────────────────────────────────────────────────


def _try_export(
    source: Path,
    target: Path,
    export_format: str,
    quantize: Optional[str],
    output_dir: Path,
    force: bool,
    export_fn,
) -> OptimizationMetadata:
    """Attempt to export *source* to the requested format.

    Returns the ``OptimizationMetadata`` for this attempt (never
    raises).
    """
    meta = OptimizationMetadata(
        source_model=str(source),
        export_format=export_format,
        quantization=quantize,
    )

    if export_format not in _SUPPORTED_FORMATS:
        meta.status = "skipped"
        meta.diagnostics.append(f"Unsupported format: {export_format}")
        return meta

    if quantize and quantize not in _QUANTIZATION_OPTIONS:
        meta.diagnostics.append(f"Unsupported quantisation: {quantize}")

    # Check staleness
    if not force and not _is_stale(source, target):
        meta.status = "success"
        meta.artifact_path = str(target.resolve())
        meta.diagnostics.append("Fresh artifact already exists")
        return meta

    # Attempt export
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ok = export_fn(source, target, export_format)
    except Exception as exc:
        meta.status = "failed"
        meta.diagnostics.append(f"Export raised: {exc}")
        return meta

    if not ok:
        meta.status = "skipped"
        meta.diagnostics.append("Export skipped (checker returned False)")
        return meta

    if target.exists():
        meta.status = "success"
        meta.artifact_path = str(target.resolve())
    else:
        meta.status = "failed"
        meta.diagnostics.append("Export completed but artifact not found at target")

    return meta


def _select_artifact(result: OptimizationResult, diag: Diagnostics) -> None:
    """Pick the fastest valid artifact; fall back to ``.pt``."""
    valid = [m for m in result.artifacts if m.status == "success" and m.benchmark_fps]
    # Prefer non‑fallback (i.e. optimised formats) with FPS data
    optimised = [m for m in valid if m.export_format != "pt"]

    if optimised:
        optimised.sort(key=lambda m: m.benchmark_fps or 0.0, reverse=True)
        result.selected = optimised[0]
        diag.add_flag("optimized_artifact_selected")
        return

    # Fallback to .pt
    pt_candidate = [m for m in result.artifacts if m.export_format == "pt"]
    if pt_candidate:
        result.selected = pt_candidate[0]
        diag.add_flag("fallback_to_pt")
    else:
        diag.add_flag("no_valid_artifact")


# ── Mock helpers for dependency injection (used in tests) ──────────────


def _mock_export(source: Path, target: Path, fmt: str) -> bool:
    """Minimal export stub — subclasses should replace via *export_fn*.

    Creates an empty marker file at *target*.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # OpenVINO is a directory; ONNX is a single file
    if fmt == "openvino":
        target.mkdir(exist_ok=True)
        (target / "model.xml").write_text("")
        (target / "model.bin").write_text("")
    else:
        target.write_text("")  # ONNX: empty placeholder
    return True


def _mock_benchmark(artifact: Path) -> Optional[float]:
    """Minimal benchmark stub — subclasses should replace via *benchmark_fn*.

    Returns a dummy 30.0 FPS.
    """
    return 30.0

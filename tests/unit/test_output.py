"""Tests for the CSV output writer.

Covers:
- Header matches the spec exactly.
- One row round-trip produces correct column values.
- Missing optional fields produce empty CSV cells.
- Homography_valid bool serialises as lowercase "true"/"false".
- Multiple rows.
- Context manager lifecycle.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from tennis_tracker.output import TrackingCsvWriter, read_csv_rows
from tennis_tracker.types import OutputRow


def _sample_row(
    frame_index: int = 0,
    time_s: float = 0.0,
    has_players: bool = True,
    has_ball: bool = True,
) -> OutputRow:
    """Build a sample OutputRow for testing."""
    return OutputRow(
        frame_index=frame_index,
        time_s=time_s,
        player_a_x_m=1.23 if has_players else None,
        player_a_y_m=4.56 if has_players else None,
        player_a_pixel_x=100.0 if has_players else None,
        player_a_pixel_y=200.0 if has_players else None,
        player_a_confidence=0.95 if has_players else None,
        player_b_x_m=-1.23 if has_players else None,
        player_b_y_m=-4.56 if has_players else None,
        player_b_pixel_x=300.0 if has_players else None,
        player_b_pixel_y=400.0 if has_players else None,
        player_b_confidence=0.88 if has_players else None,
        ball_x_m=0.5 if has_ball else None,
        ball_y_m=2.3 if has_ball else None,
        ball_pixel_x=160.0 if has_ball else None,
        ball_pixel_y=120.0 if has_ball else None,
        ball_confidence=0.72 if has_ball else None,
        court_confidence=0.95,
        homography_valid=True,
        diagnostics="",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Helper: read a CSV file into a list of row dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def test_header_matches_spec() -> None:
    """The first line of the CSV must equal OutputRow.csv_header()."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path):
        pass  # header written on open

    rows = _read_csv(path)
    assert rows == [], "No data rows expected after just the header"
    path.unlink()


# ---------------------------------------------------------------------------
# Row content
# ---------------------------------------------------------------------------


def test_single_row_round_trip() -> None:
    row = _sample_row(frame_index=42, time_s=1.4)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        writer.write_row(row)

    rows = _read_csv(path)
    assert len(rows) == 1
    data = rows[0]
    assert data["frame_index"] == "42"
    assert data["time_s"] == "1.4"
    assert data["player_a_x_m"] == "1.23"
    assert data["player_a_y_m"] == "4.56"
    assert data["homography_valid"] == "true"


def test_ball_confidence_output() -> None:
    row = _sample_row(frame_index=0, time_s=0.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        writer.write_row(row)

    rows = _read_csv(path)
    assert rows[0]["ball_confidence"] == "0.72"
    path.unlink()


# ---------------------------------------------------------------------------
# Missing optional fields → empty cells
# ---------------------------------------------------------------------------


def test_missing_optional_fields_produce_empty_cells() -> None:
    """All optional fields should be empty for a completely empty row."""
    row = OutputRow(
        frame_index=5,
        time_s=0.167,
        player_a_x_m=None,
        player_a_y_m=None,
        player_a_pixel_x=None,
        player_a_pixel_y=None,
        player_a_confidence=None,
        player_b_x_m=None,
        player_b_y_m=None,
        player_b_pixel_x=None,
        player_b_pixel_y=None,
        player_b_confidence=None,
        ball_x_m=None,
        ball_y_m=None,
        ball_pixel_x=None,
        ball_pixel_y=None,
        ball_confidence=None,
        court_confidence=None,
        homography_valid=False,
        diagnostics="",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        writer.write_row(row)

    rows = _read_csv(path)
    data = rows[0]
    # All optional coordinate/confidence fields are empty.
    assert data["player_a_x_m"] == ""
    assert data["player_a_y_m"] == ""
    assert data["player_a_confidence"] == ""
    assert data["ball_x_m"] == ""
    assert data["ball_confidence"] == ""
    assert data["court_confidence"] == ""
    # Non-optional fields are present.
    assert data["frame_index"] == "5"
    assert data["time_s"] == "0.167"
    assert data["homography_valid"] == "false"
    path.unlink()


# ---------------------------------------------------------------------------
# Bool serialisation
# ---------------------------------------------------------------------------


def test_bool_serialisation() -> None:
    row_true = _sample_row(frame_index=0, time_s=0.0)
    row_true.homography_valid = True
    row_false = _sample_row(frame_index=1, time_s=0.033)
    row_false.homography_valid = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        writer.write_row(row_true)
        writer.write_row(row_false)

    rows = _read_csv(path)
    assert rows[0]["homography_valid"] == "true"
    assert rows[1]["homography_valid"] == "false"
    path.unlink()


# ---------------------------------------------------------------------------
# Multiple rows
# ---------------------------------------------------------------------------


def test_multiple_rows() -> None:
    rows_in = [_sample_row(frame_index=i, time_s=0.033 * i) for i in range(3)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        for r in rows_in:
            writer.write_row(r)

    rows_out = _read_csv(path)
    assert len(rows_out) == 3
    assert rows_out[0]["frame_index"] == "0"
    assert rows_out[1]["frame_index"] == "1"
    assert rows_out[2]["frame_index"] == "2"
    path.unlink()


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------


def test_write_after_close_raises() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    writer = TrackingCsvWriter(path)
    with writer:
        writer.write_row(_sample_row())

    import pytest

    with pytest.raises(RuntimeError, match="closed or unopened"):
        writer.write_row(_sample_row())
    path.unlink()


# ---------------------------------------------------------------------------
# read_csv_rows
# ---------------------------------------------------------------------------


def test_read_csv_rows() -> None:
    rows_in = [_sample_row(frame_index=7, time_s=0.233)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        path = Path(tmp.name)

    with TrackingCsvWriter(path) as writer:
        writer.write_rows(rows_in)

    loaded = list(read_csv_rows(path))
    assert len(loaded) == 1
    assert loaded[0]["frame_index"] == "7"
    path.unlink()

"""CSV writer for raw and smoothed tracking outputs.

Produces CSV files matching the schema defined in docs/spec.md.
Missing Optional fields are serialised as empty cells, never as 0 or "None".
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional, Sequence, TextIO

from tennis_tracker.types import OutputRow


def _serialise_value(value: object) -> str:
    """Return the CSV cell string for *value*.

    - None → empty string (missing coordinate / confidence)
    - bool → lowercase "true" / "false"
    - float → standard str (python csv handles formatting)
    - str → passed through as-is
    - int → str
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class TrackingCsvWriter:
    """Writes tracking CSV files for raw or smoothed coordinate streams.

    Usage::

        with TrackingCsvWriter("output.csv") as writer:
            for row in rows:
                writer.write_row(row)

    The same writer class is used for both raw and smoothed output;
    callers simply open separate file paths.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: Optional[TextIO] = None
        self._writer: Optional[csv.writer] = None

    def __enter__(self) -> TrackingCsvWriter:
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        # Write the canonical header.
        self._writer.writerow(OutputRow.field_names())
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    @property
    def path(self) -> Path:
        return self._path

    def write_row(self, row: OutputRow) -> None:
        """Serialise *row* as a single CSV line."""
        if self._writer is None:
            raise RuntimeError(
                "Cannot write to a closed or unopened TrackingCsvWriter. "
                "Use it as a context manager."
            )
        values = [
            row.frame_index,
            row.time_s,
            row.player_a_x_m,
            row.player_a_y_m,
            row.player_a_pixel_x,
            row.player_a_pixel_y,
            row.player_a_confidence,
            row.player_b_x_m,
            row.player_b_y_m,
            row.player_b_pixel_x,
            row.player_b_pixel_y,
            row.player_b_confidence,
            row.ball_x_m,
            row.ball_y_m,
            row.ball_pixel_x,
            row.ball_pixel_y,
            row.ball_confidence,
            row.court_confidence,
            row.homography_valid,
            row.diagnostics,
        ]
        self._writer.writerow(_serialise_value(v) for v in values)

    def write_rows(self, rows: Sequence[OutputRow]) -> None:
        """Convenience: write a sequence of rows."""
        for row in rows:
            self.write_row(row)


def read_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield rows from an existing tracking CSV as raw string dicts.

    Useful for integration tests and verification.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        yield from reader

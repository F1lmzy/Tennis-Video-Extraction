"""Diagnostics for the tennis-tracking pipeline.

Diagnostics are compact, serialisable flag-bundles that accompany
every output row.  They make it possible to trust or reject a frame
without inspecting individual coordinates.
"""

from __future__ import annotations

from typing import Optional


VALID_FLAGS: frozenset[str] = frozenset({
    "missing_ball",
    "low_ball_confidence",
    "missing_player_a",
    "low_player_a_confidence",
    "missing_player_b",
    "low_player_b_confidence",
    "low_court_confidence",
    "homography_invalid",
    "ambigous_player_id",
    "interpolated_ball",
    "interpolated_player_a",
    "interpolated_player_b",
    "unsynchronized_frame",
})


class Diagnostics:
    """Mutable set of diagnostic flags for one output row.

    Example usage::

        diag = Diagnostics()
        diag.add_flag("missing_ball")
        diag.add_flag("low_court_confidence")
        print(diag.to_string())   # "missing_ball;low_court_confidence"
    """

    def __init__(self, initial: Optional[str] = None) -> None:
        self._flags: set[str] = set()
        if initial:
            self._flags.update(self._parse(initial))

    # --- mutators -------------------------------------------------------

    def add_flag(self, flag: str) -> Diagnostics:
        """Add a single flag.  Unknown flags are allowed as callers may
        introduce domain-specific flags at any level."""
        self._flags.add(flag)
        return self

    def merge(self, other: Diagnostics) -> Diagnostics:
        """Merge flags from *other* into this instance."""
        self._flags.update(other._flags)
        return self

    def clear(self) -> Diagnostics:
        """Remove all flags."""
        self._flags.clear()
        return self

    # --- queries --------------------------------------------------------

    def has_flag(self, flag: str) -> bool:
        """Return True if *flag* is set."""
        return flag in self._flags

    @property
    def is_empty(self) -> bool:
        """Return True when no flags are set."""
        return len(self._flags) == 0

    def __bool__(self) -> bool:
        return not self.is_empty

    def __contains__(self, flag: str) -> bool:
        return self.has_flag(flag)

    # --- serialisation --------------------------------------------------

    def to_string(self) -> str:
        """Serialise to a compact semicolon-separated string.

        Empty diagnostics produce an empty string so the CSV cell
        is empty rather than containing a meaningless token.
        """
        return ";".join(sorted(self._flags))

    @classmethod
    def from_string(cls, s: str) -> Diagnostics:
        """Deserialise from a compact semicolon-separated string."""
        return cls(initial=s)

    @staticmethod
    def _parse(s: str) -> set[str]:
        parts = s.split(";")
        return {p.strip() for p in parts if p.strip()}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Diagnostics):
            return NotImplemented
        return self._flags == other._flags

    def __repr__(self) -> str:
        flags = ";".join(sorted(self._flags)) if self._flags else "(empty)"
        return f"Diagnostics({flags})"

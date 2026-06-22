"""Tests for the diagnostics module.

Covers:
- Flag add / query / clear.
- Compact semicolon-separated serialisation.
- Round-trip from_string → to_string.
- Empty diagnostics edge case.
- Missing-detection representation using types module.
- Multiple flags.
"""

from __future__ import annotations

from tennis_tracker.diagnostics import VALID_FLAGS, Diagnostics
from tennis_tracker.types import CourtPoint, OutputRow, PixelPoint, TrackKind, TrackRow


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_empty_diagnostics_produce_empty_string() -> None:
    d = Diagnostics()
    assert d.to_string() == ""
    assert d.is_empty
    assert not d


def test_single_flag_serialises() -> None:
    d = Diagnostics().add_flag("missing_ball")
    assert d.to_string() == "missing_ball"
    assert not d.is_empty
    assert d.has_flag("missing_ball")


def test_multiple_flags_serialise_semicolon_separated() -> None:
    d = Diagnostics()
    d.add_flag("missing_ball").add_flag("low_court_confidence")
    serialized = d.to_string()
    parts = serialized.split(";")
    assert "missing_ball" in parts
    assert "low_court_confidence" in parts
    assert len(parts) == 2


def test_flags_are_sorted_in_output() -> None:
    d = Diagnostics()
    d.add_flag("low_court_confidence").add_flag("missing_ball")
    assert d.to_string() == "low_court_confidence;missing_ball"


def test_round_trip_parse() -> None:
    original = Diagnostics()
    original.add_flag("missing_ball").add_flag("homography_invalid")
    serialized = original.to_string()
    restored = Diagnostics.from_string(serialized)
    assert restored == original
    assert restored.to_string() == serialized


def test_parse_from_existing_string() -> None:
    d = Diagnostics(initial="missing_ball;low_court_confidence")
    assert d.has_flag("missing_ball")
    assert d.has_flag("low_court_confidence")
    assert not d.has_flag("homography_invalid")


def test_clear_flags() -> None:
    d = Diagnostics().add_flag("missing_ball")
    assert not d.is_empty
    d.clear()
    assert d.is_empty
    assert d.to_string() == ""


def test_contains_operator() -> None:
    d = Diagnostics().add_flag("missing_ball")
    assert "missing_ball" in d
    assert "homography_invalid" not in d


def test_merge() -> None:
    a = Diagnostics().add_flag("missing_ball")
    b = Diagnostics().add_flag("low_court_confidence")
    a.merge(b)
    assert a.has_flag("missing_ball")
    assert a.has_flag("low_court_confidence")


def test_valid_flags_contains_expected_entries() -> None:
    assert "missing_ball" in VALID_FLAGS
    assert "homography_invalid" in VALID_FLAGS
    assert "low_court_confidence" in VALID_FLAGS
    assert "ambigous_player_id" in VALID_FLAGS
    assert len(VALID_FLAGS) >= 10


# ---------------------------------------------------------------------------
# Missing-detection representation (Optional, never zero)
# ---------------------------------------------------------------------------


def test_missing_player_coordinates_are_none_not_zero() -> None:
    """Confirm that missing player fields are None, not 0.0."""
    row = OutputRow(
        frame_index=0,
        time_s=0.0,
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
    assert row.player_a_x_m is None
    assert row.player_a_y_m is None
    assert row.player_b_x_m is None
    assert row.player_b_confidence is None
    assert row.ball_x_m is None
    assert row.ball_y_m is None


def test_track_row_has_court_position_reflects_optional() -> None:
    row_with = TrackRow(
        frame_index=1,
        time_s=0.033,
        kind=TrackKind.BALL,
        pixel=PixelPoint(x=320, y=240),
        confidence=0.95,
        court=CourtPoint(x_m=1.5, y_m=5.0, confidence=0.9),
    )
    assert row_with.has_court_position is True

    row_without = TrackRow(
        frame_index=1,
        time_s=0.033,
        kind=TrackKind.BALL,
        pixel=PixelPoint(x=320, y=240),
        confidence=0.95,
        court=None,
    )
    assert row_without.has_court_position is False


def test_court_point_with_confidence() -> None:
    cp = CourtPoint(x_m=2.0, y_m=-3.5, confidence=0.85)
    assert cp.x_m == 2.0
    assert cp.y_m == -3.5
    assert cp.confidence == 0.85


def test_detection_kind_values() -> None:
    assert TrackKind.PLAYER_A.value == "player_a"
    assert TrackKind.PLAYER_B.value == "player_b"
    assert TrackKind.BALL.value == "ball"


def test_output_row_csv_header_matches_spec() -> None:
    expected = (
        "frame_index,time_s,"
        "player_a_x_m,player_a_y_m,"
        "player_a_pixel_x,player_a_pixel_y,"
        "player_a_confidence,"
        "player_b_x_m,player_b_y_m,"
        "player_b_pixel_x,player_b_pixel_y,"
        "player_b_confidence,"
        "ball_x_m,ball_y_m,"
        "ball_pixel_x,ball_pixel_y,"
        "ball_confidence,"
        "court_confidence,homography_valid,diagnostics"
    )
    assert OutputRow.csv_header() == expected

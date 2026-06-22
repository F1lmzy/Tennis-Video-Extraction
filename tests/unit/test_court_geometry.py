"""Tests for court geometry constants and coordinate convention."""

from tennis_tracker.court import (
    ALL_NAMED_POINTS,
    CENTER_MARK_FAR,
    CENTER_MARK_NEAR,
    DOUBLES_FAR_LEFT,
    DOUBLES_FAR_RIGHT,
    DOUBLES_LENGTH,
    DOUBLES_NEAR_LEFT,
    DOUBLES_NEAR_RIGHT,
    DOUBLES_WIDTH,
    NET_CENTER,
    NET_LEFT,
    NET_RIGHT,
    SERVICE_FAR_CENTER,
    SERVICE_FAR_LEFT,
    SERVICE_FAR_RIGHT,
    SERVICE_LINE_DISTANCE,
    SERVICE_NEAR_CENTER,
    SERVICE_NEAR_LEFT,
    SERVICE_NEAR_RIGHT,
    SINGLES_FAR_LEFT,
    SINGLES_FAR_RIGHT,
    SINGLES_NEAR_LEFT,
    SINGLES_NEAR_RIGHT,
    SINGLES_WIDTH,
    named_point_by_label,
)


class TestDimensions:
    def test_doubles_width(self) -> None:
        assert DOUBLES_WIDTH == 10.97

    def test_doubles_length(self) -> None:
        assert DOUBLES_LENGTH == 23.77

    def test_singles_width(self) -> None:
        assert SINGLES_WIDTH == 8.23

    def test_service_line_distance(self) -> None:
        assert SERVICE_LINE_DISTANCE == 6.40


class TestOrigin:
    """Origin must be at the center of the full doubles court."""

    def test_net_center_is_origin(self) -> None:
        assert NET_CENTER.x_m == 0.0
        assert NET_CENTER.y_m == 0.0

    def test_doubles_corners_are_symmetric(self) -> None:
        # Near corners have negative Y; far corners have positive Y.
        assert DOUBLES_NEAR_LEFT.y_m < 0
        assert DOUBLES_FAR_LEFT.y_m > 0
        # Left corners have negative X; right corners have positive X.
        assert DOUBLES_NEAR_LEFT.x_m < 0
        assert DOUBLES_NEAR_RIGHT.x_m > 0
        # X symmetry: left = -right
        assert DOUBLES_NEAR_LEFT.x_m == -DOUBLES_NEAR_RIGHT.x_m
        assert DOUBLES_FAR_LEFT.x_m == -DOUBLES_FAR_RIGHT.x_m
        # Y symmetry: near = -far
        assert DOUBLES_NEAR_LEFT.y_m == -DOUBLES_FAR_LEFT.y_m

    def test_center_marks_at_y_extents(self) -> None:
        assert CENTER_MARK_NEAR.y_m == -DOUBLES_LENGTH / 2.0
        assert CENTER_MARK_FAR.y_m == DOUBLES_LENGTH / 2.0
        assert CENTER_MARK_NEAR.x_m == 0.0
        assert CENTER_MARK_FAR.x_m == 0.0


class TestAxisOrientation:
    """Verify +X right and +Y far-baseline convention."""

    def test_positive_x_points_right(self) -> None:
        # Right corners have larger X than left corners.
        assert DOUBLES_NEAR_RIGHT.x_m > DOUBLES_NEAR_LEFT.x_m
        assert DOUBLES_FAR_RIGHT.x_m > DOUBLES_FAR_LEFT.x_m
        # Right side is positive, left side is negative.
        assert DOUBLES_NEAR_RIGHT.x_m > 0
        assert DOUBLES_NEAR_LEFT.x_m < 0

    def test_positive_y_points_to_far_baseline(self) -> None:
        # Far points have larger Y than near points.
        assert DOUBLES_FAR_LEFT.y_m > DOUBLES_NEAR_LEFT.y_m
        assert DOUBLES_FAR_RIGHT.y_m > DOUBLES_NEAR_RIGHT.y_m
        # Far side is positive Y, near side is negative Y.
        assert DOUBLES_FAR_LEFT.y_m > 0
        assert DOUBLES_NEAR_LEFT.y_m < 0


class TestNamedKeypoints:
    """Verify that named court points have expected coordinates."""

    def test_doubles_corner_coordinates(self) -> None:
        half_w = DOUBLES_WIDTH / 2.0   # 5.485
        half_l = DOUBLES_LENGTH / 2.0  # 11.885
        assert DOUBLES_NEAR_LEFT.x_m == -half_w
        assert DOUBLES_NEAR_LEFT.y_m == -half_l
        assert DOUBLES_NEAR_RIGHT.x_m == half_w
        assert DOUBLES_NEAR_RIGHT.y_m == -half_l
        assert DOUBLES_FAR_LEFT.x_m == -half_w
        assert DOUBLES_FAR_LEFT.y_m == half_l
        assert DOUBLES_FAR_RIGHT.x_m == half_w
        assert DOUBLES_FAR_RIGHT.y_m == half_l

    def test_singles_corner_coordinates(self) -> None:
        half_s = SINGLES_WIDTH / 2.0  # 4.115
        half_l = DOUBLES_LENGTH / 2.0  # 11.885
        assert SINGLES_NEAR_LEFT.x_m == -half_s
        assert SINGLES_NEAR_LEFT.y_m == -half_l
        assert SINGLES_NEAR_RIGHT.x_m == half_s
        assert SINGLES_NEAR_RIGHT.y_m == -half_l
        assert SINGLES_FAR_LEFT.x_m == -half_s
        assert SINGLES_FAR_LEFT.y_m == half_l
        assert SINGLES_FAR_RIGHT.x_m == half_s
        assert SINGLES_FAR_RIGHT.y_m == half_l

    def test_net_points(self) -> None:
        half_w = DOUBLES_WIDTH / 2.0
        assert NET_CENTER.x_m == 0.0
        assert NET_CENTER.y_m == 0.0
        assert NET_LEFT.x_m == -half_w
        assert NET_LEFT.y_m == 0.0
        assert NET_RIGHT.x_m == half_w
        assert NET_RIGHT.y_m == 0.0

    def test_service_line_points(self) -> None:
        half_s = SINGLES_WIDTH / 2.0
        d = SERVICE_LINE_DISTANCE

        # Near service line
        assert SERVICE_NEAR_LEFT.x_m == -half_s
        assert SERVICE_NEAR_LEFT.y_m == -d
        assert SERVICE_NEAR_CENTER.x_m == 0.0
        assert SERVICE_NEAR_CENTER.y_m == -d
        assert SERVICE_NEAR_RIGHT.x_m == half_s
        assert SERVICE_NEAR_RIGHT.y_m == -d

        # Far service line
        assert SERVICE_FAR_LEFT.x_m == -half_s
        assert SERVICE_FAR_LEFT.y_m == d
        assert SERVICE_FAR_CENTER.x_m == 0.0
        assert SERVICE_FAR_CENTER.y_m == d
        assert SERVICE_FAR_RIGHT.x_m == half_s
        assert SERVICE_FAR_RIGHT.y_m == d

    def test_center_marks(self) -> None:
        half_l = DOUBLES_LENGTH / 2.0
        assert CENTER_MARK_NEAR.x_m == 0.0
        assert CENTER_MARK_NEAR.y_m == -half_l
        assert CENTER_MARK_FAR.x_m == 0.0
        assert CENTER_MARK_FAR.y_m == half_l


class TestNamedPointLookup:
    def test_lookup_by_label(self) -> None:
        pt = named_point_by_label("doubles_near_left")
        assert pt is not None
        assert pt.x_m == DOUBLES_NEAR_LEFT.x_m
        assert pt.y_m == DOUBLES_NEAR_LEFT.y_m

    def test_lookup_returns_none_for_unknown_label(self) -> None:
        assert named_point_by_label("nonexistent") is None

    def test_all_named_points_have_unique_labels(self) -> None:
        labels = [pt.label for pt in ALL_NAMED_POINTS]
        assert len(labels) == len(set(labels))

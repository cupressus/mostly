import pytest

from src.mostly.membership_functions.crisp import MFCrisp


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-1.0, 0.0, id="default-open-right-oob-left"),
        pytest.param(0.0, 1.0, id="default-open-right-left-edge-included"),
        pytest.param(4.999, 1.0, id="default-open-right-interior"),
        pytest.param(5.0, 0.0, id="default-open-right-right-edge-excluded"),
        pytest.param(6.0, 0.0, id="default-open-right-oob-right"),
    ],
)
def test_default_left_closed_right_open_membership(input: float, expected: float) -> None:
    """Default MFCrisp should model [left, right)."""
    mf = MFCrisp(left=0.0, right=5.0)
    assert mf(input) == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(0.0, 1.0, id="closed-left-edge"),
        pytest.param(5.0, 1.0, id="closed-right-edge"),
    ],
)
def test_closed_interval_includes_both_edges(input: float, expected: float) -> None:
    """Closed interval [left, right] should include both boundaries."""
    mf = MFCrisp(left=0.0, right=5.0, include_left=True, include_right=True)
    assert mf(input) == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(0.0, 0.0, id="open-left-excludes-left-edge"),
        pytest.param(5.0, 1.0, id="open-left-includes-right-edge"),
    ],
)
def test_left_open_right_closed_membership(input: float, expected: float) -> None:
    """Interval (left, right] should exclude left and include right boundary."""
    mf = MFCrisp(left=0.0, right=5.0, include_left=False, include_right=True)
    assert mf(input) == expected


def test_invalid_bounds_left_greater_than_right() -> None:
    """Model should reject inverted crisp intervals."""
    with pytest.raises(ValueError, match="Crisp bounds must satisfy left ≤ right"):
        MFCrisp(left=5.0, right=0.0)


def test_degenerate_interval_requires_both_bounds_inclusive() -> None:
    """Point intervals are only valid as [x, x]."""
    with pytest.raises(
        ValueError,
        match="When left == right, both include_left and include_right must be True",
    ):
        MFCrisp(left=3.0, right=3.0, include_left=True, include_right=False)


def test_degenerate_closed_point_interval() -> None:
    """Closed point interval [x, x] should only include x."""
    mf = MFCrisp(left=3.0, right=3.0, include_left=True, include_right=True)
    assert mf(3.0) == 1.0
    assert mf(2.999) == 0.0
    assert mf(3.001) == 0.0

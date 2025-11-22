import pytest

from src.mostly.membership_funs.trapezoid import MFTrapezoid

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-100, 0, id="oob left"),
        pytest.param(0, 0, id="left edge"),
        pytest.param(2, 0.5, id="left middle"),
        pytest.param(4, 1, id="center edge l"),
        pytest.param(5, 1, id="center middle"),
        pytest.param(6, 1, id="center edge r"),
        pytest.param(8, 0.5, id="right middle"),
        pytest.param(10, 0, id="right edge"),
        pytest.param(100, 0, id="oob right"),
    ],
)
def test_regular_membership(regular_trapezoidal_mf: MFTrapezoid, input, expected) -> None:
    """Test Regular Trapezoidal Membership Function."""
    assert regular_trapezoidal_mf.shape == "regular"
    assert regular_trapezoidal_mf(input) == expected
    assert regular_trapezoidal_mf.support() == (0, 10)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-1, 1, id="left - oob neg"),
        pytest.param(0, 1, id="left - full"),
        pytest.param(5, 0.5, id="left - center"),
        pytest.param(10, 0, id="left - right edge"),
        pytest.param(15, 0, id="left - oob pos"),
    ],
)
def test_left_shoulder_membership(left_trapezoidal_triangular_mf: MFTrapezoid, input: float, expected: float) -> None:
    """Test Shoulder Triangle Membership Function."""
    assert left_trapezoidal_triangular_mf.shape == "left"
    assert left_trapezoidal_triangular_mf(input) == expected
    assert left_trapezoidal_triangular_mf.support() == (0, 10)


@pytest.mark.parametrize(
    "input,expected",
    [
        # right shoulder
        pytest.param(-1, 0, id="right - oob neg"),
        pytest.param(0, 0, id="right - edge"),
        pytest.param(5, 0.5, id="right - center"),
        pytest.param(10, 1, id="right - full"),
        pytest.param(15, 1, id="right - oob"),
    ],
)
def test_right_shoulder_membership(right_trapezoidal_triangular_mf: MFTrapezoid, input: float, expected: float) -> None:
    """Test Shoulder Triangle Membership Function."""
    assert right_trapezoidal_triangular_mf.shape == "right"
    assert right_trapezoidal_triangular_mf(input) == expected
    assert right_trapezoidal_triangular_mf.support() == (0, 10)


# region NEGATIVE TESTS


def test_compliance_validation() -> None:
    """Test compliance method for invalid triangle configuration."""
    with pytest.raises(ValueError, match="Trapezoid points must satisfy a ≤ b ≤ c ≤ d"):
        MFTrapezoid(a=2.0, b=1.0, c=3.0, d=10)
    with pytest.raises(ValueError, match="Both shoulders cannot be equal;"):
        MFTrapezoid(a=0.0, b=0.0, c=3.0, d=3.0)
    with pytest.raises(ValueError, match="All points a, b, c, d cannot be equal"):
        MFTrapezoid(a=0.0, b=0.0, c=0.0, d=0.0)

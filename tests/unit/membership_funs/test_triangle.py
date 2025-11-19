import numpy as np
import pytest
from pydantic import ValidationError

from src.mostly.membership_funs.trapezoid import MFTrapezoid
from src.mostly.membership_funs.triangle import MFTriangle

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-100, 0, id="oob left"),
        pytest.param(0, 0, id="left edge"),
        pytest.param(2.5, 0.5, id="left middle"),
        pytest.param(5, 1, id="center edge"),
        pytest.param(7.5, 0.5, id="right middle"),
        pytest.param(10, 0, id="right edge"),
        pytest.param(100, 0, id="oob right"),
    ],
)
def test_regular_membership(
    regular_triangular_mf: MFTriangle, triangular_trapezoidal_mf: MFTrapezoid, input, expected
) -> None:
    """Test Regular Triangle and Trapezoidal Triangle Membership Function."""
    assert regular_triangular_mf.shape == "regular"
    assert regular_triangular_mf(input) == expected
    assert regular_triangular_mf.support() == (0, 10)

    assert triangular_trapezoidal_mf(input) == expected
    assert triangular_trapezoidal_mf.support() == (0, 10)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-1, 1, id="left - oob neg"),
        pytest.param(0, 1, id="left - full"),
        pytest.param(5, 0.5, id="left - center"),
        pytest.param(10, 0, id="left - right edge"),
        pytest.param(15, 0, id="left - oob"),
    ],
)
def test_left_shoulder_membership(left_triangular_mf: MFTriangle, input: float, expected: float) -> None:
    """Test Shoulder Triangle Membership Function."""
    assert left_triangular_mf.shape == "left"
    assert left_triangular_mf(input) == expected
    assert left_triangular_mf.support() == (0, 10)


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
def test_right_shoulder_membership(right_triangular_mf: MFTriangle, input: float, expected: float) -> None:
    """Test Shoulder Triangle Membership Function."""
    assert right_triangular_mf.shape == "right"
    assert right_triangular_mf(input) == expected
    assert right_triangular_mf.support() == (0, 10)


# region NEGATIVE TESTS


def test_compliance_validation() -> None:
    """Test compliance method for invalid triangle configuration."""
    with pytest.raises(ValueError, match="Triangle points must satisfy a ≤ b ≤ c"):
        MFTriangle(a=2.0, b=1.0, c=3.0)  # Invalid: a > b
    with pytest.raises(ValueError, match="Triangle points must satisfy a ≤ b ≤ c"):
        MFTriangle(a=1.0, b=3.0, c=2.0)  # Invalid: b > c
    with pytest.raises(ValueError, match="Triangle points must satisfy a ≤ b ≤ c"):
        MFTriangle(a=2.0, b=3.0, c=1.0)  # Invalid: b > c
    with pytest.raises(ValueError, match="All points a, b, c cannot be equal; not a valid triangle"):
        MFTriangle(a=0.0, b=0.0, c=0.0)  # All Equal


@pytest.mark.parametrize(
    "input",
    [
        # right shoulder
        pytest.param(None, id="None"),
        pytest.param("nan", id="nan"),
        pytest.param(np.nan, id="numpy nan"),
        pytest.param("", id="empty string"),
        pytest.param([], id="empty list"),
        pytest.param({}, id="empty dict"),
        pytest.param(set(), id="empty set"),
        pytest.param(np.inf, id="numpy inf"),
        pytest.param(-np.inf, id="numpy -inf"),
    ],
)
def test_invalid_input_type(regular_triangular_mf: MFTriangle, input) -> None:
    """Test invalid input type for membership function."""
    with pytest.raises(ValidationError):
        regular_triangular_mf(input)  # type: ignore

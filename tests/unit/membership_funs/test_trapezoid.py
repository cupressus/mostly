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
    """Test Regular Triangle Membership Function."""
    assert regular_trapezoidal_mf(input) == expected
    assert regular_trapezoidal_mf.support() == (0, 10)

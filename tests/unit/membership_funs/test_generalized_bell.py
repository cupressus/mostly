import pytest

from src.mostly.membership_funs.generalized_bell import MFGeneralizedBell

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(5, 1.0, id="center"),
        pytest.param(7, 0.5, id="right half-width"),
        pytest.param(3, 0.5, id="left half-width"),
        pytest.param(9, 1 / 257, id="far right"),
        pytest.param(1, 1 / 257, id="far left"),
    ],
)
def test_regular_generalized_bell(
    regular_generalized_bell_mf: MFGeneralizedBell, input, expected
) -> None:
    """Test Standard Generalized Bell Membership Function."""
    assert regular_generalized_bell_mf(input) == pytest.approx(expected, rel=1e-3)

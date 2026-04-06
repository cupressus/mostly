import pytest

from src.mostly.membership_functions.gaussian import MFGaussian

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-100, 0, id="oob left"),
        pytest.param(4, 0.607, id="left std"),
        pytest.param(5, 1, id="center edge"),
        pytest.param(6, 0.607, id="right std"),
        pytest.param(100, 0, id="oob right"),
    ],
)
def test_regular_gaussian(regular_gaussian_mf: MFGaussian, input, expected) -> None:
    """Test Standard Gaussian Membership Function."""
    assert regular_gaussian_mf(input) == pytest.approx(expected, rel=1e-3)
    # assert regular_gaussian_mf.support() == (0, 10)

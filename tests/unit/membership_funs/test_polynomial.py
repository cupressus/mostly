import pytest
from pydantic import ValidationError

from src.mostly.membership_functions.polynomial import MFPolynomialS, MFPolynomialZ


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(0.0, 1.0, id="left_oob"),
        pytest.param(2.0, 1.0, id="shoulder"),
        pytest.param(3.0, 0.875, id="descending_first_half"),
        pytest.param(4.0, 0.5, id="midpoint"),
        pytest.param(5.0, 0.125, id="descending_second_half"),
        pytest.param(6.0, 0.0, id="foot"),
        pytest.param(10.0, 0.0, id="right_oob"),
    ],
)
def test_polynomial_z_membership(input: float, expected: float) -> None:
    """Test Z-shaped polynomial membership values across all regions."""
    mf = MFPolynomialZ(shoulder=2.0, foot=6.0)
    assert mf(input) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-1.0, 0.0, id="left_oob"),
        pytest.param(2.0, 0.0, id="foot"),
        pytest.param(3.0, 0.125, id="ascending_first_half"),
        pytest.param(4.0, 0.5, id="midpoint"),
        pytest.param(5.0, 0.875, id="ascending_second_half"),
        pytest.param(6.0, 1.0, id="shoulder"),
        pytest.param(20.0, 1.0, id="right_oob"),
    ],
)
def test_polynomial_s_membership(input: float, expected: float) -> None:
    """Test S-shaped polynomial membership values across all regions."""
    mf = MFPolynomialS(shoulder=6.0, foot=2.0)
    assert mf(input) == pytest.approx(expected, rel=1e-6)


def test_polynomial_z_invalid_order_raises_validation_error() -> None:
    """Test Z polynomial requires shoulder < foot."""
    with pytest.raises(ValueError, match="shoulder must be less than foot"):
        MFPolynomialZ(shoulder=6.0, foot=2.0)


def test_polynomial_s_invalid_order_raises_validation_error() -> None:
    """Test S polynomial requires foot < shoulder."""
    with pytest.raises(ValueError, match="foot must be less than shoulder"):
        MFPolynomialS(shoulder=2.0, foot=6.0)


@pytest.mark.parametrize("mf", [MFPolynomialZ(shoulder=2.0, foot=6.0), MFPolynomialS(shoulder=6.0, foot=2.0)])
def test_polynomial_invalid_call_input_raises_validation_error(mf) -> None:
    """Test invalid input type for polynomial call."""
    with pytest.raises(ValidationError):
        mf(None)  # type: ignore

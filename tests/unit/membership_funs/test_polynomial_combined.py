import pytest
from pydantic import ValidationError

from src.mostly.membership_functions.polynomial import MFPolynomialS, MFPolynomialZ
from src.mostly.membership_functions.polynomial_combined import MFPolynomialCombined


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-2.0, 0.0, id="left_oob"),
        pytest.param(2.0, 0.5, id="left_ramp"),
        pytest.param(4.0, 1.0, id="left_shoulder"),
        pytest.param(5.0, 1.0, id="plateau"),
        pytest.param(6.0, 1.0, id="right_shoulder"),
        pytest.param(8.0, 0.5, id="right_ramp"),
        pytest.param(10.0, 0.0, id="right_foot"),
        pytest.param(12.0, 0.0, id="right_oob"),
    ],
)
def test_polynomial_combined_membership(input: float, expected: float) -> None:
    """Test combined polynomial behavior across left, plateau, and right regions."""
    left = MFPolynomialS(shoulder=4.0, foot=0.0)
    right = MFPolynomialZ(shoulder=6.0, foot=10.0)
    mf = MFPolynomialCombined(left=left, right=right)

    assert mf(input) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("input", [pytest.param(-1.0, id="left_side"), pytest.param(11.0, id="right_side")])
def test_polynomial_combined_delegates_to_side_polynomials(input: float) -> None:
    """Test combined function delegates outside plateau to left/right components."""
    left = MFPolynomialS(shoulder=4.0, foot=0.0)
    right = MFPolynomialZ(shoulder=6.0, foot=10.0)
    mf = MFPolynomialCombined(left=left, right=right)

    if input < left.shoulder:
        assert mf(input) == pytest.approx(left(input), rel=1e-9)
    else:
        assert mf(input) == pytest.approx(right(input), rel=1e-9)


def test_polynomial_combined_invalid_shoulder_order_raises_validation_error() -> None:
    """Test combined polynomial requires left shoulder <= right shoulder."""
    left = MFPolynomialS(shoulder=8.0, foot=0.0)
    right = MFPolynomialZ(shoulder=6.0, foot=10.0)

    with pytest.raises(ValueError, match="Left shoulder must be less than right shoulder"):
        MFPolynomialCombined(left=left, right=right)


def test_polynomial_combined_invalid_call_input_raises_validation_error() -> None:
    """Test invalid input type for combined polynomial call."""
    left = MFPolynomialS(shoulder=4.0, foot=0.0)
    right = MFPolynomialZ(shoulder=6.0, foot=10.0)
    mf = MFPolynomialCombined(left=left, right=right)

    with pytest.raises(ValidationError):
        mf(None)  # type: ignore

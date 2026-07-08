from math import exp

import pytest
from pydantic import ValidationError

from src.mostly.membership_functions.sigmoidal import MFSigmoidal


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-10.0, 0.0, id="far_left"),
        pytest.param(0.0, 0.5, id="center"),
        pytest.param(10.0, 1.0, id="far_right"),
    ],
)
def test_regular_sigmoidal(input: float, expected: float) -> None:
    """Test standard increasing sigmoidal membership function."""
    mf = MFSigmoidal(a=1.0, c=0.0)
    assert mf(input) == pytest.approx(expected, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-10.0, 1.0, id="far_left"),
        pytest.param(0.0, 0.5, id="center"),
        pytest.param(10.0, 0.0, id="far_right"),
    ],
)
def test_negative_slope_sigmoidal(input: float, expected: float) -> None:
    """Test decreasing sigmoidal when slope is negative."""
    mf = MFSigmoidal(a=-1.0, c=0.0)
    assert mf(input) == pytest.approx(expected, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(3.0, 0.5, id="at_center"),
        pytest.param(4.0, 1.0 / (1.0 + exp(-2.0)), id="one_unit_right"),
        pytest.param(2.0, 1.0 / (1.0 + exp(2.0)), id="one_unit_left"),
    ],
)
def test_shifted_and_scaled_sigmoidal(input: float, expected: float) -> None:
    """Test sigmoidal with non-default center and slope."""
    mf = MFSigmoidal(a=2.0, c=3.0)
    assert mf(input) == pytest.approx(expected, rel=1e-6)


def test_invalid_call_input_raises_validation_error() -> None:
    """Test invalid input type for sigmoidal call."""
    mf = MFSigmoidal(a=1.0, c=0.0)
    with pytest.raises(ValidationError):
        mf(None)  # type: ignore

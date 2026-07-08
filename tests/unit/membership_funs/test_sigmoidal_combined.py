import pytest
from pydantic import ValidationError

from src.mostly.membership_functions.sigmoidal import MFSigmoidal
from src.mostly.membership_functions.sigmoidal_combined import MFSigmoidalCombined


@pytest.mark.parametrize(
    "input",
    [
        pytest.param(-10.0, id="far_left"),
        pytest.param(0.0, id="at_first_center"),
        pytest.param(2.5, id="between_centers"),
        pytest.param(10.0, id="far_right"),
    ],
)
def test_combined_sigmoidal_diff(input: float) -> None:
    """Test diff method against component-wise subtraction."""
    mf1 = MFSigmoidal(a=2.0, c=0.0)
    mf2 = MFSigmoidal(a=2.0, c=5.0)
    combined = MFSigmoidalCombined(mfs=(mf1, mf2), method="diff")

    assert combined(input) == pytest.approx(mf1(input) - mf2(input), rel=1e-9)


@pytest.mark.parametrize(
    "input",
    [
        pytest.param(-10.0, id="far_left"),
        pytest.param(0.0, id="at_first_center"),
        pytest.param(2.5, id="between_centers"),
        pytest.param(10.0, id="far_right"),
    ],
)
def test_combined_sigmoidal_multiply(input: float) -> None:
    """Test multiply method against component-wise product."""
    mf1 = MFSigmoidal(a=2.0, c=0.0)
    mf2 = MFSigmoidal(a=2.0, c=5.0)
    combined = MFSigmoidalCombined(mfs=(mf1, mf2), method="multiply")

    assert combined(input) == pytest.approx(mf1(input) * mf2(input), rel=1e-9)


def test_invalid_method_raises_validation_error() -> None:
    """Test invalid combination method is rejected by model validation."""
    mf1 = MFSigmoidal(a=1.0, c=0.0)
    mf2 = MFSigmoidal(a=1.0, c=2.0)

    with pytest.raises(ValidationError):
        MFSigmoidalCombined(mfs=(mf1, mf2), method="sum")  # type: ignore


def test_invalid_mfs_tuple_length_raises_validation_error() -> None:
    """Test tuple length validation for combined sigmoidal definition."""
    mf1 = MFSigmoidal(a=1.0, c=0.0)

    with pytest.raises(ValidationError):
        MFSigmoidalCombined(mfs=(mf1,), method="diff")  # type: ignore


def test_invalid_call_input_raises_validation_error() -> None:
    """Test invalid input type for combined sigmoidal call."""
    mf1 = MFSigmoidal(a=1.0, c=0.0)
    mf2 = MFSigmoidal(a=1.0, c=2.0)
    mf = MFSigmoidalCombined(mfs=(mf1, mf2), method="diff")

    with pytest.raises(ValidationError):
        mf(None)  # type: ignore

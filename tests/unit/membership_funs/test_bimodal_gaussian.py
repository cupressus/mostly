import pytest
from pydantic import ValidationError

from src.mostly.membership_funs.bimodal_gaussian import MFBimodalGaussian

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(-100, 0.0, id="way_left_oob"),
        pytest.param(1.0, 0.135, id="left_2std"),
        pytest.param(3.0, 1.0, id="left_mean"),
        pytest.param(5.0, 1.0, id="plateau_center"),
        pytest.param(7.0, 1.0, id="right_mean"),
        pytest.param(9.0, 0.135, id="right_2std"),
        pytest.param(100, 0.0, id="way_right_oob"),
    ],
)
def test_regular_bimodal_gaussian(regular_bimodal_gaussian_mf: MFBimodalGaussian, input, expected) -> None:
    """Test ordered Bimodal Gaussian (plateau case)."""
    assert regular_bimodal_gaussian_mf(input) == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(0.0, 2.54e-13, id="far_left"),
        pytest.param(3.0, 0.000335, id="right_mean_inv"),
        pytest.param(5.0, 0.0183, id="between_means"),
        pytest.param(7.0, 0.000335, id="left_mean_inv"),
        pytest.param(10.0, 2.54e-13, id="far_right"),
    ],
)
def test_inverted_bimodal_gaussian(inverted_bimodal_gaussian_mf: MFBimodalGaussian, input, expected) -> None:
    """Test inverted Bimodal Gaussian (product case)."""
    assert inverted_bimodal_gaussian_mf(input) == pytest.approx(expected, rel=1e-2)


# region EDGE CASE TESTS


def test_equal_means() -> None:
    """Test bimodal Gaussian with equal means (plateau at single point)."""
    mf = MFBimodalGaussian(left_mean=5.0, left_sigma=1.0, right_mean=5.0, right_sigma=1.0)

    # At the mean, should be 1.0
    assert mf(5.0) == pytest.approx(1.0, rel=1e-3)

    # Outside, should decay as Gaussian
    assert mf(4.0) == pytest.approx(0.607, rel=1e-2)
    assert mf(6.0) == pytest.approx(0.607, rel=1e-2)


def test_asymmetric_sigmas() -> None:
    """Test bimodal Gaussian with different left and right sigmas."""
    mf = MFBimodalGaussian(left_mean=3.0, left_sigma=0.5, right_mean=7.0, right_sigma=2.0)

    # Plateau should still be 1.0
    assert mf(5.0) == pytest.approx(1.0, rel=1e-3)

    # Left tail should decay faster (smaller sigma)
    # 1 std left of left_mean (x=2.5) with sigma=0.5
    assert mf(2.5) == pytest.approx(0.607, rel=1e-2)

    # Right tail should decay slower (larger sigma)
    # 1 std right of right_mean (x=9.0) with sigma=2.0
    assert mf(9.0) == pytest.approx(0.607, rel=1e-2)


# region VALIDATION TESTS


def test_negative_left_sigma() -> None:
    """Test that negative left_sigma raises ValidationError."""
    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=3.0, left_sigma=-1.0, right_mean=7.0, right_sigma=1.0)


def test_zero_left_sigma() -> None:
    """Test that zero left_sigma raises ValidationError."""
    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=3.0, left_sigma=0.0, right_mean=7.0, right_sigma=1.0)


def test_negative_right_sigma() -> None:
    """Test that negative right_sigma raises ValidationError."""
    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=3.0, left_sigma=1.0, right_mean=7.0, right_sigma=-1.0)


def test_zero_right_sigma() -> None:
    """Test that zero right_sigma raises ValidationError."""
    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=3.0, left_sigma=1.0, right_mean=7.0, right_sigma=0.0)


def test_non_finite_means() -> None:
    """Test that non-finite means raise ValidationError."""
    import numpy as np

    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=np.nan, left_sigma=1.0, right_mean=7.0, right_sigma=1.0)

    with pytest.raises(ValidationError):
        MFBimodalGaussian(left_mean=3.0, left_sigma=1.0, right_mean=np.inf, right_sigma=1.0)

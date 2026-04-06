import pytest

from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.membership_functions.base import MembershipFunction
from src.mostly.membership_functions.bimodal_gaussian import MFBimodalGaussian
from src.mostly.membership_functions.gaussian import MFGaussian
from src.mostly.membership_functions.generalized_bell import MFGeneralizedBell
from src.mostly.membership_functions.trapezoidal import MFTrapezoidal
from src.mostly.membership_functions.triangle import MFTriangular


# region FIXTURES BASE MF
class DummyMF(MembershipFunction):
    """Minimal concrete implementation for interface verification."""

    offset: float = 0.5

    def __call__(self, x: float) -> float:  # runtime type checks are not enforced
        """Calculate degree of Membership for a given input `x`."""
        return float(x + self.offset)

    def support(self) -> tuple[float, float]:
        """Provide 0 Cutoffs."""
        return (-1.0, 1.0)


@pytest.fixture
def dummy_mf() -> DummyMF:
    """Fixture that returns a DummyMF instance for FuzzySet tests."""
    return DummyMF()


# region FIXTURES TRIANGULAR MF


@pytest.fixture
def regular_triangular_mf() -> MFTriangular:
    """Fixture that returns a regular triangular membership function."""
    return MFTriangular(a=0, b=5, c=10)


@pytest.fixture
def left_triangular_mf() -> MFTriangular:
    """Fixture that returns a left-shoulder triangular membership function."""
    return MFTriangular(a=0, b=0, c=10)


@pytest.fixture
def right_triangular_mf() -> MFTriangular:
    """Fixture that returns a right-shoulder triangular membership function."""
    return MFTriangular(a=0, b=10, c=10)


# region FIXTURES TRAPEZOIDAL MF


@pytest.fixture
def regular_trapezoidal_mf() -> MFTrapezoidal:
    """Fixture that returns a regular trapezoidal membership function."""
    return MFTrapezoidal(a=0, b=4, c=6, d=10)


@pytest.fixture
def triangular_trapezoidal_mf() -> MFTrapezoidal:
    """Fixture that returns a trapezoidal membership function forced to be triangular."""
    return MFTrapezoidal(a=0, b=5, c=5, d=10)


@pytest.fixture
def left_trapezoidal_triangular_mf() -> MFTrapezoidal:
    """Fixture that returns a left-shoulder trapezoidal membership function forced to be triangular."""
    return MFTrapezoidal(a=0, b=0, c=0, d=10)


@pytest.fixture
def right_trapezoidal_triangular_mf() -> MFTrapezoidal:
    """Fixture that returns a right-shoulder trapezoidal membership function forced to be triangular."""
    return MFTrapezoidal(a=0, b=10, c=10, d=10)


# region FIXTURES GAUSSIAN MF
@pytest.fixture
def regular_gaussian_mf() -> "MFGaussian":
    """Fixture that returns a standard Gaussian membership function."""
    return MFGaussian(mean=5.0, sigma=1.0)


# region FIXTURES BIMODAL GAUSSIAN MF
@pytest.fixture
def regular_bimodal_gaussian_mf() -> "MFBimodalGaussian":
    """Fixture that returns a bimodal Gaussian with ordered means (plateau case)."""
    return MFBimodalGaussian(left_mean=3.0, left_sigma=1.0, right_mean=7.0, right_sigma=1.0)


@pytest.fixture
def inverted_bimodal_gaussian_mf() -> "MFBimodalGaussian":
    """Fixture that returns a bimodal Gaussian with inverted means (product case)."""
    return MFBimodalGaussian(left_mean=7.0, left_sigma=1.0, right_mean=3.0, right_sigma=1.0)


# region FIXTURES GENERALIZED BELL MF
@pytest.fixture
def regular_generalized_bell_mf() -> "MFGeneralizedBell":
    """Fixture that returns a standard Generalized Bell membership function."""
    return MFGeneralizedBell(width=2.0, slope=4.0, center=5.0)


# region FIXTURES LINGUISTIC VARIABLE
@pytest.fixture
def simple_linguistic_variable() -> "LinguisticVariable":
    """Fixture that returns a simple LinguisticVariable instance."""
    lv = LinguisticVariable(
        concept="temperature",
        uod=(0.0, 100.0),
        fuzzy_sets={
            "cold": MFTriangular(a=0.0, b=0.0, c=50.0),
            "warm": MFTriangular(a=25.0, b=50.0, c=75.0),
            "hot": MFTriangular(a=50.0, b=100.0, c=100.0),
        },
    )
    return lv

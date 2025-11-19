import pytest

from src.mostly.membership_funs.base import MembershipFunction
from src.mostly.membership_funs.trapezoid import MFTrapezoid
from src.mostly.membership_funs.triangle import MFTriangle


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
def regular_triangular_mf() -> MFTriangle:
    """Fixture that returns a regular triangular membership function."""
    return MFTriangle(a=0, b=5, c=10)


@pytest.fixture
def left_triangular_mf() -> MFTriangle:
    """Fixture that returns a left-shoulder triangular membership function."""
    return MFTriangle(a=0, b=0, c=10)


@pytest.fixture
def right_triangular_mf() -> MFTriangle:
    """Fixture that returns a right-shoulder triangular membership function."""
    return MFTriangle(a=0, b=10, c=10)


# region FIXTURES TRAPEZOIDAL MF


@pytest.fixture
def regular_trapezoidal_mf() -> MFTrapezoid:
    """Fixture that returns a regular trapezoidal membership function."""
    return MFTrapezoid(a=0, b=4, c=6, d=10)


@pytest.fixture
def triangular_trapezoidal_mf() -> MFTrapezoid:
    """Fixture that returns a trapezoidal membership function forced to be triangular."""
    return MFTrapezoid(a=0, b=5, c=5, d=10)

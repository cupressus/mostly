from pydantic import FiniteFloat, model_validator, validate_call

from .base import MembershipFunction
from .polynomial import MFPolynomialS, MFPolynomialZ


class MFPolynomialCombined(MembershipFunction):
    """Combined Polynomial Membership Function.

    A fuzzy membership function defined by the combination of two polynomials.

    Parameters
    ----------
    left : MFPolynomialS
        The left polynomial, which must be of the 'right_open' variant.
    right : MFPolynomialZ
        The right polynomial, which must be of the 'left_open' variant.

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    left: MFPolynomialS
    right: MFPolynomialZ

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialCombined":
        """Validate model for correct Combined Polynomial."""
        if self.left.shoulder > self.right.shoulder:
            raise ValueError("Left shoulder must be less than right shoulder")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match x:
            case x if self.left.shoulder <= x <= self.right.shoulder:
                return 1.0
            case x if x < self.left.shoulder:
                return self.left(x)
            case x if x > self.right.shoulder:
                return self.right(x)
            case _:
                return 0.0

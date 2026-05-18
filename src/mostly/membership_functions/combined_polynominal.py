from pydantic import FiniteFloat, model_validator, validate_call

from .base import MembershipFunction
from .polynomial import MFPolynomial


class MFCombinedPolynomial(MembershipFunction):
    """Combined Polynomial Membership Function.

    A fuzzy membership function defined by the combination of two polynomials.

    Parameters
    ----------
    left : MFPolynomial
        The left polynomial, which must be of the 'right_open' variant.
    right : MFPolynomial
        The right polynomial, which must be of the 'left_open' variant.

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    left: MFPolynomial
    right: MFPolynomial

    @model_validator(mode="after")
    def compliance(self) -> "MFCombinedPolynomial":
        """Validate model for correct Combined Polynomial."""
        if self.left.variant != "right_open":
            raise ValueError("Left polynomial must be of 'right_open' variant")
        if self.right.variant != "left_open":
            raise ValueError("Right polynomial must be of 'left_open' variant")
        if self.left.shoulder >= self.right.shoulder:
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

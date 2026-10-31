from pydantic import FiniteFloat, model_validator, validate_call

from .base import MembershipFunction
from .polynomial import MFPolynomialS, MFPolynomialZ


class MFPolynomialCombined(MembershipFunction):
    """Combine rising and falling polynomials around a plateau of 1.0.

    The left shoulder must not exceed the right shoulder; equal shoulders
    are allowed. Outside the plateau, evaluation delegates to the
    corresponding polynomial.

    Attributes:
        left: Rising S-shaped polynomial.
        right: Falling Z-shaped polynomial.

    Raises:
        ValidationError: If the polynomials are invalid or their shoulders
            are in reverse order.

    Examples:
        >>> from mostly.membership_functions import MFPolynomialCombined, MFPolynomialS, MFPolynomialZ
        >>> rising = MFPolynomialS(foot=0.0, shoulder=1.0)
        >>> falling = MFPolynomialZ(shoulder=1.0, foot=2.0)
        >>> MFPolynomialCombined(left=rising, right=falling)(1.0)
        1.0

    """

    left: MFPolynomialS
    right: MFPolynomialZ

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialCombined":
        """Validate the ordering of the component shoulders."""
        if self.left.shoulder > self.right.shoulder:
            raise ValueError("Left shoulder must be less than right shoulder")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the plateau or the appropriate component polynomial."""
        match x:
            case x if self.left.shoulder <= x <= self.right.shoulder:
                return 1.0
            case x if x < self.left.shoulder:
                return self.left(x)
            case x if x > self.right.shoulder:
                return self.right(x)
            case _:
                return 0.0

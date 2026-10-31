from math import exp

from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction


class MFSigmoidal(MembershipFunction):
    """A logistic sigmoidal membership function.

    In ordinary numeric ranges, a positive slope rises, a negative slope falls,
    and a zero slope is constant at 0.5. Membership is 0.5 at the center.

    Attributes:
        a: Signed slope of the sigmoid.
        c: Center of the sigmoid.

    Raises:
        ValidationError: If either field is non-finite.

    Examples:
        >>> from mostly.membership_functions import MFSigmoidal
        >>> MFSigmoidal(a=1.0, c=0.0)(0.0)
        0.5

    """

    a: FiniteFloat
    c: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the sigmoid at a finite scalar.

        Extreme finite inputs or fields can produce `NaN` if intermediate
        arithmetic overflows.

        Raises:
            OverflowError: If the exponential overflows for an extreme input.

        """
        z = self.a * (x - self.c)
        return 1.0 / (1.0 + exp(-z))

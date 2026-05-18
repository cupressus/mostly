from math import exp

from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction


class MFSigmoidal(MembershipFunction):
    """Sigmoidal Membership Function.

    A fuzzy membership function defined by a sigmoidal curve.

    Parameters
    ----------
    a : FiniteFloat
        Slope of the sigmoid
    c : FiniteFloat
        Center of the sigmoid

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    a: FiniteFloat
    c: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        z = self.a * (x - self.c)
        return 1.0 / (1.0 + exp(-z))

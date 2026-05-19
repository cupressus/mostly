from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction


class MFCrisp(MembershipFunction):
    """Crisp Membership Function.

    A fuzzy membership function defined by a crisp (binary) distribution.

    Parameters
    ----------
    left : FiniteFloat
        Left boundary of the crisp membership function
    right : FiniteFloat
        Right boundary of the crisp membership function

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    left: FiniteFloat
    right: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        if self.left < x < self.right:
            return 1.0
        else:
            return 0.0

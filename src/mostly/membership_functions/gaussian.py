from math import exp

from pydantic import Field, FiniteFloat, validate_call

from .base import MembershipFunction


class MFGaussian(MembershipFunction):
    """Gaussian Membership Function.

    A fuzzy membership function defined by a Gaussian (normal) distribution.

    Parameters
    ----------
    mean : FiniteFloat
        Center (mean/median/mode) of the Gaussian
    sigma : FiniteFloat
        Standard deviation of the Gaussian

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    mean: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        z = (x - self.mean) / self.sigma
        return exp(-0.5 * z * z)

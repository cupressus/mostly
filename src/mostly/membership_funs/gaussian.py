from math import exp

from pydantic import Field, FiniteFloat, validate_call

from src.mostly.membership_funs.base import MembershipFunction


class MFGaussian(MembershipFunction):
    """Gaussian Membership Function.

    A fuzzy membership function defined by a Gaussian (normal) distribution.

    Parameters
    ----------
    mean : FiniteFloat
        Center (mean/median/mode) of the Gaussian
    sigma : FiniteFloat
        Standard deviation of the Gaussian
    k_sigma : int, optional
        Truncation factor for finite support, by default `4` standard deviations

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.
    support()
        Returns the interval where the membership function is non-zero.

    """

    mean: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)
    k_sigma: int = Field(default=4, ge=1)

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        left, right = self.support()
        if x <= left or x >= right:
            return 0.0
        z = (x - self.mean) / self.sigma
        return exp(-0.5 * z * z)

    def support(self) -> tuple[FiniteFloat, FiniteFloat]:
        """Provide 0 Cutoffs."""
        left = self.mean - (self.k_sigma * self.sigma)
        right = self.mean + (self.k_sigma * self.sigma)
        return left, right


# %%

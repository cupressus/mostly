from math import exp

from pydantic import Field, FiniteFloat, validate_call

from .base import MembershipFunction


class MFGaussian(MembershipFunction):
    """A Gaussian membership function with a peak of 1.0 at its mean.

    Attributes:
        mean: Center of the curve.
        sigma: Positive standard deviation controlling the curve's width.

    Raises:
        ValidationError: If either field is non-finite or `sigma` is not positive.

    Examples:
        >>> from mostly.membership_functions import MFGaussian
        >>> MFGaussian(mean=0.0, sigma=1.0)(0.0)
        1.0

    """

    mean: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the Gaussian membership at a finite scalar."""
        z = (x - self.mean) / self.sigma
        return exp(-0.5 * z * z)

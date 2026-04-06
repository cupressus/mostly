from math import exp

from pydantic import Field, FiniteFloat, validate_call

from .base import MembershipFunction


class MFBimodalGaussian(MembershipFunction):
    """Bimodal Gaussian Membership Function.

    A fuzzy membership function combining two Gaussian distributions with piecewise behavior.

    When left_mean <= right_mean:
        - Returns Gaussian tail for x < left_mean
        - Returns 1.0 (plateau) for left_mean <= x <= right_mean
        - Returns Gaussian tail for x > right_mean

    When left_mean > right_mean:
        - Returns product of both Gaussian distributions (peak < 1.0)

    Parameters
    ----------
    left_mean : FiniteFloat
        Center (mean) of the left Gaussian
    left_sigma : FiniteFloat
        Standard deviation of the left Gaussian (must be positive)
    right_mean : FiniteFloat
        Center (mean) of the right Gaussian
    right_sigma : FiniteFloat
        Standard deviation of the right Gaussian (must be positive)

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    left_mean: FiniteFloat
    left_sigma: FiniteFloat = Field(gt=0.0)
    right_mean: FiniteFloat
    right_sigma: FiniteFloat = Field(gt=0.0)

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of membership for a given input `x`."""
        # Calculate individual Gaussian values
        z_left = (x - self.left_mean) / self.left_sigma
        gauss_left = exp(-0.5 * z_left * z_left)

        z_right = (x - self.right_mean) / self.right_sigma
        gauss_right = exp(-0.5 * z_right * z_right)

        # Piecewise logic based on mean ordering
        if self.left_mean <= self.right_mean:
            # Ordered case: plateau at 1.0 between means
            if x < self.left_mean:
                return gauss_left
            elif x > self.right_mean:
                return gauss_right
            else:
                return 1.0
        else:
            # Inverted case: product of Gaussians (max < 1.0)
            return gauss_left * gauss_right

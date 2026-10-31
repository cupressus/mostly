from math import exp

from pydantic import Field, FiniteFloat, validate_call

from .base import MembershipFunction


class MFBimodalGaussian(MembershipFunction):
    """Combine two Gaussians into a piecewise membership function.

    If `left_mean <= right_mean`, the value is 1.0 between the means and
    follows the corresponding Gaussian outside them. If the means are
    reversed, the value is the product of both Gaussians.

    Attributes:
        left_mean: Center of the left Gaussian.
        left_sigma: Positive standard deviation of the left Gaussian.
        right_mean: Center of the right Gaussian.
        right_sigma: Positive standard deviation of the right Gaussian.

    Raises:
        ValidationError: If a field is non-finite or either sigma is not positive.

    Examples:
        >>> from mostly.membership_functions import MFBimodalGaussian
        >>> mf = MFBimodalGaussian(left_mean=0.0, left_sigma=1.0, right_mean=2.0, right_sigma=1.0)
        >>> mf(1.0)
        1.0

    """

    left_mean: FiniteFloat
    left_sigma: FiniteFloat = Field(gt=0.0)
    right_mean: FiniteFloat
    right_sigma: FiniteFloat = Field(gt=0.0)

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the piecewise Gaussian membership at a finite scalar."""
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

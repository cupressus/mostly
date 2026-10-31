from pydantic import Field, FiniteFloat, validate_call

from .base import MembershipFunction


class MFGeneralizedBell(MembershipFunction):
    """A symmetric generalized bell membership function.

    Membership is 1.0 at `center` and 0.5 at a distance of `width` from it.
    The `slope` controls the steepness of the transition.

    Attributes:
        width: Positive half-width at half-maximum.
        slope: Positive parameter controlling steepness.
        center: Center of the curve.

    Raises:
        ValidationError: If a field is non-finite or `width` or `slope` is not positive.

    Examples:
        >>> from mostly.membership_functions import MFGeneralizedBell
        >>> MFGeneralizedBell(width=2.0, slope=1.0, center=0.0)(2.0)
        0.5

    """

    width: FiniteFloat = Field(gt=0.0)
    slope: FiniteFloat = Field(gt=0.0)
    center: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the bell curve at a finite scalar.

        Raises:
            OverflowError: If exponentiation overflows for extreme values.

        """
        return 1.0 / (1.0 + abs((x - self.center) / self.width) ** (2.0 * self.slope))

from pydantic import Field, FiniteFloat, validate_call

from src.mostly.membership_funs.base import MembershipFunction


class MFGeneralizedBell(MembershipFunction):
    """Generalized Bell-Shaped Membership Function.

    A fuzzy membership function defined by the generalized bell curve.

    Parameters
    ----------
    width : FiniteFloat
        Half-width at half-maximum; controls the width of the bell curve.
    slope : FiniteFloat
        Controls the steepness of the transition slopes.
    center : FiniteFloat
        Center of the bell curve.

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    Notes
    -----
    Formula: 1 / (1 + (|(x - center) / width|)^(2 * slope))

    """

    width: FiniteFloat = Field(gt=0.0)
    slope: FiniteFloat = Field(gt=0.0)
    center: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        return 1.0 / (1.0 + abs((x - self.center) / self.width) ** (2.0 * self.slope))

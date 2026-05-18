from typing import Literal

from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction


class MFPolynomial(MembershipFunction):
    """Polynomial Membership Function.

    A fuzzy membership function defined by a polynomial.

    Parameters
    ----------
    variant : Literal['left_open','right_open']
        Variant of the polynomial membership function.
        Can be either 'left_open', which is a Z-shaped function, or 'right_open' which is an S-shaped function.
    shoulder : FiniteFloat
        *Shoulder* of the membership function
    foot : FiniteFloat
        *Foot* of the membership function

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    variant: Literal["left_open", "right_open"]
    shoulder: FiniteFloat
    foot: FiniteFloat

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        if self.variant == "left_open":
            match x:
                case x if x <= self.foot:
                    return 1.0
                case x if x >= self.shoulder:
                    return 0.0
                case x if self.shoulder <= x <= (self.foot + self.shoulder) / 2:
                    return 1 - 2 * ((x - self.foot) / (self.foot - self.shoulder)) ** 2
                case x if (self.foot + self.shoulder) / 2 <= x <= self.shoulder:
                    return 2 * ((x - self.foot) / (self.foot - self.shoulder)) ** 2
                case _:
                    return 0.0
        else:
            match x:
                case x if x <= self.foot:
                    return 0.0
                case x if x >= self.shoulder:
                    return 1.0
                case x if self.foot <= x <= (self.foot + self.shoulder) / 2:
                    return 2 * ((x - self.foot) / (self.shoulder - self.foot)) ** 2
                case x if (self.foot + self.shoulder) / 2 <= x <= self.shoulder:
                    return 1 - 2 * ((x - self.shoulder) / (self.shoulder - self.foot)) ** 2
                case _:
                    return 0.0

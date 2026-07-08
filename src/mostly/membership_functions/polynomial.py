from pydantic import FiniteFloat, model_validator, validate_call

from .base import MembershipFunction


class MFPolynomialZ(MembershipFunction):
    """Z-shaped Polynomial Membership Function.

    A fuzzy membership function defined by a polynomial.

    Parameters
    ----------
    shoulder : FiniteFloat
        *Shoulder* of the membership function
    foot : FiniteFloat
        *Foot* of the membership function

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    shoulder: FiniteFloat
    foot: FiniteFloat

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialZ":
        """Validate model for correct Polynomial."""
        if self.shoulder >= self.foot:
            raise ValueError("For 'left_open' variant, shoulder must be less than foot")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match x:
            case x if x <= self.shoulder:
                return 1.0
            case x if self.shoulder <= x <= (self.foot + self.shoulder) / 2:
                return 1 - 2 * ((x - self.shoulder) / (self.foot - self.shoulder)) ** 2
            case x if (self.shoulder + self.foot) / 2 <= x <= self.foot:
                return 2 * ((x - self.foot) / (self.foot - self.shoulder)) ** 2
            case x if x >= self.foot:
                return 0.0
            case _:
                return 0.0


class MFPolynomialS(MembershipFunction):
    """S-shaped Polynomial Membership Function.

    A fuzzy membership function defined by a polynomial.

    Parameters
    ----------
    shoulder : FiniteFloat
        *Shoulder* of the membership function
    foot : FiniteFloat
        *Foot* of the membership function

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    shoulder: FiniteFloat
    foot: FiniteFloat

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialS":
        """Validate model for correct Polynomial."""
        if self.foot >= self.shoulder:
            raise ValueError("For 'right_open' variant, foot must be less than shoulder")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match x:
            case x if x <= self.foot:
                return 0.0
            case x if self.foot <= x <= (self.foot + self.shoulder) / 2:
                return 2 * ((x - self.foot) / (self.shoulder - self.foot)) ** 2
            case x if (self.foot + self.shoulder) / 2 <= x <= self.shoulder:
                return 1 - 2 * ((x - self.shoulder) / (self.shoulder - self.foot)) ** 2
            case x if x >= self.shoulder:
                return 1.0
            case _:
                return 0.0

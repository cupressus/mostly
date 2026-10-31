from pydantic import FiniteFloat, model_validator, validate_call

from .base import MembershipFunction


class MFPolynomialZ(MembershipFunction):
    """A Z-shaped polynomial that falls smoothly from 1.0 to 0.0.

    Membership is 1.0 at and below `shoulder` and 0.0 at and above `foot`.

    Attributes:
        shoulder: Start of the falling transition.
        foot: End of the transition; must be greater than `shoulder`.

    Raises:
        ValidationError: If either point is non-finite or `shoulder >= foot`.

    Examples:
        >>> from mostly.membership_functions import MFPolynomialZ
        >>> MFPolynomialZ(shoulder=0.0, foot=2.0)(1.0)
        0.5

    """

    shoulder: FiniteFloat
    foot: FiniteFloat

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialZ":
        """Validate that the shoulder precedes the foot."""
        if self.shoulder >= self.foot:
            raise ValueError("For 'left_open' variant, shoulder must be less than foot")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the falling polynomial at a finite scalar."""
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
    """An S-shaped polynomial that rises smoothly from 0.0 to 1.0.

    Membership is 0.0 at and below `foot` and 1.0 at and above `shoulder`.

    Attributes:
        shoulder: End of the rising transition.
        foot: Start of the transition; must be less than `shoulder`.

    Raises:
        ValidationError: If either point is non-finite or `foot >= shoulder`.

    Examples:
        >>> from mostly.membership_functions import MFPolynomialS
        >>> MFPolynomialS(foot=0.0, shoulder=2.0)(1.0)
        0.5

    """

    shoulder: FiniteFloat
    foot: FiniteFloat

    @model_validator(mode="after")
    def compliance(self) -> "MFPolynomialS":
        """Validate that the foot precedes the shoulder."""
        if self.foot >= self.shoulder:
            raise ValueError("For 'right_open' variant, foot must be less than shoulder")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the rising polynomial at a finite scalar."""
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

from pydantic import Field, FiniteFloat, model_validator, validate_call

from src.mostly.membership_funs.base import MembershipFunction


class MFTrapezoid(MembershipFunction):
    """Trapezoidal Membership Function.

    A fuzzy membership function defined by four points (`a`, `b`, `c`, `d`) forming a trapezoidal shape.

    Parameters
    ----------
    a : FiniteFloat
        The left foot of the trapezoid (start of the rising edge).
    b : FiniteFloat
        The left shoulder of the trapezoid (start of the plateau).
    c : FiniteFloat
        The right shoulder of the trapezoid (end of the plateau).
    d : FiniteFloat
        The right foot of the trapezoid (end of the falling edge).

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.
    support()
        Returns the interval [a, d] where the membership function is non-zero.

    Raises
    ------
    ValueError
        If the trapezoid points do not satisfy a ≤ b ≤ c ≤ d.

    """

    a: FiniteFloat = Field(description="The Left Foot of the Trapezoid")
    b: FiniteFloat = Field(description="The Left Shoulder of the Trapezoid")
    c: FiniteFloat = Field(description="The Right Shoulder of the Trapezoid")
    d: FiniteFloat = Field(description="The Right Foot of the Trapezoid")

    @model_validator(mode="after")
    def compliance(self) -> "MFTrapezoid":
        """Validate model for correct Trapezoid."""
        if self.a > self.b or self.c > self.d or self.b > self.c:
            raise ValueError("Trapezoid points must satisfy a ≤ b ≤ c ≤ d")
        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match x:
            case x if x < self.a or x > self.d:
                return 0.0
            case x if self.a <= x <= self.b:
                return (x - self.a) / (self.b - self.a)
            case x if self.b < x < self.c:
                return 1.0
            case x if self.c <= x <= self.d:
                return (self.d - x) / (self.d - self.c)
            case _:
                return 0.0

    def support(self) -> tuple[FiniteFloat, FiniteFloat]:
        """Provide 0 Cutoffs."""
        return self.a, self.d

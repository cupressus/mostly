from typing import Literal

from pydantic import Field, FiniteFloat, computed_field, model_validator, validate_call

from src.mostly.membership_funs.base import MembershipFunction


class MFTrapezoid(MembershipFunction):
    """Trapezoidal Membership Function.

    A fuzzy membership function defined by four points (`a`, `b`, `c`, `d`) forming a trapezoidal shape.
    Derived shapes (e.g. Triangles) are valid.

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
        If the trapezoid is not valid (e.g., points not in order, all points equal, etc.).

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
        elif self.a == self.b == self.c == self.d:
            raise ValueError("All points a, b, c, d cannot be equal; not a valid trapezoid")
        elif self.a == self.b and self.c == self.d:
            raise ValueError("Both shoulders cannot be equal; not a valid trapezoid")
        else:
            return self

    @computed_field
    @property
    def shape(self) -> Literal["left", "regular", "right"]:
        """Trapezoid Type."""
        match True:
            case _ if self.a == self.b and self.c != self.d:
                return "left"
            case _ if self.a != self.b and self.c == self.d:
                return "right"
            case _:
                return "regular"

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match self.shape:
            case "left":
                if x <= self.a:
                    return 1.0
                return max(min((self.d - x) / (self.d - self.c), 1.0), 0.0)
            case "right":
                if x >= self.d:
                    return 1.0
                return max(min((x - self.a) / (self.b - self.a), 1.0), 0.0)
            case "regular":
                if x <= self.a or x >= self.d:
                    return 0.0
                return max(min((x - self.a) / (self.b - self.a), 1.0, (self.d - x) / (self.d - self.c)), 0.0)
            case _:  # pragma: no cover
                raise RuntimeError(f"Unexpected trapezoid shape: {self.shape}")

    def support(self) -> tuple[FiniteFloat, FiniteFloat]:
        """Provide 0 Cutoffs."""
        return self.a, self.d

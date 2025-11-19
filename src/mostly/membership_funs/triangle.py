from typing import Literal

from pydantic import Field, FiniteFloat, computed_field, model_validator, validate_call

from src.mostly.membership_funs.base import MembershipFunction


class MFTriangle(MembershipFunction):
    """Triangular Membership Function.

    This class represents a triangular membership function, commonly used in fuzzy logic systems.
    The triangle is defined by three points: `a` (left foot), `b` (peak), and `c` (right foot).
    It is possible for `a` and `b` to be equal, resulting in a left-shoulder triangle,
    and for `b` and `c` to be equal, resulting in a right-shoulder triangle.

    Parameters
    ----------
    a : FiniteFloat
        The left foot of the triangle.
    b : FiniteFloat
        The peak of the triangle.
    c : FiniteFloat
        The right foot of the triangle.

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.
    support()
        Returns the interval `[a, c]` where the membership function is nonzero.

    Raises
    ------
    ValueError
        If the triangle points do not satisfy a ≤ b ≤ c.

    """

    a: FiniteFloat = Field(description="The Left Foot of a Triangle")
    b: FiniteFloat = Field(description="The Peak of a Triangle")
    c: FiniteFloat = Field(description="The Right Foot of a Triangle")

    @model_validator(mode="after")
    def compliance(self) -> "MFTriangle":
        """Validate model for correct Triangle."""
        if self.a > self.b or self.b > self.c:
            raise ValueError("Triangle points must satisfy a ≤ b ≤ c")
        if self.a == self.b == self.c:
            raise ValueError("All points a, b, c cannot be equal; not a valid triangle")
        return self

    @computed_field
    @property
    def shape(self) -> Literal["left", "regular", "right"]:
        """Triangle Type."""
        match True:
            case _ if self.a == self.b:
                return "left"
            case _ if self.b == self.c:
                return "right"
            case _:
                return "regular"

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        match self.shape:
            case "regular":
                return max(min((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)
            case "left":
                return max(min((self.c - x) / (self.c - self.a), 1), 0)
            case "right":
                return max(min((x - self.a) / (self.c - self.a), 1), 0)
            case _:
                raise RuntimeError(f"Unexpected triangle shape: {self.shape}")

    def support(self) -> tuple[FiniteFloat, FiniteFloat]:
        """Provide 0 Cutoffs."""
        return self.a, self.c

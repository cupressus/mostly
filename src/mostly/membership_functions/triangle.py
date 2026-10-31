from typing import Literal

from pydantic import Field, FiniteFloat, computed_field, model_validator, validate_call

from .base import MembershipFunction


class MFTriangular(MembershipFunction):
    """A triangular membership function, optionally with one shoulder.

    Points must satisfy `a <= b <= c` and cannot all coincide. Equal `a` and
    `b` create a left shoulder; equal `b` and `c` create a right shoulder.
    A shoulder remains at 1.0 beyond its vertical edge in ordinary numeric
    ranges. Extreme finite inputs or points can produce `NaN` through overflow.

    Attributes:
        a: Left foot.
        b: Peak.
        c: Right foot.

    Raises:
        ValidationError: If points are non-finite, out of order, or all equal.

    Examples:
        >>> from mostly.membership_functions import MFTriangular
        >>> MFTriangular(a=0.0, b=1.0, c=2.0)(0.5)
        0.5

    """

    a: FiniteFloat = Field(description="The Left Foot of a Triangle")
    b: FiniteFloat = Field(description="The Peak of a Triangle")
    c: FiniteFloat = Field(description="The Right Foot of a Triangle")

    @model_validator(mode="after")
    def compliance(self) -> "MFTriangular":
        """Validate point ordering and reject a fully degenerate triangle."""
        if self.a > self.b or self.b > self.c:
            raise ValueError("Triangle points must satisfy a ≤ b ≤ c")
        if self.a == self.b == self.c:
            raise ValueError("All points a, b, c cannot be equal; not a valid triangle")
        return self

    @computed_field
    @property
    def shape(self) -> Literal["left", "regular", "right"]:
        """Classify the shape as `left`, `right`, or `regular`."""
        match True:
            case _ if self.a == self.b:
                return "left"
            case _ if self.b == self.c:
                return "right"
            case _:
                return "regular"

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the triangular membership at a finite scalar."""
        match self.shape:
            case "regular":
                return max(min((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)
            case "left":
                return max(min((self.c - x) / (self.c - self.a), 1), 0)
            case "right":
                return max(min((x - self.a) / (self.c - self.a), 1), 0)
            case _:  # pragma: no cover
                raise RuntimeError(f"Unexpected triangle shape: {self.shape}")

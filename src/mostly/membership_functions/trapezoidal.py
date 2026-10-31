from typing import Literal

from pydantic import Field, FiniteFloat, computed_field, model_validator, validate_call

from .base import MembershipFunction


class MFTrapezoidal(MembershipFunction):
    """A trapezoidal membership function, optionally with one shoulder.

    Points must satisfy `a <= b <= c <= d`. A single vertical outer edge
    creates a shoulder; both outer edges cannot be vertical. When `b == c`,
    the plateau reduces to a triangular peak.

    Attributes:
        a: Left foot.
        b: Start of the plateau.
        c: End of the plateau.
        d: Right foot.

    Raises:
        ValidationError: If points are non-finite, out of order, or both
            outer edges have zero width.

    Examples:
        >>> from mostly.membership_functions import MFTrapezoidal
        >>> MFTrapezoidal(a=0.0, b=1.0, c=2.0, d=3.0)(1.5)
        1.0

    """

    a: FiniteFloat = Field(description="The Left Foot of the Trapezoid")
    b: FiniteFloat = Field(description="The Left Shoulder of the Trapezoid")
    c: FiniteFloat = Field(description="The Right Shoulder of the Trapezoid")
    d: FiniteFloat = Field(description="The Right Foot of the Trapezoid")

    @model_validator(mode="after")
    def compliance(self) -> "MFTrapezoidal":
        """Validate point ordering and reject invalid vertical edges."""
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
        """Classify the shape as `left`, `right`, or `regular`."""
        match True:
            case _ if self.a == self.b and self.c != self.d:
                return "left"
            case _ if self.a != self.b and self.c == self.d:
                return "right"
            case _:
                return "regular"

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the trapezoidal membership at a finite scalar."""
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

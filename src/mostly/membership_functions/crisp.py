from pydantic import Field, FiniteFloat, model_validator, validate_call

from .base import MembershipFunction


class MFCrisp(MembershipFunction):
    """Crisp Membership Function.

    A fuzzy membership function defined by a crisp (binary) distribution.

    Parameters
    ----------
    left : FiniteFloat
        Left boundary of the crisp membership function
    right : FiniteFloat
        Right boundary of the crisp membership function
    include_left : bool
        Whether the left boundary is inclusive (default: True)
    include_right : bool
        Whether the right boundary is inclusive (default: False)

    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    left: FiniteFloat
    right: FiniteFloat
    include_left: bool = Field(default=True)
    include_right: bool = Field(default=False)

    @model_validator(mode="after")
    def compliance(self) -> "MFCrisp":
        """Validate interval bounds and boundary-inclusion consistency."""
        if self.left > self.right:
            raise ValueError("Crisp bounds must satisfy left ≤ right")

        # Degenerate interval [x, x] is only valid when both bounds are inclusive.
        if self.left == self.right and not (self.include_left and self.include_right):
            raise ValueError(
                "When left == right, both include_left and include_right must be True"
            )

        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of Membership for a given input `x`."""
        left_ok = x >= self.left if self.include_left else x > self.left
        right_ok = x <= self.right if self.include_right else x < self.right

        if left_ok and right_ok:
            return 1.0
        else:
            return 0.0

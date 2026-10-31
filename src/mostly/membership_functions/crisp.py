from pydantic import Field, FiniteFloat, model_validator, validate_call

from .base import MembershipFunction


class MFCrisp(MembershipFunction):
    """Binary membership over an interval with configurable endpoints.

    By default the interval is `(left, right]`: the left endpoint is excluded
    and the right endpoint is included. Equal bounds are valid only when both
    endpoints are included.

    Attributes:
        left: Lower interval boundary.
        right: Upper interval boundary; must be at least `left`.
        include_left: Whether the left endpoint is included.
        include_right: Whether the right endpoint is included.

    Raises:
        ValidationError: If bounds are non-finite or reversed, or equal bounds
            do not include both endpoints.

    Examples:
        >>> from mostly.membership_functions import MFCrisp
        >>> interval = MFCrisp(left=0.0, right=1.0)
        >>> interval(0.0), interval(1.0)
        (0.0, 1.0)

    """

    left: FiniteFloat
    right: FiniteFloat
    include_left: bool = Field(default=False)
    include_right: bool = Field(default=True)

    @model_validator(mode="after")
    def compliance(self) -> "MFCrisp":
        """Validate bound ordering and inclusion of equal endpoints."""
        if self.left > self.right:
            raise ValueError("Crisp bounds must satisfy left ≤ right")

        # Degenerate interval [x, x] is only valid when both bounds are inclusive.
        if self.left == self.right and not (self.include_left and self.include_right):
            raise ValueError("When left == right, both include_left and include_right must be True")

        return self

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Return 1.0 inside the interval and 0.0 outside it."""
        left_ok = x >= self.left if self.include_left else x > self.left
        right_ok = x <= self.right if self.include_right else x < self.right

        if left_ok and right_ok:
            return 1.0
        else:
            return 0.0

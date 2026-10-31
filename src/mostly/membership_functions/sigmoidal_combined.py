from typing import Literal

from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction
from .sigmoidal import MFSigmoidal


class MFSigmoidalCombined(MembershipFunction):
    """Combine two sigmoids by subtraction or multiplication.

    The `diff` method subtracts the second value from the first and can
    produce negative results. The `multiply` method returns their product.

    Attributes:
        mfs: Ordered pair of sigmoidal membership functions.
        method: Combination method, either `diff` or `multiply`.

    Raises:
        ValidationError: If `mfs` cannot be parsed as a pair of sigmoids or
            `method` is unsupported.

    Examples:
        >>> from mostly.membership_functions import MFSigmoidal, MFSigmoidalCombined
        >>> half = MFSigmoidal(a=0.0, c=0.0)
        >>> rising = MFSigmoidal(a=1.0, c=0.0)
        >>> combined = MFSigmoidalCombined(mfs=(half, rising), method="diff")
        >>> combined(1.0) < 0.0
        True

    """

    mfs: tuple[MFSigmoidal, MFSigmoidal]
    method: Literal["diff", "multiply"]

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the configured difference or product.

        The result is `NaN` if either component produces `NaN`.

        Raises:
            ValueError: If `method` is not `diff` or `multiply`.
            OverflowError: If evaluating a sigmoid overflows.

        """
        mf1, mf2 = self.mfs
        if self.method == "diff":
            return mf1(x) - mf2(x)
        elif self.method == "multiply":
            return mf1(x) * mf2(x)  # pyright: ignore[reportOperatorIssue]
        else:
            raise ValueError("Invalid method. Use 'diff' or 'multiply'.")

from typing import Literal

from pydantic import FiniteFloat, validate_call

from .base import MembershipFunction
from .sigmoidal import MFSigmoidal


class MFSigmoidalCombined(MembershipFunction):
    """Combined Sigmoidal Membership Function.

    A fuzzy membership function defined by combining two sigmoidal curves.

    Parameters
    ----------
    mfs : tuple[MFSigmoidal, MFSigmoidal]
        Tuple of two sigmoidal membership functions
    method : Literal["diff", "multiply"]
        The combination method to use


    Methods
    -------
    __call__
        Calculates the degree of membership for the input `x`.

    """

    mfs: tuple[MFSigmoidal, MFSigmoidal]
    method: Literal["diff", "multiply"]

    @validate_call
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate degree of membership for a given input `x`."""
        mf1, mf2 = self.mfs
        if self.method == "diff":
            return mf1(x) - mf2(x)
        elif self.method == "multiply":
            return mf1(x) * mf2(x)  # pyright: ignore[reportOperatorIssue]
        else:
            raise ValueError("Invalid method. Use 'diff' or 'multiply'.")

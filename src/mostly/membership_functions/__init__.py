"""Scalar membership functions for fuzzy inference.

Import concrete models from this package and call them with a finite scalar
to evaluate membership. Some computations can overflow or produce non-finite
results for extreme finite inputs or model fields.
"""

from .base import MembershipFunction
from .bimodal_gaussian import MFBimodalGaussian
from .crisp import MFCrisp
from .gaussian import MFGaussian
from .generalized_bell import MFGeneralizedBell
from .polynomial import MFPolynomialS, MFPolynomialZ
from .polynomial_combined import MFPolynomialCombined
from .sigmoidal import MFSigmoidal
from .sigmoidal_combined import MFSigmoidalCombined
from .trapezoidal import MFTrapezoidal
from .triangle import MFTriangular

__all__ = [
    "MFBimodalGaussian",
    "MFGaussian",
    "MFGeneralizedBell",
    "MFTrapezoidal",
    "MFTriangular",
    "MembershipFunction",
    "MFPolynomialZ",
    "MFPolynomialS",
    "MFPolynomialCombined",
    "MFSigmoidal",
    "MFSigmoidalCombined",
    "MFCrisp",
]

"""Public API for building fuzzy inference systems."""

from .fuzzy_rules import And, FuzzyRule, Is, Not, Or
from .inference import InferenceConfig, MamdaniFIS
from .linguistic_variable import LinguisticVariable
from .membership_functions import MembershipFunction

__all__ = [
    "And",
    "FuzzyRule",
    "InferenceConfig",
    "Is",
    "LinguisticVariable",
    "MamdaniFIS",
    "MembershipFunction",
    "Not",
    "Or",
]

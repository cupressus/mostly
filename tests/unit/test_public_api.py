"""Tests for the documented public import surface."""

from mostly import (
    And,
    FuzzyRule,
    InferenceConfig,
    Is,
    LinguisticVariable,
    MamdaniFIS,
    MembershipFunction,
    Not,
    Or,
)
from mostly.fuzzy_rules import And as FuzzyRulesAnd
from mostly.fuzzy_rules import FuzzyRule as FuzzyRulesFuzzyRule
from mostly.fuzzy_rules import Is as FuzzyRulesIs
from mostly.fuzzy_rules import Not as FuzzyRulesNot
from mostly.fuzzy_rules import Or as FuzzyRulesOr
from mostly.inference import InferenceConfig as InferenceInferenceConfig
from mostly.inference import MamdaniFIS as InferenceMamdaniFIS
from mostly.membership_functions import MFTriangular


def test_root_exports_primary_workflow_types() -> None:
    """The package root exposes the documented core workflow."""
    assert FuzzyRule is FuzzyRulesFuzzyRule
    assert And is FuzzyRulesAnd
    assert Is is FuzzyRulesIs
    assert Not is FuzzyRulesNot
    assert Or is FuzzyRulesOr
    assert MamdaniFIS is InferenceMamdaniFIS
    assert InferenceConfig is InferenceInferenceConfig
    assert LinguisticVariable.__module__ == "mostly.linguistic_variable"
    assert MembershipFunction.__module__ == "mostly.membership_functions.base"
    assert MFTriangular.__module__ == "mostly.membership_functions.triangle"

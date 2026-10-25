"""Tests for the documented public import surface."""

import subprocess
import sys

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
from mostly.plotting.altair import (
    mostly_light,
    plot_fuzzy_set,
    plot_inference_inputs,
    plot_inference_outputs,
    plot_linguistic_variable,
)


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


def test_core_import_does_not_load_plotting_dependencies() -> None:
    """Importing the core API must not require or initialize plotting packages."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mostly; "
                "assert 'altair' not in sys.modules; "
                "assert 'pandas' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_altair_exports_documented_plotting_utilities() -> None:
    """The Altair backend exposes all documented plotting utilities."""
    assert callable(mostly_light)
    assert callable(plot_fuzzy_set)
    assert callable(plot_inference_inputs)
    assert callable(plot_inference_outputs)
    assert callable(plot_linguistic_variable)

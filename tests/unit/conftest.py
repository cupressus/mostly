"""Shared fixtures for unit tests."""

import pytest

from mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from mostly.fuzzy_rules.logical_operators import Is, Or
from mostly.inference.mamdani import MamdaniFIS
from mostly.linguistic_variable import LinguisticVariable
from mostly.membership_functions.triangle import MFTriangular


@pytest.fixture
def fis() -> MamdaniFIS:
    """Construct a tipping system for inference and plotting tests."""
    input_variables = {
        concept: LinguisticVariable(
            concept=concept,
            uod=(0.0, 10.0),
            fuzzy_sets={
                "poor": MFTriangular(a=0.0, b=0.0, c=5.0),
                "good": MFTriangular(a=0.0, b=5.0, c=10.0),
                "excellent": MFTriangular(a=5.0, b=10.0, c=10.0),
            },
        )
        for concept in ("food_quality", "service_quality")
    }
    tip = LinguisticVariable(
        concept="tip_amount",
        uod=(0.0, 25.0),
        fuzzy_sets={
            "low": MFTriangular(a=0.0, b=0.0, c=13.0),
            "medium": MFTriangular(a=0.0, b=13.0, c=25.0),
            "high": MFTriangular(a=13.0, b=25.0, c=25.0),
        },
    )
    rules = [
        FuzzyRule(
            antecedent=Or(
                [
                    Is(concept="food_quality", term="poor"),
                    Is(concept="service_quality", term="poor"),
                ],
            ),
            consequences={"tip_amount": "low"},
        ),
        FuzzyRule(
            antecedent=Is(concept="service_quality", term="good"),
            consequences={"tip_amount": "medium"},
        ),
        FuzzyRule(
            antecedent=Or(
                [
                    Is(concept="food_quality", term="excellent"),
                    Is(concept="service_quality", term="excellent"),
                ],
            ),
            consequences={"tip_amount": "high"},
        ),
    ]
    return MamdaniFIS(
        input_variables=input_variables,
        output_variables={"tip_amount": tip},
        fuzzy_rules=rules,
    )

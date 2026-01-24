from src.mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from src.mostly.fuzzy_rules.logical_operators import Is, Or
from src.mostly.inference.mamdani import MamdaniFIS
from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.membership_funs.triangle import MFTriangle

service = LinguisticVariable(
    concept="service_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangle(a=0.0, b=0.0, c=5.0),
        "good": MFTriangle(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangle(a=5.0, b=10.0, c=10.0),
    },
)

quality = LinguisticVariable(
    concept="food_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangle(a=0.0, b=0.0, c=5.0),
        "good": MFTriangle(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangle(a=5.0, b=10.0, c=10.0),
    },
)

tip = LinguisticVariable(
    concept="tip_amount",
    uod=(0.0, 25.0),
    fuzzy_sets={
        "low": MFTriangle(a=0.0, b=0.0, c=13.0),
        "medium": MFTriangle(a=0.0, b=13.0, c=25.0),
        "high": MFTriangle(a=13.0, b=25.0, c=25.0),
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

fis = MamdaniFIS(
    input_variables={
        "food_quality": quality,
        "service_quality": service,
    },
    output_variables={"tip_amount": tip},
    fuzzy_rules=rules,
)


def test_mamdani_inference():
    """Test Mamdani fuzzy inference system."""
    assert isinstance(
        fis.infer(
            crisp_inputs={"food_quality": 6.5, "service_quality": 9.8},
            defuzzification="centroid",
        ).get("tip_amount"),
        float,
    )


def test_pretty_rules():
    """Test pretty string representation of fuzzy rules."""
    rule = rules[0]
    pretty_str = rule.pretty()
    expected_str = "IF ((food_quality IS poor) OR (service_quality IS poor)) THEN (tip_amount IS low) [weight: 1.0]"
    assert pretty_str == expected_str

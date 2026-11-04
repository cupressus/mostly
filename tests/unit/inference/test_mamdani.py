import altair as alt
import pytest

from mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from mostly.fuzzy_rules.logical_operators import Is, Or
from mostly.inference.mamdani import MamdaniFIS
from mostly.linguistic_variable import LinguisticVariable
from mostly.membership_functions.triangle import MFTriangular
from mostly.plotting.altair.plot_fis_inputs import plot_inference_inputs
from mostly.plotting.altair.plot_fis_outputs import plot_inference_outputs
from mostly.plotting.altair.plot_fis_rules import plot_inference_rules
from mostly.plotting.altair.themes import mostly_light  # noqa: F401

service = LinguisticVariable(
    concept="service_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangular(a=0.0, b=0.0, c=5.0),
        "good": MFTriangular(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangular(a=5.0, b=10.0, c=10.0),
    },
)

quality = LinguisticVariable(
    concept="food_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangular(a=0.0, b=0.0, c=5.0),
        "good": MFTriangular(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangular(a=5.0, b=10.0, c=10.0),
    },
)

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
        ).get("tip_amount"),
        float,
    )


def test_pretty_rules():
    """Test pretty string representation of fuzzy rules."""
    rule = rules[0]
    pretty_str = rule.pretty()
    expected_str = "IF ((food_quality IS poor) OR (service_quality IS poor)) THEN (tip_amount IS low) [weight: 1.0]"
    assert pretty_str == expected_str


def test_mamdani_inference_output_plotting():
    """Test Mamdani inference with plotting of inputs and outputs."""
    alt.theme.enable("mostly_light")
    crisp_inputs = {"food_quality": 6.5, "service_quality": 9.8}
    output_plot = plot_inference_outputs(fis, crisp_inputs=crisp_inputs)
    assert output_plot is not None
    assert isinstance(output_plot, (alt.Chart, alt.LayerChart, alt.VConcatChart))


@pytest.mark.parametrize(("aggregation", "expected_aggregate"), [("max", 0.96), ("sum", 1.66)])
def test_mamdani_rule_tables_show_strengths_and_aggregated_consequences(aggregation, expected_aggregate):
    """The rule tables expose firing strengths and the aggregate for shared consequences."""
    test_fis = fis.model_copy(deep=True)
    test_fis.inference_config.implication = "scale"
    test_fis.inference_config.aggregation = aggregation
    test_fis.fuzzy_rules.append(
        FuzzyRule(
            antecedent=Is(concept="food_quality", term="good"),
            consequences={"tip_amount": "high"},
        )
    )
    crisp_inputs = {"food_quality": 6.5, "service_quality": 9.8}

    rule_table, consequence_table = plot_inference_rules(test_fis, crisp_inputs=crisp_inputs)

    assert rule_table.columns.tolist() == ["Pretty rule", "Firing strength"]
    assert rule_table.iloc[2]["Pretty rule"] == rules[2].pretty()
    assert rule_table.iloc[2]["Firing strength"] == pytest.approx(0.96)
    assert rule_table.iloc[3]["Pretty rule"] == test_fis.fuzzy_rules[3].pretty()
    assert rule_table.iloc[3]["Firing strength"] == pytest.approx(0.7)

    high_consequence = consequence_table[consequence_table["Consequence"] == "tip_amount IS high"]
    assert set(high_consequence["Pretty rule"]) == {
        rules[2].pretty(),
        test_fis.fuzzy_rules[3].pretty(),
    }
    assert set(high_consequence["Implication"]) == {"scale"}
    assert set(high_consequence["Aggregation"]) == {aggregation}
    at_maximum = high_consequence[high_consequence["Output"] == 25.0]
    assert at_maximum["Aggregated membership"].unique().tolist() == pytest.approx([expected_aggregate])
    assert at_maximum.loc[at_maximum["Pretty rule"] == rules[2].pretty(), "Implied membership"].iloc[
        0
    ] == pytest.approx(0.96)
    assert at_maximum.loc[at_maximum["Pretty rule"] == test_fis.fuzzy_rules[3].pretty(), "Implied membership"].iloc[
        0
    ] == pytest.approx(0.7)


def test_mamdani_inference_input_plotting():
    """Test Mamdani inference with plotting of inputs and outputs."""
    alt.theme.enable("mostly_light")
    crisp_inputs = {"food_quality": 6.5, "service_quality": 9.8}
    input_plot = plot_inference_inputs(fis, crisp_inputs=crisp_inputs)
    assert input_plot is not None
    assert isinstance(input_plot, (alt.Chart, alt.LayerChart, alt.HConcatChart))

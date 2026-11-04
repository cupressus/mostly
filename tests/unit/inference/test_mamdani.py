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
def test_mamdani_rule_plot_shows_firing_strengths_and_aggregated_consequences(aggregation, expected_aggregate):
    """The rule plot exposes firing strengths and the aggregate for shared consequences."""
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

    rule_plot = plot_inference_rules(test_fis, crisp_inputs=crisp_inputs)
    spec = rule_plot.to_dict()
    serialized = str(spec)

    assert isinstance(rule_plot, alt.VConcatChart)
    assert "Rule Firing Strengths" in serialized
    assert "Consequence: tip_amount IS high" in serialized
    assert f"implication: scale; aggregation: {aggregation}" in serialized
    assert f"Dashed black line: {aggregation}-aggregated result" in serialized

    strength_data = spec["datasets"][spec["vconcat"][0]["data"]["name"]]
    assert strength_data[2]["strength"] == pytest.approx(0.96)
    assert strength_data[3]["strength"] == pytest.approx(0.7)
    assert strength_data[2]["rule"] == rules[2].pretty()
    assert strength_data[3]["rule"] == test_fis.fuzzy_rules[3].pretty()
    assert spec["vconcat"][0]["encoding"]["y"]["field"] == "rule"
    assert spec["vconcat"][0]["encoding"]["x"]["field"] == "strength"

    high_chart = next(
        chart
        for chart in spec["vconcat"]
        if chart.get("title", {}).get("text") == "Consequence: tip_amount IS high"
    )
    aggregate_layer = next(layer for layer in high_chart["layer"] if layer["mark"].get("color") == "black")
    aggregate_data = spec["datasets"][aggregate_layer["data"]["name"]]
    assert aggregate_data[-1]["membership"] == pytest.approx(expected_aggregate)
    rule_layer = next(layer for layer in high_chart["layer"] if layer["mark"].get("color") != "black")
    rule_data = spec["datasets"][rule_layer["data"]["name"]]
    assert rule_data[0]["rule"] == rules[2].pretty()
    assert rule_layer["encoding"]["color"]["field"] == "rule"


def test_mamdani_inference_input_plotting():
    """Test Mamdani inference with plotting of inputs and outputs."""
    alt.theme.enable("mostly_light")
    crisp_inputs = {"food_quality": 6.5, "service_quality": 9.8}
    input_plot = plot_inference_inputs(fis, crisp_inputs=crisp_inputs)
    assert input_plot is not None
    assert isinstance(input_plot, (alt.Chart, alt.LayerChart, alt.HConcatChart))

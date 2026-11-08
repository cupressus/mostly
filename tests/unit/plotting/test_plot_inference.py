"""Tests for plotting fuzzy inference inputs, outputs, and rules."""

from collections.abc import Callable
from typing import Literal

import altair as alt
import pytest

from mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from mostly.fuzzy_rules.logical_operators import Is
from mostly.inference.mamdani import MamdaniFIS
from mostly.plotting.altair import plot_inference_inputs, plot_inference_outputs, plot_inference_rules


@pytest.mark.parametrize(
    ("crisp_inputs", "layer_counts", "subtitles"),
    [
        (None, (3, 3), ("", "")),
        ({}, (3, 3), ("", "")),
        ({"food_quality": 6.5}, (5, 3), ("Input: 6.5", "")),
        ({"service_quality": 9.8}, (3, 5), ("", "Input: 9.8")),
        ({"food_quality": 0.0}, (5, 3), ("Input: 0.0", "")),
        ({"food_quality": 6.5, "service_quality": 9.8}, (5, 5), ("Input: 6.5", "Input: 9.8")),
    ],
)
def test_mamdani_inference_input_plotting(
    fis: MamdaniFIS,
    crisp_inputs: dict[str, float] | None,
    layer_counts: tuple[int, int],
    subtitles: tuple[str, str],
) -> None:
    """Verify input charts, including absent, partial, and zero highlights."""
    input_plot = plot_inference_inputs(fis, crisp_inputs=crisp_inputs, resolution=5)
    assert isinstance(input_plot, alt.HConcatChart)

    spec = input_plot.to_dict()
    assert spec["resolve"]["scale"] == {"color": "independent", "x": "independent", "y": "independent"}
    assert len(spec["hconcat"]) == 2
    for chart, concept, layer_count, subtitle in zip(
        spec["hconcat"], ("Food_quality", "Service_quality"), layer_counts, subtitles, strict=True
    ):
        assert chart["title"]["text"] == f'Fuzzy Sets of Linguistic Variable "{concept}"'
        assert chart["title"]["subtitle"] == subtitle
        assert len(chart["layer"]) == layer_count
        samples = spec["datasets"][chart["layer"][0]["data"]["name"]]
        assert [sample["x"] for sample in samples] == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])


def test_mamdani_inference_output_plotting(fis: MamdaniFIS) -> None:
    """Verify the aggregated output curve and defuzzified value."""
    fis.inference_config.resolution = 6
    output_plot = plot_inference_outputs(fis, crisp_inputs={"food_quality": 6.5, "service_quality": 9.8})
    assert isinstance(output_plot, alt.VConcatChart)

    spec = output_plot.to_dict()
    assert spec["resolve"]["scale"] == {"color": "independent", "x": "independent", "y": "independent"}
    assert len(spec["vconcat"]) == 1
    curve, marker, label = spec["vconcat"][0]["layer"]
    assert [curve["mark"]["type"], marker["mark"]["type"], label["mark"]["type"]] == ["line", "rule", "text"]
    assert curve["title"] == {
        "text": "Output Variable: Tip_amount",
        "subtitle": "aggregation: max; defuzzification: centroid",
    }
    assert curve["encoding"]["x"]["title"] == "UoD: [0.0, 25.0]"
    assert curve["encoding"]["y"]["scale"]["domain"] == [0, 1]

    samples = spec["datasets"][curve["data"]["name"]]
    assert [sample["x"] for sample in samples] == pytest.approx([0, 5, 10, 15, 20, 25])
    assert [sample["membership"] for sample in samples] == pytest.approx([0, 0.04, 0.04, 1 / 6, 7 / 12, 0.96])
    expected_centroid = 21.657355679702047
    marker_data = spec["datasets"][marker["data"]["name"]]
    label_data = spec["datasets"][label["data"]["name"]]
    assert marker_data[0]["x"] == pytest.approx(expected_centroid)
    assert marker["title"] == "Defuzzified Value: 21.66"
    assert label_data[0]["x"] == pytest.approx(expected_centroid)
    assert label_data[0]["label"] == "tip_amount: 21.66"


def test_inference_outputs_include_every_output_variable(fis: MamdaniFIS) -> None:
    """Verify that each output variable receives a separate chart and label."""
    fis.inference_config.resolution = 6
    fis.output_variables["bonus"] = fis.output_variables["tip_amount"].model_copy(update={"concept": "bonus"})
    for rule in fis.fuzzy_rules:
        rule.consequences["bonus"] = rule.consequences["tip_amount"]

    spec = plot_inference_outputs(fis, {"food_quality": 6.5, "service_quality": 9.8}).to_dict()
    assert len(spec["vconcat"]) == 2
    assert [chart["layer"][0]["title"]["text"] for chart in spec["vconcat"]] == [
        "Output Variable: Tip_amount",
        "Output Variable: Bonus",
    ]
    labels = [spec["datasets"][chart["layer"][2]["data"]["name"]][0]["label"] for chart in spec["vconcat"]]
    assert labels == ["tip_amount: 21.66", "bonus: 21.66"]


def test_inference_outputs_with_no_rules(fis: MamdaniFIS) -> None:
    """Verify that no rules produce a zero curve and a zero defuzzified value."""
    fis.fuzzy_rules.clear()
    fis.inference_config.resolution = 3
    spec = plot_inference_outputs(fis, {}).to_dict()
    curve, marker, label = spec["vconcat"][0]["layer"]

    samples = spec["datasets"][curve["data"]["name"]]
    assert [sample["membership"] for sample in samples] == [0.0, 0.0, 0.0]
    assert spec["datasets"][marker["data"]["name"]] == [{"x": 0.0}]
    assert spec["datasets"][label["data"]["name"]] == [{"x": 0.0, "label": "tip_amount: 0.00"}]


@pytest.mark.parametrize(
    ("implication", "aggregation", "expected_result"),
    [
        ("scale", "max", 0.96),
        ("clip", "max", 0.96),
        ("scale", "sum", 1.66),
        ("clip", "sum", 1.66),
        ("scale", "probor", 0.988),
        ("clip", "probor", 0.988),
    ],
)
def test_mamdani_rule_tables_show_strengths_and_aggregated_consequences(
    fis: MamdaniFIS,
    implication: Literal["clip", "scale"],
    aggregation: Literal["max", "sum", "probor"],
    expected_result: float,
) -> None:
    """Verify firing strengths and all supported consequence aggregations."""
    fis.inference_config.implication = implication
    fis.inference_config.aggregation = aggregation
    fis.fuzzy_rules.append(
        FuzzyRule(
            antecedent=Is(concept="food_quality", term="good"),
            consequences={"tip_amount": "high"},
        )
    )
    rule_table, consequence_table = plot_inference_rules(fis, {"food_quality": 6.5, "service_quality": 9.8})

    assert rule_table.columns.tolist() == ["Pretty rule", "Firing strength"]
    assert rule_table.iloc[2]["Pretty rule"] == fis.fuzzy_rules[2].pretty()
    assert rule_table.iloc[2]["Firing strength"] == pytest.approx(0.96)
    assert rule_table.iloc[3]["Pretty rule"] == fis.fuzzy_rules[3].pretty()
    assert rule_table.iloc[3]["Firing strength"] == pytest.approx(0.7)

    assert consequence_table.columns.tolist() == ["Consequence", "Result", "Implication", "Aggregation"]
    assert len(consequence_table) == 3
    high_consequence = consequence_table[consequence_table["Consequence"] == "tip_amount IS high"]
    assert len(high_consequence) == 1
    assert high_consequence.iloc[0]["Result"] == pytest.approx(expected_result)
    assert high_consequence.iloc[0]["Implication"] == implication
    assert high_consequence.iloc[0]["Aggregation"] == aggregation


@pytest.mark.parametrize(("implication", "expected_result"), [("clip", 0.04), ("scale", 1 / 30)])
def test_rule_consequence_respects_implication_and_resolution(
    fis: MamdaniFIS,
    implication: Literal["clip", "scale"],
    expected_result: float,
) -> None:
    """Verify the consequence peak uses the configured sampling and implication."""
    fis.inference_config.implication = implication
    fis.inference_config.resolution = 6
    _, consequence_table = plot_inference_rules(fis, {"food_quality": 6.5, "service_quality": 9.8})

    medium = consequence_table[consequence_table["Consequence"] == "tip_amount IS medium"]
    assert len(medium) == 1
    assert medium.iloc[0]["Result"] == pytest.approx(expected_result)


def test_rule_tables_with_no_rules(fis: MamdaniFIS) -> None:
    """Verify that empty rule tables retain their documented columns."""
    fis.fuzzy_rules.clear()
    rule_table, consequence_table = plot_inference_rules(fis, {})

    assert rule_table.empty
    assert rule_table.columns.tolist() == ["Pretty rule", "Firing strength"]
    assert consequence_table.empty
    assert consequence_table.columns.tolist() == ["Consequence", "Result", "Implication", "Aggregation"]


@pytest.mark.parametrize("plotter", [plot_inference_outputs, plot_inference_rules], ids=["outputs", "rules"])
def test_inference_plotting_rejects_unknown_inputs(
    fis: MamdaniFIS,
    plotter: Callable[[MamdaniFIS, dict[str, float]], object],
) -> None:
    """Verify inference input errors are propagated to plotting callers."""
    with pytest.raises(ValueError, match="Input variable 'unknown' not defined in FIS"):
        plotter(fis, {"unknown": 1.0})


@pytest.mark.parametrize("plotter", [plot_inference_outputs, plot_inference_rules], ids=["outputs", "rules"])
def test_inference_plotting_rejects_unknown_aggregation(
    fis: MamdaniFIS,
    plotter: Callable[[MamdaniFIS, dict[str, float]], object],
) -> None:
    """Verify unsupported aggregation methods raise rather than producing zeros."""
    fis.inference_config = fis.inference_config.model_copy(update={"aggregation": "invalid"})
    with pytest.raises(ValueError, match="Unknown aggregation method: invalid"):
        plotter(fis, {"food_quality": 6.5, "service_quality": 9.8})

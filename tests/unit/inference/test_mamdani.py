from typing import Literal

import pytest

from mostly.inference.mamdani import MamdaniFIS


def test_mamdani_inference(fis: MamdaniFIS) -> None:
    """Test Mamdani fuzzy inference system."""
    assert isinstance(
        fis.infer(
            crisp_inputs={"food_quality": 6.5, "service_quality": 9.8},
        ).get("tip_amount"),
        float,
    )


def test_implication_rejects_unknown_method(fis: MamdaniFIS) -> None:
    """Test implication rejects methods outside the supported choices."""
    mf = fis.output_variables["tip_amount"].get_fuzzy_set("low")
    x_vals, _ = fis._aggregation([], resolution=1)["tip_amount"]

    with pytest.raises(ValueError, match="Unknown implication method: invalid"):
        fis._implication(mf, 0.5, x_vals, "invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("sum", 0.8), ("probor", 0.64)],
)
def test_aggregation_sum_and_probor(
    fis: MamdaniFIS, aggregation: Literal["sum", "probor"], expected: float
) -> None:
    """Test sum and probabilistic OR combine repeated rule outputs."""
    consequences = [(fis.fuzzy_rules[0], 0.4), (fis.fuzzy_rules[0], 0.4)]

    _, aggregated = fis._aggregation(consequences, resolution=2, aggregation=aggregation)["tip_amount"]

    assert aggregated[0] == pytest.approx(expected)


def test_aggregation_skips_rules_without_output_consequences(fis: MamdaniFIS) -> None:
    """Test outputs with no matching rule consequence retain zero membership."""
    fis.output_variables["bonus"] = fis.output_variables["tip_amount"].model_copy(update={"concept": "bonus"})

    _, aggregated = fis._aggregation([(fis.fuzzy_rules[0], 0.5)], resolution=2)["bonus"]

    assert aggregated.tolist() == [0.0, 0.0]


def test_defuzzification_returns_zero_for_empty_aggregate(fis: MamdaniFIS) -> None:
    """Test centroid defuzzification returns zero when no membership is present."""
    aggregated_outputs = fis._aggregation([], resolution=2)

    assert fis._defuzzification(aggregated_outputs) == {"tip_amount": 0.0}


def test_defuzzification_rejects_unknown_output_variable(fis: MamdaniFIS) -> None:
    """Test defuzzification rejects outputs not declared by the FIS."""
    aggregated_outputs = {"unknown": fis._aggregation([], resolution=1)["tip_amount"]}

    with pytest.raises(ValueError, match="Output variable 'unknown' not defined in FIS"):
        fis._defuzzification(aggregated_outputs)


def test_defuzzification_rejects_unknown_method(fis: MamdaniFIS) -> None:
    """Test defuzzification rejects methods outside the supported choices."""
    aggregated_outputs = fis._aggregation([(fis.fuzzy_rules[0], 0.5)], resolution=1)

    with pytest.raises(ValueError, match="Unknown defuzzification method: invalid"):
        fis._defuzzification(aggregated_outputs, method="invalid")  # type: ignore[arg-type]


def test_pretty_rules(fis: MamdaniFIS) -> None:
    """Test pretty string representation of fuzzy rules."""
    rule = fis.fuzzy_rules[0]
    pretty_str = rule.pretty()
    expected_str = "IF ((food_quality IS poor) OR (service_quality IS poor)) THEN (tip_amount IS low) [weight: 1.0]"
    assert pretty_str == expected_str

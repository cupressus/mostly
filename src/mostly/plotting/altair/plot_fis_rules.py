import numpy as np
import pandas as pd

from ...inference.mamdani import MamdaniFIS


def plot_inference_rules(
    fis: MamdaniFIS,
    crisp_inputs: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tables of rule firing strengths and implied and aggregated consequences."""
    fuzzified_inputs = fis._fuzzification(crisp_inputs)
    rule_strengths = fis._rule_evaluation(fuzzified_inputs)

    rule_table = pd.DataFrame(
        [
            {
                "Pretty rule": rule.pretty(),
                "Firing strength": strength,
            }
            for rule, strength in rule_strengths
        ],
        columns=["Pretty rule", "Firing strength"],
    )
    consequence_tables = []

    for concept, term in dict.fromkeys(
        (concept, term) for rule, _ in rule_strengths for concept, term in rule.consequences.items()
    ):
        lv = fis.output_variables[concept]
        x_min, x_max = lv.uod
        x_vals = np.linspace(x_min, x_max, fis.inference_config.resolution)
        aggregate_vals = np.zeros_like(x_vals)
        rule_curves = []

        for rule, strength in rule_strengths:
            if rule.consequences.get(concept) != term:
                continue

            implied_vals = fis._implication(
                lv.get_fuzzy_set(term),
                strength,
                x_vals,
                fis.inference_config.implication,
            )
            match fis.inference_config.aggregation:
                case "max":
                    aggregate_vals = np.maximum(aggregate_vals, implied_vals)
                case "sum":
                    aggregate_vals += implied_vals
                case "probor":
                    aggregate_vals += implied_vals - aggregate_vals * implied_vals

            rule_curves.append((rule.pretty(), strength, implied_vals))

        for rule, strength, implied_vals in rule_curves:
            consequence_tables.append(
                pd.DataFrame(
                    {
                        "Consequence": f"{concept} IS {term}",
                        "Output": x_vals,
                        "Pretty rule": rule,
                        "Firing strength": strength,
                        "Implication": fis.inference_config.implication,
                        "Implied membership": implied_vals,
                        "Aggregation": fis.inference_config.aggregation,
                        "Aggregated membership": aggregate_vals,
                    }
                )
            )

    consequence_table = (
        pd.concat(consequence_tables, ignore_index=True)
        if consequence_tables
        else pd.DataFrame(
            columns=[
                "Consequence",
                "Output",
                "Pretty rule",
                "Firing strength",
                "Implication",
                "Implied membership",
                "Aggregation",
                "Aggregated membership",
            ]
        )
    )
    return rule_table, consequence_table

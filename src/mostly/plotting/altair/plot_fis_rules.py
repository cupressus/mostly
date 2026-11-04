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
    consequence_rows = []

    for concept, term in dict.fromkeys(
        (concept, term) for rule, _ in rule_strengths for concept, term in rule.consequences.items()
    ):
        lv = fis.output_variables[concept]
        x_min, x_max = lv.uod
        x_vals = np.linspace(x_min, x_max, fis.inference_config.resolution)
        aggregate_vals = np.zeros_like(x_vals)
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

        consequence_rows.append(
            {
                "Consequence": f"{concept} IS {term}",
                "Result": float(np.max(aggregate_vals, initial=0.0)),
                "Implication": fis.inference_config.implication,
                "Aggregation": fis.inference_config.aggregation,
            }
        )

    consequence_table = pd.DataFrame(
        consequence_rows,
        columns=["Consequence", "Result", "Implication", "Aggregation"],
    )
    return rule_table, consequence_table

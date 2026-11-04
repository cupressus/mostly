import altair as alt
import numpy as np
import pandas as pd

from ...inference.mamdani import MamdaniFIS


def plot_inference_rules(
    fis: MamdaniFIS,
    crisp_inputs: dict[str, float],
) -> alt.VConcatChart:
    """Plot rule firing strengths and each consequence's implied and aggregated membership."""
    fuzzified_inputs = fis._fuzzification(crisp_inputs)
    rule_strengths = fis._rule_evaluation(fuzzified_inputs)

    strength_data = pd.DataFrame(
        [
            {
                "rule_id": f"Rule {index}",
                "rule": rule.pretty(),
                "strength": strength,
            }
            for index, (rule, strength) in enumerate(rule_strengths, start=1)
        ],
        columns=["rule_id", "rule", "strength"],
    )
    charts = [
        alt.Chart(strength_data)
        .mark_bar()
        .encode(
            x=alt.X("rule_id:N", sort=None, title="Rule"),
            y=alt.Y("strength:Q", scale=alt.Scale(domain=[0, 1]), title="Firing Strength"),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("rule:N", title="Rule definition"),
                alt.Tooltip("strength:Q", title="Firing strength", format=".3f"),
            ],
        )
        .properties(
            title=alt.Title(
                "Rule Firing Strengths",
                subtitle=(
                    f"implication: {fis.inference_config.implication}; "
                    f"aggregation: {fis.inference_config.aggregation}"
                ),
            ),
            width=450,
            height=120,
        )
    ]

    consequences = dict.fromkeys(
        (concept, term)
        for rule, _ in rule_strengths
        for concept, term in rule.consequences.items()
    )

    for concept, term in consequences:
        lv = fis.output_variables[concept]
        x_min, x_max = lv.uod
        x_vals = np.linspace(x_min, x_max, fis.inference_config.resolution)
        aggregate_vals = np.zeros_like(x_vals)
        rule_charts = []

        for index, (rule, strength) in enumerate(rule_strengths, start=1):
            if rule.consequences.get(concept) != term:
                continue

            mf = lv.get_fuzzy_set(term)
            implied_vals = fis._implication(
                mf,
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

            rule_data = pd.DataFrame(
                {
                    "x": x_vals,
                    "membership": implied_vals,
                    "rule_id": f"Rule {index}",
                    "rule": rule.pretty(),
                    "strength": strength,
                }
            )
            rule_charts.append(
                alt.Chart(rule_data)
                .mark_line()
                .encode(
                    x=alt.X("x:Q", title=f"UoD: [{x_min}, {x_max}]"),
                    y=alt.Y(
                        "membership:Q",
                        title="Degree of Membership",
                    ),
                    color=alt.Color("rule_id:N", title="Rule"),
                    tooltip=[
                        alt.Tooltip("rule_id:N", title="Rule"),
                        alt.Tooltip("rule:N", title="Rule definition"),
                        alt.Tooltip("strength:Q", title="Firing strength", format=".3f"),
                        alt.Tooltip("x:Q", title="Output"),
                        alt.Tooltip("membership:Q", title="Membership", format=".3f"),
                    ],
                )
            )

        aggregate_data = pd.DataFrame(
            {
                "x": x_vals,
                "membership": aggregate_vals,
            }
        )
        aggregate_chart = (
            alt.Chart(aggregate_data)
            .mark_line(color="black", strokeDash=[6, 3], strokeWidth=3)
            .encode(
                x=alt.X("x:Q", title=f"UoD: [{x_min}, {x_max}]"),
                y=alt.Y(
                    "membership:Q",
                    title="Degree of Membership",
                ),
                tooltip=[
                    alt.Tooltip("x:Q", title="Output"),
                    alt.Tooltip("membership:Q", title="Aggregated membership", format=".3f"),
                ],
            )
        )
        charts.append(
            (alt.layer(*rule_charts, aggregate_chart) if rule_charts else aggregate_chart).properties(
                title=alt.Title(
                    f"Consequence: {concept} IS {term}",
                    subtitle=f"Dashed black line: {fis.inference_config.aggregation}-aggregated result",
                ),
                width=450,
                height=300,
            )
        )

    return alt.vconcat(*charts, spacing=15).resolve_scale(
        color="independent",
        x="independent",
        y="independent",
    )

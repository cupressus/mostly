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
                "rule": rule.pretty(),
                "strength": strength,
            }
            for rule, strength in rule_strengths
        ],
        columns=["rule", "strength"],
    )
    charts: list[alt.Chart | alt.LayerChart] = [
        alt.Chart(strength_data)
        .mark_bar()
        .encode(
            x=alt.X("strength:Q", scale=alt.Scale(domain=[0, 1]), title="Firing Strength"),
            y=alt.Y("rule:N", sort=None, title="Rule"),
            tooltip=[
                alt.Tooltip("rule:N", title="Rule"),
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
            width=600,
            height=max(120, len(rule_strengths) * 40),
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

        for rule, strength in rule_strengths:
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
                    color=alt.Color("rule:N", title="Rule"),
                    tooltip=[
                        alt.Tooltip("rule:N", title="Rule"),
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

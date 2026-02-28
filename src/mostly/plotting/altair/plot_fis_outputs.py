import altair as alt
import pandas as pd

from src.mostly.inference.mamdani import MamdaniFIS


def plot_inference_outputs(
    fis: MamdaniFIS,
    crisp_inputs: dict[str, float],
) -> alt.VConcatChart:
    """Plot the aggregated outputs of the fuzzy inference system."""
    fuzzified_inputs = fis._fuzzification(crisp_inputs)
    rule_strengths = fis._rule_evaluation(fuzzified_inputs)
    aggregated_outputs = fis._aggregation(
        rule_strengths,
        fis.inference_config.resolution,
        fis.inference_config.aggregation,
        fis.inference_config.implication,
    )
    defuzzified_outputs = fis._defuzzification(
        aggregated_outputs,
        fis.inference_config.defuzzification,
    )

    charts = []
    for concept, (x_vals, agg_vals) in aggregated_outputs.items():
        defuzzified_value = defuzzified_outputs[concept]
        chart = (
            alt.Chart(pd.DataFrame({"x": x_vals, "membership": agg_vals}))
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=f"UoD: [{', '.join(map(str, fis.output_variables[concept].uod))}]"),
                y=alt.Y("membership:Q", title="Degree of Membership", scale=alt.Scale(domain=[0, 1])),
            )
            .properties(
                title=alt.Title(
                    f"Output Variable: {str.capitalize(concept)}",
                    subtitle=(
                        f"aggregation: {fis.inference_config.aggregation}; "
                        f"defuzzification: {fis.inference_config.defuzzification}"
                    ),
                ),
                width=450,
                height=400,
            )
        )

        defuzzified_line = (
            alt.Chart(pd.DataFrame({"x": [defuzzified_value]}))
            .mark_rule(strokeDash=[2.5, 5], strokeWidth=2.5)
            .encode(x="x:Q")
            .properties(title=f"Defuzzified Value: {defuzzified_value:.2f}")
        )

        defuzzified_label = (
            alt.Chart(pd.DataFrame({"x": [defuzzified_value], "label": [f"{concept}: {defuzzified_value:.2f}"]}))
            .mark_text(align="left", dx=5, dy=-10)
            .encode(x="x:Q", text="label:N")
        )

        charts.append(chart + defuzzified_line + defuzzified_label)

    return alt.vconcat(*charts, spacing=15).resolve_scale(
        color="independent",
        x="independent",
        y="independent",
    )

import altair as alt
import numpy as np
import pandas as pd

from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.plotting.altair.plot_fuzzy_set import plot_fuzzy_set


def plot_linguistic_variable(
    lv: LinguisticVariable,
    resolution: int = 1000,
    highlight: float | None = None,
) -> alt.Chart | alt.LayerChart:
    """Plot the fuzzy sets of a linguistic variable.

    Args:
        lv: The linguistic variable to plot.
        resolution: Optional. The number of points to use for plotting the fuzzy sets.
        highlight: Optional. A value to highlight on the plot, showing its degree of membership in each fuzzy set.

    """
    charts = [plot_fuzzy_set(term, mf, lv.uod, resolution) for term, mf in lv.fuzzy_sets.items()]
    chart = alt.layer(*charts).properties(
        title=alt.Title(
            f'Fuzzy Sets of Linguistic Variable "{str.capitalize(lv.concept)}"',
            subtitle=f"Input: {float(highlight)}" if highlight else "",
        )
    )

    if highlight is not None:
        highlight_df = pd.DataFrame(
            {
                "x": [np.clip(highlight, *lv.uod)] * len(lv.fuzzy_sets),
                "term": list(lv.fuzzy_sets.keys()),
                "membership": [mf(highlight) for mf in lv.fuzzy_sets.values()],
            }
        )
        dots = (
            alt.Chart(highlight_df)
            .mark_circle(filled=True, size=100)
            .encode(
                x=alt.X("x:Q", title=f"UoD: [{', '.join(map(str, lv.uod))}]"),
                y=alt.Y("membership:Q", title="Degree of Membership"),
                color="term:N",
            )
        )
        labels = (
            alt.Chart(highlight_df)
            .mark_text(align="left", dx=5, dy=-5)
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("membership:Q"),
                text=alt.Text("membership:Q", format=".2f"),
                color="term:N",
            )
        )
        chart = chart + dots + labels

    return chart.properties(
        width=450,
        height=400,
    )

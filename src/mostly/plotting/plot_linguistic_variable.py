import altair as alt
import numpy as np
import pandas as pd

from src.mostly.linguistic_variable import LinguisticVariable


class PlotLinguisticVariable:
    """Mixin class to provide a plot method for linguistic variables."""

    lv: LinguisticVariable

    def plot(
        self,
        resolution: int = 1000,
        highlight: float | None = None,
    ) -> alt.Chart | alt.LayerChart:
        """Plot the fuzzy sets in the linguistic variable."""
        x_vals = np.linspace(*self.lv.uod, resolution)
        plot_data = []

        for term, mf in self.lv.fuzzy_sets.items():
            y_vals = [mf(x) for x in x_vals]
            plot_data.extend(
                {
                    "x": x,
                    "membership": y,
                    "term": term,
                }
                for x, y in zip(x_vals, y_vals, strict=True)
            )

        chart = (
            alt.Chart(pd.DataFrame(plot_data))
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=f"UoD: [{', '.join(map(str, self.lv.uod))}]"),
                y=alt.Y("membership:Q", title="Degree of Membership"),
                color="term:N",
            )
            .properties(
                title=alt.Title(
                    f'Fuzzy Sets of Linguistic Variable "{str.capitalize(self.lv.concept)}"',
                    subtitle=f"Input: {float(highlight)}" if highlight else "",
                )
            )
        )

        if highlight is not None:
            highlight_df = pd.DataFrame(
                {
                    "x": [np.clip(highlight, *self.lv.uod)] * len(self.lv.fuzzy_sets),
                    "term": list(self.lv.fuzzy_sets.keys()),
                    "membership": [mf(highlight) for mf in self.lv.fuzzy_sets.values()],
                }
            )
            dots = (
                alt.Chart(highlight_df)
                .mark_circle(filled=True, size=100)
                .encode(
                    x=alt.X("x:Q", title=f"UoD: [{', '.join(map(str, self.lv.uod))}]"),
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

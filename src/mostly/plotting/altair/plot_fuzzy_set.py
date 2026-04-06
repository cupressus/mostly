import altair as alt
import numpy as np
import pandas as pd

from ...membership_functions.base import MembershipFunction


def plot_fuzzy_set(
    term: str,
    mf: MembershipFunction,
    uod: tuple[float, float],
    resolution: int = 1000,
) -> alt.Chart:
    """Plot a single fuzzy set as a line chart.

    Args:
        term: The linguistic term name for the fuzzy set.
        mf: Membership function of the fuzzy set.
        uod: Universe of discourse bounds.
        resolution: Optional. Number of points used to sample the fuzzy set.

    Returns:
        Chart: An Altair line chart for the fuzzy set.

    """
    x_vals = np.linspace(*uod, resolution)
    y_vals = [mf(x) for x in x_vals]
    plot_data = pd.DataFrame(
        {
            "x": x_vals,
            "membership": y_vals,
            "term": [term] * len(x_vals),
        }
    )

    return (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            x=alt.X("x:Q", title=f"UoD: [{', '.join(map(str, uod))}]"),
            y=alt.Y("membership:Q", title="Degree of Membership"),
            color="term:N",
        )
        .properties(
            width=450,
            height=400,
        )
    )

import altair as alt

from src.mostly.inference.mamdani import MamdaniFIS
from src.mostly.plotting.altair.plot_linguistic_variable import plot_linguistic_variable


def plot_inference_inputs(
    fis: MamdaniFIS,
    crisp_inputs: dict[str, float] | None = None,
    resolution: int = 1000,
) -> alt.HConcatChart:
    """Plot the fuzzy inference systems' linguistic variables and their fuzzy sets.

    Args:
        fis: A Mamdani fuzzy inference system.
        crisp_inputs: Optional dictionary of crisp input values to highlight on the plots.
        resolution: The resolution of the plots.

    Returns:
        An Altair HConcatChart containing the plots of the input variables.

    """
    charts = []
    for concept, lv in fis.input_variables.items():
        highlight = crisp_inputs[concept] if crisp_inputs and concept in crisp_inputs else None
        chart: alt.LayerChart = plot_linguistic_variable(lv, resolution, highlight)  # type: ignore
        charts.append([chart])

    return alt.hconcat(*[chart for sublist in charts for chart in sublist], spacing=15).resolve_scale(
        color="independent",
        x="independent",
        y="independent",
    )

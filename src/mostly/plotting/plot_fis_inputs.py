from typing import Any

import altair as alt
from pydantic import BaseModel, ConfigDict

from src.mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.plotting.plot_linguistic_variable import PlotLinguisticVariable


class PlotInferenceInputs(BaseModel):
    """A mixin for plotting input variables of a fuzzy inference system (FIS)."""

    input_variables: dict[str, LinguisticVariable]
    output_variables: dict[str, LinguisticVariable]
    fuzzy_rules: list[FuzzyRule]
    meta_fields: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def plot_inputs(
        self,
        inputs: dict[str, float] | None = None,
        resolution: int = 1000,
    ) -> alt.VConcatChart:
        """Plot the fuzzy inference systems' linguistic variables and their fuzzy sets."""
        charts = []
        for concept, lv in self.input_variables.items():
            highlight = inputs[concept] if inputs and concept in inputs else None
            chart: alt.LayerChart = PlotLinguisticVariable(lv).plot(resolution, highlight)  # type: ignore
            charts.append([chart])

        return alt.vconcat(*[chart for sublist in charts for chart in sublist], spacing=15).resolve_scale(
            color="independent",
            x="independent",
            y="independent",
        )

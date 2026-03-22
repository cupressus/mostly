import altair as alt  # noqa: D104

from .plot_fis_inputs import plot_inference_inputs
from .plot_fis_outputs import plot_inference_outputs
from .plot_fuzzy_set import plot_fuzzy_set
from .plot_linguistic_variable import plot_linguistic_variable
from .themes import mostly_light  # noqa: F401

alt.theme.enable("mostly_light")

__all__ = [
    "plot_inference_inputs",
    "plot_inference_outputs",
    "plot_fuzzy_set",
    "plot_linguistic_variable",
]

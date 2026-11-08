"""Tests for the Mostly Altair theme."""

import altair as alt

from mostly.linguistic_variable import LinguisticVariable
from mostly.plotting.altair import mostly_light, plot_fuzzy_set


def test_mostly_light_theme_is_applied(simple_linguistic_variable: LinguisticVariable) -> None:
    """Verify the theme configuration and its application to chart output."""
    theme = mostly_light()
    assert theme.get("background") == "#ffffff"
    config = theme.get("config")
    assert isinstance(config, dict)
    title = config.get("title")
    line = config.get("line")
    ranges = config.get("range")
    assert isinstance(title, dict)
    assert isinstance(line, dict)
    assert isinstance(ranges, dict)
    category = ranges.get("category")
    assert isinstance(category, dict)
    assert title.get("font") == "Lato"
    assert line.get("stroke") == "#b6377a"
    assert category.get("scheme") == "observable10"

    with alt.theme.enable("mostly_light"):
        chart = plot_fuzzy_set(
            "cold",
            simple_linguistic_variable.get_fuzzy_set("cold"),
            simple_linguistic_variable.uod,
            resolution=5,
        )
        spec = chart.to_dict()

    assert spec["background"] == theme.get("background")
    assert spec["config"]["title"]["font"] == "Lato"
    assert spec["config"]["line"]["stroke"] == "#b6377a"

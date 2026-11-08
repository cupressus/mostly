"""Tests for plotting linguistic variables."""

import altair as alt
import pytest

from mostly.linguistic_variable import LinguisticVariable
from mostly.plotting.altair import plot_linguistic_variable


def test_plotting(simple_linguistic_variable: LinguisticVariable) -> None:
    """Verify that an unhighlighted variable has one line per fuzzy set."""
    chart = plot_linguistic_variable(simple_linguistic_variable)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))

    spec = chart.to_dict()
    assert spec["title"] == {
        "text": 'Fuzzy Sets of Linguistic Variable "Temperature"',
        "subtitle": "",
    }
    assert (spec["width"], spec["height"]) == (450, 400)
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["line", "line", "line"]
    assert [len(spec["datasets"][layer["data"]["name"]]) for layer in spec["layer"]] == [1000, 1000, 1000]


@pytest.mark.parametrize(
    ("highlight", "expected_x", "expected_memberships"),
    [
        (0.0, 0.0, (1.0, 0.0, 0.0)),
        (37.5, 37.5, (0.25, 0.5, 0.0)),
        (100.0, 100.0, (0.0, 0.0, 1.0)),
        (-25.0, 0.0, (1.0, 0.0, 0.0)),
        (125.0, 100.0, (0.0, 0.0, 1.0)),
    ],
)
def test_highlight_shows_memberships_and_clips_position(
    simple_linguistic_variable: LinguisticVariable,
    highlight: float,
    expected_x: float,
    expected_memberships: tuple[float, float, float],
) -> None:
    """Verify highlighted points, labels, and positions at and beyond the UoD."""
    chart = plot_linguistic_variable(simple_linguistic_variable, resolution=5, highlight=highlight)
    spec = chart.to_dict()
    lines, dots, labels = spec["layer"][:3], spec["layer"][3], spec["layer"][4]

    assert spec["title"]["subtitle"] == f"Input: {highlight}"
    assert [len(spec["datasets"][layer["data"]["name"]]) for layer in lines] == [5, 5, 5]
    assert dots["mark"]["type"] == "circle"
    assert labels["mark"]["type"] == "text"
    assert labels["encoding"]["text"]["field"] == "membership"
    assert labels["encoding"]["text"]["format"] == ".2f"
    assert dots["data"] == labels["data"]

    points = spec["datasets"][dots["data"]["name"]]
    assert [point["x"] for point in points] == pytest.approx([expected_x] * 3)
    assert [point["term"] for point in points] == ["cold", "warm", "hot"]
    assert [point["membership"] for point in points] == pytest.approx(expected_memberships)

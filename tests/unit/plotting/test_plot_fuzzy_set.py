"""Tests for plotting individual fuzzy sets."""

import altair as alt
import pytest

from mostly.linguistic_variable import LinguisticVariable
from mostly.plotting.altair import plot_fuzzy_set


def test_plot_single_fuzzy_set(simple_linguistic_variable: LinguisticVariable) -> None:
    """Verify the chart uses the default sampling and expected encodings."""
    chart = plot_fuzzy_set(
        term="cold",
        mf=simple_linguistic_variable.get_fuzzy_set("cold"),
        uod=simple_linguistic_variable.uod,
    )

    assert isinstance(chart, alt.Chart)
    spec = chart.to_dict()
    samples = spec["datasets"][spec["data"]["name"]]
    assert len(samples) == 1000
    assert samples[0] == {"x": 0.0, "membership": 1.0, "term": "cold"}
    assert samples[-1] == {"x": 100.0, "membership": 0.0, "term": "cold"}
    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["x"]["title"] == "UoD: [0.0, 100.0]"
    assert spec["encoding"]["y"]["field"] == "membership"
    assert spec["encoding"]["color"]["field"] == "term"
    assert (spec["width"], spec["height"]) == (450, 400)


@pytest.mark.parametrize(
    ("resolution", "expected_x", "expected_membership"),
    [
        (1, [0.0], [1.0]),
        (5, [0.0, 25.0, 50.0, 75.0, 100.0], [1.0, 0.5, 0.0, 0.0, 0.0]),
    ],
)
def test_plot_fuzzy_set_respects_resolution(
    simple_linguistic_variable: LinguisticVariable,
    resolution: int,
    expected_x: list[float],
    expected_membership: list[float],
) -> None:
    """Verify sample positions and membership values at custom resolutions."""
    chart = plot_fuzzy_set(
        "cold",
        simple_linguistic_variable.get_fuzzy_set("cold"),
        simple_linguistic_variable.uod,
        resolution=resolution,
    )
    spec = chart.to_dict()
    samples = spec["datasets"][spec["data"]["name"]]
    assert [sample["x"] for sample in samples] == pytest.approx(expected_x)
    assert [sample["membership"] for sample in samples] == pytest.approx(expected_membership)
    assert [sample["term"] for sample in samples] == ["cold"] * resolution

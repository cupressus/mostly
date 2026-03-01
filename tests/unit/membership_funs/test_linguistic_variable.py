import altair as alt
import pytest

from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.membership_funs.triangle import MFTriangle
from src.mostly.plotting.altair.plot_linguistic_variable import plot_linguistic_variable

# region POSITIVE TESTS


@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(0, (1, 0, 0), id="left"),
        pytest.param(25, (1, 0, 0), id="left fuzzy"),
        pytest.param(37.5, (0.5, 0.5, 0), id="left center"),
        pytest.param(50, (0, 1, 0), id="center"),
        pytest.param(62.5, (0, 0.5, 0.5), id="right center"),
        pytest.param(75, (0, 0, 1), id="right fuzzy"),
        pytest.param(100, (0, 0, 1), id="right"),
    ],
)
def test_regular_gaussian(simple_linguistic_variable: LinguisticVariable, input, expected) -> None:
    """Test Standard Gaussian Membership Function."""
    assert simple_linguistic_variable.fuzzify(input)["cold"] == pytest.approx(expected[0])
    assert simple_linguistic_variable.fuzzify(input)["warm"] == pytest.approx(expected[1])
    assert simple_linguistic_variable.fuzzify(input)["hot"] == pytest.approx(expected[2])


def test_get_fuzzy_set(simple_linguistic_variable: LinguisticVariable) -> None:
    """Test retrieval of fuzzy sets by term."""
    cold_fs = simple_linguistic_variable.get_fuzzy_set("cold")
    warm_fs = simple_linguistic_variable.get_fuzzy_set("warm")
    hot_fs = simple_linguistic_variable.get_fuzzy_set("hot")

    assert isinstance(cold_fs, MFTriangle)
    assert isinstance(warm_fs, MFTriangle)
    assert isinstance(hot_fs, MFTriangle)


def test_get_fuzzy_set_invalid_term(simple_linguistic_variable: LinguisticVariable) -> None:
    """Test retrieval of fuzzy sets with an invalid term."""
    with pytest.raises(ValueError) as exc:
        simple_linguistic_variable.get_fuzzy_set("freezing")

    err = exc.value
    assert "Fuzzy set for term 'freezing' not found in linguistic variable 'temperature'" in str(err)


# region PLOTTING TESTS
def test_plotting(simple_linguistic_variable: LinguisticVariable) -> None:
    """Test that the plot method returns an Altair chart without errors."""
    chart = plot_linguistic_variable(simple_linguistic_variable)
    assert chart is not None
    assert isinstance(chart, (alt.Chart, alt.LayerChart))

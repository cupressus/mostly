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
        pytest.param(25, (0.5, 0, 0), id="left fuzzy"),
        pytest.param(37.5, (0.25, 0.5, 0), id="left center"),
        pytest.param(50, (0, 1, 0), id="center"),
        pytest.param(62.5, (0, 0.5, 0.25), id="right center"),
        pytest.param(75, (0, 0, 0.5), id="right fuzzy"),
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


# region NEGATIVE TESTS - VALIDATION


def test_invalid_uod_inverted_bounds() -> None:
    """Test that construction fails when UOD bounds are inverted (min >= max)."""
    with pytest.raises(ValueError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(100.0, 0.0),
            fuzzy_sets={
                "cold": MFTriangle(a=0.0, b=0.0, c=50.0),
                "hot": MFTriangle(a=50.0, b=100.0, c=100.0),
            },
        )

    err = exc.value
    assert "Invalid UOD bounds" in str(err)
    assert "minimum (100.0) must be strictly less than maximum (0.0)" in str(err)


def test_invalid_uod_equal_bounds() -> None:
    """Test that construction fails when UOD bounds are equal."""
    with pytest.raises(ValueError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(50.0, 50.0),
            fuzzy_sets={
                "medium": MFTriangle(a=25.0, b=50.0, c=75.0),
            },
        )

    err = exc.value
    assert "Invalid UOD bounds" in str(err)
    assert "minimum (50.0) must be strictly less than maximum (50.0)" in str(err)


def test_incomplete_coverage_gap_at_boundaries() -> None:
    """Test that construction fails when membership functions don't cover the entire UOD."""
    with pytest.raises(ValueError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(0.0, 100.0),
            fuzzy_sets={
                # Using regular triangles (not shoulders) to create gaps
                "cold": MFTriangle(a=20.0, b=30.0, c=40.0),  # Gap at [0, 20)
                "warm": MFTriangle(a=30.0, b=50.0, c=70.0),
                "hot": MFTriangle(a=60.0, b=75.0, c=85.0),  # Gap at (85, 100]
            },
        )

    err = exc.value
    assert "Incomplete coverage" in str(err)
    assert "have zero membership to all terms" in str(err)


def test_incomplete_coverage_gap_in_middle() -> None:
    """Test that construction fails when there's a gap in the middle of the UOD."""
    with pytest.raises(ValueError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(0.0, 100.0),
            fuzzy_sets={
                "cold": MFTriangle(a=0.0, b=0.0, c=30.0),
                # Gap between 30 and 60
                "hot": MFTriangle(a=60.0, b=100.0, c=100.0),
            },
        )

    err = exc.value
    assert "Incomplete coverage" in str(err)
    assert "temperature" in str(err)


class BrokenMF(MFTriangle):
    """A broken membership function that returns NaN."""

    def __call__(self, x: float) -> float:
        """Return NaN to simulate a broken membership function."""
        return float("nan")


def test_nan_membership_detected() -> None:
    """Test that construction fails when a membership function returns NaN."""
    with pytest.raises(ValueError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(0.0, 100.0),
            fuzzy_sets={
                "cold": MFTriangle(a=0.0, b=0.0, c=50.0),
                "broken": BrokenMF(a=25.0, b=50.0, c=75.0),
                "hot": MFTriangle(a=50.0, b=100.0, c=100.0),
            },
        )

    err = exc.value
    assert "returns NaN" in str(err)
    assert "broken" in str(err)
    assert "broken membership function" in str(err)


def test_fuzzify_input_below_uod() -> None:
    """Test that fuzzify raises an error when input is below UOD minimum."""
    lv = LinguisticVariable(
        concept="temperature",
        uod=(0.0, 100.0),
        fuzzy_sets={
            "cold": MFTriangle(a=0.0, b=0.0, c=55.0),
            "hot": MFTriangle(a=45.0, b=100.0, c=100.0),
        },
    )

    with pytest.raises(ValueError) as exc:
        lv.fuzzify(-10.0)

    err = exc.value
    assert "Input value -10.0 is outside the UOD bounds [0.0, 100.0]" in str(err)


def test_fuzzify_input_above_uod() -> None:
    """Test that fuzzify raises an error when input is above UOD maximum."""
    lv = LinguisticVariable(
        concept="temperature",
        uod=(0.0, 100.0),
        fuzzy_sets={
            "cold": MFTriangle(a=0.0, b=0.0, c=55.0),
            "hot": MFTriangle(a=45.0, b=100.0, c=100.0),
        },
    )

    with pytest.raises(ValueError) as exc:
        lv.fuzzify(110.0)

    err = exc.value
    assert "Input value 110.0 is outside the UOD bounds [0.0, 100.0]" in str(err)


def test_valid_complete_coverage() -> None:
    """Test that a valid LinguisticVariable with complete coverage is accepted."""
    # This should not raise any errors
    lv = LinguisticVariable(
        concept="speed",
        uod=(0.0, 120.0),
        fuzzy_sets={
            "slow": MFTriangle(a=0.0, b=0.0, c=60.0),
            "medium": MFTriangle(a=30.0, b=60.0, c=90.0),
            "fast": MFTriangle(a=60.0, b=120.0, c=120.0),
        },
    )

    # Should be able to fuzzify values within UOD
    result = lv.fuzzify(30.0)
    assert "slow" in result
    assert "medium" in result
    assert "fast" in result

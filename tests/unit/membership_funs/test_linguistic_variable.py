import pytest
from pydantic import ValidationError

from src.mostly.linguistic_variable import LinguisticVariable
from src.mostly.membership_funs.triangle import MFTriangle

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


# region NEGATIVE TESTS
def test_uod_compliance_message() -> None:
    """Ensure the validator's message is surfaced in Pydantic's ValidationError."""
    with pytest.raises(ValidationError) as exc:
        LinguisticVariable(
            concept="temperature",
            uod=(0.0, 100.0),
            fuzzy_sets={
                "cold": MFTriangle(a=25.0, b=25.0, c=50.0),
                "warm": MFTriangle(a=25.0, b=50.0, c=75.0),
                "hot": MFTriangle(a=50.0, b=150.0, c=150.0),
            },
        )

    err = exc.value
    # Pydantic wraps the ValueError raised in the model validator into a ValidationError
    assert hasattr(err, "errors"), "Expected a Pydantic ValidationError"
    errs = err.errors()
    # There should be at least one error of type 'value_error'
    ve = next((e for e in errs if e.get("type") == "value_error"), None)
    assert ve is not None
    # Check the custom message substring for stability across versions
    assert "outside the linguistic variable UOD" in ve.get("msg", "")


def test_get_fuzzy_set_invalid_term(simple_linguistic_variable: LinguisticVariable) -> None:
    """Test retrieval of fuzzy sets with an invalid term."""
    with pytest.raises(ValueError) as exc:
        simple_linguistic_variable.get_fuzzy_set("freezing")

    err = exc.value
    assert "Fuzzy set for term 'freezing' not found in linguistic variable 'temperature'" in str(err)

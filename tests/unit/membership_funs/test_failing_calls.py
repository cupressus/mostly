import numpy as np
import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    "mf_fixture_name",
    [
        "regular_triangular_mf",
        "triangular_trapezoidal_mf",
        "regular_gaussian_mf",
        "regular_bimodal_gaussian_mf",
    ],
)
@pytest.mark.parametrize(
    "input",
    [
        pytest.param(None, id="None"),
        pytest.param("nan", id="nan"),
        pytest.param(np.nan, id="numpy nan"),
        pytest.param("", id="empty string"),
        pytest.param([], id="empty list"),
        pytest.param({}, id="empty dict"),
        pytest.param(set(), id="empty set"),
        pytest.param(np.inf, id="numpy inf"),
        pytest.param(-np.inf, id="numpy -inf"),
    ],
)
def test_invalid_input_type(request, mf_fixture_name, input) -> None:
    """Test invalid input type for membership function."""
    mf = request.getfixturevalue(mf_fixture_name)
    with pytest.raises(ValidationError):
        mf(input)

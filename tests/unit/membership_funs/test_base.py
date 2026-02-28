import math

import pytest

from src.mostly.membership_funs.base import MembershipFunction


def test_cannot_instantiate_abstract_membership_function():
    """Ensure MembershipFunction is abstract and cannot be instantiated directly."""
    # Abstract methods (__call__, support) must be implemented by subclasses
    with pytest.raises(TypeError):
        MembershipFunction()  # type: ignore[abstract]


def test_concrete_subclass_implements_interface(dummy_mf):
    """Verify that a concrete subclass of MembershipFunction implements the interface correctly."""
    assert isinstance(dummy_mf, MembershipFunction)

    # __call__ returns a numeric value
    val = dummy_mf(1.0)
    assert isinstance(val, float)
    assert math.isfinite(val)
    assert val == pytest.approx(1.5)

    # support returns a length-2 tuple of finite floats with ordered bounds
    left, right = dummy_mf.support()
    assert isinstance(left, float) and isinstance(right, float)
    assert math.isfinite(left) and math.isfinite(right)
    assert left < right


def test_support_and_call_contract_consistency(dummy_mf):
    """Verify that the support and __call__ methods are consistent in a subclass."""
    left, right = dummy_mf.support()

    # Outside support, this dummy returns linear values; for the abstract contract
    # we only assert that support is a pair of finite floats and callable executes.
    # This guards the base class expectations without over-constraining behavior.
    for x in (left, (left + right) / 2, right):
        val = dummy_mf(x)
        assert isinstance(val, float)
        assert math.isfinite(val)

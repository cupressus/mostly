from abc import ABC, abstractmethod

from pydantic import BaseModel, FiniteFloat


class MembershipFunction(BaseModel, ABC):
    """Abstract base for callable scalar membership functions.

    Subclasses define how to evaluate a value and validate their inputs.
    """

    @abstractmethod
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Evaluate the membership function at a scalar value.

        Args:
            x: Value to evaluate.

        Returns:
            The membership degree for `x`.

        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__ method")  # pragma: no cover

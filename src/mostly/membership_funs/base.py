from abc import ABC, abstractmethod

from pydantic import BaseModel, FiniteFloat


class MembershipFunction(BaseModel, ABC):
    """Abstract Base Class for Membership Functions."""

    @abstractmethod
    def __call__(self, x: FiniteFloat) -> FiniteFloat:
        """Calculate Degree of Membership for a given input `x`."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__ method")

    @abstractmethod
    def support(self) -> tuple[FiniteFloat, FiniteFloat]:
        """Provide 0 Cutoffs for the Membership Function."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement support method")

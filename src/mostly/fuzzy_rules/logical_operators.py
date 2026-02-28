from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints
from pydantic.dataclasses import dataclass

SnakedStr = Annotated[
    str,
    Field(StringConstraints(strip_whitespace=True, to_lower=True)),
    AfterValidator(lambda v: v.replace(" ", "_")),
]


@dataclass(frozen=True)
class Is:
    """Represents a fuzzy condition.

    Attributes:
        concept : str
            The concept to which the term belongs e.g. 'temperature'.
        term : str
            The term representing the fuzzy state of the concept e.g. 'hot', 'cold'.

    Examples:
        >>> condition = Is(concept="temperature", term="hot")
        >>> condition.pretty()
        "(temperature IS hot)"

    """

    concept: SnakedStr
    term: SnakedStr

    def eval(self, fuzzified: dict[str, dict[str, float]]) -> float:
        """Evaluate the condition against fuzzified input.

        Attributes:
            fuzzified : dict[str, dict[str, float]]
                A dictionary mapping concepts to their terms and corresponding degrees of membership.
                e.g. {'temperature': {'hot': 0.8, 'cold': 0.2}}

        """
        return fuzzified.get(self.concept, {}).get(self.term, 0.0)

    def get_variables(self) -> set[str]:
        """Return the set of variable names used in this condition."""
        return {self.concept}

    def pretty(self) -> str:
        """Return a human-readable string representation of the condition."""
        return f"({self.concept} IS {self.term})"


@dataclass(frozen=True)
class And:
    """Represents a conjunction of fuzzy conditions."""

    children: list["Is | And | Or | Not"]

    def eval(self, fuzzified: dict[str, dict[str, float]]) -> float:
        """Evaluate the conjunction against fuzzified input."""
        return min(child.eval(fuzzified) for child in self.children)

    def get_variables(self) -> set[str]:
        """Return the set of variable names used in the conjunction."""
        return set().union(*(c.get_variables() for c in self.children))

    def pretty(self) -> str:
        """Return a human-readable string representation of the conjunction."""
        return "(" + " AND ".join(child.pretty() for child in self.children) + ")"


@dataclass(frozen=True)
class Or:
    """Represents a disjunction of fuzzy conditions."""

    children: list["Is | And | Or | Not"]

    def eval(self, fuzzified: dict[str, dict[str, float]]) -> float:
        """Evaluate the disjunction against fuzzified input."""
        return max(child.eval(fuzzified) for child in self.children)

    def get_variables(self) -> set[str]:
        """Return the set of variable names used in the disjunction."""
        return set().union(*(c.get_variables() for c in self.children))

    def pretty(self) -> str:
        """Return a human-readable string representation of the disjunction."""
        return "(" + " OR ".join(child.pretty() for child in self.children) + ")"


@dataclass(frozen=True)
class Not:
    """Represents a negation of a fuzzy condition."""

    child: "Is | And | Or"

    def eval(self, fuzzified: dict[str, dict[str, float]]) -> float:
        """Evaluate the negation against fuzzified input."""
        return 1.0 - self.child.eval(fuzzified)

    def get_variables(self) -> set[str]:
        """Return the set of variable names used in the negation."""
        return self.child.get_variables()

    def pretty(self) -> str:
        """Return a human-readable string representation of the negation."""
        return f"NOT {self.child.pretty()}"

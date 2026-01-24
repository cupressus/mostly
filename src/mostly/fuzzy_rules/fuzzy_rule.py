# %%
from pydantic import BaseModel, ConfigDict, FiniteFloat

from .logical_operators import And, Is, Not, Or


class FuzzyRule(BaseModel):
    """A single fuzzy rule with an antecedent and consequences.

    Attributes:
        antecedent : Is | And | Or | Not
            The antecedent (IF part) of the rule, which can be a simple condition or
            a combination of conditions using AND, OR, and NOT.
        consequences : dict[str, str]
            The consequences (THEN part) of the rule, mapping concepts to their resulting terms.
            For example {"temperature": "hot"}.
        weight : FiniteFloat, optional
            The weight of the rule, default is 1.0. This can be used to adjust the influence of the rule
            on the output.

    Example:
    >>> rule = FuzzyRule(
    ...     antecedent=And(
    ...         children=[
    ...             Is(concept="temperature", term="hot"),
    ...             Or([
    ...                 Is(concept="humidity", term="high"),
    ...                 Not(Is(concept="wind", term="strong"))
    ...             ]),
    ...         ]
    ...     ),
    ...     consequences={"fan_speed": "high"},
    ...     weight=1.0,
    ... )

    """

    model_config = ConfigDict(str_strip_whitespace=True, str_to_lower=True)

    antecedent: Is | And | Or | Not
    consequences: dict[str, str]
    weight: FiniteFloat = 1.0

    def eval(self, fuzzified_input: dict[str, dict[str, float]]) -> FiniteFloat:
        """Evaluate the rule against a fuzzified input."""
        return self.weight * self.antecedent.eval(fuzzified_input)

    def get_variable_names(self) -> set[str]:
        """Return the set of variable names used in the antecedent."""
        return self.antecedent.get_variables()

    def pretty(self) -> str:
        """Return a human-readable string representation of the rule."""
        cons_str = "(" + " AND ".join(f"{var} IS {term}" for var, term in self.consequences.items()) + ")"
        return f"IF {self.antecedent.pretty()} THEN {cons_str} [weight: {self.weight}]"

    def __str__(self):
        """Return a string representation of the rule."""
        return self.pretty()

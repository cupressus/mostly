from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, FiniteFloat, StringConstraints, model_validator, validate_call

from .membership_funs.base import MembershipFunction

# force Concept to be snake_case using pydantic AfterValidator
SnakedStr = Annotated[
    str,
    Field(StringConstraints(strip_whitespace=True, to_lower=True)),
    AfterValidator(lambda v: v.replace(" ", "_")),
]


class LinguisticVariable(BaseModel):
    """A concept (e.g. 'temperature') described by fuzzy terms (e.g. 'hot', 'cold').

    Attributes
    ----------
    concept : str
        The *concept* represented by the linguistic variable (e.g. 'temperature').
        Strings forced to lowercase and stripped whitespace.
    uod : tuple[FiniteFloat, FiniteFloat]
        The range of accepted values, known as *Universe of Discourse (UOD)*, e.g. (0.0, 100.0).
    fuzzy_sets : dict[Term, MembershipFunction]
        A mapping of *terms* (e.g. 'hot', 'cold') to their corresponding *membership functions* - *Fuzzy Set*.

    Methods
    -------
    fuzzify(x)
        Fuzzify a *crisp finite float* into *degrees of membership* to each *term*.
    get_fuzzy_set(term)
        Retrieve the *membership function* corresponding to a given *term*.

    """

    concept: SnakedStr
    uod: tuple[FiniteFloat, FiniteFloat]
    fuzzy_sets: dict[SnakedStr, MembershipFunction]

    @model_validator(mode="after")
    def uod_compliance(self) -> "LinguisticVariable":
        """Ensure all fuzzy sets are compliant with the Universe of Discourse (UOD)."""
        uod_min, uod_max = self.uod
        for term, fs in self.fuzzy_sets.items():
            fs_uod_min, fs_uod_max = fs.support()
            if fs_uod_min < uod_min or fs_uod_max > uod_max:
                raise ValueError(
                    f"Fuzzy set for term '{term}' has UOD ({fs_uod_min}, {fs_uod_max}) "
                    f"which is outside the linguistic variable UOD ({uod_min}, {uod_max})."
                )
        return self

    @validate_call
    def fuzzify(self, x: FiniteFloat) -> dict[SnakedStr, FiniteFloat]:
        """Fuzzify a given input value into a dictionary of *terms* and their *degree of membership*.

        Parameters
        ----------
        x : FiniteFloat
            The input value to be fuzzified.

        Returns
        -------
        dict[Term, FiniteFloat]
            A dictionary where the keys are terms and the values are their degrees of membership.
            For instance `{'cold': 0.8, 'medium': 0.2, 'warm': 0.0}`

        """
        return {term: fs(x) for term, fs in self.fuzzy_sets.items()}

    def get_fuzzy_set(self, term: SnakedStr) -> MembershipFunction:
        """Retrieve the fuzzy set associated with a given term.

        Parameters
        ----------
        term : str
            The linguistic term (e.g. 'hot') for which the corresponding fuzzy set is to be retrieved.

        Returns
        -------
        MembershipFunction
            The membership function associated with the specified term.

        Raises
        ------
        ValueError
            If no membership function is found for the given term in the linguistic variable.

        """
        try:
            return self.fuzzy_sets[term]
        except KeyError:
            raise ValueError(
                f"Fuzzy set for term '{term}' not found in linguistic variable '{self.concept}'. "
                f"Available terms: {', '.join(self.fuzzy_sets.keys())}."
            ) from None

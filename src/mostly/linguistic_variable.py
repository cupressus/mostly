from typing import Annotated

import numpy as np
from pydantic import AfterValidator, BaseModel, FiniteFloat, StringConstraints, model_validator, validate_call

from .membership_functions import MembershipFunction

SnakedStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True),
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
    def validate_uod_bounds(self) -> "LinguisticVariable":
        """Validate that UOD bounds are properly ordered.

        Raises
        ------
        ValueError
            If the minimum UOD value is greater than or equal to the maximum.

        """
        if self.uod[0] >= self.uod[1]:
            raise ValueError(
                f"Invalid UOD bounds for '{self.concept}': minimum ({self.uod[0]}) "
                f"must be strictly less than maximum ({self.uod[1]})."
            )
        return self

    @model_validator(mode="after")
    def validate_membership_quality(self) -> "LinguisticVariable":
        """Validate that membership functions return valid values and provide full UOD coverage.

        Checks that:
        1. No membership function returns NaN (critical - breaks inference)
        2. Every point in the UOD has non-zero membership to at least one term (prevents gaps)

        Raises
        ------
        ValueError
            If any membership function returns NaN, or if any point in the UOD has zero membership to all terms.

        """
        # Sample UOD with adaptive resolution: scale with UOD distance, bounded [50, 1000]
        # This ensures consistent coverage detection across different UOD scales
        uod_distance = self.uod[1] - self.uod[0]
        num_samples = max(50, min(int(np.ceil(uod_distance)) + 1, 1000))
        sample_points = np.linspace(self.uod[0], self.uod[1], num=num_samples)
        uncovered_points = []

        for x in sample_points:
            memberships = []
            for term, fs in self.fuzzy_sets.items():
                membership_value = fs(float(x))

                # Priority 1: Check for NaN (critical failure)
                if np.isnan(membership_value):
                    raise ValueError(
                        f"Membership function for term '{term}' in linguistic variable '{self.concept}' "
                        f"returns NaN at x={x}. This indicates a broken membership function that will "
                        f"corrupt inference calculations."
                    )

                memberships.append(membership_value)

            # Priority 2: Check for coverage gaps
            if max(memberships) == 0.0:
                uncovered_points.append(x)

        # Report uncovered ranges if any exist
        if uncovered_points:
            # Group consecutive points into ranges for clearer error message
            ranges = []
            start = uncovered_points[0]
            prev = uncovered_points[0]

            for point in uncovered_points[1:]:
                if not np.isclose(point - prev, sample_points[1] - sample_points[0], rtol=1e-9):
                    # Gap detected, close current range
                    ranges.append((start, prev))
                    start = point
                prev = point
            ranges.append((start, prev))

            range_strs = [f"[{r[0]:.2f}, {r[1]:.2f}]" for r in ranges]
            raise ValueError(
                f"Incomplete coverage in linguistic variable '{self.concept}': "
                f"the following ranges in the UOD have zero membership to all terms: {', '.join(range_strs)}. "
                f"Every point in the UOD [{self.uod[0]}, {self.uod[1]}] must have non-zero membership "
                f"to at least one term."
            )

        return self

    @validate_call
    def fuzzify(self, x: FiniteFloat) -> dict[SnakedStr, FiniteFloat]:
        """Fuzzify a given input value into a dictionary of *terms* and their *degree of membership*.

        Parameters
        ----------
        x : FiniteFloat
            The input value to be fuzzified. Must be within the UOD bounds.

        Returns
        -------
        dict[Term, FiniteFloat]
            A dictionary where the keys are terms and the values are their degrees of membership.
            For instance `{'cold': 0.8, 'medium': 0.2, 'warm': 0.0}`

        Raises
        ------
        ValueError
            If the input value is outside the UOD bounds.

        """
        # Validate input is within UOD bounds
        if not (self.uod[0] <= x <= self.uod[1]):
            raise ValueError(
                f"Input value {x} is outside the UOD bounds [{self.uod[0]}, {self.uod[1]}] "
                f"for linguistic variable '{self.concept}'."
            )

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

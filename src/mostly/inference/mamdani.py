from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, FiniteFloat, validate_call

from ..fuzzy_rules.fuzzy_rule import FuzzyRule
from ..linguistic_variable import LinguisticVariable
from ..membership_funs.base import MembershipFunction


class MamdaniFIS(BaseModel):
    """A Mamdani Fuzzy Inference System (FIS).

    Attributes
    ----------
    input_variables : dict[str, LinguisticVariable]
        Concepts mapped to their linguistic variables to be used as input variables.

    output_variables : dict[str, LinguisticVariable]
        Concepts mapped to their linguistic variables to be used as output variables.

    fuzzy_rules : list[FuzzyRule]
        A list of fuzzy rules defining the inference logic.

    meta_fields : dict[str, Any], optional
        Additional metadata fields for the FIS.

    """

    input_variables: dict[str, LinguisticVariable]
    output_variables: dict[str, LinguisticVariable]
    fuzzy_rules: list[FuzzyRule]
    meta_fields: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _fuzzification(self, crisp_inputs: dict[str, FiniteFloat]) -> dict[str, dict[str, FiniteFloat]]:
        """Fuzzify crisp inputs based on the linguistic variables membership functions.

        Parameters
        ----------
        crisp_inputs : dict[str, FiniteFloat]
            A dictionary mapping input concept names to their crisp values, e.g. {'temperature': 25.0}.

        Returns
        -------
        dict[str, dict[str, FiniteFloat]]
            A dictionary mapping concepts to their fuzzified terms and their degrees of membership,
            e.g. {'temperature': {'hot': 0.8, 'warm': 0.2}}.

        """
        fuzzified = {}
        for concept, value in crisp_inputs.items():
            if concept not in self.input_variables:
                raise ValueError(
                    f"Input variable '{concept}' not defined in FIS. "
                    f"Valid concepts are: {list(self.input_variables.keys())}."
                )
            lv = self.input_variables[concept]
            fuzzified[concept] = lv.fuzzify(value)
        return fuzzified

    def _rule_evaluation(self, fuzzified: dict[str, dict[str, FiniteFloat]]) -> list[tuple[FuzzyRule, FiniteFloat]]:
        """Calculate the strength of each rule based on the fuzzified inputs.

        Returns
        -------
        list[tuple[FuzzyRule, FiniteFloat]]
            A list of tuples containing each rule and its corresponding firing strength.

        """
        strengths = []
        for rule in self.fuzzy_rules:
            strength: FiniteFloat = rule.eval(fuzzified)
            strengths.append((rule, strength))
        return strengths

    def _implication(
        self,
        mf: MembershipFunction,
        strength: FiniteFloat,
        x_vals: np.ndarray,
        implication: Literal["clip", "scale"] = "clip",
    ) -> np.ndarray:
        """Apply the implication method to modify the membership function."""
        match implication:
            case "clip":
                return np.array([min(mf(x), strength) for x in x_vals])
            case "scale":
                return np.array([mf(x) * strength for x in x_vals])
            case _:
                raise ValueError(f"Unknown implication method: {implication}")

    def _aggregation(
        self,
        consequences: list[tuple[FuzzyRule, FiniteFloat]],
        resolution: int = 500,
        aggregation: Literal["max", "sum", "probor"] = "max",
        implication: Literal["clip", "scale"] = "clip",
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Aggregate the outputs of the rules based on the specified method.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray]]
            A dictionary mapping concepts to their aggregated x values and membership values,
            e.g. {'fan_speed': (x_vals, agg_vals)}.

        """
        output_aggregation = {}

        for concept, lv in self.output_variables.items():
            x_min, x_max = lv.uod
            x_vals = np.linspace(x_min, x_max, resolution)
            agg_vals = np.zeros_like(x_vals)

            for rule, strength in consequences:
                if concept not in rule.consequences:
                    continue

                mf: MembershipFunction = lv.get_fuzzy_set(rule.consequences[concept])
                clipped_vals = self._implication(mf, strength, x_vals, implication)

                # Apply the aggregation method using match-case
                match aggregation:
                    case "max":
                        agg_vals = np.maximum(agg_vals, clipped_vals)
                    case "sum":
                        agg_vals += clipped_vals
                    case "probor":
                        agg_vals += clipped_vals - (agg_vals * clipped_vals)
                    case _:
                        raise ValueError(f"Unknown aggregation method: {aggregation}")

            output_aggregation[concept] = (x_vals, agg_vals)

        return output_aggregation

    def _defuzzification(
        self,
        aggregated_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
        method: Literal["centroid"] = "centroid",
    ) -> dict[str, float]:
        """Defuzzify the aggregated outputs to get crisp values using the specified method.

        Returns
        -------
        dict[str, float]
            A dictionary mapping concepts to their defuzzified crisp values, e.g. {'fan_speed': 22.5}.

        """
        defuzzified = {}
        for concept, (x_vals, agg_vals) in aggregated_outputs.items():
            if concept not in self.output_variables:
                raise ValueError(f"Output variable '{concept}' not defined in FIS.")

            match method:
                case "centroid":
                    # Use centroid method for defuzzification
                    numerator = np.sum(x_vals * agg_vals)
                    denominator = np.sum(agg_vals)
                    if denominator == 0:
                        defuzzified[concept] = 0.0
                    else:
                        defuzzified[concept] = numerator / denominator
                case _:
                    raise ValueError(f"Unknown defuzzification method: {method}")

        return defuzzified

    @validate_call
    def infer(
        self,
        crisp_inputs: dict[str, float],
        resolution: int = 500,
        aggregation: Literal["max", "sum", "probor"] = "max",
        implication: Literal["clip", "scale"] = "clip",
        defuzzification: Literal["centroid"] = "centroid",
    ) -> dict[str, float]:
        """Perform fuzzy inference on the given inputs.

        Parameters
        ----------
        crisp_inputs : dict[str, float]
            A dictionary mapping input concept names to their crisp values, e.g. {'temperature': 25.0}.
        resolution: int
            The number of points to use for output aggregation.
        aggregation: Literal["max", "sum", "probor"]
            The method to use for aggregating rule outputs.
        implication: Literal["clip", "scale"]
            The method to use for applying rule strengths to output membership functions.
        defuzzification: Literal["centroid"]
            The method to use for defuzzifying the aggregated outputs.

        Returns
        -------
        dict[str, float]
            A dictionary mapping concepts to their defuzzified crisp values, e.g. {'fan_speed': 22.5}.

        """
        fuzzified_inputs = self._fuzzification(crisp_inputs)
        rule_strengths = self._rule_evaluation(fuzzified_inputs)
        aggregated_outputs = self._aggregation(rule_strengths, resolution, aggregation, implication)
        return self._defuzzification(aggregated_outputs, defuzzification)

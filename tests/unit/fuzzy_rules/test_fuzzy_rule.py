from src.mostly.fuzzy_rules.fuzzy_rule import FuzzyRule
from src.mostly.fuzzy_rules.logical_operators import And, Is, Not, Or


class TestGetVariableNames:
    """Tests for FuzzyRule.get_variable_names() method."""

    def test_simple_is_condition(self):
        """Test get_variable_names with a simple Is condition."""
        rule = FuzzyRule(
            antecedent=Is(concept="temperature", term="hot"),
            consequences={"fan_speed": "high"},
        )
        assert rule.get_variable_names() == {"temperature"}

    def test_and_with_two_variables(self):
        """Test get_variable_names with And combining two different variables."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Is(concept="humidity", term="high"),
                ]
            ),
            consequences={"fan_speed": "high"},
        )
        assert rule.get_variable_names() == {"temperature", "humidity"}

    def test_and_with_same_variable(self):
        """Test get_variable_names with And combining the same variable twice."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Is(concept="temperature", term="very_hot"),
                ]
            ),
            consequences={"fan_speed": "high"},
        )
        assert rule.get_variable_names() == {"temperature"}

    def test_or_with_multiple_variables(self):
        """Test get_variable_names with Or combining multiple variables."""
        rule = FuzzyRule(
            antecedent=Or(
                children=[
                    Is(concept="temperature", term="hot"),
                    Is(concept="humidity", term="high"),
                    Is(concept="pressure", term="low"),
                ]
            ),
            consequences={"alert": "on"},
        )
        assert rule.get_variable_names() == {"temperature", "humidity", "pressure"}

    def test_not_condition(self):
        """Test get_variable_names with Not wrapping a condition."""
        rule = FuzzyRule(
            antecedent=Not(child=Is(concept="wind", term="strong")),
            consequences={"window": "open"},
        )
        assert rule.get_variable_names() == {"wind"}

    def test_nested_and_or(self):
        """Test get_variable_names with nested And/Or conditions."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Or(
                        children=[
                            Is(concept="humidity", term="high"),
                            Is(concept="wind", term="strong"),
                        ]
                    ),
                ]
            ),
            consequences={"fan_speed": "high"},
        )
        assert rule.get_variable_names() == {"temperature", "humidity", "wind"}

    def test_complex_nested_structure(self):
        """Test get_variable_names with complex nested structure including Not."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Or(
                        children=[
                            Is(concept="humidity", term="high"),
                            Not(child=Is(concept="wind", term="strong")),
                        ]
                    ),
                ]
            ),
            consequences={"fan_speed": "high"},
        )
        assert rule.get_variable_names() == {"temperature", "humidity", "wind"}

    def test_multiple_occurrences_in_nested_structure(self):
        """Test that variables appearing multiple times in nested structures are counted once."""
        rule = FuzzyRule(
            antecedent=Or(
                children=[
                    And(
                        children=[
                            Is(concept="temperature", term="hot"),
                            Is(concept="humidity", term="high"),
                        ]
                    ),
                    And(
                        children=[
                            Is(concept="temperature", term="very_hot"),
                            Is(concept="pressure", term="low"),
                        ]
                    ),
                ]
            ),
            consequences={"alert": "critical"},
        )
        assert rule.get_variable_names() == {"temperature", "humidity", "pressure"}


class TestStr:
    """Tests for FuzzyRule.__str__() method."""

    def test_str_simple_rule(self):
        """Test __str__ with a simple rule."""
        rule = FuzzyRule(
            antecedent=Is(concept="temperature", term="hot"),
            consequences={"fan_speed": "high"},
        )
        expected = "IF (temperature IS hot) THEN (fan_speed IS high) [weight: 1.0]"
        assert str(rule) == expected

    def test_str_with_and(self):
        """Test __str__ with And condition."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Is(concept="humidity", term="high"),
                ]
            ),
            consequences={"fan_speed": "high"},
        )
        expected = "IF ((temperature IS hot) AND (humidity IS high)) " "THEN (fan_speed IS high) [weight: 1.0]"
        assert str(rule) == expected

    def test_str_with_or(self):
        """Test __str__ with Or condition."""
        rule = FuzzyRule(
            antecedent=Or(
                children=[
                    Is(concept="food_quality", term="poor"),
                    Is(concept="service_quality", term="poor"),
                ]
            ),
            consequences={"tip_amount": "low"},
        )
        expected = "IF ((food_quality IS poor) OR (service_quality IS poor)) " "THEN (tip_amount IS low) [weight: 1.0]"
        assert str(rule) == expected

    def test_str_with_not(self):
        """Test __str__ with Not condition."""
        rule = FuzzyRule(
            antecedent=Not(child=Is(concept="wind", term="strong")),
            consequences={"window": "open"},
        )
        expected = "IF NOT (wind IS strong) THEN (window IS open) [weight: 1.0]"
        assert str(rule) == expected

    def test_str_with_custom_weight(self):
        """Test __str__ with a custom weight."""
        rule = FuzzyRule(
            antecedent=Is(concept="temperature", term="hot"),
            consequences={"fan_speed": "high"},
            weight=0.8,
        )
        expected = "IF (temperature IS hot) THEN (fan_speed IS high) [weight: 0.8]"
        assert str(rule) == expected

    def test_str_with_multiple_consequences(self):
        """Test __str__ with multiple consequences."""
        rule = FuzzyRule(
            antecedent=Is(concept="temperature", term="hot"),
            consequences={"fan_speed": "high", "ac_mode": "cool"},
        )
        result = str(rule)
        # Check that both consequences are present (order may vary in dict)
        assert result.startswith("IF (temperature IS hot) THEN (")
        assert "fan_speed IS high" in result
        assert "ac_mode IS cool" in result
        assert result.endswith(") [weight: 1.0]")

    def test_str_nested_complex_rule(self):
        """Test __str__ with complex nested conditions."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Or(
                        children=[
                            Is(concept="humidity", term="high"),
                            Not(child=Is(concept="wind", term="strong")),
                        ]
                    ),
                ]
            ),
            consequences={"fan_speed": "high"},
            weight=0.9,
        )
        expected = (
            "IF ((temperature IS hot) AND "
            "((humidity IS high) OR NOT (wind IS strong))) "
            "THEN (fan_speed IS high) [weight: 0.9]"
        )
        assert str(rule) == expected

    def test_str_matches_pretty(self):
        """Test that __str__ returns the same as pretty()."""
        rule = FuzzyRule(
            antecedent=And(
                children=[
                    Is(concept="temperature", term="hot"),
                    Is(concept="humidity", term="high"),
                ]
            ),
            consequences={"fan_speed": "high"},
            weight=0.75,
        )
        assert str(rule) == rule.pretty()

    def test_str_with_snake_case_conversion(self):
        """Test __str__ with string normalization (spaces to underscores)."""
        rule = FuzzyRule(
            antecedent=Is(concept="air temperature", term="very hot"),
            consequences={"fan speed": "very high"},
        )
        # Pydantic config should convert "air temperature" to "air_temperature"
        expected = "IF (air_temperature IS very_hot) THEN (fan_speed IS very_high) [weight: 1.0]"
        assert str(rule) == expected

# mostly

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

`mostly` is a lightweight, typed Python library for defining and evaluating
Mamdani fuzzy inference systems. Describe linguistic variables and rules in
plain Python, then turn crisp inputs into crisp outputs.

## Features

- Build linguistic variables with validated, complete membership coverage.
- Choose from triangular, trapezoidal, Gaussian, bell, sigmoidal, polynomial,
  crisp, and bimodal Gaussian membership functions.
- Compose readable fuzzy rules with `Is`, `And`, `Or`, and `Not`.
- Run Mamdani inference with configurable implication and aggregation.
- Visualize variables and inference with the optional Altair backend.

## Installation

`mostly` requires Python 3.13 or later.

```bash
pip install mostly
```

To include the optional Altair plotting tools:

```bash
pip install "mostly[plotting]"
```

<details>
<summary>Using <code>uv</code> or <code>poetry</code> </summary>

```bash
# uv
uv add mostly
uv add "mostly[plotting]"

# Poetry
poetry add mostly
poetry add "mostly[plotting]"
```

</details>

## Quick start

The example below defines a small tipping system with two input variables,
one output variable, and three rules:

```python
from mostly import FuzzyRule, Is, LinguisticVariable, MamdaniFIS, Or
from mostly.membership_functions import MFTriangular

food_quality = LinguisticVariable(
    concept="food_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangular(a=0.0, b=0.0, c=5.0),
        "good": MFTriangular(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangular(a=5.0, b=10.0, c=10.0),
    },
)

service_quality = LinguisticVariable(
    concept="service_quality",
    uod=(0.0, 10.0),
    fuzzy_sets={
        "poor": MFTriangular(a=0.0, b=0.0, c=5.0),
        "good": MFTriangular(a=0.0, b=5.0, c=10.0),
        "excellent": MFTriangular(a=5.0, b=10.0, c=10.0),
    },
)

tip_amount = LinguisticVariable(
    concept="tip_amount",
    uod=(0.0, 25.0),
    fuzzy_sets={
        "low": MFTriangular(a=0.0, b=0.0, c=13.0),
        "medium": MFTriangular(a=0.0, b=13.0, c=25.0),
        "high": MFTriangular(a=13.0, b=25.0, c=25.0),
    },
)

rules = [
    FuzzyRule(
        antecedent=Or(
            [
                Is(concept="food_quality", term="poor"),
                Is(concept="service_quality", term="poor"),
            ]
        ),
        consequences={"tip_amount": "low"},
    ),
    FuzzyRule(
        antecedent=Is(concept="service_quality", term="good"),
        consequences={"tip_amount": "medium"},
    ),
    FuzzyRule(
        antecedent=Or(
            [
                Is(concept="food_quality", term="excellent"),
                Is(concept="service_quality", term="excellent"),
            ]
        ),
        consequences={"tip_amount": "high"},
    ),
]

fis = MamdaniFIS(
    input_variables={
        "food_quality": food_quality,
        "service_quality": service_quality,
    },
    output_variables={"tip_amount": tip_amount},
    fuzzy_rules=rules,
)

result = fis.infer({"food_quality": 6.5, "service_quality": 9.8})
print(result["tip_amount"])
```

`infer()` returns a dictionary of defuzzified output values. By default,
`mostly` uses centroid defuzzification, clipped implication, and maximum
aggregation.

## Configuring inference

Pass an `InferenceConfig` to tune the output resolution, implication, and
aggregation:

```python
from mostly import InferenceConfig, MamdaniFIS

config = InferenceConfig(
    resolution=1_000,
    implication="clip",
    aggregation="max",
)

# Pass config alongside input_variables, output_variables, and fuzzy_rules.
fis = MamdaniFIS(
    input_variables={...},
    output_variables={...},
    fuzzy_rules=[...],
    inference_config=config,
)
```

Supported aggregation methods are `"max"`, `"sum"`, and `"probor"`;
implication can be `"clip"` or `"scale"`.

## Plotting

After installing the `plotting` extra, import the Altair backend explicitly:

```python
from mostly.plotting.altair import (
    plot_fuzzy_set,
    plot_inference_inputs,
    plot_inference_outputs,
    plot_linguistic_variable,
)

chart = plot_linguistic_variable(food_quality)
chart
```

The same backend can visualize an individual fuzzy set as well as a system's
fuzzified inputs and aggregated outputs.

## Documentation and examples

- [Introduction](./docs/index.md)
- [Membership functions tutorial](./docs/mf.ipynb)
- [Tipping-problem tutorial](./docs/tipping_problem.ipynb)
- [API source](./src/mostly)

## Development

Install all dependency groups, then run the tests:

```bash
uv sync --all-groups
uv run pytest
```

To preview the documentation locally:

```bash
uv run mkdocs serve
```

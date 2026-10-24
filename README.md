# mostly

A lightweight Python library for building fuzzy inference systems.

---

## Installation

Install the core fuzzy-inference workflow:

```bash
pip install mostly
```

Install Altair plotting support when needed:

```bash
pip install "mostly[plotting]"
```

## Usage

Import the primary building blocks directly from `mostly`; import the complete
membership-function catalog from its dedicated namespace:

```python
from mostly import And, FuzzyRule, Is, LinguisticVariable, MamdaniFIS
from mostly.membership_functions import MFTriangular
```

Altair utilities are backend-specific and available after installing the
`plotting` extra:

```python
from mostly.plotting.altair import (
    plot_fuzzy_set,
    plot_inference_inputs,
    plot_inference_outputs,
    plot_linguistic_variable,
)
```

The names exported by `mostly`, `mostly.inference`, `mostly.fuzzy_rules`, and
`mostly.membership_functions` are the stable public API. Concrete
implementation modules are internal and may change between releases.

## Documentation

## Changelog

## License

The project is licensed under the MIT License.
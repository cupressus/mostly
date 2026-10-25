# mostly

A lightweight Python library for building fuzzy inference systems.

## Public API

The primary building blocks are available directly from `mostly`:

```python
from mostly import And, FuzzyRule, Is, LinguisticVariable, MamdaniFIS
from mostly.membership_functions import MFTriangular
```

Membership-function implementations remain in
`mostly.membership_functions`. The names exported by `mostly`,
`mostly.inference`, `mostly.fuzzy_rules`, and
`mostly.membership_functions` are stable public API; concrete implementation
modules are internal.

## Plotting

Install the optional plotting dependencies:

```bash
pip install "mostly[plotting]"
```

Altair utilities use an explicit backend namespace:

```python
from mostly.plotting.altair import plot_linguistic_variable
```       # Other markdown pages, images and other files.

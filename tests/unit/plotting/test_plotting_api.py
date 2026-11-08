"""Tests for the plotting import boundary and public API."""

import subprocess
import sys

from mostly.plotting.altair import (
    mostly_light,
    plot_fuzzy_set,
    plot_inference_inputs,
    plot_inference_outputs,
    plot_inference_rules,
    plot_linguistic_variable,
)


def test_core_import_does_not_load_plotting_dependencies() -> None:
    """Ensure importing the core API does not initialize plotting packages."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; import mostly; assert 'altair' not in sys.modules; assert 'pandas' not in sys.modules"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_altair_exports_documented_plotting_utilities() -> None:
    """Verify that the Altair backend exposes the documented utilities."""
    assert callable(mostly_light)
    assert callable(plot_fuzzy_set)
    assert callable(plot_inference_inputs)
    assert callable(plot_inference_outputs)
    assert callable(plot_inference_rules)
    assert callable(plot_linguistic_variable)

"""Shared RAPID helpers for synthetic and state-based workflows.

Import concrete functionality from submodules, for example:

- `rapid_tools.engine`
- `rapid_tools.prep`
- `rapid_tools.adapters.synthetic`

Selected high-level helpers are exposed lazily so importing `rapid_tools` does
not eagerly pull in the full dependency stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "HydrographMetricConfig": ("hydrograph", "HydrographMetricConfig"),
    "RapidStateContext": ("registry", "RapidStateContext"),
    "iter_preparable_states": ("registry", "iter_preparable_states"),
    "load_state_registry": ("registry", "load_state_registry"),
    "prepare_experiment": ("prep", "prepare_experiment"),
    "prepare_state": ("prep", "prepare_state"),
    "run_prepared_experiment": ("engine", "run_prepared_experiment"),
    "run_prepared_state": ("engine", "run_prepared_state"),
    "summarize_outlet_hydrograph": ("hydrograph", "summarize_outlet_hydrograph"),
    "write_hydrograph_outputs": ("hydrograph", "write_hydrograph_outputs"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)

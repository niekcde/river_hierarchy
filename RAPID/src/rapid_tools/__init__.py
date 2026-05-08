"""Shared RAPID engine, prep helpers, and workflow adapters.

Import concrete functionality from submodules, for example:

- `rapid_tools.engine`
- `rapid_tools.prep`
- `rapid_tools.adapters.synthetic`

This package `__init__` stays lightweight so adapter/prep imports do not force
the full RAPID engine dependency stack in environments that only need partial
functionality.
"""

__all__ = []

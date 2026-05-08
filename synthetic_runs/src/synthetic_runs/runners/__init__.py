"""Entry points and control helpers for sampled synthetic runs and sensitivity runs."""

from .controls import (
    default_single_edge_control_path,
    load_single_edge_control,
    resolve_single_edge_control,
)


def run_sampled_realizations(*args, **kwargs):
    from .sampled import run_sampled_realizations as _impl

    return _impl(*args, **kwargs)


def run_sensitivity_grid(*args, **kwargs):
    from .sensitivity import run_sensitivity_grid as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "default_single_edge_control_path",
    "load_single_edge_control",
    "resolve_single_edge_control",
    "run_sampled_realizations",
    "run_sensitivity_grid",
]

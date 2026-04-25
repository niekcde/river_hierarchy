"""Config-driven workflow entry points for regular and sensitivity pipelines."""


def run_geometry_from_config(*args, **kwargs):
    from .regular import run_geometry_from_config as _impl

    return _impl(*args, **kwargs)


def run_sampling_from_config(*args, **kwargs):
    from .regular import run_sampling_from_config as _impl

    return _impl(*args, **kwargs)


def run_routing_from_config(*args, **kwargs):
    from .regular import run_routing_from_config as _impl

    return _impl(*args, **kwargs)


def run_k_metrics_from_config(*args, **kwargs):
    from .regular import run_k_metrics_from_config as _impl

    return _impl(*args, **kwargs)


def build_recipes_from_config(*args, **kwargs):
    from .sensitivity import build_recipes_from_config as _impl

    return _impl(*args, **kwargs)


def run_grid_from_config(*args, **kwargs):
    from .sensitivity import run_grid_from_config as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "build_recipes_from_config",
    "run_geometry_from_config",
    "run_grid_from_config",
    "run_k_metrics_from_config",
    "run_routing_from_config",
    "run_sampling_from_config",
]

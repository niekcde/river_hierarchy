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
    "build_all_matched_link_summary": ("analysis", "build_all_matched_link_summary"),
    "build_combined_analysis_frame": ("analysis", "build_combined_analysis_frame"),
    "build_matched_corridor_hydrograph_frame": ("analysis", "build_matched_corridor_hydrograph_frame"),
    "build_experiment_analysis_frame": ("analysis", "build_experiment_analysis_frame"),
    "build_parent_link_difference_timeseries": ("analysis", "build_parent_link_difference_timeseries"),
    "build_reach_timing_comparison": ("analysis", "build_reach_timing_comparison"),
    "compare_hydrographs": ("analysis", "compare_hydrographs"),
    "compute_wave_intensity_matrix": ("analysis", "compute_wave_intensity_matrix"),
    "compute_reach_timing_metrics": ("analysis", "compute_reach_timing_metrics"),
    "build_geometry_context_frame": ("plotting", "build_geometry_context_frame"),
    "build_multi_example_timing_frame": ("plotting", "build_multi_example_timing_frame"),
    "HydrographMetricConfig": ("hydrograph", "HydrographMetricConfig"),
    "load_reach_wave_stack": ("analysis", "load_reach_wave_stack"),
    "make_timing_scatter": ("plotting", "make_timing_scatter"),
    "RapidStateContext": ("registry", "RapidStateContext"),
    "iter_preparable_states": ("registry", "iter_preparable_states"),
    "load_state_registry": ("registry", "load_state_registry"),
    "load_state_hydrograph": ("analysis", "load_state_hydrograph"),
    "plot_corridor_difference_heatmap": ("analysis", "plot_corridor_difference_heatmap"),
    "plot_effect_signature": ("analysis", "plot_effect_signature"),
    "plot_hydrograph_overlay": ("analysis", "plot_hydrograph_overlay"),
    "plot_matched_link_difference_scatter": ("analysis", "plot_matched_link_difference_scatter"),
    "plot_matched_corridor_hydrographs": ("analysis", "plot_matched_corridor_hydrographs"),
    "plot_reach_timing_map": ("analysis", "plot_reach_timing_map"),
    "summarize_matched_corridor_peak_timing": ("analysis", "summarize_matched_corridor_peak_timing"),
    "prepare_experiment": ("prep", "prepare_experiment"),
    "prepare_state": ("prep", "prepare_state"),
    "run_prepared_experiment": ("engine", "run_prepared_experiment"),
    "run_prepared_state": ("engine", "run_prepared_state"),
    "supported_timing_x_metrics": ("plotting", "supported_timing_x_metrics"),
    "supported_timing_y_metrics": ("plotting", "supported_timing_y_metrics"),
    "supported_timing_metrics": ("plotting", "supported_timing_metrics"),
    "summarize_outlet_hydrograph": ("hydrograph", "summarize_outlet_hydrograph"),
    "write_wave_comparison_mp4": ("analysis", "write_wave_comparison_mp4"),
    "write_wave_gif": ("analysis", "write_wave_gif"),
    "write_wave_mp4": ("analysis", "write_wave_mp4"),
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

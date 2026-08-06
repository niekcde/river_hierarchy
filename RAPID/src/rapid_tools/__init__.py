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
    "build_actual_sword_node_frame": ("slope_investigation", "build_actual_sword_node_frame"),
    "build_low_k_parent_summary": ("slope_investigation", "build_low_k_parent_summary"),
    "build_slope_link_frame": ("slope_investigation", "build_slope_link_frame"),
    "build_slope_node_frame": ("slope_investigation", "build_slope_node_frame"),
    "build_slope_profile_long_frame": ("slope_investigation", "build_slope_profile_long_frame"),
    "identify_slope_seed_links": ("slope_investigation", "identify_slope_seed_links"),
    "list_slope_states": ("slope_investigation", "list_slope_states"),
    "plot_actual_sword_vs_state_vs_monotone": ("slope_investigation", "plot_actual_sword_vs_state_vs_monotone"),
    "plot_raw_vs_monotone_profile": ("slope_investigation", "plot_raw_vs_monotone_profile"),
    "plot_slope_plan_view": ("slope_investigation", "plot_slope_plan_view"),
    "plot_slope_profiles": ("slope_investigation", "plot_slope_profiles"),
    "plot_slope_ratio_scatter": ("slope_investigation", "plot_slope_ratio_scatter"),
    "plot_sword_vs_monotone_profile": ("slope_investigation", "plot_sword_vs_monotone_profile"),
    "build_timestep_tail_frame": ("timestep_stability", "build_timestep_tail_frame"),
    "list_stability_states": ("timestep_stability", "list_stability_states"),
    "plot_effective_length_vs_dt": ("timestep_stability", "plot_effective_length_vs_dt"),
    "plot_timestep_tail_suite": ("timestep_stability", "plot_timestep_tail_suite"),
    "summarize_outlet_hydrograph": ("hydrograph", "summarize_outlet_hydrograph"),
    "write_wave_comparison_mp4": ("analysis", "write_wave_comparison_mp4"),
    "write_hydro_pulse_gif": ("analysis", "write_hydro_pulse_gif"),
    "write_reference_peak_wave_gif": ("analysis", "write_reference_peak_wave_gif"),
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

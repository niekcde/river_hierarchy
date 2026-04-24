"""Post-run analysis and inspection utilities for synthetic experiments."""

from .k_metrics import compute_k_metrics_for_network, compute_metrics
from .plotting import (
    iter_recipe_records,
    load_recipe_records,
    plot_network,
    plot_recipe,
    plot_recipe_grid,
    recipe_break_summary,
    recipe_summary_frame,
    recipe_title,
)
from .run_level_rf import (
    RunLevelRFResult,
    canonicalize_run_columns,
    merge_edge_summary_features,
    prepare_run_level_dataset,
    run_random_forest_from_files,
    run_random_forest_regression,
    save_run_level_result,
    summarize_edge_geometry,
)

__all__ = [
    "RunLevelRFResult",
    "canonicalize_run_columns",
    "compute_k_metrics_for_network",
    "compute_metrics",
    "iter_recipe_records",
    "load_recipe_records",
    "merge_edge_summary_features",
    "plot_network",
    "plot_recipe",
    "plot_recipe_grid",
    "prepare_run_level_dataset",
    "recipe_break_summary",
    "recipe_summary_frame",
    "recipe_title",
    "run_random_forest_from_files",
    "run_random_forest_regression",
    "save_run_level_result",
    "summarize_edge_geometry",
]

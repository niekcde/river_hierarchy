from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.collapse_decisions import (
    compute_collapse_decisions_from_unit_metrics,
    summarize_group_count_selection,
)
from hierarchy_level_definition.metrics import (
    compute_unit_metrics,
    summarize_by_hierarchy_level,
)


@dataclass
class UnitWorkflowOutputs:
    unit_summary: pd.DataFrame
    unit_metrics: pd.DataFrame
    path_metrics: pd.DataFrame
    hierarchy_level_metrics: pd.DataFrame
    collapse_ranking: pd.DataFrame
    merge_tree: pd.DataFrame
    bubble_summary: pd.DataFrame
    group_count_summary: pd.DataFrame
    selected_groups: pd.DataFrame


def _git_revision() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _default_output_dir(links_path: str | Path) -> Path:
    links_path = Path(links_path)
    network_name = links_path.stem
    for suffix in ("_links", "directed_links"):
        if network_name.endswith(suffix):
            network_name = network_name[: -len(suffix)]
            break
    return Path(__file__).resolve().parent / "outputs" / network_name


def select_optimal_groups(
    merge_tree: pd.DataFrame,
    group_count_summary: pd.DataFrame,
) -> pd.DataFrame:
    if merge_tree.empty or group_count_summary.empty:
        return pd.DataFrame(columns=list(merge_tree.columns))

    selected = group_count_summary.loc[group_count_summary["is_optimal_n_groups"]]
    if selected.empty:
        return pd.DataFrame(columns=list(merge_tree.columns))

    optimal_n_groups = int(selected["n_groups"].iloc[0])
    return (
        merge_tree.loc[merge_tree["n_groups"] == optimal_n_groups]
        .sort_values("group_index", kind="mergesort")
        .reset_index(drop=True)
    )


def run_unit_workflow(
    links_path: str | Path,
    nodes_path: str | Path,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
    classification_thresholds: Mapping[str, float] | None = None,
    ranking_sequence: Sequence[tuple[str, bool]] | None = None,
    merge_feature_columns: Sequence[str] | None = None,
    merge_log_feature_columns: Sequence[str] | None = None,
    debug: bool = False,
) -> UnitWorkflowOutputs:
    unit_summary, unit_metrics, path_metrics = compute_unit_metrics(
        links_path,
        nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
        pixel_width_fields=pixel_width_fields,
        pixel_width_percentiles=pixel_width_percentiles,
        use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
        classification_thresholds=classification_thresholds,
        debug=debug,
    )
    hierarchy_level_metrics = summarize_by_hierarchy_level(unit_metrics)
    collapse_ranking, merge_tree, bubble_summary = compute_collapse_decisions_from_unit_metrics(
        unit_metrics,
        ranking_sequence=ranking_sequence,
        merge_feature_columns=merge_feature_columns,
        merge_log_feature_columns=merge_log_feature_columns,
    )
    group_count_summary = summarize_group_count_selection(merge_tree)
    selected_groups = select_optimal_groups(merge_tree, group_count_summary)

    return UnitWorkflowOutputs(
        unit_summary=unit_summary,
        unit_metrics=unit_metrics,
        path_metrics=path_metrics,
        hierarchy_level_metrics=hierarchy_level_metrics,
        collapse_ranking=collapse_ranking,
        merge_tree=merge_tree,
        bubble_summary=bubble_summary,
        group_count_summary=group_count_summary,
        selected_groups=selected_groups,
    )


def write_unit_workflow_outputs(
    output_dir: str | Path,
    results: UnitWorkflowOutputs,
    *,
    manifest_overrides: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results.unit_summary.to_csv(output_path / "unit_summary.csv", index=False)
    results.path_metrics.to_csv(output_path / "path_metrics.csv", index=False)
    results.unit_metrics.to_csv(output_path / "unit_metrics.csv", index=False)
    results.hierarchy_level_metrics.to_csv(output_path / "hierarchy_level_metrics.csv", index=False)
    results.collapse_ranking.to_csv(output_path / "collapse_ranking.csv", index=False)
    results.merge_tree.to_csv(output_path / "merge_tree.csv", index=False)
    results.merge_tree.to_csv(output_path / "ordered_group_partitions.csv", index=False)
    results.group_count_summary.to_csv(output_path / "group_count_summary.csv", index=False)
    results.bubble_summary.to_csv(output_path / "bubble_summary.csv", index=False)
    results.selected_groups.to_csv(output_path / "selected_groups.csv", index=False)

    metrics_config = dict(results.unit_metrics.attrs.get("metrics_config", {}))
    collapse_config = dict(results.merge_tree.attrs.get("collapse_config", {}))
    if manifest_overrides:
        collapse_config.update(dict(manifest_overrides))

    optimal_n_groups = None
    if not results.group_count_summary.empty:
        selected = results.group_count_summary.loc[results.group_count_summary["is_optimal_n_groups"]]
        if not selected.empty:
            optimal_n_groups = int(selected["n_groups"].iloc[0])

    manifest = {
        "files": [
            "unit_summary.csv",
            "path_metrics.csv",
            "unit_metrics.csv",
            "hierarchy_level_metrics.csv",
            "collapse_ranking.csv",
            "merge_tree.csv",
            "ordered_group_partitions.csv",
            "group_count_summary.csv",
            "bubble_summary.csv",
            "selected_groups.csv",
        ],
        "n_units": int(len(results.unit_metrics)),
        "n_paths": int(len(results.path_metrics)),
        "n_group_rows": int(len(results.merge_tree)),
        "n_selected_groups": int(len(results.selected_groups)),
        "optimal_n_groups": optimal_n_groups,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
        **metrics_config,
        **collapse_config,
    }
    with (output_path / "unit_workflow_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full hierarchy-unit workflow on reviewed graph-like links/nodes files: "
            "unit detection, path metrics, unit metrics, collapse ranking, ordered group partitions, "
            "group-count selection, and final selected groups."
        )
    )
    parser.add_argument("links_gpkg", help="Reviewed links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Reviewed nodes GeoPackage.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to hierarchy_level_definition/outputs/<network_name>/")
    parser.add_argument("--max-path-cutoff", type=int, default=100, help="Maximum edge-count cutoff for unit path enumeration.")
    parser.add_argument("--max-paths", type=int, default=5000, help="Maximum number of simple paths per unit.")
    parser.add_argument(
        "--pixel-width-fields",
        nargs="*",
        default=None,
        help="Optional candidate field names containing per-link width samples.",
    )
    parser.add_argument(
        "--pixel-width-percentiles",
        nargs="*",
        type=float,
        default=None,
        help="Optional percentile set for width diagnostics, e.g. 5 50 95.",
    )
    parser.add_argument(
        "--disable-pixel-width-extremes",
        action="store_true",
        help="Force path width min/max diagnostics to use representative widths instead of pixel samples.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable unit-detection debug output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = run_unit_workflow(
        args.links_gpkg,
        args.nodes_gpkg,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        pixel_width_fields=args.pixel_width_fields,
        pixel_width_percentiles=args.pixel_width_percentiles,
        use_pixel_widths_for_extremes=not args.disable_pixel_width_extremes,
        debug=args.debug,
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(args.links_gpkg)
    write_unit_workflow_outputs(output_dir, results)

    selected = results.group_count_summary.loc[results.group_count_summary["is_optimal_n_groups"]]
    optimal_n_groups = int(selected["n_groups"].iloc[0]) if not selected.empty else None

    print(f"Wrote workflow outputs to {output_dir}")
    print(f"Units: {len(results.unit_metrics)}")
    print(f"Paths: {len(results.path_metrics)}")
    print(f"Ordered group rows: {len(results.merge_tree)}")
    print(f"Optimal n_groups: {optimal_n_groups}")
    print("Selected groups:")
    if results.selected_groups.empty:
        print("(none)")
    else:
        print(results.selected_groups.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

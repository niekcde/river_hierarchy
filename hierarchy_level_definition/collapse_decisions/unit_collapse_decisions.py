from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.metrics import compute_unit_metrics

REQUIRED_UNIT_COLUMNS = (
    "unit_id",
    "compound_bubble_id",
    "primary_parent_id",
    "compound_unit_id",
    "compound_bubble_role",
    "unit_node_ids",
    "n_valid_paths",
    "effective_n_paths_width",
    "path_disparity_width",
    "equivalent_length",
    "elongation",
    "topologic_complexity_score",
)

DEFAULT_RANKING_SEQUENCE: tuple[tuple[str, bool], ...] = (
    ("path_disparity_width", True),
    ("elongation", False),
    ("equivalent_length", False),
    ("topologic_complexity_score", False),
)

DEFAULT_MERGE_FEATURE_COLUMNS = (
    "path_disparity_width",
    "elongation",
    "equivalent_length",
    "topologic_complexity_score",
)

DEFAULT_LOG_FEATURE_COLUMNS = ("equivalent_length",)
DEFAULT_MAX_GROUP_COUNT = 6


def _ensure_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def _with_collapse_attrs(frame: pd.DataFrame, collapse_config: Mapping[str, Any]) -> pd.DataFrame:
    metrics_config = dict(frame.attrs.get("metrics_config", {}))
    frame.attrs = {
        "metrics_config": metrics_config,
        "collapse_config": dict(collapse_config),
    }
    return frame


def _normalize_feature_columns(columns: Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return DEFAULT_MERGE_FEATURE_COLUMNS
    normalized = tuple(dict.fromkeys(str(column) for column in columns))
    if not normalized:
        raise ValueError("feature_columns must contain at least one column name.")
    return normalized


def _normalize_log_feature_columns(columns: Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return DEFAULT_LOG_FEATURE_COLUMNS
    return tuple(dict.fromkeys(str(column) for column in columns))


def _normalize_bubble_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _normalize_optional_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _parse_node_ids(value: object) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        if value == "":
            return frozenset()
        return frozenset(int(token.strip()) for token in value.split(",") if token.strip())
    if pd.isna(value):
        return frozenset()
    if isinstance(value, Iterable):
        return frozenset(int(token) for token in value)
    return frozenset()


def _normalized_rank_component(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid_mask = numeric.notna() & np.isfinite(numeric)
    result = pd.Series(0.0, index=series.index, dtype=float)
    valid = numeric.loc[valid_mask]
    if valid.empty:
        return result
    if len(valid) == 1:
        result.loc[valid.index] = 1.0
        return result
    ranks = valid.rank(method="average", ascending=not higher_is_better)
    scale = len(valid) - 1
    result.loc[valid.index] = 1.0 - (ranks - 1.0) / float(scale)
    return result


def _transform_numeric_series(series: pd.Series, *, log_transform: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if not log_transform:
        return numeric
    transformed = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = numeric.notna() & np.isfinite(numeric) & (numeric > -1.0)
    transformed.loc[valid_mask] = np.log1p(numeric.loc[valid_mask])
    return transformed


def _zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        valid_mask = values.notna() & np.isfinite(values)
        if not valid_mask.any():
            standardized[column] = np.nan
            continue
        valid = values.loc[valid_mask]
        std = float(valid.std(ddof=0))
        if std <= 0.0 or math.isnan(std):
            standardized[column] = 0.0
            standardized.loc[~valid_mask, column] = np.nan
            continue
        mean = float(valid.mean())
        standardized[column] = (values - mean) / std
    return standardized


def _euclidean_distance_ignore_nan(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return float("inf")
    diff = left[mask] - right[mask]
    return float(np.sqrt(np.sum(diff * diff)))


def _bubble_sort_value(value: int | None) -> tuple[int, int]:
    if value is None:
        return (1, 0)
    return (0, int(value))


def _build_unit_adjacency(unit_metrics: pd.DataFrame) -> nx.Graph:
    _ensure_columns(unit_metrics, REQUIRED_UNIT_COLUMNS)
    graph = nx.Graph()
    records = unit_metrics.sort_values("unit_id", kind="mergesort").to_dict("records")
    node_sets = {int(record["unit_id"]): _parse_node_ids(record["unit_node_ids"]) for record in records}

    for record in records:
        unit_id = int(record["unit_id"])
        graph.add_node(
            unit_id,
            compound_bubble_id=_normalize_bubble_id(record["compound_bubble_id"]),
            primary_parent_id=_normalize_optional_int(record["primary_parent_id"]),
            compound_unit_id=_normalize_optional_int(record["compound_unit_id"]),
            node_ids=node_sets[unit_id],
        )

    for index, left in enumerate(records):
        left_unit_id = int(left["unit_id"])
        left_bubble_id = _normalize_bubble_id(left["compound_bubble_id"])
        left_parent_id = _normalize_optional_int(left["primary_parent_id"])
        left_compound_unit_id = _normalize_optional_int(left["compound_unit_id"])
        left_nodes = node_sets[left_unit_id]

        for right in records[index + 1 :]:
            right_unit_id = int(right["unit_id"])
            right_bubble_id = _normalize_bubble_id(right["compound_bubble_id"])
            if left_bubble_id != right_bubble_id:
                continue

            right_parent_id = _normalize_optional_int(right["primary_parent_id"])
            right_compound_unit_id = _normalize_optional_int(right["compound_unit_id"])
            right_nodes = node_sets[right_unit_id]

            reasons: list[str] = []
            if left_nodes & right_nodes:
                reasons.append("shared_nodes")
            if left_parent_id == right_unit_id or right_parent_id == left_unit_id:
                reasons.append("parent_child")
            if left_parent_id is not None and left_parent_id == right_parent_id:
                reasons.append("siblings")
            if (
                left_compound_unit_id is not None
                and right_compound_unit_id is not None
                and left_compound_unit_id == right_compound_unit_id
            ):
                reasons.append("same_outer_unit")

            if reasons:
                graph.add_edge(
                    left_unit_id,
                    right_unit_id,
                    reasons=tuple(sorted(set(reasons))),
                )

    return graph


def rank_unit_collapse_priority(
    unit_metrics: pd.DataFrame,
    *,
    ranking_sequence: Sequence[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    _ensure_columns(unit_metrics, REQUIRED_UNIT_COLUMNS)
    ranking_sequence = tuple(ranking_sequence or DEFAULT_RANKING_SEQUENCE)
    ranking = unit_metrics.copy()

    component_columns: list[str] = []
    for column, higher_is_better in ranking_sequence:
        component_column = f"{column}_rank_component"
        ranking[component_column] = _normalized_rank_component(ranking[column], higher_is_better=higher_is_better)
        component_columns.append(component_column)

    ranking["collapse_priority_score"] = ranking[component_columns].fillna(0.0).mean(axis=1)

    ranking = ranking.sort_values(
        [
            "collapse_priority_score",
            "path_disparity_width",
            "elongation",
            "equivalent_length",
            "topologic_complexity_score",
            "n_valid_paths",
            "unit_id",
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranking["collapse_order_global"] = pd.Series(np.arange(1, len(ranking) + 1), dtype="Int64")
    ranking["collapse_order_in_bubble"] = (
        ranking.groupby("compound_bubble_id", sort=False, dropna=False).cumcount() + 1
    ).astype("Int64")

    collapse_config = {
        "decision_scope": "unit collapse ranking and constrained within-bubble merge tree",
        "ranking_sequence": [
            {"column": column, "higher_is_better": higher_is_better}
            for column, higher_is_better in ranking_sequence
        ],
        "ranking_score_method": "mean of rank-normalized decision components",
    }
    return _with_collapse_attrs(ranking, collapse_config)


def _cluster_bridge_reasons(
    left_members: Sequence[int],
    right_members: Sequence[int],
    adjacency_graph: nx.Graph,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    for left_unit_id in left_members:
        for right_unit_id in right_members:
            if not adjacency_graph.has_edge(left_unit_id, right_unit_id):
                continue
            reasons.update(adjacency_graph.edges[left_unit_id, right_unit_id].get("reasons", ()))
    return tuple(sorted(reasons))


def _cluster_centroid(
    member_unit_ids: Sequence[int],
    standardized_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    subset = standardized_features.loc[list(member_unit_ids), list(feature_columns)]
    return subset.mean(axis=0, skipna=True).to_numpy(dtype=float)


def _cluster_bubble_ids(
    member_unit_ids: Sequence[int],
    ranking_by_id: pd.DataFrame,
) -> tuple[int, ...]:
    bubble_ids = {
        _normalize_bubble_id(ranking_by_id.loc[int(unit_id), "compound_bubble_id"])
        for unit_id in member_unit_ids
    }
    normalized = [bubble_id for bubble_id in bubble_ids if bubble_id is not None]
    return tuple(sorted(normalized))


def _cluster_rank_bounds(
    member_unit_ids: Sequence[int],
    ranking_by_id: pd.DataFrame,
) -> tuple[int, int]:
    ranks = [int(ranking_by_id.loc[int(unit_id), "collapse_order_global"]) for unit_id in member_unit_ids]
    return min(ranks), max(ranks)


def _ordered_partition_costs(
    standardized_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    matrix = standardized_features.loc[:, list(feature_columns)].to_numpy(dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    n_units = matrix.shape[0]
    costs = np.zeros((n_units, n_units), dtype=float)
    for start in range(n_units):
        segment = matrix[start:]
        cumulative = np.cumsum(segment, axis=0)
        cumulative_sq = np.cumsum(segment * segment, axis=0)
        for offset in range(segment.shape[0]):
            length = offset + 1
            sums = cumulative[offset]
            sums_sq = cumulative_sq[offset]
            sse = sums_sq - (sums * sums) / float(length)
            costs[start, start + offset] = float(np.sum(sse))
    return costs


def _optimal_contiguous_partitions(costs: np.ndarray, n_groups: int) -> tuple[float, list[tuple[int, int]]]:
    n_units = costs.shape[0]
    if n_groups <= 0 or n_groups > n_units:
        raise ValueError("n_groups must be between 1 and the number of ranked units.")

    dp = np.full((n_groups + 1, n_units + 1), np.inf, dtype=float)
    prev = np.full((n_groups + 1, n_units + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for group_count in range(1, n_groups + 1):
        for end_index in range(group_count, n_units + 1):
            best_cost = np.inf
            best_start = -1
            for start_index in range(group_count - 1, end_index):
                cost = dp[group_count - 1, start_index] + costs[start_index, end_index - 1]
                if cost < best_cost:
                    best_cost = cost
                    best_start = start_index
            dp[group_count, end_index] = best_cost
            prev[group_count, end_index] = best_start

    boundaries: list[tuple[int, int]] = []
    end_index = n_units
    group_count = n_groups
    while group_count > 0:
        start_index = int(prev[group_count, end_index])
        if start_index < 0:
            raise RuntimeError("Failed to reconstruct contiguous group partition.")
        boundaries.append((start_index, end_index))
        end_index = start_index
        group_count -= 1
    boundaries.reverse()
    return float(dp[n_groups, n_units]), boundaries


def _perpendicular_distance_to_line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    denominator = math.hypot(x2 - x1, y2 - y1)
    if denominator <= 0.0 or math.isnan(denominator):
        return 0.0
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    return float(numerator / denominator)


def build_constrained_merge_tree(
    unit_metrics: pd.DataFrame,
    *,
    ranking_sequence: Sequence[tuple[str, bool]] | None = None,
    feature_columns: Sequence[str] | None = None,
    log_feature_columns: Sequence[str] | None = None,
    max_group_count: int = DEFAULT_MAX_GROUP_COUNT,
) -> pd.DataFrame:
    _ensure_columns(unit_metrics, REQUIRED_UNIT_COLUMNS)
    feature_columns = _normalize_feature_columns(feature_columns)
    log_feature_columns = _normalize_log_feature_columns(log_feature_columns)
    _ensure_columns(unit_metrics, feature_columns)

    if "collapse_order_global" in unit_metrics.columns:
        ranking = unit_metrics.copy()
    else:
        ranking = rank_unit_collapse_priority(unit_metrics, ranking_sequence=ranking_sequence)

    ranking = ranking.sort_values("collapse_order_global", kind="mergesort").reset_index(drop=True)
    ranking_by_id = ranking.set_index(ranking["unit_id"].astype(int), drop=False)

    feature_frame = pd.DataFrame(index=ranking["unit_id"].astype(int))
    for column in feature_columns:
        feature_frame[column] = _transform_numeric_series(
            ranking_by_id[column],
            log_transform=column in log_feature_columns,
        )

    standardized_features = _zscore_frame(feature_frame.loc[:, list(feature_columns)])
    records: list[dict[str, Any]] = []
    n_units = len(ranking)
    if n_units <= 1:
        grouping = pd.DataFrame(
            columns=[
                "n_groups",
                "group_index",
                "group_label",
                "group_size",
                "unit_ids",
                "rank_start",
                "rank_end",
                "compound_bubble_ids",
                "mean_collapse_priority_score",
                "mean_path_disparity_width",
                "mean_effective_n_paths_width",
                "mean_n_valid_paths",
                "mean_equivalent_length",
                "mean_elongation",
                "mean_topologic_complexity_score",
                "within_group_cost",
                "partition_total_cost",
                "grouping_rule",
            ]
        )
        return _with_collapse_attrs(
            grouping,
            {
                "group_scope": "global ordered contiguous group partitions across the full collapse ranking",
                "group_constraint": "groups must be contiguous in global collapse ranking",
                "merge_feature_columns": list(feature_columns),
                "merge_log_feature_columns": list(log_feature_columns),
                "group_distance": "within-group sum of squared distances in globally standardized feature space",
                "group_count_range": [2, min(max_group_count, n_units)],
            },
        )

    costs = _ordered_partition_costs(standardized_features, feature_columns)
    max_group_count = max(2, min(int(max_group_count), n_units))

    for n_groups in range(2, max_group_count + 1):
        partition_total_cost, boundaries = _optimal_contiguous_partitions(costs, n_groups)
        for group_index, (start_index, end_index) in enumerate(boundaries, start=1):
            group_frame = ranking.iloc[start_index:end_index]
            unit_ids = [int(unit_id) for unit_id in group_frame["unit_id"].tolist()]
            rank_start = int(group_frame["collapse_order_global"].min())
            rank_end = int(group_frame["collapse_order_global"].max())
            bubble_ids = _cluster_bubble_ids(unit_ids, ranking_by_id)
            within_group_cost = float(costs[start_index, end_index - 1])

            records.append(
                {
                    "n_groups": n_groups,
                    "group_index": group_index,
                    "group_label": f"G{n_groups}_{group_index}",
                    "group_size": len(unit_ids),
                    "unit_ids": ",".join(str(unit_id) for unit_id in unit_ids),
                    "rank_start": rank_start,
                    "rank_end": rank_end,
                    "compound_bubble_ids": ",".join(str(bubble_id) for bubble_id in bubble_ids),
                    "mean_collapse_priority_score": float(group_frame["collapse_priority_score"].mean()),
                    "mean_path_disparity_width": float(group_frame["path_disparity_width"].mean()),
                    "mean_effective_n_paths_width": float(group_frame["effective_n_paths_width"].mean()),
                    "mean_n_valid_paths": float(group_frame["n_valid_paths"].mean()),
                    "mean_equivalent_length": float(group_frame["equivalent_length"].mean()),
                    "mean_elongation": float(group_frame["elongation"].mean()),
                    "mean_topologic_complexity_score": float(group_frame["topologic_complexity_score"].mean()),
                    "within_group_cost": within_group_cost,
                    "partition_total_cost": partition_total_cost,
                    "grouping_rule": "optimal_contiguous_partition",
                }
            )

    merge_tree = pd.DataFrame.from_records(
        records,
        columns=[
            "n_groups",
            "group_index",
            "group_label",
            "group_size",
            "unit_ids",
            "rank_start",
            "rank_end",
            "compound_bubble_ids",
            "mean_collapse_priority_score",
            "mean_path_disparity_width",
            "mean_effective_n_paths_width",
            "mean_n_valid_paths",
            "mean_equivalent_length",
            "mean_elongation",
            "mean_topologic_complexity_score",
            "within_group_cost",
            "partition_total_cost",
            "grouping_rule",
        ],
    )
    if merge_tree.empty:
        merge_tree = pd.DataFrame(
            columns=[
                "n_groups",
                "group_index",
                "group_label",
                "group_size",
                "unit_ids",
                "rank_start",
                "rank_end",
                "compound_bubble_ids",
                "mean_collapse_priority_score",
                "mean_path_disparity_width",
                "mean_effective_n_paths_width",
                "mean_n_valid_paths",
                "mean_equivalent_length",
                "mean_elongation",
                "mean_topologic_complexity_score",
                "within_group_cost",
                "partition_total_cost",
                "grouping_rule",
            ]
        )
    else:
        for column in (
            "n_groups",
            "group_index",
            "group_size",
            "rank_start",
            "rank_end",
        ):
            merge_tree[column] = merge_tree[column].astype("Int64")

    collapse_config = {
        "group_scope": "global ordered contiguous group partitions across the full collapse ranking",
        "group_constraint": "groups must be contiguous in global collapse ranking",
        "merge_feature_columns": list(feature_columns),
        "merge_log_feature_columns": list(log_feature_columns),
        "group_distance": "within-group sum of squared distances in globally standardized feature space",
        "group_count_range": [2, max_group_count],
    }
    return _with_collapse_attrs(merge_tree, collapse_config)


def summarize_group_count_selection(group_partitions: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(group_partitions, ("n_groups", "partition_total_cost"))

    if group_partitions.empty:
        summary = pd.DataFrame(
            columns=[
                "n_groups",
                "partition_total_cost",
                "cost_reduction_from_prev",
                "relative_cost_reduction_from_prev",
                "normalized_partition_total_cost",
                "elbow_score",
                "is_optimal_n_groups",
            ]
        )
        collapse_config = dict(group_partitions.attrs.get("collapse_config", {}))
        collapse_config.setdefault("group_count_selection_method", "global elbow on partition_total_cost")
        collapse_config.setdefault("optimal_n_groups", None)
        return _with_collapse_attrs(summary, collapse_config)

    summary = (
        group_partitions.groupby("n_groups", as_index=False, sort=True)
        .agg(partition_total_cost=("partition_total_cost", "first"))
        .sort_values("n_groups", kind="mergesort")
        .reset_index(drop=True)
    )

    summary["cost_reduction_from_prev"] = (
        summary["partition_total_cost"].shift(1) - summary["partition_total_cost"]
    )
    summary["relative_cost_reduction_from_prev"] = (
        summary["cost_reduction_from_prev"] / summary["partition_total_cost"].shift(1)
    )

    first_cost = float(summary["partition_total_cost"].iloc[0])
    if math.isfinite(first_cost) and first_cost != 0.0:
        summary["normalized_partition_total_cost"] = summary["partition_total_cost"] / first_cost
    else:
        summary["normalized_partition_total_cost"] = np.nan

    summary["elbow_score"] = 0.0
    if len(summary) >= 3:
        x1 = float(summary["n_groups"].iloc[0])
        y1 = float(summary["partition_total_cost"].iloc[0])
        x2 = float(summary["n_groups"].iloc[-1])
        y2 = float(summary["partition_total_cost"].iloc[-1])
        summary["elbow_score"] = summary.apply(
            lambda row: _perpendicular_distance_to_line(
                float(row["n_groups"]),
                float(row["partition_total_cost"]),
                x1,
                y1,
                x2,
                y2,
            ),
            axis=1,
        )
        summary.loc[summary.index[[0, len(summary) - 1]], "elbow_score"] = 0.0

    optimal_row = summary.sort_values(
        ["elbow_score", "n_groups"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    optimal_n_groups = int(optimal_row["n_groups"])
    summary["is_optimal_n_groups"] = summary["n_groups"].eq(optimal_n_groups)

    summary["n_groups"] = summary["n_groups"].astype("Int64")
    summary["is_optimal_n_groups"] = summary["is_optimal_n_groups"].astype(bool)

    collapse_config = dict(group_partitions.attrs.get("collapse_config", {}))
    collapse_config.setdefault("group_count_selection_method", "global elbow on partition_total_cost")
    collapse_config.setdefault("optimal_n_groups", optimal_n_groups)
    return _with_collapse_attrs(summary, collapse_config)


def summarize_collapse_bubbles(
    collapse_ranking: pd.DataFrame,
    merge_tree: pd.DataFrame,
) -> pd.DataFrame:
    _ensure_columns(
        collapse_ranking,
        (
            "compound_bubble_id",
            "compound_unit_id",
            "compound_bubble_role",
            "collapse_priority_score",
            "path_disparity_width",
            "effective_n_paths_width",
            "n_valid_paths",
            "equivalent_length",
            "elongation",
            "topologic_complexity_score",
            "unit_id",
        ),
    )

    records: list[dict[str, Any]] = []
    for bubble_value, group in collapse_ranking.groupby("compound_bubble_id", sort=True, dropna=False):
        bubble_id = _normalize_bubble_id(bubble_value)
        bubble_root_rows = group.loc[group["compound_bubble_role"] == "bubble_root", "unit_id"]
        if bubble_root_rows.empty:
            bubble_root_unit_id = int(group["unit_id"].min())
        else:
            bubble_root_unit_id = int(bubble_root_rows.min())

        if merge_tree.empty or bubble_id is None or "compound_bubble_ids" not in merge_tree.columns:
            merge_rows = merge_tree.iloc[0:0]
        else:
            bubble_token = str(bubble_id)
            merge_rows = merge_tree.loc[
                merge_tree["compound_bubble_ids"].fillna("").map(
                    lambda value: bubble_token in {token.strip() for token in str(value).split(",") if token.strip()}
                )
            ]
        n_group_rows = int(len(merge_rows))
        min_n_groups = int(merge_rows["n_groups"].min()) if n_group_rows > 0 else pd.NA
        max_n_groups = int(merge_rows["n_groups"].max()) if n_group_rows > 0 else pd.NA

        records.append(
            {
                "compound_bubble_id": bubble_id,
                "bubble_root_unit_id": bubble_root_unit_id,
                "n_units": int(len(group)),
                "n_compound_units": int(group["compound_unit_id"].notna().sum()),
                "n_group_rows": n_group_rows,
                "min_n_groups": min_n_groups,
                "max_n_groups": max_n_groups,
                "mean_collapse_priority_score": float(group["collapse_priority_score"].mean()),
                "mean_path_disparity_width": float(group["path_disparity_width"].mean()),
                "mean_effective_n_paths_width": float(group["effective_n_paths_width"].mean()),
                "mean_n_valid_paths": float(group["n_valid_paths"].mean()),
                "mean_equivalent_length": float(group["equivalent_length"].mean()),
                "mean_elongation": float(group["elongation"].mean()),
                "mean_topologic_complexity_score": float(group["topologic_complexity_score"].mean()),
            }
        )

    bubble_summary = pd.DataFrame.from_records(
        records,
        columns=[
            "compound_bubble_id",
            "bubble_root_unit_id",
            "n_units",
            "n_compound_units",
            "n_group_rows",
            "min_n_groups",
            "max_n_groups",
            "mean_collapse_priority_score",
            "mean_path_disparity_width",
            "mean_effective_n_paths_width",
            "mean_n_valid_paths",
            "mean_equivalent_length",
            "mean_elongation",
            "mean_topologic_complexity_score",
        ],
    )
    if not bubble_summary.empty:
        bubble_summary = bubble_summary.sort_values(
            "compound_bubble_id",
            key=lambda series: series.map(_bubble_sort_value),
            kind="mergesort",
        ).reset_index(drop=True)
    for column in ("compound_bubble_id", "bubble_root_unit_id", "n_units", "n_compound_units", "n_group_rows", "min_n_groups", "max_n_groups"):
        if column in bubble_summary:
            bubble_summary[column] = bubble_summary[column].astype("Int64")

    collapse_config = dict(collapse_ranking.attrs.get("collapse_config", {}))
    collapse_config.setdefault("bubble_summary_scope", "one row per compound_bubble_id")
    collapse_config.setdefault("bubble_summary_note", "bubble summaries are annotations on top of global contiguous group partitions")
    return _with_collapse_attrs(bubble_summary, collapse_config)


def compute_collapse_decisions_from_unit_metrics(
    unit_metrics: pd.DataFrame,
    *,
    ranking_sequence: Sequence[tuple[str, bool]] | None = None,
    merge_feature_columns: Sequence[str] | None = None,
    merge_log_feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    collapse_ranking = rank_unit_collapse_priority(
        unit_metrics,
        ranking_sequence=ranking_sequence,
    )
    merge_tree = build_constrained_merge_tree(
        unit_metrics,
        feature_columns=merge_feature_columns,
        log_feature_columns=merge_log_feature_columns,
    )
    bubble_summary = summarize_collapse_bubbles(collapse_ranking, merge_tree)
    return collapse_ranking, merge_tree, bubble_summary


def compute_collapse_decisions(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, unit_metrics, path_metrics = compute_unit_metrics(
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
    collapse_ranking, merge_tree, bubble_summary = compute_collapse_decisions_from_unit_metrics(
        unit_metrics,
        ranking_sequence=ranking_sequence,
        merge_feature_columns=merge_feature_columns,
        merge_log_feature_columns=merge_log_feature_columns,
    )
    return summary, unit_metrics, path_metrics, collapse_ranking, merge_tree, bubble_summary


def _git_revision() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
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


def write_collapse_outputs(
    output_dir: str | Path,
    collapse_ranking: pd.DataFrame,
    merge_tree: pd.DataFrame,
    bubble_summary: pd.DataFrame,
    *,
    group_count_summary: pd.DataFrame | None = None,
    manifest_overrides: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if group_count_summary is None:
        group_count_summary = summarize_group_count_selection(merge_tree)

    collapse_ranking.to_csv(output_path / "collapse_ranking.csv", index=False)
    merge_tree.to_csv(output_path / "constrained_merge_tree.csv", index=False)
    merge_tree.to_csv(output_path / "ordered_group_partitions.csv", index=False)
    group_count_summary.to_csv(output_path / "group_count_selection.csv", index=False)
    bubble_summary.to_csv(output_path / "bubble_summary.csv", index=False)

    metrics_config = dict(collapse_ranking.attrs.get("metrics_config", {}))
    collapse_config = dict(collapse_ranking.attrs.get("collapse_config", {}))
    if manifest_overrides:
        collapse_config.update(dict(manifest_overrides))

    manifest = {
        "files": [
            "collapse_ranking.csv",
            "constrained_merge_tree.csv",
            "ordered_group_partitions.csv",
            "group_count_selection.csv",
            "bubble_summary.csv",
        ],
        "n_ranked_units": int(len(collapse_ranking)),
        "n_group_rows": int(len(merge_tree)),
        "n_bubbles": int(len(bubble_summary)),
        "optimal_n_groups": group_count_summary.loc[group_count_summary["is_optimal_n_groups"], "n_groups"].iloc[0]
        if not group_count_summary.empty
        else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
        **metrics_config,
        **collapse_config,
    }
    with (output_path / "collapse_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build unit-level collapse ranking tables and ordered contiguous group partitions. "
            "The ranking is unit-based and uses direct hierarchy-aware metrics, while the second output "
            "contains non-overlapping contiguous groups along the global collapse ranking."
        )
    )
    parser.add_argument("links_gpkg", help="Reviewed links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Reviewed nodes GeoPackage.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for collapse outputs.")
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

    summary, unit_metrics, path_metrics, collapse_ranking, merge_tree, bubble_summary = compute_collapse_decisions(
        args.links_gpkg,
        args.nodes_gpkg,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        pixel_width_fields=args.pixel_width_fields,
        pixel_width_percentiles=args.pixel_width_percentiles,
        use_pixel_widths_for_extremes=not args.disable_pixel_width_extremes,
        debug=args.debug,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        network_name = Path(args.links_gpkg).stem.replace("_links", "")
        output_dir = Path(__file__).resolve().parent / "outputs" / network_name

    write_collapse_outputs(output_dir, collapse_ranking, merge_tree, bubble_summary)

    print(f"Wrote collapse outputs to {output_dir}")
    print(f"Units ranked: {len(collapse_ranking)}")
    print(f"Group rows: {len(merge_tree)}")
    print(f"Bubbles: {len(bubble_summary)}")
    print("Top-ranked units:")
    preview_columns = [
        "unit_id",
        "compound_bubble_id",
        "collapse_priority_score",
        "path_disparity_width",
        "n_valid_paths",
        "equivalent_length",
        "elongation",
        "topologic_complexity_score",
    ]
    print(collapse_ranking.loc[:, preview_columns].head(10).to_string(index=False))
    print("Top group rows:")
    print(merge_tree.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

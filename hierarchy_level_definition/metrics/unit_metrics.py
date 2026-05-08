from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    analyze_network,
    build_unit_context_frame,
    load_network,
)


LENGTH_COLUMNS = ("len_adj", "len")
REPRESENTATIVE_WIDTH_COLUMNS = ("wid_adj", "wid", "wid_med")
DEFAULT_PIXEL_WIDTH_FIELDS = ("width_samples", "width_px", "pixel_widths", "wid_pix", "wid_samples")
DEFAULT_PIXEL_WIDTH_PERCENTILES = (5.0, 50.0, 95.0)
DEFAULT_CLASSIFICATION_THRESHOLDS = {
    "high_evenness_threshold": 0.75,
    "low_evenness_threshold": 0.4,
    "dominant_fraction_threshold": 0.7,
    "balanced_two_path_smaller_fraction_threshold": 0.4,
    "high_topologic_complexity_threshold": 3.0,
}
PATH_WIDTH_EQ_METHOD = "length_weighted_harmonic_mean_of_representative_link_widths"
PATH_WIDTH_EXTREME_METHOD = "pixel_or_sample_widths_when_available_else_representative_link_widths"
UNIT_WIDTH_EQ_METHOD = "sum_of_path_equivalent_widths"
UNIT_LENGTH_EQ_METHOD = "path_width_weighted_mean_of_path_lengths"
DYNAMIC_PROXY_METHOD = "path_equivalent_width_fraction"
DYNAMIC_PROXY_WEIGHT_FIELD = "path_width_eq"
DIRECT_METRICS_ONLY = True
RECURSIVE_COLLAPSED_GEOMETRY_IMPLEMENTED = False


def infer_network_name(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_links", "_nodes"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_output_dir_for_links(links_path: str | Path) -> Path:
    return Path(__file__).resolve().parent / "outputs" / infer_network_name(links_path)


def edge_length_from_attrs(attrs: Mapping[str, Any], geometry: Any | None = None) -> float:
    for key in LENGTH_COLUMNS:
        value = attrs.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric

    geometry = geometry if geometry is not None else attrs.get("geometry")
    if geometry is not None:
        try:
            numeric = float(geometry.length)
        except (AttributeError, TypeError, ValueError):
            return float("nan")
        if math.isfinite(numeric):
            return numeric
    return float("nan")


def edge_width_from_attrs(attrs: Mapping[str, Any]) -> float:
    for key in REPRESENTATIVE_WIDTH_COLUMNS:
        value = attrs.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric > 0:
            return numeric
    return float("nan")


def _coerce_positive_numeric_array(values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    valid = np.isfinite(array) & (array > 0)
    return array[valid]


def edge_width_samples_from_attrs(
    attrs: Mapping[str, Any],
    pixel_width_fields: Sequence[str] | None = None,
) -> np.ndarray:
    fields = tuple(pixel_width_fields or DEFAULT_PIXEL_WIDTH_FIELDS)
    for field in fields:
        if field not in attrs:
            continue

        raw_value = attrs.get(field)
        if raw_value is None:
            continue
        if isinstance(raw_value, float) and pd.isna(raw_value):
            continue

        if isinstance(raw_value, np.ndarray):
            samples = _coerce_positive_numeric_array(raw_value)
        elif isinstance(raw_value, (list, tuple)):
            samples = _coerce_positive_numeric_array(raw_value)
        elif isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                continue
            parsed: Any = None
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None

            if isinstance(parsed, np.ndarray):
                samples = _coerce_positive_numeric_array(parsed)
            elif isinstance(parsed, (list, tuple)):
                samples = _coerce_positive_numeric_array(parsed)
            elif parsed is not None and not isinstance(parsed, str):
                samples = _coerce_positive_numeric_array([parsed])
            else:
                parts = [part.strip() for part in text.split(",") if part.strip()]
                samples = _coerce_positive_numeric_array(parts)
        else:
            samples = _coerce_positive_numeric_array([raw_value])

        if samples.size:
            return samples

    return np.asarray([], dtype=float)


def weighted_harmonic_mean_width(lengths: Sequence[float], widths: Sequence[float]) -> float:
    if len(lengths) != len(widths) or not lengths:
        return float("nan")
    lengths_array = np.asarray(lengths, dtype=float)
    widths_array = np.asarray(widths, dtype=float)
    if lengths_array.size == 0 or widths_array.size == 0:
        return float("nan")
    if np.any(~np.isfinite(lengths_array)) or np.any(lengths_array < 0):
        return float("nan")
    if np.any(~np.isfinite(widths_array)) or np.any(widths_array <= 0):
        return float("nan")
    total_length = float(lengths_array.sum())
    if total_length <= 0:
        return float("nan")
    return total_length / float(np.sum(lengths_array / widths_array))


def normalized_entropy(values: Sequence[float]) -> tuple[float, float, float]:
    valid = _coerce_positive_numeric_array(values)
    n_valid = int(valid.size)
    if n_valid == 0:
        return float("nan"), float("nan"), float("nan")
    if n_valid == 1:
        return 0.0, 1.0, 1.0

    shares = valid / valid.sum()
    entropy = float(-np.sum(shares * np.log(shares)))
    evenness = entropy / math.log(n_valid)
    effective_n = float(math.exp(entropy))
    return entropy, evenness, effective_n


def coefficient_of_variation(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    mean_value = float(array.mean())
    if mean_value == 0:
        return float("nan")
    return float(array.std(ddof=0) / mean_value)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if values_array.size == 0 or weights_array.size == 0 or values_array.size != weights_array.size:
        return float("nan")
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values_array[valid], weights=weights_array[valid]))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return float("nan")
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _normalize_thresholds(classification_thresholds: Mapping[str, float] | None = None) -> dict[str, float]:
    thresholds = dict(DEFAULT_CLASSIFICATION_THRESHOLDS)
    if classification_thresholds:
        thresholds.update({key: float(value) for key, value in classification_thresholds.items()})
    return thresholds


def _normalize_pixel_width_fields(pixel_width_fields: Sequence[str] | None = None) -> tuple[str, ...]:
    return tuple(pixel_width_fields or DEFAULT_PIXEL_WIDTH_FIELDS)


def _normalize_percentiles(pixel_width_percentiles: Sequence[float] | None = None) -> tuple[float, float, float]:
    percentiles = tuple(float(value) for value in (pixel_width_percentiles or DEFAULT_PIXEL_WIDTH_PERCENTILES))
    if len(percentiles) != 3:
        raise ValueError("pixel_width_percentiles must contain exactly three values.")
    return percentiles


def _build_metrics_config(
    *,
    input_links_path: str | Path | None = None,
    input_nodes_path: str | Path | None = None,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
    classification_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = _normalize_thresholds(classification_thresholds)
    percentiles = _normalize_percentiles(pixel_width_percentiles)
    return {
        "input_links_path": str(input_links_path) if input_links_path is not None else None,
        "input_nodes_path": str(input_nodes_path) if input_nodes_path is not None else None,
        "length_precedence": [*LENGTH_COLUMNS, "geometry.length"],
        "representative_width_precedence": list(REPRESENTATIVE_WIDTH_COLUMNS),
        "pixel_width_fields_used": list(_normalize_pixel_width_fields(pixel_width_fields)),
        "pixel_width_percentiles": list(percentiles),
        "use_pixel_widths_for_extremes": bool(use_pixel_widths_for_extremes),
        "path_width_equivalent_method": PATH_WIDTH_EQ_METHOD,
        "path_width_min_max_method": PATH_WIDTH_EXTREME_METHOD if use_pixel_widths_for_extremes else "representative_link_widths_only",
        "equivalent_unit_width_method": UNIT_WIDTH_EQ_METHOD,
        "equivalent_unit_length_method": UNIT_LENGTH_EQ_METHOD,
        "entropy_base": "natural_logarithm",
        "effective_path_method": "exp(shannon_entropy)",
        "dynamic_proxy_method": DYNAMIC_PROXY_METHOD,
        "dynamic_proxy_weight_field": DYNAMIC_PROXY_WEIGHT_FIELD,
        "classification_thresholds": thresholds,
        "direct_metrics_only": DIRECT_METRICS_ONLY,
        "recursive_collapsed_geometry_implemented": RECURSIVE_COLLAPSED_GEOMETRY_IMPLEMENTED,
        "notes": (
            "Direct unit metrics are computed from the unit's detected paths in the current graph. "
            "Hierarchy metadata such as collapse_level is included, but recursive collapsed geometry is future work."
        ),
    }


def _with_metrics_attrs(
    frame: pd.DataFrame,
    metrics_config: Mapping[str, Any],
) -> pd.DataFrame:
    frame.attrs["metrics_config"] = dict(metrics_config)
    return frame


def _build_primary_tree_metadata(units: list[StructuralUnit]) -> pd.DataFrame:
    return build_unit_context_frame(units)


def _valid_path_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["path_width_eq"].notna()
        & (frame["path_width_eq"] > 0)
        & frame["path_length"].notna()
        & (frame["path_length"] > 0)
    )


def _rank_descending_stable(values: pd.Series, ids: pd.Series) -> pd.Series:
    valid = values.notna() & np.isfinite(values) & (values > 0)
    ranks = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if not valid.any():
        return ranks
    ordered_index = (
        pd.DataFrame({"value": values.loc[valid], "path_id": ids.loc[valid]}, index=values.loc[valid].index)
        .sort_values(["value", "path_id"], ascending=[False, True], kind="mergesort")
        .index
    )
    ranks.loc[ordered_index] = pd.Series(range(1, len(ordered_index) + 1), index=ordered_index, dtype="Int64")
    return ranks


def _path_width_extreme_stats(
    representative_widths: Sequence[float],
    sample_arrays: Sequence[np.ndarray],
    *,
    pixel_width_percentiles: Sequence[float],
    use_pixel_widths_for_extremes: bool,
) -> dict[str, float]:
    percentiles = _normalize_percentiles(pixel_width_percentiles)
    valid_rep_widths = _coerce_positive_numeric_array(representative_widths)
    valid_samples = [samples for samples in sample_arrays if isinstance(samples, np.ndarray) and samples.size > 0]

    source_values = np.asarray([], dtype=float)
    if use_pixel_widths_for_extremes and valid_samples:
        source_values = np.concatenate(valid_samples)
    elif valid_rep_widths.size:
        source_values = valid_rep_widths

    if source_values.size == 0:
        return {
            "path_width_min": float("nan"),
            "path_width_max": float("nan"),
            "path_width_p05": float("nan"),
            "path_width_p50": float("nan"),
            "path_width_p95": float("nan"),
        }

    p_low, p_mid, p_high = np.percentile(source_values, percentiles)
    return {
        "path_width_min": float(source_values.min()),
        "path_width_max": float(source_values.max()),
        "path_width_p05": float(p_low),
        "path_width_p50": float(p_mid),
        "path_width_p95": float(p_high),
    }


def _path_metrics_for_unit(
    unit: StructuralUnit,
    link_lookup: dict[int, dict[str, Any]],
    *,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
) -> list[dict[str, Any]]:
    percentiles = _normalize_percentiles(pixel_width_percentiles)
    records: list[dict[str, Any]] = []

    for path in unit.paths:
        lengths: list[float] = []
        representative_widths: list[float] = []
        sample_arrays: list[np.ndarray] = []

        for link_id in path.id_links:
            attrs = link_lookup[int(link_id)]
            lengths.append(edge_length_from_attrs(attrs, geometry=attrs.get("geometry")))
            representative_widths.append(edge_width_from_attrs(attrs))
            sample_arrays.append(edge_width_samples_from_attrs(attrs, pixel_width_fields=pixel_width_fields))

        lengths_array = np.asarray(lengths, dtype=float)
        path_length = (
            float(lengths_array.sum())
            if lengths_array.size > 0 and np.all(np.isfinite(lengths_array)) and np.all(lengths_array >= 0)
            else float("nan")
        )
        path_width_eq = weighted_harmonic_mean_width(lengths, representative_widths)
        width_extremes = _path_width_extreme_stats(
            representative_widths,
            sample_arrays,
            pixel_width_percentiles=percentiles,
            use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
        )

        records.append(
            {
                "unit_id": unit.unit_id,
                "path_id": path.path_id,
                "n_links": len(path.id_links),
                "path_length": path_length,
                "path_width_eq": path_width_eq,
                "id_links": ",".join(str(link_id) for link_id in path.id_links),
                **width_extremes,
            }
        )

    return records


def _build_path_metrics_frame(
    units: list[StructuralUnit],
    link_lookup: dict[int, dict[str, Any]],
    *,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
) -> pd.DataFrame:
    path_records: list[dict[str, Any]] = []
    for unit in units:
        path_records.extend(
            _path_metrics_for_unit(
                unit,
                link_lookup,
                pixel_width_fields=pixel_width_fields,
                pixel_width_percentiles=pixel_width_percentiles,
                use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
            )
        )

    columns = [
        "unit_id",
        "path_id",
        "path_length",
        "path_width_eq",
        "path_width_min",
        "path_width_max",
        "path_width_p05",
        "path_width_p50",
        "path_width_p95",
        "n_links",
        "id_links",
        "path_width_fraction",
        "path_length_fraction",
        "path_rank_by_width",
        "path_rank_by_length",
    ]
    if not path_records:
        empty = pd.DataFrame(columns=columns)
        empty["path_rank_by_width"] = empty["path_rank_by_width"].astype("Int64")
        empty["path_rank_by_length"] = empty["path_rank_by_length"].astype("Int64")
        return empty

    path_metrics = pd.DataFrame.from_records(path_records)
    path_metrics["path_width_fraction"] = np.nan
    path_metrics["path_length_fraction"] = np.nan
    path_metrics["path_rank_by_width"] = pd.Series(pd.NA, index=path_metrics.index, dtype="Int64")
    path_metrics["path_rank_by_length"] = pd.Series(pd.NA, index=path_metrics.index, dtype="Int64")

    for unit_id, unit_paths in path_metrics.groupby("unit_id", sort=False):
        valid_mask = _valid_path_mask(unit_paths)
        valid_paths = unit_paths.loc[valid_mask].copy()
        if valid_paths.empty:
            continue

        equivalent_width = float(valid_paths["path_width_eq"].sum())
        total_valid_path_length = float(valid_paths["path_length"].sum())

        if equivalent_width > 0:
            path_metrics.loc[valid_paths.index, "path_width_fraction"] = valid_paths["path_width_eq"] / equivalent_width
        if total_valid_path_length > 0:
            path_metrics.loc[valid_paths.index, "path_length_fraction"] = valid_paths["path_length"] / total_valid_path_length

        path_metrics.loc[unit_paths.index, "path_rank_by_width"] = _rank_descending_stable(
            unit_paths["path_width_eq"],
            unit_paths["path_id"],
        )
        path_metrics.loc[unit_paths.index, "path_rank_by_length"] = _rank_descending_stable(
            unit_paths["path_length"],
            unit_paths["path_id"],
        )

    path_metrics = path_metrics[columns].sort_values(["unit_id", "path_id"], kind="mergesort").reset_index(drop=True)
    path_metrics["path_rank_by_width"] = path_metrics["path_rank_by_width"].astype("Int64")
    path_metrics["path_rank_by_length"] = path_metrics["path_rank_by_length"].astype("Int64")
    return path_metrics


def classify_unit_topodynamic(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> str:
    config = _normalize_thresholds(thresholds)
    n_valid_paths = int(row.get("n_valid_paths", 0) or 0)
    internal_bifurcation_count = int(row.get("internal_bifurcation_count", 0) or 0)
    internal_confluence_count = int(row.get("internal_confluence_count", 0) or 0)
    smaller_width_fraction_2 = row.get("smaller_width_fraction_2")
    dominant_width_fraction = row.get("dominant_width_fraction")
    width_evenness = row.get("width_evenness")
    topologic_complexity_score = row.get("topologic_complexity_score")
    is_compound = bool(row.get("is_compound", False) or row.get("compound_indicator", 0))

    if n_valid_paths <= 0:
        return "invalid_or_no_valid_paths"
    if n_valid_paths == 1:
        return "single_valid_path"

    if (
        n_valid_paths == 2
        and internal_bifurcation_count == 0
        and internal_confluence_count == 0
        and pd.notna(smaller_width_fraction_2)
        and smaller_width_fraction_2 >= config["balanced_two_path_smaller_fraction_threshold"]
    ):
        return "balanced_simple_split"

    if (
        n_valid_paths == 2
        and internal_bifurcation_count == 0
        and internal_confluence_count == 0
        and pd.notna(smaller_width_fraction_2)
        and smaller_width_fraction_2 < config["balanced_two_path_smaller_fraction_threshold"]
    ):
        return "dominant_simple_split"

    if (
        is_compound
        and pd.notna(topologic_complexity_score)
        and topologic_complexity_score >= config["high_topologic_complexity_threshold"]
        and pd.notna(width_evenness)
        and width_evenness >= config["high_evenness_threshold"]
    ):
        return "compound_topologically_complex_dynamically_complex"

    if (
        is_compound
        and pd.notna(topologic_complexity_score)
        and topologic_complexity_score >= config["high_topologic_complexity_threshold"]
        and pd.notna(dominant_width_fraction)
        and dominant_width_fraction >= config["dominant_fraction_threshold"]
    ):
        return "compound_topologically_complex_dynamically_simple"

    if n_valid_paths > 2 and pd.notna(width_evenness) and width_evenness >= config["high_evenness_threshold"]:
        return "balanced_multi_thread_unit"

    if n_valid_paths > 2 and pd.notna(dominant_width_fraction) and dominant_width_fraction >= config["dominant_fraction_threshold"]:
        return "dominant_multi_thread_unit"

    if (
        pd.notna(topologic_complexity_score)
        and topologic_complexity_score >= config["high_topologic_complexity_threshold"]
        and pd.notna(width_evenness)
        and width_evenness < config["low_evenness_threshold"]
    ):
        return "topologically_complex_dynamically_simple_unit"

    if (
        pd.notna(topologic_complexity_score)
        and topologic_complexity_score >= config["high_topologic_complexity_threshold"]
        and pd.notna(width_evenness)
        and width_evenness >= config["high_evenness_threshold"]
    ):
        return "topologically_complex_dynamically_complex_unit"

    return "intermediate_unit"


def summarize_by_hierarchy_level(unit_metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "collapse_level",
        "n_units",
        "n_compound_units",
        "n_leaf_units",
        "mean_equivalent_width",
        "median_equivalent_width",
        "sum_equivalent_width",
        "mean_equivalent_length",
        "median_equivalent_length",
        "mean_elongation",
        "median_elongation",
        "mean_n_paths",
        "mean_n_valid_paths",
        "mean_effective_n_paths_width",
        "mean_width_evenness",
        "mean_largest_path_width_fraction",
        "mean_topologic_complexity_score",
        "mean_dynamic_proxy_complexity_score",
        "width_weighted_mean_elongation",
        "width_weighted_mean_effective_n_paths_width",
        "width_weighted_mean_width_evenness",
        "width_weighted_mean_topologic_complexity_score",
        "width_weighted_mean_dynamic_proxy_complexity_score",
    ]
    if unit_metrics_df.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for collapse_level, group in unit_metrics_df.groupby("collapse_level", sort=True, dropna=False):
        records.append(
            {
                "collapse_level": collapse_level,
                "n_units": int(len(group)),
                "n_compound_units": int(group["compound_indicator"].fillna(0).sum()),
                "n_leaf_units": int(len(group) - group["compound_indicator"].fillna(0).sum()),
                "mean_equivalent_width": float(group["equivalent_width"].mean()),
                "median_equivalent_width": float(group["equivalent_width"].median()),
                "sum_equivalent_width": float(group["equivalent_width"].sum(min_count=1)),
                "mean_equivalent_length": float(group["equivalent_length"].mean()),
                "median_equivalent_length": float(group["equivalent_length"].median()),
                "mean_elongation": float(group["elongation"].mean()),
                "median_elongation": float(group["elongation"].median()),
                "mean_n_paths": float(group["n_paths"].mean()),
                "mean_n_valid_paths": float(group["n_valid_paths"].mean()),
                "mean_effective_n_paths_width": float(group["effective_n_paths_width"].mean()),
                "mean_width_evenness": float(group["width_evenness"].mean()),
                "mean_largest_path_width_fraction": float(group["largest_path_width_fraction"].mean()),
                "mean_topologic_complexity_score": float(group["topologic_complexity_score"].mean()),
                "mean_dynamic_proxy_complexity_score": float(group["dynamic_proxy_complexity_score"].mean()),
                "width_weighted_mean_elongation": weighted_mean(group["elongation"], group["equivalent_width"]),
                "width_weighted_mean_effective_n_paths_width": weighted_mean(
                    group["effective_n_paths_width"],
                    group["equivalent_width"],
                ),
                "width_weighted_mean_width_evenness": weighted_mean(group["width_evenness"], group["equivalent_width"]),
                "width_weighted_mean_topologic_complexity_score": weighted_mean(
                    group["topologic_complexity_score"],
                    group["equivalent_width"],
                ),
                "width_weighted_mean_dynamic_proxy_complexity_score": weighted_mean(
                    group["dynamic_proxy_complexity_score"],
                    group["equivalent_width"],
                ),
            }
        )

    summary = pd.DataFrame.from_records(records, columns=columns).sort_values("collapse_level", kind="mergesort").reset_index(drop=True)
    if "collapse_level" in summary:
        summary["collapse_level"] = summary["collapse_level"].astype("Int64")
    return summary


def compute_unit_metrics_from_units(
    links: gpd.GeoDataFrame,
    units: list[StructuralUnit],
    *,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
    classification_thresholds: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pixel_width_fields = _normalize_pixel_width_fields(pixel_width_fields)
    pixel_width_percentiles = _normalize_percentiles(pixel_width_percentiles)
    thresholds = _normalize_thresholds(classification_thresholds)
    metrics_config = _build_metrics_config(
        pixel_width_fields=pixel_width_fields,
        pixel_width_percentiles=pixel_width_percentiles,
        use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
        classification_thresholds=thresholds,
    )

    link_lookup = {int(row.id_link): row._asdict() for row in links.itertuples(index=False)}
    path_metrics = _build_path_metrics_frame(
        units,
        link_lookup,
        pixel_width_fields=pixel_width_fields,
        pixel_width_percentiles=pixel_width_percentiles,
        use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
    )
    tree_metadata = _build_primary_tree_metadata(units)
    unit_metric_columns = [
        "unit_id",
        "bifurcation",
        "confluence",
        "class",
        "n_paths",
        "n_valid_paths",
        "equivalent_width",
        "equivalent_length",
        "elongation",
        "path_length_min",
        "path_length_max",
        "path_length_mean",
        "path_length_range",
        "path_length_range_norm",
        "path_length_cv",
        "path_width_eq_min",
        "path_width_eq_max",
        "path_width_eq_mean",
        "path_width_range",
        "path_width_range_norm",
        "largest_path_width_fraction",
        "dominant_width_fraction",
        "width_entropy",
        "width_evenness",
        "effective_n_paths_width",
        "path_disparity_width",
        "width_ratio_2",
        "smaller_width_fraction_2",
        "dominant_width_fraction_2",
        "length_ratio_2",
        "internal_bifurcation_count",
        "internal_confluence_count",
        "total_bifurcation_count",
        "total_confluence_count",
        "internal_branch_node_count",
        "branching_density_by_length",
        "path_redundancy",
        "compound_indicator",
        "topologic_complexity_score",
        "dynamic_proxy_method",
        "dynamic_proxy_weight_field",
        "dynamic_proxy_entropy",
        "effective_n_paths_dyn_width",
        "dominant_dyn_fraction_width",
        "dynamic_proxy_complexity_score",
    ]

    unit_records: list[dict[str, Any]] = []
    for unit in units:
        unit_paths = path_metrics.loc[path_metrics["unit_id"] == unit.unit_id].copy()
        valid_paths = unit_paths.loc[_valid_path_mask(unit_paths)].copy()

        n_valid_paths = int(len(valid_paths))
        valid_path_lengths = valid_paths["path_length"].to_numpy(dtype=float)
        valid_path_widths = valid_paths["path_width_eq"].to_numpy(dtype=float)

        equivalent_width = float(valid_path_widths.sum()) if n_valid_paths > 0 else float("nan")
        equivalent_length = weighted_mean(valid_path_lengths, valid_path_widths)
        elongation = _safe_divide(equivalent_length, equivalent_width)

        path_length_min = float(valid_paths["path_length"].min()) if n_valid_paths > 0 else float("nan")
        path_length_max = float(valid_paths["path_length"].max()) if n_valid_paths > 0 else float("nan")
        path_length_mean = float(valid_paths["path_length"].mean()) if n_valid_paths > 0 else float("nan")
        path_length_range = path_length_max - path_length_min if n_valid_paths > 0 else float("nan")
        path_length_range_norm = _safe_divide(path_length_range, equivalent_length)
        path_length_cv = coefficient_of_variation(valid_path_lengths)

        path_width_eq_min = float(valid_paths["path_width_eq"].min()) if n_valid_paths > 0 else float("nan")
        path_width_eq_max = float(valid_paths["path_width_eq"].max()) if n_valid_paths > 0 else float("nan")
        path_width_eq_mean = float(valid_paths["path_width_eq"].mean()) if n_valid_paths > 0 else float("nan")
        path_width_range = path_width_eq_max - path_width_eq_min if n_valid_paths > 0 else float("nan")
        path_width_range_norm = _safe_divide(path_width_range, equivalent_width)
        largest_path_width_fraction = _safe_divide(path_width_eq_max, equivalent_width)
        dominant_width_fraction = largest_path_width_fraction

        width_entropy, width_evenness, effective_n_paths_width = normalized_entropy(valid_path_widths)
        path_disparity_width = _safe_divide(float(n_valid_paths), effective_n_paths_width)

        width_ratio_2 = float("nan")
        smaller_width_fraction_2 = float("nan")
        dominant_width_fraction_2 = float("nan")
        length_ratio_2 = float("nan")
        if n_valid_paths == 2:
            width_min = float(valid_path_widths.min())
            width_max = float(valid_path_widths.max())
            length_min = float(valid_path_lengths.min())
            length_max = float(valid_path_lengths.max())
            width_ratio_2 = _safe_divide(width_min, width_max)
            smaller_width_fraction_2 = _safe_divide(width_min, float(valid_path_widths.sum()))
            dominant_width_fraction_2 = _safe_divide(width_max, float(valid_path_widths.sum()))
            length_ratio_2 = _safe_divide(length_min, length_max)

        internal_bifurcation_count = len(unit.internal_bifurcations)
        internal_confluence_count = len(unit.internal_confluences)
        total_bifurcation_count = internal_bifurcation_count + 1
        total_confluence_count = internal_confluence_count + 1
        internal_branch_node_count = internal_bifurcation_count + internal_confluence_count
        branching_density_by_length = _safe_divide(internal_branch_node_count, equivalent_length)
        path_redundancy = max(n_valid_paths - 1, 0)
        compound_indicator = int(len(unit.children) > 0)
        topologic_complexity_score = (
            math.log1p(n_valid_paths)
            + math.log1p(internal_bifurcation_count)
            + math.log1p(internal_confluence_count)
        )

        dynamic_proxy_entropy = width_entropy
        effective_n_paths_dyn_width = effective_n_paths_width
        dominant_dyn_fraction_width = dominant_width_fraction
        dynamic_proxy_complexity_score = width_evenness

        unit_records.append(
            {
                "unit_id": unit.unit_id,
                "bifurcation": unit.bifurcation,
                "confluence": unit.confluence,
                "class": unit.unit_class,
                "n_paths": unit.n_paths,
                "n_valid_paths": n_valid_paths,
                # Direct metrics are computed from the currently detected paths of the unit.
                # Compound units carry hierarchy labels, but their geometry is not recursively rebuilt yet.
                "equivalent_width": equivalent_width,
                "equivalent_length": equivalent_length,
                "elongation": elongation,
                "path_length_min": path_length_min,
                "path_length_max": path_length_max,
                "path_length_mean": path_length_mean,
                "path_length_range": path_length_range,
                "path_length_range_norm": path_length_range_norm,
                "path_length_cv": path_length_cv,
                "path_width_eq_min": path_width_eq_min,
                "path_width_eq_max": path_width_eq_max,
                "path_width_eq_mean": path_width_eq_mean,
                "path_width_range": path_width_range,
                "path_width_range_norm": path_width_range_norm,
                "largest_path_width_fraction": largest_path_width_fraction,
                "dominant_width_fraction": dominant_width_fraction,
                "width_entropy": width_entropy,
                "width_evenness": width_evenness,
                "effective_n_paths_width": effective_n_paths_width,
                "path_disparity_width": path_disparity_width,
                "width_ratio_2": width_ratio_2,
                "smaller_width_fraction_2": smaller_width_fraction_2,
                "dominant_width_fraction_2": dominant_width_fraction_2,
                "length_ratio_2": length_ratio_2,
                "internal_bifurcation_count": internal_bifurcation_count,
                "internal_confluence_count": internal_confluence_count,
                "total_bifurcation_count": total_bifurcation_count,
                "total_confluence_count": total_confluence_count,
                "internal_branch_node_count": internal_branch_node_count,
                "branching_density_by_length": branching_density_by_length,
                "path_redundancy": path_redundancy,
                "compound_indicator": compound_indicator,
                "topologic_complexity_score": topologic_complexity_score,
                "dynamic_proxy_method": DYNAMIC_PROXY_METHOD,
                "dynamic_proxy_weight_field": DYNAMIC_PROXY_WEIGHT_FIELD,
                "dynamic_proxy_entropy": dynamic_proxy_entropy,
                "effective_n_paths_dyn_width": effective_n_paths_dyn_width,
                "dominant_dyn_fraction_width": dominant_dyn_fraction_width,
                "dynamic_proxy_complexity_score": dynamic_proxy_complexity_score,
            }
        )

    unit_metrics = pd.DataFrame.from_records(unit_records, columns=unit_metric_columns).merge(
        tree_metadata,
        on="unit_id",
        how="left",
    )
    unit_metrics["compound_indicator"] = unit_metrics["compound_indicator"].astype("Int64")
    if unit_metrics.empty:
        unit_metrics["unit_topodynamic_class"] = pd.Series(dtype=object)
    else:
        unit_metrics["unit_topodynamic_class"] = unit_metrics.apply(
            lambda row: classify_unit_topodynamic(row, thresholds),
            axis=1,
        )
        unit_metrics = unit_metrics.sort_values(
            ["root_unit_id", "depth_from_root", "collapse_level", "bifurcation", "confluence", "unit_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    _with_metrics_attrs(path_metrics, metrics_config)
    _with_metrics_attrs(unit_metrics, metrics_config)
    return unit_metrics, path_metrics


def compute_unit_metrics(
    links_path: str | Path,
    nodes_path: str | Path,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    pixel_width_fields: Sequence[str] | None = None,
    pixel_width_percentiles: Sequence[float] | None = None,
    use_pixel_widths_for_extremes: bool = True,
    classification_thresholds: Mapping[str, float] | None = None,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links, _ = load_network(links_path, nodes_path)
    summary, units, _ = analyze_network(
        links_path,
        nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
        debug=debug,
    )
    unit_metrics, path_metrics = compute_unit_metrics_from_units(
        links,
        units,
        pixel_width_fields=pixel_width_fields,
        pixel_width_percentiles=pixel_width_percentiles,
        use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
        classification_thresholds=classification_thresholds,
    )

    metrics_config = dict(unit_metrics.attrs.get("metrics_config", {}))
    metrics_config.update(
        _build_metrics_config(
            input_links_path=links_path,
            input_nodes_path=nodes_path,
            pixel_width_fields=pixel_width_fields,
            pixel_width_percentiles=pixel_width_percentiles,
            use_pixel_widths_for_extremes=use_pixel_widths_for_extremes,
            classification_thresholds=classification_thresholds,
        )
    )
    _with_metrics_attrs(summary, metrics_config)
    _with_metrics_attrs(unit_metrics, metrics_config)
    _with_metrics_attrs(path_metrics, metrics_config)
    return summary, unit_metrics, path_metrics


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


def write_metrics_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    unit_metrics: pd.DataFrame,
    path_metrics: pd.DataFrame,
    *,
    hierarchy_level_metrics: pd.DataFrame | None = None,
    manifest_overrides: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if hierarchy_level_metrics is None:
        hierarchy_level_metrics = summarize_by_hierarchy_level(unit_metrics)

    summary.to_csv(output_path / "unit_summary.csv", index=False)
    unit_metrics.to_csv(output_path / "unit_metrics.csv", index=False)
    path_metrics.to_csv(output_path / "path_metrics.csv", index=False)
    hierarchy_level_metrics.to_csv(output_path / "hierarchy_level_metrics.csv", index=False)

    metrics_config = dict(unit_metrics.attrs.get("metrics_config", {}))
    if manifest_overrides:
        metrics_config.update(dict(manifest_overrides))

    manifest = {
        "files": [
            "unit_summary.csv",
            "unit_metrics.csv",
            "path_metrics.csv",
            "hierarchy_level_metrics.csv",
        ],
        "n_units": int(len(unit_metrics)),
        "n_paths": int(len(path_metrics)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_revision(),
        **metrics_config,
    }
    with (output_path / "metrics_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute direct, hierarchy-aware metrics for detected bifurcation-confluence units. "
            "The metrics are computed from each unit's currently detected paths and do not yet "
            "implement recursive collapsed geometry."
        )
    )
    parser.add_argument("links_gpkg", help="Reviewed links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Reviewed nodes GeoPackage.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for CSV outputs.")
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
        nargs=3,
        type=float,
        default=None,
        metavar=("LOW", "MID", "HIGH"),
        help="Percentiles used for path_width_p05/p50/p95 style diagnostics.",
    )
    parser.add_argument(
        "--disable-pixel-widths-for-extremes",
        action="store_true",
        help="Use representative link widths only for min/max and percentile diagnostics.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information during unit detection.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, unit_metrics, path_metrics = compute_unit_metrics(
        args.links_gpkg,
        args.nodes_gpkg,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        pixel_width_fields=args.pixel_width_fields,
        pixel_width_percentiles=args.pixel_width_percentiles,
        use_pixel_widths_for_extremes=not args.disable_pixel_widths_for_extremes,
        debug=args.debug,
    )

    print(unit_metrics.to_string(index=False))
    print()
    print(path_metrics.to_string(index=False))

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir_for_links(args.links_gpkg)
    write_metrics_outputs(output_dir, summary, unit_metrics, path_metrics)
    print()
    print(f"Wrote outputs to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()

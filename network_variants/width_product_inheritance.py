from __future__ import annotations

from collections import Counter
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


# These values describe the physical width product.  For a geometrically
# unchanged link they must be copied, not recalculated from a child raster.
MEASURED_WIDTH_FIELDS = (
    "wid_pix",
    "wid",
    "wid_adj",
    "wid_med",
    "wid_pctdiff",
    "wid_pix_total",
    "wid_pix_wet",
    "wid_pix_dry",
    "wid_adj_total",
    "wid_med_total",
    "wid_p05_total",
    "wid_p50_total",
    "wid_p95_total",
    "wid_adj_wet",
    "wid_med_wet",
    "wid_p05_wet",
    "wid_p50_wet",
    "wid_p95_wet",
    "wid_adj_dry",
    "wid_med_dry",
    "wid_p05_dry",
    "wid_p50_dry",
    "wid_p95_dry",
    "wet_fraction_adj",
    "dry_fraction_adj",
    "n_wet_threads_mean",
    "n_wet_threads_max",
    "n_channels_mean",
    "n_channels_max",
    "n_invalid_width_samples",
    "qc_mean_relative_dt_diff",
    "qc_max_relative_dt_diff",
    "qc_mean_width_chord_diff",
    "qc_mean_opposition_cos_trimmed",
    "qc_min_opposition_cos",
    "width_method",
)

# These fields record the reviewed/model decision and the effective value used
# by routing.  They are meaningful only after the child link has either
# inherited an unchanged parent decision or undergone its own evaluation.
ROUTING_WIDTH_FIELDS = (
    "width_reviewed_flag",
    "width_review_decision",
    "width_update_applied",
    "width_update_method",
    "width_update_source_link_id",
    "width_update_scale_factor",
    "width_update_target_context_width",
    "wid_orig",
    "wid_adj_orig",
    "wid_med_orig",
    "manual_width_review_candidate",
    "manual_width_review_decision",
    "manual_width_review_requires_correction",
    "manual_width_review_source",
    "width_model_candidate",
    "length_over_raw_wet_width",
    "split_sensitive",
    "opposition_cos_trimmed_mean",
    "opposition_cos_trimmed_median",
    "invalid_sample_fraction",
    "upstream_continuity_ratio",
    "upstream_continuity_abs_deviation",
    "upstream_hybrid_continuity_abs_deviation",
    "width_model_excluded",
    "width_model_probability",
    "width_model_threshold",
    "width_model_prediction",
    "routing_context_width_us",
    "routing_context_width_ds",
    "routing_context_uses_endpoint",
    "comparison_scale_factor",
    "comparison_routing_width",
    "comparison_status",
    "comparison_flag",
    "model_confidence_strength",
    "upstream_deviation_strength",
    "opposition_badness",
    "adaptive_relaxation_strength",
    "adaptive_min_scale",
    "adaptive_max_scale",
    "adaptive_raw_penalty_weight",
    "wid_adj_routing",
    "width_correction_scale_factor",
    "width_correction_applied",
    "width_correction_flag",
    "width_qc_decision",
    "width_qc_reason",
    "width_correction_absolute_change",
    "width_correction_relative_change",
    "dry_corridor_fraction",
    "dry_corridor_meaningful",
    "routing_to_wet_width_ratio",
    "routing_to_total_width_ratio",
    "measured_wet_exceeds_total_corridor",
    "routing_exceeds_total_corridor",
    "routing_corridor_consistency_flag",
    "routing_corridor_consistency_reason",
    "width_correction_source",
    "width_algorithm_version",
    "width_model_version",
    "width_model_name",
)

WIDTH_INHERITANCE_FIELDS = MEASURED_WIDTH_FIELDS + ROUTING_WIDTH_FIELDS


def _as_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _geometries_are_unchanged(child_geometry: object, parent_geometry: object) -> bool:
    if child_geometry is None or parent_geometry is None:
        return False
    try:
        if bool(child_geometry.is_empty) or bool(parent_geometry.is_empty):
            return False
        # Topological equality is independent of coordinate order, which is
        # required because flow alignment may reverse a LineString.
        return bool(child_geometry.equals(parent_geometry))
    except (AttributeError, TypeError):
        return False


def _missing_value_for(series: pd.Series) -> object:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return np.nan
    return None


def inherit_unchanged_link_width_products(
    *,
    child_links: gpd.GeoDataFrame,
    child_width_samples: pd.DataFrame,
    parent_links: gpd.GeoDataFrame,
    parent_width_samples: pd.DataFrame,
    link_lineage: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    """Copy the complete parent width product for truly unchanged links.

    ``unchanged_1to1`` is treated as a candidate rather than proof: the child
    and parent geometries must also be topologically equal, and a parent link
    may map to only one child.  All other child links retain their newly
    measured physical fields but have routing/model fields cleared until the
    child-width evaluator processes them.
    """

    required_link_columns = {"id_link", "geometry"}
    for label, frame in (("child_links", child_links), ("parent_links", parent_links)):
        missing = required_link_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
        if frame["id_link"].duplicated().any():
            raise ValueError(f"{label} contains duplicate id_link values.")
    required_lineage = {"id_link", "lineage_type", "dominant_parent_link_id"}
    missing_lineage = required_lineage.difference(link_lineage.columns)
    if missing_lineage:
        raise ValueError(f"link_lineage is missing required columns: {sorted(missing_lineage)}")

    result_links = child_links.copy()
    result_samples = child_width_samples.copy()
    parent_by_id = parent_links.set_index(parent_links["id_link"].astype(int), drop=False)
    child_by_id = result_links.set_index(result_links["id_link"].astype(int), drop=False)

    candidates: list[tuple[int, int]] = []
    rejected: dict[int, str] = {}
    for row in link_lineage.itertuples(index=False):
        child_id = _as_int(getattr(row, "id_link", None))
        if child_id is None or child_id not in child_by_id.index:
            continue
        if str(getattr(row, "lineage_type", "")) != "unchanged_1to1":
            rejected[child_id] = "lineage_not_unchanged_1to1"
            continue
        parent_id = _as_int(getattr(row, "dominant_parent_link_id", None))
        if parent_id is None or parent_id not in parent_by_id.index:
            rejected[child_id] = "parent_link_missing"
            continue
        candidates.append((child_id, parent_id))

    parent_counts = Counter(parent_id for _, parent_id in candidates)
    inherited: dict[int, int] = {}
    for child_id, parent_id in candidates:
        if parent_counts[parent_id] != 1:
            rejected[child_id] = "parent_maps_to_multiple_children"
            continue
        if not _geometries_are_unchanged(
            child_by_id.loc[child_id, "geometry"],
            parent_by_id.loc[parent_id, "geometry"],
        ):
            rejected[child_id] = "geometry_changed"
            continue
        inherited[child_id] = parent_id

    inherited_fields = [field for field in WIDTH_INHERITANCE_FIELDS if field in parent_links.columns]
    routing_fields_present = [field for field in ROUTING_WIDTH_FIELDS if field in parent_links.columns]
    child_ids = result_links["id_link"].astype(int)
    inherited_mask = child_ids.isin(inherited)

    # A changed link must never retain a parent model decision merely because
    # an earlier merge happened to carry that column forward.
    for field in routing_fields_present:
        if field in result_links.columns:
            result_links.loc[~inherited_mask, field] = _missing_value_for(result_links[field])

    for field in inherited_fields:
        values_by_parent = parent_by_id[field].to_dict()
        inherited_values = child_ids.map(
            lambda child_id: values_by_parent.get(inherited.get(int(child_id)))
        )
        if field not in result_links.columns:
            result_links[field] = inherited_values
        elif inherited_mask.any():
            result_links.loc[inherited_mask, field] = inherited_values.loc[inherited_mask].to_numpy()

    result_links["width_inheritance_status"] = np.where(
        inherited_mask,
        "inherited_unchanged_1to1",
        "changed_requires_evaluation",
    )
    result_links["width_inherited_from_parent_link_id"] = child_ids.map(inherited).astype("Int64")
    result_links["width_inherited_field_count"] = np.where(
        inherited_mask,
        len(inherited_fields),
        0,
    ).astype(int)

    inherited_child_ids = set(inherited)
    if inherited_child_ids:
        if "id_link" in result_samples.columns:
            result_samples = result_samples.loc[
                ~pd.to_numeric(result_samples["id_link"], errors="coerce").isin(inherited_child_ids)
            ].copy()
        if "id_link" in parent_width_samples.columns:
            copied_samples: list[pd.DataFrame] = []
            parent_sample_ids = pd.to_numeric(parent_width_samples["id_link"], errors="coerce")
            for child_id, parent_id in sorted(inherited.items()):
                copied = parent_width_samples.loc[parent_sample_ids.eq(parent_id)].copy()
                copied["id_link"] = child_id
                copied_samples.append(copied)
            if copied_samples:
                result_samples = pd.concat([result_samples, *copied_samples], ignore_index=True, sort=False)
                sort_columns = [
                    column for column in ("id_link", "sample_index") if column in result_samples.columns
                ]
                if sort_columns:
                    result_samples = result_samples.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    reason_counts = dict(sorted(Counter(rejected.values()).items()))
    summary: dict[str, Any] = {
        "contract_version": "unchanged-link-width-inheritance-v1",
        "child_link_count": int(len(result_links)),
        "lineage_candidate_count": int(len(candidates)),
        "inherited_link_count": int(len(inherited)),
        "changed_link_count": int(len(result_links) - len(inherited)),
        "inherited_link_ids": sorted(inherited),
        "parent_link_by_child_link": {
            str(child_id): int(parent_id) for child_id, parent_id in sorted(inherited.items())
        },
        "rejected_candidate_reasons": reason_counts,
        "inherited_fields": inherited_fields,
        "missing_expected_parent_fields": [
            field for field in WIDTH_INHERITANCE_FIELDS if field not in parent_links.columns
        ],
        "routing_width_available_on_parent": "wid_adj_routing" in parent_links.columns,
    }
    return result_links, result_samples, summary

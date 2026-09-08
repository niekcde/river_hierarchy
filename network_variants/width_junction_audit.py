from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256
from network_variants.width_review_dataset import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_OUTPUT_DIR,
    DEFAULT_REVIEW_ROOT,
    DATASET_FILENAME,
)
from network_variants.width_review_gui import compute_sample_qc_frame


AUDIT_SCHEMA_VERSION = "width-junction-audit-v1"
DEFAULT_DATASET_PATH = DEFAULT_DATASET_OUTPUT_DIR / DATASET_FILENAME
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_junction_audit"

ENDPOINTS_FILENAME = "junction_endpoint_widths_v1.csv"
COMPARISONS_FILENAME = "reviewed_junction_comparisons_v1.csv"
PROFILE_FILENAME = "junction_window_profile_v1.csv"
CONSENSUS_FILENAME = "junction_review_consensus_v1.csv"
MANIFEST_FILENAME = "junction_width_audit_manifest_v1.json"


@dataclass(frozen=True)
class WindowSpecification:
    fraction: float
    minimum_length: float
    maximum_length: float

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("Window fraction must be in (0, 1].")
        if self.minimum_length <= 0 or self.maximum_length <= 0:
            raise ValueError("Window minimum and maximum must be positive.")
        if self.minimum_length > self.maximum_length:
            raise ValueError("Window minimum cannot exceed its maximum.")

    @property
    def window_id(self) -> str:
        percentage = int(round(self.fraction * 100))
        return f"p{percentage:02d}_min{self.minimum_length:g}_max{self.maximum_length:g}"

    def length_for(self, link_length: float) -> float:
        return float(np.clip(float(link_length) * self.fraction, self.minimum_length, self.maximum_length))


DEFAULT_WINDOWS = (
    WindowSpecification(0.05, 100.0, 500.0),
    WindowSpecification(0.10, 250.0, 1000.0),
    WindowSpecification(0.15, 250.0, 1500.0),
)


def parse_window(value: str) -> WindowSpecification:
    try:
        fraction_text, minimum_text, maximum_text = (part.strip() for part in value.split(","))
        fraction = float(fraction_text)
        if fraction > 1:
            fraction /= 100.0
        return WindowSpecification(fraction, float(minimum_text), float(maximum_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Windows must be PERCENT_OR_FRACTION,MIN_LENGTH,MAX_LENGTH; for example 10,250,1000."
        ) from exc


def _point_xy(geometry: Any) -> tuple[float, float]:
    if geometry is None or geometry.is_empty:
        return float("nan"), float("nan")
    if geometry.geom_type == "Point":
        return float(geometry.x), float(geometry.y)
    point = geometry.representative_point()
    return float(point.x), float(point.y)


def _distance(x: float, y: float, point: tuple[float, float]) -> float:
    return float(np.hypot(float(x) - point[0], float(y) - point[1]))


def _prepare_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if "used_in_selected_width" in samples.columns:
        prepared = samples.copy()
        prepared["used_in_trimmed_width"] = (
            prepared["used_in_selected_width"].fillna(False).astype(bool)
        )
        return prepared
    if "used_in_trimmed_width" in samples.columns:
        return samples.copy()
    marked, _ = compute_sample_qc_frame(samples)
    return marked


def calculate_endpoint_widths(
    *,
    example_id: str,
    families: pd.DataFrame,
    samples: pd.DataFrame,
    directed_links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
    windows: Sequence[WindowSpecification],
    whole_width_field: str | None = None,
) -> pd.DataFrame:
    required_family = {"id_link", "id_us_node", "id_ds_node", "len", "wid_adj_wet"}
    required_sample = {"id_link", "sample_index", "x", "y", "width_wet"}
    required_link = {"id_link", "id_us_node", "id_ds_node", "geometry"}
    missing = sorted(required_family.difference(families.columns))
    if missing:
        raise ValueError(f"Width families are missing fields: {missing}")
    missing = sorted(required_sample.difference(samples.columns))
    if missing:
        raise ValueError(f"Width samples are missing fields: {missing}")
    missing = sorted(required_link.difference(directed_links.columns))
    if missing:
        raise ValueError(f"Directed links are missing fields: {missing}")
    if "id_node" not in directed_nodes.columns:
        raise ValueError("Directed nodes are missing id_node.")

    family = families.copy()
    for column in ("id_link", "id_us_node", "id_ds_node"):
        family[column] = pd.to_numeric(family[column], errors="coerce").astype("Int64")
    family = family.dropna(subset=["id_link", "id_us_node", "id_ds_node"]).copy()
    family[["id_link", "id_us_node", "id_ds_node"]] = family[
        ["id_link", "id_us_node", "id_ds_node"]
    ].astype(int)
    family["len"] = pd.to_numeric(family["len"], errors="coerce")
    # The automatic fallback is retained only for reproducing historical
    # offline audits created while manual targets overwrote wid_adj_wet. New
    # production routing explicitly requests the measured wid_adj_wet field.
    if whole_width_field is None:
        whole_width_field = "wid_adj_wet_raw" if "wid_adj_wet_raw" in family.columns else "wid_adj_wet"
    if whole_width_field not in family.columns:
        raise ValueError(f"Width families are missing requested whole-width field {whole_width_field!r}.")
    family[whole_width_field] = pd.to_numeric(family[whole_width_field], errors="coerce")
    if family.duplicated("id_link").any():
        raise ValueError(f"Example {example_id} has duplicate width-family link IDs.")

    links = directed_links.copy()
    links["id_link"] = pd.to_numeric(links["id_link"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["id_link"]).copy()
    links["id_link"] = links["id_link"].astype(int)
    links = links.drop_duplicates("id_link").set_index("id_link", drop=False)
    nodes = directed_nodes.copy()
    nodes["id_node"] = pd.to_numeric(nodes["id_node"], errors="coerce").astype("Int64")
    nodes = nodes.dropna(subset=["id_node"]).copy()
    nodes["id_node"] = nodes["id_node"].astype(int)
    nodes = nodes.drop_duplicates("id_node").set_index("id_node", drop=False)

    marked = _prepare_samples(samples)
    marked["id_link"] = pd.to_numeric(marked["id_link"], errors="coerce").astype("Int64")
    marked = marked.dropna(subset=["id_link"]).copy()
    marked["id_link"] = marked["id_link"].astype(int)

    rows: list[dict[str, Any]] = []
    grouped_samples = {int(link_id): group for link_id, group in marked.groupby("id_link", sort=False)}
    for link in family.itertuples(index=False):
        link_id = int(link.id_link)
        link_length = float(link.len)
        whole_width = float(getattr(link, whole_width_field))
        link_samples = grouped_samples.get(link_id)
        if link_samples is None or link_samples.empty or link_id not in links.index:
            for spec in windows:
                rows.append(
                    {
                        "audit_schema_version": AUDIT_SCHEMA_VERSION,
                        "example_id": str(example_id).zfill(2),
                        "id_link": link_id,
                        "id_us_node": int(link.id_us_node),
                        "id_ds_node": int(link.id_ds_node),
                        "window_id": spec.window_id,
                        "window_fraction": spec.fraction,
                        "window_minimum_length": spec.minimum_length,
                        "window_maximum_length": spec.maximum_length,
                        "requested_window_length": spec.length_for(link_length),
                        "link_length": link_length,
                        "whole_link_mean_wet_width": whole_width,
                        "whole_link_mean_wet_width_source": whole_width_field,
                        "length_over_wet_width": link_length / whole_width if whole_width > 0 else np.nan,
                        "width_near_us_node_wet": np.nan,
                        "width_near_ds_node_wet": np.nan,
                        "us_endpoint_sample_count": 0,
                        "ds_endpoint_sample_count": 0,
                        "us_trim_gap_length": np.nan,
                        "ds_trim_gap_length": np.nan,
                        "us_endpoint_sample_span": np.nan,
                        "ds_endpoint_sample_span": np.nan,
                        "endpoint_windows_nonoverlapping": False,
                        "sample_order_orientation": "unavailable",
                        "sample_orientation_score_margin": np.nan,
                        "width_trim_method": getattr(link, "width_trim_method", "legacy_reconstructed"),
                    }
                )
            continue

        ordered = link_samples.sort_values("sample_index", kind="mergesort").copy()
        x = pd.to_numeric(ordered["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(ordered["y"], errors="coerce").to_numpy(dtype=float)
        step = np.hypot(np.diff(x), np.diff(y))
        step = np.where(np.isfinite(step), step, 0.0)
        cumulative = np.insert(np.cumsum(step), 0, 0.0)
        sampled_length = float(cumulative[-1]) if len(cumulative) else 0.0

        us_geometry = nodes.loc[int(link.id_us_node), "geometry"] if int(link.id_us_node) in nodes.index else None
        ds_geometry = nodes.loc[int(link.id_ds_node), "geometry"] if int(link.id_ds_node) in nodes.index else None
        us_point, ds_point = _point_xy(us_geometry), _point_xy(ds_geometry)
        forward_score = _distance(x[0], y[0], us_point) + _distance(x[-1], y[-1], ds_point)
        reverse_score = _distance(x[0], y[0], ds_point) + _distance(x[-1], y[-1], us_point)
        sample_order_forward = bool(forward_score <= reverse_score)
        denominator = max(forward_score, reverse_score, 1e-12)
        orientation_margin = abs(reverse_score - forward_score) / denominator
        distance_from_us = cumulative if sample_order_forward else sampled_length - cumulative
        distance_from_ds = sampled_length - distance_from_us

        valid = ordered["used_in_trimmed_width"].fillna(False).astype(bool).to_numpy()
        if "width_invalid" in ordered.columns:
            valid &= ~ordered["width_invalid"].fillna(False).astype(bool).to_numpy()
        widths = pd.to_numeric(ordered["width_wet"], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(widths) & (widths > 0) & np.isfinite(distance_from_us) & np.isfinite(distance_from_ds)
        us_gap = float(np.min(distance_from_us[valid])) if valid.any() else np.nan
        ds_gap = float(np.min(distance_from_ds[valid])) if valid.any() else np.nan

        for spec in windows:
            window_length = spec.length_for(link_length)
            us_mask = valid & (distance_from_us <= us_gap + window_length) if np.isfinite(us_gap) else valid & False
            ds_mask = valid & (distance_from_ds <= ds_gap + window_length) if np.isfinite(ds_gap) else valid & False
            us_distances = distance_from_us[us_mask]
            ds_distances = distance_from_ds[ds_mask]
            rows.append(
                {
                    "audit_schema_version": AUDIT_SCHEMA_VERSION,
                    "example_id": str(example_id).zfill(2),
                    "id_link": link_id,
                    "id_us_node": int(link.id_us_node),
                    "id_ds_node": int(link.id_ds_node),
                    "window_id": spec.window_id,
                    "window_fraction": spec.fraction,
                    "window_minimum_length": spec.minimum_length,
                    "window_maximum_length": spec.maximum_length,
                    "requested_window_length": window_length,
                    "link_length": link_length,
                    "whole_link_mean_wet_width": whole_width,
                    "whole_link_mean_wet_width_source": whole_width_field,
                    "length_over_wet_width": link_length / whole_width if whole_width > 0 else np.nan,
                    "width_near_us_node_wet": float(np.median(widths[us_mask])) if us_mask.any() else np.nan,
                    "width_near_ds_node_wet": float(np.median(widths[ds_mask])) if ds_mask.any() else np.nan,
                    "us_endpoint_sample_count": int(us_mask.sum()),
                    "ds_endpoint_sample_count": int(ds_mask.sum()),
                    "us_trim_gap_length": us_gap,
                    "ds_trim_gap_length": ds_gap,
                    "us_endpoint_sample_span": float(us_distances.max() - us_distances.min()) if len(us_distances) else np.nan,
                    "ds_endpoint_sample_span": float(ds_distances.max() - ds_distances.min()) if len(ds_distances) else np.nan,
                    "endpoint_windows_nonoverlapping": bool(sampled_length >= us_gap + ds_gap + 2 * window_length),
                    "sample_order_orientation": "sample_index_us_to_ds" if sample_order_forward else "sample_index_ds_to_us",
                    "sample_orientation_score_margin": orientation_margin,
                    "width_trim_method": getattr(link, "width_trim_method", "legacy_reconstructed"),
                }
            )
    return pd.DataFrame(rows)


def build_reviewed_junction_comparisons(
    endpoint_widths: pd.DataFrame,
    reviewed_dataset: pd.DataFrame,
) -> pd.DataFrame:
    review = reviewed_dataset.copy()
    review["example_id"] = review["example_id"].astype("string").str.zfill(2)
    review["id_link"] = pd.to_numeric(review["id_link"], errors="raise").astype(int)
    review_columns = [
        "example_id",
        "id_link",
        "manual_decision",
        "requires_correction",
        "split_sensitive",
        "upstream_continuity_ratio",
        "upstream_continuity_abs_deviation",
        "review_archive_dir",
        "review_manifest_path",
    ]
    missing = sorted(set(review_columns).difference(review.columns))
    if missing:
        raise ValueError(f"Reviewed dataset is missing fields: {missing}")

    output_rows: list[pd.DataFrame] = []
    for (example_id, window_id), links in endpoint_widths.groupby(["example_id", "window_id"], sort=True):
        links = links.copy()
        incoming_link_ids = links.groupby("id_ds_node")["id_link"].apply(
            lambda values: ",".join(str(int(value)) for value in sorted(values))
        )
        outgoing_link_ids = links.groupby("id_us_node")["id_link"].apply(
            lambda values: ",".join(str(int(value)) for value in sorted(values))
        )
        incoming_max_length = links.groupby("id_ds_node")["link_length"].max()
        outgoing_max_length = links.groupby("id_us_node")["link_length"].max()
        incoming_max_length_over_width = links.groupby("id_ds_node")["length_over_wet_width"].max()
        outgoing_max_length_over_width = links.groupby("id_us_node")["length_over_wet_width"].max()
        incoming_windows_nonoverlapping = links.groupby("id_ds_node")["endpoint_windows_nonoverlapping"].all()
        outgoing_windows_nonoverlapping = links.groupby("id_us_node")["endpoint_windows_nonoverlapping"].all()
        incoming_mean = links.groupby("id_ds_node")["whole_link_mean_wet_width"].sum(min_count=1)
        outgoing_mean = links.groupby("id_us_node")["whole_link_mean_wet_width"].sum(min_count=1)

        incoming_expected = links.groupby("id_ds_node")["id_link"].count()
        outgoing_expected = links.groupby("id_us_node")["id_link"].count()
        incoming_available = links.groupby("id_ds_node")["width_near_ds_node_wet"].count()
        outgoing_available = links.groupby("id_us_node")["width_near_us_node_wet"].count()
        incoming_junction = links.groupby("id_ds_node")["width_near_ds_node_wet"].sum(min_count=1)
        outgoing_junction = links.groupby("id_us_node")["width_near_us_node_wet"].sum(min_count=1)
        incoming_junction = incoming_junction.where(incoming_available.eq(incoming_expected))
        outgoing_junction = outgoing_junction.where(outgoing_available.eq(outgoing_expected))

        selected = review.loc[review["example_id"].eq(example_id), review_columns].copy()
        selected = selected.merge(
            links[
                [
                    "id_link",
                    "link_length",
                    "whole_link_mean_wet_width",
                    "length_over_wet_width",
                    "width_near_us_node_wet",
                    "width_near_ds_node_wet",
                    "requested_window_length",
                    "endpoint_windows_nonoverlapping",
                    "us_endpoint_sample_count",
                    "ds_endpoint_sample_count",
                    "us_trim_gap_length",
                    "ds_trim_gap_length",
                    "sample_order_orientation",
                    "sample_orientation_score_margin",
                ]
            ],
            on="id_link",
            how="left",
            validate="1:1",
        )
        selected["window_id"] = window_id
        upstream_node_by_link = links.set_index("id_link")["id_us_node"]
        selected["incoming_link_ids"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_link_ids).to_dict()
        )
        selected["outgoing_link_ids"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_link_ids).to_dict()
        )
        selected["incoming_max_link_length"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_max_length).to_dict()
        )
        selected["outgoing_max_link_length"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_max_length).to_dict()
        )
        selected["incoming_max_length_over_width"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_max_length_over_width).to_dict()
        )
        selected["outgoing_max_length_over_width"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_max_length_over_width).to_dict()
        )
        selected["incoming_endpoint_windows_nonoverlapping"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_windows_nonoverlapping).to_dict()
        )
        selected["outgoing_endpoint_windows_nonoverlapping"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_windows_nonoverlapping).to_dict()
        )
        selected["all_junction_endpoint_windows_nonoverlapping"] = (
            selected["incoming_endpoint_windows_nonoverlapping"].astype("boolean").fillna(False)
            & selected["outgoing_endpoint_windows_nonoverlapping"].astype("boolean").fillna(False)
        )
        selected["incoming_mean_width_sum"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_mean).to_dict()
        )
        selected["outgoing_mean_width_sum"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_mean).to_dict()
        )
        selected["incoming_junction_width_sum"] = selected["id_link"].map(
            upstream_node_by_link.map(incoming_junction).to_dict()
        )
        selected["outgoing_junction_width_sum"] = selected["id_link"].map(
            upstream_node_by_link.map(outgoing_junction).to_dict()
        )
        selected["mean_continuity_ratio_recomputed"] = (
            selected["outgoing_mean_width_sum"] / selected["incoming_mean_width_sum"]
        )
        selected["mean_continuity_abs_deviation_recomputed"] = (
            selected["mean_continuity_ratio_recomputed"] - 1.0
        ).abs()
        selected["junction_continuity_ratio"] = (
            selected["outgoing_junction_width_sum"] / selected["incoming_junction_width_sum"]
        )
        selected["junction_continuity_abs_deviation"] = (
            selected["junction_continuity_ratio"] - 1.0
        ).abs()
        selected["junction_minus_mean_abs_deviation"] = (
            selected["junction_continuity_abs_deviation"]
            - selected["mean_continuity_abs_deviation_recomputed"]
        )
        selected["absolute_deviation_change"] = selected["junction_minus_mean_abs_deviation"].abs()
        selected["mean_high_junction_low"] = (
            selected["mean_continuity_abs_deviation_recomputed"].ge(0.5)
            & selected["junction_continuity_abs_deviation"].lt(0.25)
        )
        selected["junction_high_mean_low"] = (
            selected["junction_continuity_abs_deviation"].ge(0.5)
            & selected["mean_continuity_abs_deviation_recomputed"].lt(0.25)
        )
        selected["material_continuity_change"] = selected["absolute_deviation_change"].ge(0.25)
        output_rows.append(selected)
    return pd.concat(output_rows, ignore_index=True).sort_values(
        ["window_id", "absolute_deviation_change", "example_id", "id_link"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_window_profile(comparisons: pd.DataFrame, endpoint_widths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_id, reviewed in comparisons.groupby("window_id", sort=True):
        endpoints = endpoint_widths.loc[endpoint_widths["window_id"].eq(window_id)]
        local = reviewed["junction_continuity_abs_deviation"].notna()
        orientation_reverse = endpoints["sample_order_orientation"].eq("sample_index_ds_to_us")
        rows.append(
            {
                "window_id": window_id,
                "window_fraction": float(endpoints["window_fraction"].iloc[0]),
                "window_minimum_length": float(endpoints["window_minimum_length"].iloc[0]),
                "window_maximum_length": float(endpoints["window_maximum_length"].iloc[0]),
                "all_link_count": int(len(endpoints)),
                "reviewed_link_count": int(len(reviewed)),
                "reviewed_junction_comparison_available": int(local.sum()),
                "reviewed_junction_comparison_coverage": float(local.mean()),
                "nonoverlapping_endpoint_windows": int(endpoints["endpoint_windows_nonoverlapping"].sum()),
                "nonoverlapping_endpoint_window_fraction": float(endpoints["endpoint_windows_nonoverlapping"].mean()),
                "median_us_endpoint_sample_count": float(endpoints["us_endpoint_sample_count"].median()),
                "median_ds_endpoint_sample_count": float(endpoints["ds_endpoint_sample_count"].median()),
                "median_us_trim_gap_length": float(endpoints["us_trim_gap_length"].median()),
                "median_ds_trim_gap_length": float(endpoints["ds_trim_gap_length"].median()),
                "reversed_sample_order_links": int(orientation_reverse.sum()),
                "material_continuity_changes": int(reviewed["material_continuity_change"].sum()),
                "mean_high_junction_low": int(reviewed["mean_high_junction_low"].sum()),
                "junction_high_mean_low": int(reviewed["junction_high_mean_low"].sum()),
                "manual_good_material_changes": int(
                    (reviewed["manual_decision"].eq("good") & reviewed["material_continuity_change"]).sum()
                ),
                "manual_bad_material_changes": int(
                    (reviewed["manual_decision"].eq("bad") & reviewed["material_continuity_change"]).sum()
                ),
                "reviewed_with_incoming_length_over_width_ge_3": int(
                    reviewed["incoming_max_length_over_width"].ge(3).sum()
                ),
                "material_changes_with_incoming_length_over_width_ge_3": int(
                    (reviewed["material_continuity_change"] & reviewed["incoming_max_length_over_width"].ge(3)).sum()
                ),
                "reviewed_with_incoming_length_over_width_ge_5": int(
                    reviewed["incoming_max_length_over_width"].ge(5).sum()
                ),
                "material_changes_with_incoming_length_over_width_ge_5": int(
                    (reviewed["material_continuity_change"] & reviewed["incoming_max_length_over_width"].ge(5)).sum()
                ),
                "reviewed_with_incoming_length_over_width_ge_10": int(
                    reviewed["incoming_max_length_over_width"].ge(10).sum()
                ),
                "material_changes_with_incoming_length_over_width_ge_10": int(
                    (reviewed["material_continuity_change"] & reviewed["incoming_max_length_over_width"].ge(10)).sum()
                ),
                "median_absolute_deviation_change": float(reviewed["absolute_deviation_change"].median()),
                "p95_absolute_deviation_change": float(reviewed["absolute_deviation_change"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def build_review_consensus(comparisons: pd.DataFrame) -> pd.DataFrame:
    keys = ["example_id", "id_link", "manual_decision", "requires_correction", "split_sensitive"]
    grouped = comparisons.groupby(keys, dropna=False, sort=True)
    consensus = grouped.agg(
        tested_window_count=("window_id", "nunique"),
        incoming_link_ids=("incoming_link_ids", "first"),
        outgoing_link_ids=("outgoing_link_ids", "first"),
        incoming_max_link_length=("incoming_max_link_length", "max"),
        incoming_max_length_over_width=("incoming_max_length_over_width", "max"),
        mean_continuity_abs_deviation=("mean_continuity_abs_deviation_recomputed", "first"),
        junction_continuity_abs_deviation_min=("junction_continuity_abs_deviation", "min"),
        junction_continuity_abs_deviation_max=("junction_continuity_abs_deviation", "max"),
        junction_continuity_abs_deviation_median=("junction_continuity_abs_deviation", "median"),
        maximum_absolute_deviation_change=("absolute_deviation_change", "max"),
        material_change_window_count=("material_continuity_change", "sum"),
        all_junction_endpoint_windows_nonoverlapping_all_windows=(
            "all_junction_endpoint_windows_nonoverlapping",
            "all",
        ),
        all_junction_endpoint_windows_nonoverlapping_any_window=(
            "all_junction_endpoint_windows_nonoverlapping",
            "any",
        ),
        mean_high_junction_low_all_windows=("mean_high_junction_low", "all"),
        mean_high_junction_low_any_window=("mean_high_junction_low", "any"),
        junction_high_mean_low_all_windows=("junction_high_mean_low", "all"),
        junction_high_mean_low_any_window=("junction_high_mean_low", "any"),
    ).reset_index()
    consensus["junction_window_sensitivity"] = (
        consensus["junction_continuity_abs_deviation_max"]
        - consensus["junction_continuity_abs_deviation_min"]
    )
    consensus["junction_result_stable_within_0_1"] = consensus["junction_window_sensitivity"].le(0.1)
    consensus["targeted_review_reason"] = ""
    old_may_overcall = (
        consensus["manual_decision"].eq("bad")
        & consensus["mean_high_junction_low_all_windows"]
    )
    junction_new_concern = (
        consensus["manual_decision"].eq("good")
        & consensus["junction_high_mean_low_all_windows"]
    )
    mean_issue_resolved = (
        consensus["manual_decision"].eq("good")
        & consensus["mean_high_junction_low_all_windows"]
    )
    consensus.loc[old_may_overcall, "targeted_review_reason"] = (
        "manual_bad_but_mean_discontinuity_resolves_near_junction"
    )
    consensus.loc[junction_new_concern, "targeted_review_reason"] = (
        "manual_good_but_new_near_junction_discontinuity"
    )
    consensus.loc[mean_issue_resolved, "targeted_review_reason"] = (
        "manual_good_and_mean_discontinuity_resolves_near_junction"
    )
    consensus["targeted_review_priority"] = np.select(
        [old_may_overcall | junction_new_concern, mean_issue_resolved, consensus["material_change_window_count"].gt(0)],
        [1, 2, 3],
        default=4,
    )
    return consensus.sort_values(
        ["targeted_review_priority", "maximum_absolute_deviation_change", "example_id", "id_link"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_junction_width_audit(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    review_root: str | Path = DEFAULT_REVIEW_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    windows: Sequence[WindowSpecification] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    review_root = Path(review_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    dataset = pd.read_csv(dataset_path, dtype={"example_id": "string"})
    dataset["example_id"] = dataset["example_id"].str.zfill(2)
    if not windows:
        raise ValueError("At least one junction window is required.")
    if len({spec.window_id for spec in windows}) != len(windows):
        raise ValueError("Junction window identifiers must be unique.")

    endpoint_frames: list[pd.DataFrame] = []
    source_files: list[Path] = [dataset_path]
    for example_id, review_rows in dataset.groupby("example_id", sort=True):
        relative_manifest = Path(str(review_rows["review_manifest_path"].iloc[0]))
        archive_manifest_path = (
            relative_manifest.resolve()
            if relative_manifest.is_absolute()
            else (review_root / relative_manifest).resolve()
        )
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        variant_manifest_path = Path(str(archive_manifest["source"]["variant_manifest"])).expanduser().resolve()
        variant_manifest = json.loads(variant_manifest_path.read_text(encoding="utf-8"))
        paths = variant_manifest.get("output_paths", {}) or {}
        families_path = Path(paths["link_width_families"]).expanduser().resolve()
        samples_path = Path(paths["link_width_samples"]).expanduser().resolve()
        links_path = Path(paths["directed_links"]).expanduser().resolve()
        nodes_path = Path(paths["directed_nodes"]).expanduser().resolve()
        for path in (archive_manifest_path, variant_manifest_path, families_path, samples_path, links_path, nodes_path):
            if not path.exists():
                raise FileNotFoundError(path)
            source_files.append(path)
        endpoint_frames.append(
            calculate_endpoint_widths(
                example_id=example_id,
                families=pd.read_csv(families_path),
                samples=pd.read_csv(samples_path),
                directed_links=gpd.read_file(links_path),
                directed_nodes=gpd.read_file(nodes_path),
                windows=windows,
            )
        )

    endpoints = pd.concat(endpoint_frames, ignore_index=True).sort_values(
        ["window_id", "example_id", "id_link"], kind="mergesort"
    ).reset_index(drop=True)
    comparisons = build_reviewed_junction_comparisons(endpoints, dataset)
    profile = build_window_profile(comparisons, endpoints)
    consensus = build_review_consensus(comparisons)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "endpoint_widths": output_dir / ENDPOINTS_FILENAME,
        "reviewed_comparisons": output_dir / COMPARISONS_FILENAME,
        "window_profile": output_dir / PROFILE_FILENAME,
        "review_consensus": output_dir / CONSENSUS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    _atomic_write_csv(endpoints, output_paths["endpoint_widths"])
    _atomic_write_csv(comparisons, output_paths["reviewed_comparisons"])
    _atomic_write_csv(profile, output_paths["window_profile"])
    _atomic_write_csv(consensus, output_paths["review_consensus"])
    unique_sources = tuple(dict.fromkeys(source_files))
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "code_revision": _git_revision(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "review_root": str(review_root),
        "example_count": int(dataset["example_id"].nunique()),
        "reviewed_link_count": int(len(dataset)),
        "all_link_count": int(endpoints[["example_id", "id_link"]].drop_duplicates().shape[0]),
        "windows": [
            {
                "window_id": spec.window_id,
                "fraction": spec.fraction,
                "minimum_length": spec.minimum_length,
                "maximum_length": spec.maximum_length,
                "definition": "median valid trimmed wet-width samples within requested length inward from closest retained endpoint sample",
            }
            for spec in windows
        ],
        "source_files": {str(path): _sha256(path) for path in unique_sources},
        "files": {key: str(path) for key, path in output_paths.items() if key != "manifest"},
        "file_sha256": {
            key: _sha256(path) for key, path in output_paths.items() if key != "manifest"
        },
        "production_authorized": False,
        "approval_note": "Exploratory audit only; select and approve a junction-window definition before model integration.",
    }
    _atomic_write_json(manifest, output_paths["manifest"])
    return {
        "output_dir": str(output_dir),
        "manifest": str(output_paths["manifest"]),
        "endpoint_rows": int(len(endpoints)),
        "comparison_rows": int(len(comparisons)),
        "priority_one_review_candidates": int(consensus["targeted_review_priority"].eq(1).sum()),
        "priority_two_interpretation_cases": int(consensus["targeted_review_priority"].eq(2).sum()),
        "profile": profile.to_dict("records"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whole-link versus near-junction wet widths across permanent width-review examples."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--window",
        action="append",
        type=parse_window,
        default=None,
        help=(
            "Candidate PERCENT_OR_FRACTION,MIN_LENGTH,MAX_LENGTH. Repeat for multiple windows. "
            "Defaults: 5,100,500; 10,250,1000; 15,250,1500."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_junction_width_audit(
        dataset_path=args.dataset,
        review_root=args.review_root,
        output_dir=args.output_dir,
        windows=tuple(args.window) if args.window else DEFAULT_WINDOWS,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

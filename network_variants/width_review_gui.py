from __future__ import annotations

import argparse
import json
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from rasterio.features import shapes
from rasterio.windows import from_bounds
from shapely.geometry import Point, box, shape

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from network_variants.width_review_archive import (
    ARCHIVE_DECISIONS_NAME,
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_SUMMARY_NAME,
    _atomic_write_csv,
    _atomic_write_json,
    finalize_width_review_archive,
    review_archive_dir,
    version_existing_review_archive,
)
from network_variants.width_review_fingerprint import (
    CORRUPT_STATUS,
    CURRENT_STATUS,
    INCOMPLETE_REVIEW_STATUS,
    MISSING_STATUS,
    upgrade_review_fingerprints,
    validate_review_fingerprints,
)


RIVGRAPH_WIDTH_TRIM_MULTIPLIER = 1.1
DEFAULT_WIDTH_FIELD = "wid_adj_wet"
DEFAULT_LEN_OVER_WIDTH_THRESHOLD = 1.5
CORE_WIDTH_FIELDS = ("wid", "wid_adj", "wid_med")
DEFAULT_REVIEW_CSV_NAME = ARCHIVE_DECISIONS_NAME
DEFAULT_REVIEW_SUMMARY_NAME = ARCHIVE_SUMMARY_NAME
QC_TRIMMED_FIELD = "qc_mean_opposition_cos_trimmed"
QC_TRIMMED_FALLBACK_FIELD = "qc_mean_opposition_cos_trimmed_from_samples"
QC_MEDIAN_FALLBACK_FIELD = "qc_median_opposition_cos_trimmed_from_samples"
DISPLAY_METRIC_FIELDS = (
    "len_over_width",
    "us_signed_dev",
    "ds_signed_dev",
    QC_TRIMMED_FALLBACK_FIELD,
    QC_MEDIAN_FALLBACK_FIELD,
    "qc_mean_opposition_cos",
    "n_trimmed_qc_samples",
    "split_sensitive",
)


@dataclass(frozen=True)
class VariantReviewPaths:
    variant_output_dir: Path
    manifest_path: Path
    collapsed_mask_path: Path
    directed_links_path: Path
    directed_nodes_path: Path
    link_width_families_path: Path
    link_width_samples_path: Path
    link_lineage_path: Path | None
    review_output_dir: Path
    review_csv_path: Path
    review_summary_path: Path
    source_links_path: Path
    source_nodes_path: Path
    review_mask_path: Path
    example_id: str
    variant_id: str


def normalize_example_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def infer_source_links_path(
    *,
    example_id: str,
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    normalized = normalize_example_id(example_id)
    return root / "rivgraph_centerline" / "outputs" / "runs" / normalized / "rivgraph" / f"{normalized}_links.gpkg"


def _to_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _mark_trimmed_sample_rows(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("sample_index", kind="mergesort").copy()
    n = len(group)
    if n == 0:
        group["used_in_trimmed_width"] = False
        return group

    widths = pd.to_numeric(group["width_total"], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(group["x"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(group["y"], errors="coerce").to_numpy(dtype=float)

    if n <= 1 or not np.isfinite(widths[[0, -1]]).all():
        trim = slice(0, n)
    else:
        step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        step = np.where(np.isfinite(step), step, 0.0)
        distances = np.insert(np.cumsum(step), 0, 0.0)
        revdistances = np.insert(np.cumsum(np.flipud(np.diff(distances))), 0, 0.0)
        start_index = int(np.argmin(np.abs(distances - widths[0] / 2.0 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER)))
        end_index = int(
            len(distances)
            - np.argmin(np.abs(revdistances - widths[-1] / 2.0 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER))
            - 1
        )
        trim = slice(0, n) if start_index >= end_index else slice(start_index, end_index)

    mask = np.zeros(n, dtype=bool)
    mask[trim] = True
    group["used_in_trimmed_width"] = mask
    return group


def compute_sample_qc_frame(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if samples.empty:
        return (
            samples.assign(used_in_trimmed_width=False).copy(),
            pd.DataFrame(
                columns=[
                    "id_link",
                    "qc_mean_opposition_cos",
                    QC_TRIMMED_FALLBACK_FIELD,
                    QC_MEDIAN_FALLBACK_FIELD,
                    "n_trimmed_qc_samples",
                ]
            ),
        )

    ordered = (
        samples.sort_values(["id_link", "sample_index"], kind="mergesort")
        .groupby("id_link", group_keys=False)
        .apply(_mark_trimmed_sample_rows)
        .reset_index(drop=True)
    )
    mean_frame = (
        ordered.groupby("id_link", as_index=False)
        .agg(qc_mean_opposition_cos=("qc_opposition_cos", "mean"))
    )
    trimmed_mask = ordered["used_in_trimmed_width"].fillna(False)
    if "width_invalid" in ordered.columns:
        trimmed_mask &= ~ordered["width_invalid"].fillna(False).astype(bool)
    trimmed_frame = (
        ordered.loc[trimmed_mask]
        .groupby("id_link", as_index=False)
        .agg(
            **{
                QC_TRIMMED_FALLBACK_FIELD: ("qc_opposition_cos", "mean"),
                QC_MEDIAN_FALLBACK_FIELD: ("qc_opposition_cos", "median"),
                "n_trimmed_qc_samples": ("qc_opposition_cos", "count"),
            }
        )
    )
    qc_frame = mean_frame.merge(trimmed_frame, on="id_link", how="left")
    return ordered, qc_frame


def build_candidate_frame(
    families: pd.DataFrame,
    sample_qc: pd.DataFrame,
    *,
    width_field: str,
    len_over_width_threshold: float,
) -> pd.DataFrame:
    frame = families.copy()
    frame["id_link"] = _to_numeric_series(frame, "id_link").astype("Int64")
    frame = frame.dropna(subset=["id_link"]).copy()
    frame["id_link"] = frame["id_link"].astype(int)
    frame["id_us_node"] = _to_numeric_series(frame, "id_us_node")
    frame["id_ds_node"] = _to_numeric_series(frame, "id_ds_node")
    frame["n_units"] = _to_numeric_series(frame, "n_units").fillna(0).astype(int)
    frame["len"] = _to_numeric_series(frame, "len")
    frame[width_field] = _to_numeric_series(frame, width_field)
    frame = frame.merge(sample_qc, on="id_link", how="left")

    if QC_TRIMMED_FIELD not in frame.columns:
        frame[QC_TRIMMED_FIELD] = frame[QC_TRIMMED_FALLBACK_FIELD]
    else:
        frame[QC_TRIMMED_FIELD] = pd.to_numeric(frame[QC_TRIMMED_FIELD], errors="coerce")
        frame[QC_TRIMMED_FIELD] = frame[QC_TRIMMED_FIELD].fillna(frame[QC_TRIMMED_FALLBACK_FIELD])

    frame["len_over_width"] = frame["len"] / frame[width_field]

    node_in_sum = frame.groupby("id_ds_node")[width_field].sum(min_count=1).to_dict()
    node_out_sum = frame.groupby("id_us_node")[width_field].sum(min_count=1).to_dict()
    node_in_count = frame.groupby("id_ds_node")["id_link"].nunique().to_dict()
    node_out_count = frame.groupby("id_us_node")["id_link"].nunique().to_dict()

    frame["us_in_width_sum"] = frame["id_us_node"].map(node_in_sum)
    frame["us_out_width_sum"] = frame["id_us_node"].map(node_out_sum)
    frame["ds_in_width_sum"] = frame["id_ds_node"].map(node_in_sum)
    frame["ds_out_width_sum"] = frame["id_ds_node"].map(node_out_sum)
    frame["us_in_count"] = frame["id_us_node"].map(node_in_count)
    frame["us_out_count"] = frame["id_us_node"].map(node_out_count)
    frame["ds_in_count"] = frame["id_ds_node"].map(node_in_count)
    frame["ds_out_count"] = frame["id_ds_node"].map(node_out_count)

    frame["us_node_out_over_in"] = frame["us_out_width_sum"] / frame["us_in_width_sum"]
    frame["ds_node_out_over_in"] = frame["ds_out_width_sum"] / frame["ds_in_width_sum"]
    frame["us_signed_dev"] = frame["us_node_out_over_in"] - 1.0
    frame["ds_signed_dev"] = frame["ds_node_out_over_in"] - 1.0
    frame["us_dev"] = (frame["us_node_out_over_in"] - 1.0).abs()
    frame["ds_dev"] = (frame["ds_node_out_over_in"] - 1.0).abs()
    frame["node_dev_max"] = frame[["us_dev", "ds_dev"]].max(axis=1)

    frame["candidate_flag"] = frame["n_units"].gt(0) & frame["len_over_width"].lt(float(len_over_width_threshold))
    frame["link_share_of_us_out"] = frame[width_field] / frame["us_out_width_sum"]
    frame["us_only_outgoing"] = frame["us_out_count"].fillna(0).le(1)
    frame["split_sensitive"] = frame["us_out_count"].fillna(0).gt(1)
    frame["review_order"] = np.arange(len(frame), dtype=int)
    return frame


def build_neighbor_lookup(frame: pd.DataFrame) -> dict[int, set[int]]:
    lookup: dict[int, set[int]] = {}
    for row in frame.itertuples(index=False):
        node_u = pd.to_numeric(pd.Series([getattr(row, "id_us_node", np.nan)]), errors="coerce").iloc[0]
        node_v = pd.to_numeric(pd.Series([getattr(row, "id_ds_node", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(node_u) or pd.isna(node_v):
            continue
        link_id = int(row.id_link)
        lookup.setdefault(int(node_u), set()).add(link_id)
        lookup.setdefault(int(node_v), set()).add(link_id)
    return lookup


def local_neighbor_ids(
    candidate_row: pd.Series,
    node_to_links: dict[int, set[int]],
) -> list[int]:
    link_ids = {int(candidate_row["id_link"])}
    for node_field in ("id_us_node", "id_ds_node"):
        node_id = pd.to_numeric(pd.Series([candidate_row.get(node_field)]), errors="coerce").iloc[0]
        if pd.isna(node_id):
            continue
        link_ids.update(node_to_links.get(int(node_id), set()))
    return sorted(link_ids)


def continuity_objective(
    width_value: float,
    *,
    us_in: float,
    us_out_other: float,
    ds_in_other: float,
    ds_out: float,
) -> tuple[float, float, float]:
    deviations: list[float] = []
    us_ratio = float("nan")
    ds_ratio = float("nan")

    if np.isfinite(us_in) and us_in > 0:
        us_ratio = (us_out_other + width_value) / us_in
        deviations.append(abs(us_ratio - 1.0))
    if np.isfinite(ds_out) and ds_out > 0 and np.isfinite(ds_in_other + width_value) and ds_in_other + width_value > 0:
        ds_ratio = ds_out / (ds_in_other + width_value)
        deviations.append(abs(ds_ratio - 1.0))

    if not deviations:
        return float("inf"), us_ratio, ds_ratio
    return float(max(deviations)), us_ratio, ds_ratio


def solve_continuity_replacement(
    candidate_row: pd.Series,
    frame: pd.DataFrame,
    *,
    width_field: str,
) -> dict[str, float | int | str]:
    def _sum_width(mask: pd.Series, *, empty_as_zero: bool) -> float:
        values = pd.to_numeric(frame.loc[mask, width_field], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            return 0.0 if empty_as_zero else float("nan")
        return float(values.sum())

    link_id = int(candidate_row["id_link"])
    us_node = int(candidate_row["id_us_node"])
    ds_node = int(candidate_row["id_ds_node"])
    current_width = float(candidate_row[width_field])

    us_in = _sum_width(frame["id_ds_node"] == us_node, empty_as_zero=False)
    us_out_other = _sum_width((frame["id_us_node"] == us_node) & (frame["id_link"] != link_id), empty_as_zero=True)
    ds_in_other = _sum_width((frame["id_ds_node"] == ds_node) & (frame["id_link"] != link_id), empty_as_zero=True)
    ds_out = _sum_width(frame["id_us_node"] == ds_node, empty_as_zero=False)

    local_mask = (
        (frame["id_us_node"].isin([us_node, ds_node])) |
        (frame["id_ds_node"].isin([us_node, ds_node]))
    ) & (frame["id_link"] != link_id)
    local_widths = pd.to_numeric(frame.loc[local_mask, width_field], errors="coerce")
    local_widths = local_widths[np.isfinite(local_widths) & local_widths.gt(0)]
    if local_widths.empty:
        lower = current_width * 0.5 if np.isfinite(current_width) and current_width > 0 else 1.0
        upper = current_width * 2.0 if np.isfinite(current_width) and current_width > 0 else lower
    else:
        lower = float(min(local_widths.min(), current_width))
        upper = float(max(local_widths.max(), current_width))
    if not np.isfinite(lower) or lower <= 0:
        lower = max(current_width * 0.5, 1e-6)
    if not np.isfinite(upper) or upper < lower:
        upper = max(lower, current_width)

    target_up = us_in - us_out_other if np.isfinite(us_in) and np.isfinite(us_out_other) else float("nan")
    target_down = ds_out - ds_in_other if np.isfinite(ds_out) and np.isfinite(ds_in_other) else float("nan")

    candidate_values: list[float] = [current_width, lower, upper]
    for value in local_widths.tolist():
        candidate_values.append(float(value))
    for target in (target_up, target_down):
        if np.isfinite(target) and target > 0:
            candidate_values.append(float(np.clip(target, lower, upper)))
    if upper > lower:
        candidate_values.extend(np.linspace(lower, upper, 401).tolist())

    best: dict[str, float | int | str] | None = None
    for width_value in sorted({round(float(value), 12) for value in candidate_values if np.isfinite(value) and value > 0}):
        objective, us_ratio_new, ds_ratio_new = continuity_objective(
            width_value,
            us_in=us_in,
            us_out_other=us_out_other,
            ds_in_other=ds_in_other,
            ds_out=ds_out,
        )
        if not np.isfinite(objective):
            continue
        tie_break = abs(np.log(width_value / current_width)) if np.isfinite(current_width) and current_width > 0 else abs(width_value)
        candidate = {
            "proposed_width": float(width_value),
            "predicted_node_dev_max_after": float(objective),
            "predicted_us_ratio_after": float(us_ratio_new) if np.isfinite(us_ratio_new) else float("nan"),
            "predicted_ds_ratio_after": float(ds_ratio_new) if np.isfinite(ds_ratio_new) else float("nan"),
            "tie_break": float(tie_break),
        }
        if best is None:
            best = candidate
            continue
        if candidate["predicted_node_dev_max_after"] < best["predicted_node_dev_max_after"] - 1e-12:
            best = candidate
            continue
        if abs(candidate["predicted_node_dev_max_after"] - best["predicted_node_dev_max_after"]) <= 1e-12 and candidate["tie_break"] < best["tie_break"]:
            best = candidate

    if best is None:
        raise ValueError(f"Could not solve continuity-based replacement for link {link_id}.")

    best["predicted_improvement"] = float(candidate_row["node_dev_max"]) - float(best["predicted_node_dev_max_after"])
    best["current_width_context"] = current_width
    best["width_update_method"] = "continuity_local_endpoint_context"
    return best


def _copy_backup(path: Path, *, suffix: str, overwrite: bool) -> Path:
    destination = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if destination.exists() and not overwrite:
        return destination
    shutil.copy2(path, destination)
    return destination


def resolve_variant_review_paths(
    variant_output_dir: str | Path,
    *,
    source_links_path: str | Path | None = None,
    review_output_dir: str | Path | None = None,
    review_csv_path: str | Path | None = None,
    review_summary_path: str | Path | None = None,
) -> VariantReviewPaths:
    variant_output_dir = Path(variant_output_dir).expanduser().resolve()
    manifest_path = variant_output_dir / "summary" / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_paths = manifest.get("output_paths", {}) or {}

    directed_links_path = Path(output_paths["directed_links"]).expanduser().resolve()
    directed_nodes_path = Path(output_paths["directed_nodes"]).expanduser().resolve()
    collapsed_mask_path = Path(output_paths["collapsed_mask"]).expanduser().resolve()
    link_width_families_path = Path(output_paths["link_width_families"]).expanduser().resolve()
    link_width_samples_path = Path(output_paths["link_width_samples"]).expanduser().resolve()
    link_lineage_candidate = variant_output_dir / "matching" / "link_lineage.csv"
    link_lineage_path = link_lineage_candidate if link_lineage_candidate.exists() else None

    families = pd.read_csv(link_width_families_path, nrows=1)
    if families.empty:
        raise ValueError(f"No rows found in {link_width_families_path}")
    example_id = normalize_example_id(families.iloc[0].get("example_id"))
    variant_id = str(families.iloc[0].get("variant_id"))

    resolved_source_links_path = (
        Path(source_links_path).expanduser().resolve()
        if source_links_path is not None
        else infer_source_links_path(example_id=example_id)
    )
    resolved_review_output_dir = (
        Path(review_output_dir).expanduser().resolve()
        if review_output_dir is not None
        else review_archive_dir(resolved_source_links_path)
    )
    resolved_review_csv_path = (
        Path(review_csv_path).expanduser().resolve()
        if review_csv_path is not None
        else (resolved_review_output_dir / DEFAULT_REVIEW_CSV_NAME).resolve()
    )
    resolved_review_summary_path = (
        Path(review_summary_path).expanduser().resolve()
        if review_summary_path is not None
        else (resolved_review_output_dir / DEFAULT_REVIEW_SUMMARY_NAME).resolve()
    )
    source_nodes_path = resolved_source_links_path.with_name(
        resolved_source_links_path.name.replace("_links.gpkg", "_nodes.gpkg")
    )
    cleaned_mask_path = (
        resolved_source_links_path.parent.parent
        / "masks_cleaned"
        / f"{example_id}_cleaned.tif"
    )
    review_mask_path = cleaned_mask_path if cleaned_mask_path.exists() else collapsed_mask_path

    return VariantReviewPaths(
        variant_output_dir=variant_output_dir,
        manifest_path=manifest_path,
        collapsed_mask_path=collapsed_mask_path,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        link_width_families_path=link_width_families_path,
        link_width_samples_path=link_width_samples_path,
        link_lineage_path=link_lineage_path,
        review_output_dir=resolved_review_output_dir,
        review_csv_path=resolved_review_csv_path,
        review_summary_path=resolved_review_summary_path,
        source_links_path=resolved_source_links_path,
        source_nodes_path=source_nodes_path,
        review_mask_path=review_mask_path,
        example_id=example_id,
        variant_id=variant_id,
    )


def build_parent_link_mapping(
    *,
    link_lineage_path: Path | None,
    families: pd.DataFrame,
) -> dict[int, int]:
    if link_lineage_path is None or not link_lineage_path.exists():
        return {int(link_id): int(link_id) for link_id in pd.to_numeric(families["id_link"], errors="coerce").dropna().astype(int)}

    lineage = pd.read_csv(link_lineage_path)
    mapping: dict[int, int] = {}
    if "id_link" not in lineage.columns:
        return {int(link_id): int(link_id) for link_id in pd.to_numeric(families["id_link"], errors="coerce").dropna().astype(int)}
    lineage["id_link"] = pd.to_numeric(lineage["id_link"], errors="coerce").astype("Int64")
    if "dominant_parent_link_id" in lineage.columns:
        lineage["dominant_parent_link_id"] = pd.to_numeric(lineage["dominant_parent_link_id"], errors="coerce").astype("Int64")
    for row in lineage.itertuples(index=False):
        child_id = getattr(row, "id_link")
        if pd.isna(child_id):
            continue
        parent_id = getattr(row, "dominant_parent_link_id", pd.NA)
        if pd.isna(parent_id):
            mapping[int(child_id)] = int(child_id)
        else:
            mapping[int(child_id)] = int(parent_id)
    for link_id in pd.to_numeric(families["id_link"], errors="coerce").dropna().astype(int):
        mapping.setdefault(int(link_id), int(link_id))
    return mapping


def apply_width_review_updates(
    *,
    review_decisions: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    width_field: str,
    source_links_path: Path,
    parent_link_mapping: dict[int, int],
    overwrite_backup: bool,
) -> dict[str, Any]:
    review_lookup = review_decisions.copy()
    review_lookup["id_link"] = pd.to_numeric(review_lookup["id_link"], errors="coerce").astype(int)
    review_lookup = review_lookup.drop_duplicates(subset=["id_link"], keep="last")
    candidate_lookup = candidate_frame.set_index("id_link", drop=False)

    source_links_backup = _copy_backup(source_links_path, suffix="_width_review_old", overwrite=overwrite_backup)
    source_links = gpd.read_file(source_links_path)
    source_links["id_link"] = pd.to_numeric(source_links["id_link"], errors="coerce").astype("Int64")
    source_links = source_links.dropna(subset=["id_link"]).copy()
    source_links["id_link"] = source_links["id_link"].astype(int)
    source_index = dict(zip(source_links["id_link"], source_links.index))

    for column, default in (
        ("width_reviewed_flag", False),
        ("width_review_decision", pd.NA),
        ("width_update_applied", False),
        ("width_update_method", pd.NA),
        ("width_update_source_link_id", pd.NA),
        ("width_update_scale_factor", pd.NA),
        ("width_update_target_context_width", pd.NA),
    ):
        if column not in source_links.columns:
            source_links[column] = default
    for field in CORE_WIDTH_FIELDS:
        orig_field = f"{field}_orig"
        if orig_field not in source_links.columns:
            source_links[orig_field] = pd.NA

    applied_rows: list[dict[str, Any]] = []
    mapped_bad_targets: dict[int, list[int]] = {}
    for row in review_lookup.itertuples(index=False):
        link_id = int(row.id_link)
        decision = str(getattr(row, "review_decision", "") or "").strip().lower()
        if decision != "bad":
            continue
        target_id = parent_link_mapping.get(link_id)
        if target_id is None:
            raise ValueError(f"No parent/source link mapping found for reviewed bad link {link_id}.")
        mapped_bad_targets.setdefault(int(target_id), []).append(link_id)
    ambiguous_targets = {target: children for target, children in mapped_bad_targets.items() if len(children) > 1}
    if ambiguous_targets:
        raise ValueError(f"Multiple bad reviewed links map to the same source link: {ambiguous_targets}")

    for row in review_lookup.itertuples(index=False):
        link_id = int(row.id_link)
        decision = str(getattr(row, "review_decision", "") or "").strip().lower()
        target_id = parent_link_mapping.get(link_id)
        if target_id is None or target_id not in source_index:
            continue
        row_index = source_index[target_id]
        source_links.at[row_index, "width_reviewed_flag"] = True
        source_links.at[row_index, "width_review_decision"] = decision
        source_links.at[row_index, "width_update_source_link_id"] = link_id
        existing_orig_values: dict[str, float] = {}
        for field in CORE_WIDTH_FIELDS:
            orig_field = f"{field}_orig"
            orig_value = pd.to_numeric(pd.Series([source_links.at[row_index, orig_field]]), errors="coerce").iloc[0]
            if np.isfinite(orig_value):
                existing_orig_values[field] = float(orig_value)

        if decision != "bad":
            if existing_orig_values:
                for field, orig_value in existing_orig_values.items():
                    source_links.at[row_index, field] = float(orig_value)
                    source_links.at[row_index, f"{field}_orig"] = np.nan
            source_links.at[row_index, "width_update_applied"] = False
            source_links.at[row_index, "width_update_method"] = None
            source_links.at[row_index, "width_update_scale_factor"] = np.nan
            source_links.at[row_index, "width_update_target_context_width"] = np.nan
            continue

        candidate_row = candidate_lookup.loc[link_id]
        replacement = solve_continuity_replacement(candidate_row, candidate_frame, width_field=width_field)
        current_context_width = float(candidate_row[width_field])
        proposed_width = float(replacement["proposed_width"])
        if not np.isfinite(current_context_width) or current_context_width <= 0:
            raise ValueError(f"Non-positive current context width for reviewed bad link {link_id}.")
        scale_factor = proposed_width / current_context_width

        source_links.at[row_index, "width_update_applied"] = True
        source_links.at[row_index, "width_update_method"] = str(replacement["width_update_method"])
        source_links.at[row_index, "width_update_scale_factor"] = float(scale_factor)
        source_links.at[row_index, "width_update_target_context_width"] = proposed_width

        for field in CORE_WIDTH_FIELDS:
            orig_field = f"{field}_orig"
            base_value = existing_orig_values.get(field)
            if base_value is None:
                current_value = pd.to_numeric(pd.Series([source_links.at[row_index, field]]), errors="coerce").iloc[0]
                if not np.isfinite(current_value):
                    continue
                base_value = float(current_value)
                source_links.at[row_index, orig_field] = base_value
            source_links.at[row_index, field] = float(base_value) * float(scale_factor)

        applied_rows.append(
            {
                "reviewed_link_id": link_id,
                "source_link_id": target_id,
                "decision": decision,
                "current_context_width": current_context_width,
                "proposed_context_width": proposed_width,
                "scale_factor": float(scale_factor),
                "predicted_node_dev_max_after": float(replacement["predicted_node_dev_max_after"]),
                "predicted_improvement": float(replacement["predicted_improvement"]),
            }
        )

    source_links.to_file(source_links_path, driver="GPKG")
    return {
        "source_links_path": str(source_links_path),
        "source_links_backup_path": str(source_links_backup),
        "n_reviewed_links": int(len(review_lookup)),
        "n_applied_updates": int(len(applied_rows)),
        "applied_updates": applied_rows,
    }


class WidthReviewGui:
    def __init__(
        self,
        root: tk.Tk,
        *,
        paths: VariantReviewPaths,
        candidate_frame: pd.DataFrame,
        metric_frame: pd.DataFrame,
        directed_links: gpd.GeoDataFrame,
        marked_samples: pd.DataFrame,
        width_field: str,
        show_dt_points: bool,
        initial_decisions: pd.DataFrame | None = None,
    ) -> None:
        self.root = root
        self.paths = paths
        self.width_field = width_field
        self.show_dt_points = bool(show_dt_points)
        self.root.title(f"Width review {paths.example_id} / {paths.variant_id}")
        self.collapsed_mask_path = paths.collapsed_mask_path

        self.metric_frame = metric_frame.copy()
        self.marked_samples = marked_samples.copy()
        self.candidate_frame = candidate_frame.loc[candidate_frame["candidate_flag"]].copy()
        self.candidate_frame = self.candidate_frame.sort_values(
            ["len_over_width", "node_dev_max", QC_TRIMMED_FIELD],
            ascending=[True, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        self.directed_links = directed_links.copy()
        self.directed_links["id_link"] = pd.to_numeric(self.directed_links["id_link"], errors="coerce").astype("Int64")
        self.directed_links = self.directed_links.dropna(subset=["id_link"]).copy()
        self.directed_links["id_link"] = self.directed_links["id_link"].astype(int)
        self.directed_links["id_us_node"] = pd.to_numeric(self.directed_links["id_us_node"], errors="coerce")
        self.directed_links["id_ds_node"] = pd.to_numeric(self.directed_links["id_ds_node"], errors="coerce")
        merge_columns = [
            column
            for column in ("id_link", self.width_field, *DISPLAY_METRIC_FIELDS, "n_units")
            if column in self.metric_frame.columns
        ]
        if merge_columns:
            metric_lookup = self.metric_frame[merge_columns].copy()
            drop_columns = [column for column in merge_columns if column != "id_link" and column in self.directed_links.columns]
            if drop_columns:
                self.directed_links = self.directed_links.drop(columns=drop_columns, errors="ignore")
            self.directed_links = self.directed_links.merge(metric_lookup, on="id_link", how="left", validate="1:1")
        self.directed_lookup = self.directed_links.set_index("id_link", drop=False)
        self.node_to_links = build_neighbor_lookup(self.directed_links)

        self.decisions = pd.DataFrame(
            {
                "id_link": self.candidate_frame["id_link"].astype(int),
                "review_decision": pd.Series([pd.NA] * len(self.candidate_frame), dtype="object"),
            }
        )
        if initial_decisions is not None and not initial_decisions.empty:
            initial = initial_decisions.copy()
            initial["id_link"] = pd.to_numeric(initial["id_link"], errors="coerce").astype("Int64")
            initial = initial.dropna(subset=["id_link"]).copy()
            initial["id_link"] = initial["id_link"].astype(int)
            if "review_decision" in initial.columns:
                initial["review_decision"] = initial["review_decision"].astype("string").str.strip().str.lower()
                initial["review_decision"] = initial["review_decision"].mask(
                    initial["review_decision"].isin(["", "skip"]), pd.NA
                )
                initial = initial.loc[initial["review_decision"].notna(), ["id_link", "review_decision"]]
                initial = initial.drop_duplicates(subset=["id_link"], keep="last")
                self.decisions = self.decisions.drop(columns=["review_decision"]).merge(
                    initial,
                    on="id_link",
                    how="left",
                    validate="1:1",
                )
        if self.candidate_frame.empty:
            self.current_pos = None
        else:
            undecided = self.decisions["review_decision"].isna()
            self.current_pos = int(undecided.idxmax()) if undecided.any() else 0
        self.saved = False
        self.map_view_mode = "local"

        self._build_ui()
        self._bind_shortcuts()
        self._draw()

    def _build_ui(self) -> None:
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.info_var = tk.StringVar()
        info_label = tk.Label(
            controls,
            textvariable=self.info_var,
            anchor="w",
            justify="left",
            wraplength=1200,
        )
        info_label.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 6))

        button_row = tk.Frame(controls)
        button_row.pack(side=tk.TOP, fill=tk.X)
        tk.Button(button_row, text="Good [G]", command=lambda: self._set_decision_and_advance("good")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Bad [B]", command=lambda: self._set_decision_and_advance("bad")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Skip [S]", command=lambda: self._set_decision("skip")).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Prev [Left]", command=lambda: self._move_relative(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Next [Right]", command=lambda: self._move_relative(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Zoom Focal [Z]", command=self._zoom_to_focal).pack(side=tk.LEFT, padx=8)
        tk.Button(button_row, text="Zoom Reset [X]", command=self._reset_zoom).pack(side=tk.LEFT, padx=2)
        tk.Button(button_row, text="Save and Close [W]", command=self._save_and_close).pack(side=tk.LEFT, padx=8)

        self.figure = Figure(figsize=(14, 7), dpi=100)
        self.ax_map = self.figure.add_subplot(121)
        self.ax_table = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Key>", self._on_key_press)

    def _on_key_press(self, event) -> None:
        key = (event.keysym or "").lower()
        if key == "g":
            self._set_decision_and_advance("good")
        elif key == "b":
            self._set_decision_and_advance("bad")
        elif key == "s":
            self._set_decision("skip")
        elif key == "left":
            self._move_relative(-1)
        elif key == "right":
            self._move_relative(1)
        elif key == "z":
            self._zoom_to_focal()
        elif key == "x":
            self._reset_zoom()
        elif key == "w":
            self._save_and_close()

    def _current_row(self) -> pd.Series | None:
        if self.current_pos is None or self.candidate_frame.empty:
            return None
        return self.candidate_frame.iloc[self.current_pos]

    def _current_decision(self, link_id: int) -> str:
        series = self.decisions.loc[self.decisions["id_link"] == link_id, "review_decision"]
        if series.empty:
            return ""
        value = series.iloc[0]
        return "" if pd.isna(value) else str(value)

    def _set_decision(self, decision: str) -> None:
        row = self._current_row()
        if row is None:
            return
        link_id = int(row["id_link"])
        self.decisions.loc[self.decisions["id_link"] == link_id, "review_decision"] = decision
        self._draw()

    def _set_decision_and_advance(self, decision: str) -> None:
        self._set_decision(decision)
        self._move_relative(1)

    def _move_relative(self, step: int) -> None:
        if self.current_pos is None:
            return
        self.current_pos = max(0, min(len(self.candidate_frame) - 1, self.current_pos + step))
        self.map_view_mode = "local"
        self._draw()

    def _zoom_to_focal(self) -> None:
        self.map_view_mode = "focal"
        self._draw()

    def _reset_zoom(self) -> None:
        self.map_view_mode = "local"
        self._draw()

    def _neighbor_table_text(self, local_links: pd.DataFrame, focal_link_id: int) -> str:
        rows = []
        ordered = local_links.assign(
            _is_focal=local_links["id_link"].astype(int).eq(int(focal_link_id))
        ).sort_values(["_is_focal", "id_link"], ascending=[False, True], kind="mergesort")
        for row in ordered.itertuples(index=False):
            role = "FOCAL" if int(row.id_link) == focal_link_id else "nbr"
            wid = pd.to_numeric(pd.Series([getattr(row, "wid", np.nan)]), errors="coerce").iloc[0]
            wid_adj = pd.to_numeric(pd.Series([getattr(row, "wid_adj", np.nan)]), errors="coerce").iloc[0]
            rows.append(
                f"{role:5s} "
                f"{'link=':<7s}{int(row.id_link):<7d}  "
                f"{'dir=':<9s}{int(row.id_us_node):<3d}->{int(row.id_ds_node):<3d}  "
            )
            rows.append("      " f"{'wid=':<7s}{wid:<7.1f}  " f"{'wid_adj=':<9s}{wid_adj:<7.1f}")
            if role == "FOCAL":
                len_over_width = pd.to_numeric(pd.Series([getattr(row, "len_over_width", np.nan)]), errors="coerce").iloc[0]
                us_signed_dev = pd.to_numeric(pd.Series([getattr(row, "us_signed_dev", np.nan)]), errors="coerce").iloc[0]
                ds_signed_dev = pd.to_numeric(pd.Series([getattr(row, "ds_signed_dev", np.nan)]), errors="coerce").iloc[0]
                qc_trimmed = pd.to_numeric(
                    pd.Series([getattr(row, QC_TRIMMED_FALLBACK_FIELD, np.nan)]),
                    errors="coerce",
                ).iloc[0]
                qc_median = pd.to_numeric(
                    pd.Series([getattr(row, QC_MEDIAN_FALLBACK_FIELD, np.nan)]),
                    errors="coerce",
                ).iloc[0]
                n_trimmed = pd.to_numeric(pd.Series([getattr(row, "n_trimmed_qc_samples", np.nan)]), errors="coerce").iloc[0]
                split_sensitive = bool(pd.to_numeric(pd.Series([getattr(row, "split_sensitive", np.nan)]), errors="coerce").fillna(0).iloc[0])
                rows.append(
                    "      "
                    f"{'len/w=':<7s}{len_over_width:<7.3f}  "
                    f"{'us_sd=':<7s}{us_signed_dev:<7.3f}  "
                    f"{'ds_sd=':<7s}{ds_signed_dev:<7.3f}"
                )
                rows.append(
                    "      "
                    f"{'qc_tm=':<7s}{qc_trimmed:<7.3f}  "
                    f"{'qc_md=':<7s}{qc_median:<7.3f}  "
                    f"{'n_trim=':<8s}{n_trimmed:<6.0f}"
                )
                rows.append("      " f"{'split=':<7s}{str(split_sensitive):<7s}")
            rows.append("")
        return "\n".join(rows)

    def _routing_impact_text(self, row: pd.Series) -> str:
        us_out_count = int(pd.to_numeric(pd.Series([row.get("us_out_count")]), errors="coerce").fillna(0).iloc[0])
        us_only_outgoing = bool(pd.to_numeric(pd.Series([row.get("us_only_outgoing")]), errors="coerce").fillna(0).iloc[0])
        link_share = pd.to_numeric(pd.Series([row.get("link_share_of_us_out")]), errors="coerce").iloc[0]
        if us_only_outgoing or us_out_count <= 1:
            return "routing: local-only | upstream node has no alternative outgoing split"
        share_text = f"{link_share:.3f}" if np.isfinite(link_share) else "nan"
        return f"routing: split-sensitive | us_out_count={us_out_count} | share_us_out={share_text}"

    def _current_dt_points(self, link_id: int) -> pd.DataFrame:
        if self.marked_samples.empty or "id_link" not in self.marked_samples.columns:
            return self.marked_samples.iloc[0:0].copy()
        samples = self.marked_samples.loc[self.marked_samples["id_link"] == link_id].copy()
        if samples.empty:
            return samples
        if "used_in_trimmed_width" in samples.columns:
            samples = samples.loc[samples["used_in_trimmed_width"].fillna(False)].copy()
        if "width_invalid" in samples.columns:
            samples = samples.loc[~samples["width_invalid"].fillna(False).astype(bool)].copy()
        return samples

    def _map_bounds(
        self,
        local_links: gpd.GeoDataFrame,
        dt_points: pd.DataFrame,
    ) -> tuple[float, float, float, float]:
        minx, miny, maxx, maxy = local_links.total_bounds
        bounds = [minx, miny, maxx, maxy]
        if self.show_dt_points and not dt_points.empty:
            xs = pd.to_numeric(dt_points["x"], errors="coerce")
            ys = pd.to_numeric(dt_points["y"], errors="coerce")
            xs = xs[np.isfinite(xs)]
            ys = ys[np.isfinite(ys)]
            if not xs.empty and not ys.empty:
                bounds[0] = min(bounds[0], float(xs.min()))
                bounds[1] = min(bounds[1], float(ys.min()))
                bounds[2] = max(bounds[2], float(xs.max()))
                bounds[3] = max(bounds[3], float(ys.max()))
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        pad = span * 0.15
        return (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    def _focal_bounds(
        self,
        focal_frame: gpd.GeoDataFrame,
        dt_points: pd.DataFrame,
    ) -> tuple[float, float, float, float]:
        if focal_frame.empty:
            return self._map_bounds(focal_frame, dt_points)
        minx, miny, maxx, maxy = focal_frame.total_bounds
        bounds = [minx, miny, maxx, maxy]
        if self.show_dt_points and not dt_points.empty:
            xs = [pd.to_numeric(dt_points.get(col, np.nan), errors="coerce") for col in ("x", "left_bank_x", "right_bank_x")]
            ys = [pd.to_numeric(dt_points.get(col, np.nan), errors="coerce") for col in ("y", "left_bank_y", "right_bank_y")]
            xs = pd.concat(xs, ignore_index=True)
            ys = pd.concat(ys, ignore_index=True)
            xs = xs[np.isfinite(xs)]
            ys = ys[np.isfinite(ys)]
            if not xs.empty and not ys.empty:
                bounds[0] = min(bounds[0], float(xs.min()))
                bounds[1] = min(bounds[1], float(ys.min()))
                bounds[2] = max(bounds[2], float(xs.max()))
                bounds[3] = max(bounds[3], float(ys.max()))
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        pad = span * 0.24
        return (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    def _plot_direction_arrow(
        self,
        geometry: Any,
        *,
        color: str,
        linewidth: float,
        zorder: int,
        position_frac: float = 0.3,
    ) -> None:
        if geometry is None or geometry.is_empty:
            return
        try:
            half_step = 0.03
            start = geometry.interpolate(max(0.0, position_frac - half_step), normalized=True)
            end = geometry.interpolate(min(1.0, position_frac + half_step), normalized=True)
        except Exception:
            print(f"Warning: could not compute arrow for geometry {geometry}")
            return

        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return

        self.ax_map.annotate(
            "",
            xy=(float(end.x), float(end.y)),
            xytext=(float(start.x), float(start.y)),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": linewidth,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 22,
            },
            zorder=zorder,
        )

    def _label_anchor(
        self,
        geometry: Any,
        *,
        span: float,
        position_frac: float,
        side: float,
        offset_scale: float = 0.03,
    ) -> tuple[float, float]:
        if geometry is None or geometry.is_empty:
            return (float("nan"), float("nan"))
        try:
            point = geometry.interpolate(position_frac, normalized=True)
            left = geometry.interpolate(max(0.0, position_frac - 0.03), normalized=True)
            right = geometry.interpolate(min(1.0, position_frac + 0.03), normalized=True)
        except Exception:
            point = geometry.representative_point()
            return float(point.x), float(point.y)

        dx = float(right.x - left.x)
        dy = float(right.y - left.y)
        norm = float(np.hypot(dx, dy))
        if not np.isfinite(norm) or norm == 0:
            return float(point.x), float(point.y)
        nx = -dy / norm
        ny = dx / norm
        offset = span * offset_scale * side
        return float(point.x + nx * offset), float(point.y + ny * offset)

    def _annotation_layout(
        self,
        *,
        feature_id: int,
        focal_link_id: int,
    ) -> dict[str, float]:
        is_focal = int(feature_id) == int(focal_link_id)
        if is_focal:
            return {
                "arrow_frac": 0.6 if self.map_view_mode == "focal" else 0.42,
                "label_frac": 0.78 if self.map_view_mode == "focal" else 0.68,
                "label_side": -1.0,
                "label_offset_scale": 0.042 if self.map_view_mode == "focal" else 0.034,
            }
        alternating_side = -1.0 if int(feature_id) % 2 == 0 else 1.0
        return {
            "arrow_frac": 0.34 if self.map_view_mode == "focal" else 0.27,
            "label_frac": 0.58 if self.map_view_mode == "focal" else 0.48,
            "label_side": alternating_side,
            "label_offset_scale": 0.038 if self.map_view_mode == "focal" else 0.03,
        }

    def _local_island_geometries(
        self,
        local_links: pd.DataFrame,
        dt_points: pd.DataFrame,
        ) -> list[Any]:
        if local_links.empty:
            return []
        bounds = self._map_bounds(local_links, dt_points)
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        bbox = box(*bounds)

        with rasterio.open(self.collapsed_mask_path) as ds:
            window = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                return []
            data      = ds.read(1, window=window, boundless=True, fill_value=0)
            transform = ds.window_transform(window)

        land_mask = (data == 0).astype(np.uint8)
        if not land_mask.any():
            return []

        islands: list[Any] = []
        eps = span * 1e-6 + 1e-9
        corners = [
            Point(bounds[0], bounds[1]),
            Point(bounds[0], bounds[3]),
            Point(bounds[2], bounds[1]),
            Point(bounds[2], bounds[3]),
        ]
        for geom_mapping, value in shapes(land_mask, mask=land_mask.astype(bool), transform=transform):
            if int(value) != 1:
                continue
            geom = shape(geom_mapping).intersection(bbox)
            if geom.is_empty:
                continue
            # if any(geom.intersects(corner.buffer(eps)) for corner in corners):
            #     continue
            islands.append(geom)
        return islands

    def _draw(self) -> None:
        self.ax_map.clear()
        self.ax_table.clear()

        row = self._current_row()
        if row is None:
            self.info_var.set("No candidate links to review.")
            self.canvas.draw_idle()
            return

        link_id = int(row["id_link"])
        decision = self._current_decision(link_id) or "unreviewed"
        neighbor_ids = local_neighbor_ids(row, self.node_to_links)
        local_links = self.directed_links.loc[self.directed_links["id_link"].isin(neighbor_ids)].copy()
        focal_frame = local_links.loc[local_links["id_link"] == link_id]
        neighbor_frame = local_links.loc[local_links["id_link"] != link_id]
        dt_points = self._current_dt_points(link_id)
        local_bounds = self._map_bounds(local_links, dt_points)
        focal_bounds = self._focal_bounds(focal_frame, dt_points)
        bounds = focal_bounds if self.map_view_mode == "focal" else local_bounds
        span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        island_geoms = self._local_island_geometries(local_links, dt_points)

        if island_geoms:
            island_gdf = gpd.GeoDataFrame(geometry=island_geoms, crs=local_links.crs)
            island_gdf.plot(
                ax=self.ax_map,
                facecolor="#d9d9d9",
                edgecolor="#636363",
                linewidth=1.0,
                alpha=0.8,
                zorder=0,
            )

        if not neighbor_frame.empty:
            neighbor_frame.plot(ax=self.ax_map, color="#9ecae1", linewidth=2, zorder=1)
        if not focal_frame.empty:
            focal_frame.plot(ax=self.ax_map, color="#001f87", linewidth=4, zorder=3)
        for feature in neighbor_frame.itertuples(index=False):
            layout = self._annotation_layout(feature_id=int(feature.id_link), focal_link_id=link_id)
            self._plot_direction_arrow(
                feature.geometry,
                color="#3182bd",
                linewidth=1.5,
                zorder=2,
                position_frac=layout["arrow_frac"],
            )
        for feature in focal_frame.itertuples(index=False):
            layout = self._annotation_layout(feature_id=int(feature.id_link), focal_link_id=link_id)
            self._plot_direction_arrow(
                feature.geometry,
                color="#626262",
                linewidth=2.1,
                zorder=4,
                position_frac=layout["arrow_frac"],
            )

        for feature in local_links.itertuples(index=False):
            layout = self._annotation_layout(feature_id=int(feature.id_link), focal_link_id=link_id)
            label_x, label_y = self._label_anchor(
                feature.geometry,
                span=span,
                position_frac=layout["label_frac"],
                side=layout["label_side"],
                offset_scale=layout["label_offset_scale"],
            )
            if not np.isfinite(label_x) or not np.isfinite(label_y):
                continue
            width_value = pd.to_numeric(pd.Series([getattr(feature, self.width_field, np.nan)]), errors="coerce").iloc[0]
            self.ax_map.text(
                label_x,
                label_y,
                f"{int(feature.id_link)}\n{width_value:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "0.3", "alpha": 0.8, "boxstyle": "round,pad=0.2"},
                zorder=4,
            )

        if self.show_dt_points and not dt_points.empty:
            xvals = pd.to_numeric(dt_points["x"], errors="coerce")
            yvals = pd.to_numeric(dt_points["y"], errors="coerce")
            point_values = pd.to_numeric(dt_points.get("qc_opposition_cos", np.nan), errors="coerce")
            valid = np.isfinite(xvals) & np.isfinite(yvals)
            xvals = xvals[valid]
            yvals = yvals[valid]
            point_values = point_values[valid]
            # self.ax_map.scatter(
            #     xvals,
            #     yvals,
            #     s=32,
            #     c=point_values if np.isfinite(point_values).any() else "#222222",
            #     cmap="coolwarm",
            #     edgecolors="black",
            #     linewidths=0.4,
            #     zorder=5,
            # )
            left_x = pd.to_numeric(dt_points.get("left_bank_x", np.nan), errors="coerce")
            left_y = pd.to_numeric(dt_points.get("left_bank_y", np.nan), errors="coerce")
            right_x = pd.to_numeric(dt_points.get("right_bank_x", np.nan), errors="coerce")
            right_y = pd.to_numeric(dt_points.get("right_bank_y", np.nan), errors="coerce")
            transect_valid = valid & np.isfinite(left_x) & np.isfinite(left_y) & np.isfinite(right_x) & np.isfinite(right_y)
            if transect_valid.any():
                for cx, cy, lx, ly, rx, ry in zip(
                    xvals[transect_valid],
                    yvals[transect_valid],
                    left_x[transect_valid],
                    left_y[transect_valid],
                    right_x[transect_valid],
                    right_y[transect_valid],
                ):
                    self.ax_map.plot(
                        [float(cx), float(lx)],
                        [float(cy), float(ly)],
                        color="#2ca25f",
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=4,
                    )
                    self.ax_map.plot(
                        [float(cx), float(rx)],
                        [float(cy), float(ry)],
                        color="#de2d26",
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=4,
                    )
                self.ax_map.scatter(
                    left_x[transect_valid],
                    left_y[transect_valid],
                    s=18,
                    c="#2ca25f",
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=5,
                    label="left bank",
                )
                self.ax_map.scatter(
                    right_x[transect_valid],
                    right_y[transect_valid],
                    s=18,
                    c="#de2d26",
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=5,
                    label="right bank",
                )
            # for sample_row in dt_points.loc[valid].itertuples(index=False):
            #     sx = pd.to_numeric(pd.Series([getattr(sample_row, "x", np.nan)]), errors="coerce").iloc[0]
            #     sy = pd.to_numeric(pd.Series([getattr(sample_row, "y", np.nan)]), errors="coerce").iloc[0]
            #     if not np.isfinite(sx) or not np.isfinite(sy):
            #         continue
            #     sample_index = getattr(sample_row, "sample_index", None)
            #     label = f"{int(sample_index)}" if sample_index is not None and not pd.isna(sample_index) else ""
            #     if label:
            #         self.ax_map.text(
            #             float(sx),
            #             float(sy),
            #             label,
            #             fontsize=7,
            #             ha="left",
            #             va="bottom",
            #             color="black",
            #             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 0.1},
            #             zorder=6,
            #         )

        self.ax_map.set_aspect("equal", adjustable="box")
        self.ax_map.set_xlim(bounds[0], bounds[2])
        self.ax_map.set_ylim(bounds[1], bounds[3])
        self.ax_map.set_title("Local link neighborhood")

        metrics_lines = [
            f"link {link_id} | review {self.current_pos + 1}/{len(self.candidate_frame)} | decision={decision} | view={self.map_view_mode}",
            f"len_over_width={float(row['len_over_width']):.3f} | us_signed_dev={float(row['us_signed_dev']):.3f} | ds_signed_dev={float(row['ds_signed_dev']):.3f}",
            f"qc_trimmed={float(row[QC_TRIMMED_FIELD]):.3f} | qc_median={float(row[QC_MEDIAN_FALLBACK_FIELD]):.3f} | n_trimmed={float(row['n_trimmed_qc_samples']):.0f}",
            f"context width field={self.width_field} value={float(row[self.width_field]):.3f} | n_units={int(row['n_units'])}",
        ]
        self.info_var.set("\n".join(metrics_lines))

        self.ax_table.axis("off")
        self.ax_table.set_title("Local widths and metrics")
        right_panel_text = self._neighbor_table_text(local_links, link_id) + "\n" 
        # + self._routing_impact_text(row)
        self.ax_table.text(
            0.0,
            1.0,
            right_panel_text,
            fontsize=14,
            family="monospace",
            va="top",
            linespacing=1.5,
        )
        self.canvas.draw_idle()

    def _save_and_close(self) -> None:
        normalized = self.decisions["review_decision"].astype("string").str.strip().str.lower()
        normalized = normalized.mask(normalized.isin(["", "skip"]), pd.NA)
        invalid = normalized.notna() & ~normalized.isin(["good", "bad"])
        if invalid.any():
            messagebox.showerror(
                "Invalid review decision",
                "Review decisions must be good, bad, or undecided.",
            )
            return
        self.decisions["review_decision"] = normalized
        undecided = normalized.isna().sum()
        if undecided:
            if not messagebox.askyesno(
                "Save partial review",
                f"{undecided} candidate link(s) are still unreviewed. Save anyway and apply only the reviewed decisions?",
            ):
                return

        output = self.candidate_frame.merge(self.decisions, on="id_link", how="left", validate="1:1")
        self.paths.review_output_dir.mkdir(parents=True, exist_ok=True)
        version_existing_review_archive(self.paths.review_output_dir)
        _atomic_write_csv(output, self.paths.review_csv_path)
        self.saved = True
        self.root.destroy()


def load_review_context(
    paths: VariantReviewPaths,
    *,
    width_field: str,
    len_over_width_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, dict[int, int]]:
    families = pd.read_csv(paths.link_width_families_path)
    samples = pd.read_csv(paths.link_width_samples_path)
    marked_samples, sample_qc = compute_sample_qc_frame(samples)
    resolved_width_field = width_field if width_field in families.columns else "wid_adj"
    candidate_frame = build_candidate_frame(
        families,
        sample_qc,
        width_field=resolved_width_field,
        len_over_width_threshold=len_over_width_threshold,
    )
    directed_links = gpd.read_file(paths.directed_links_path)
    parent_link_mapping = build_parent_link_mapping(link_lineage_path=paths.link_lineage_path, families=families)
    return candidate_frame, marked_samples, sample_qc, directed_links, parent_link_mapping


def inspect_existing_review(paths: VariantReviewPaths) -> dict[str, Any]:
    """Validate a permanent review before its decisions are joined by link ID."""
    manifest_path = paths.review_output_dir / ARCHIVE_MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "archive_dir": str(paths.review_output_dir),
            "status": MISSING_STATUS,
            "issues": ["No permanent review manifest exists yet."],
        }
    return validate_review_fingerprints(
        paths.review_output_dir,
        mask_path=paths.review_mask_path,
        links_path=paths.source_links_path,
        nodes_path=paths.source_nodes_path,
    )


def _allow_existing_review(root: tk.Tk, inspection: dict[str, Any]) -> bool:
    """Return whether saved decisions may be resumed after freshness inspection."""
    status = str(inspection.get("status", MISSING_STATUS))
    if status in {CURRENT_STATUS, INCOMPLETE_REVIEW_STATUS, MISSING_STATUS}:
        return True
    issues = inspection.get("issues", []) or []
    detail = "\n".join(f"- {issue}" for issue in issues)
    if status == CORRUPT_STATUS:
        title = "Damaged width-review archive"
        prompt = (
            "The existing archive failed its integrity check and cannot safely be resumed. "
            "Start a fresh review? The prior files will be preserved in versions/ when you save."
        )
    else:
        title = "Stale width review"
        prompt = (
            f"The existing width review is {status} and cannot safely be joined to the current links by ID. "
            "Start a fresh review? The prior files will be preserved in versions/ when you save."
        )
    if detail:
        prompt += f"\n\n{detail}"
    return bool(messagebox.askyesno(title, prompt, parent=root))


def run_width_review_gui(
    paths: VariantReviewPaths,
    *,
    width_field: str,
    len_over_width_threshold: float,
    overwrite_backup: bool,
    show_dt_points: bool,
    start_over: bool,
) -> dict[str, Any] | None:
    root = tk.Tk()
    inspection = inspect_existing_review(paths)
    resume_existing = not start_over
    if resume_existing and paths.review_csv_path.exists():
        status = str(inspection.get("status", MISSING_STATUS))
        if status not in {CURRENT_STATUS, INCOMPLETE_REVIEW_STATUS, MISSING_STATUS}:
            if not _allow_existing_review(root, inspection):
                root.destroy()
                return None
            resume_existing = False

    candidate_frame, marked_samples, _, directed_links, parent_link_mapping = load_review_context(
        paths,
        width_field=width_field,
        len_over_width_threshold=len_over_width_threshold,
    )
    resolved_width_field = width_field if width_field in candidate_frame.columns else "wid_adj"
    initial_decisions = None
    if resume_existing and paths.review_csv_path.exists():
        initial_decisions = pd.read_csv(paths.review_csv_path)

    gui = WidthReviewGui(
        root,
        paths=paths,
        candidate_frame=candidate_frame,
        metric_frame=candidate_frame,
        directed_links=directed_links,
        marked_samples=marked_samples,
        width_field=resolved_width_field,
        show_dt_points=show_dt_points,
        initial_decisions=initial_decisions,
    )
    root.mainloop()

    if not gui.saved:
        return None

    review_decisions = pd.read_csv(paths.review_csv_path)
    review_decisions["review_decision"] = review_decisions["review_decision"].astype("string").str.strip().str.lower()
    reviewed = review_decisions.loc[
        review_decisions["review_decision"].isin(["good", "bad"])
    ].copy()
    apply_summary = apply_width_review_updates(
        review_decisions=reviewed,
        candidate_frame=candidate_frame,
        width_field=resolved_width_field,
        source_links_path=paths.source_links_path,
        parent_link_mapping=parent_link_mapping,
        overwrite_backup=overwrite_backup,
    )
    summary = {
        "example_id": paths.example_id,
        "variant_id": paths.variant_id,
        "variant_output_dir": str(paths.variant_output_dir),
        "review_csv_path": str(paths.review_csv_path),
        "source_links_path": str(paths.source_links_path),
        "width_field": resolved_width_field,
        "len_over_width_threshold": float(len_over_width_threshold),
        "review_completion_status": (
            "complete" if len(reviewed) == len(review_decisions) else "incomplete"
        ),
        "reviewed_candidate_count": int(len(reviewed)),
        "candidate_count": int(len(review_decisions)),
        "apply_summary": apply_summary,
    }
    _atomic_write_json(summary, paths.review_summary_path)
    archive_result = finalize_width_review_archive(
        archive_dir=paths.review_output_dir,
        decisions_path=paths.review_csv_path,
        summary_path=paths.review_summary_path,
        source_links_path=paths.source_links_path,
        example_id=paths.example_id,
        variant_output_dir=paths.variant_output_dir,
        variant_manifest_path=paths.manifest_path,
    )
    fingerprint_result = upgrade_review_fingerprints(
        paths.review_output_dir,
        mask_path=paths.review_mask_path,
        links_path=paths.source_links_path,
        nodes_path=paths.source_nodes_path,
    )
    summary["archive"] = archive_result
    summary["review_status"] = fingerprint_result["status"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review candidate bad width estimates for one variant output directory, classify links as good/bad "
            "in a GUI, and automatically apply continuity-based width updates to the original RivGraph links file."
        )
    )
    parser.add_argument("variant_output_dir", help="Variant output directory containing summary/variant_manifest.json.")
    parser.add_argument("--width-field", default=DEFAULT_WIDTH_FIELD, help="Width field used for candidate filtering, continuity metrics, and local labels.")
    parser.add_argument(
        "--len-over-width-threshold",
        type=float,
        default=DEFAULT_LEN_OVER_WIDTH_THRESHOLD,
        help="Candidate filter threshold: only review links with n_units > 0 and len / width_field below this value.",
    )
    parser.add_argument("--source-links", default=None, help="Optional path to the source RivGraph links GeoPackage to update in place.")
    parser.add_argument(
        "--review-output-dir",
        default=None,
        help=(
            "Optional debugging/export directory. By default reviews are saved permanently under "
            "rivgraph_centerline/outputs/runs/<example>/width_review/."
        ),
    )
    parser.add_argument("--review-csv", default=None, help="Optional explicit review CSV output path.")
    parser.add_argument("--review-summary-json", default=None, help="Optional explicit apply-summary JSON output path.")
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help=(
            "Replace the existing *_width_review_old.gpkg source backup. By default an existing "
            "original backup is retained and reused across partial-review saves."
        ),
    )
    parser.add_argument(
        "--start-over",
        action="store_true",
        help=(
            "Start with all links undecided. Existing permanent review files are retained and are "
            "copied to width_review/versions/ when the new review is saved."
        ),
    )
    parser.add_argument(
        "--dt-points",
        action="store_true",
        help="Overlay the trimmed distance-transform sample points used for the focal link width/QC computations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = resolve_variant_review_paths(
        args.variant_output_dir,
        source_links_path=args.source_links,
        review_output_dir=args.review_output_dir,
        review_csv_path=args.review_csv,
        review_summary_path=args.review_summary_json,
    )
    summary = run_width_review_gui(
        paths,
        width_field=args.width_field,
        len_over_width_threshold=args.len_over_width_threshold,
        overwrite_backup=args.overwrite_backup,
        show_dt_points=args.dt_points,
        start_over=args.start_over,
    )
    if summary is None:
        print("Review closed without saving.")
        return
    print(f"Wrote review CSV to {paths.review_csv_path}")
    print(f"Wrote apply summary to {paths.review_summary_path}")
    print(f"Updated source links at {paths.source_links_path}")


if __name__ == "__main__":
    main()

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
from shapely.geometry import box, shape

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


RIVGRAPH_WIDTH_TRIM_MULTIPLIER = 1.1
DEFAULT_WIDTH_FIELD = "wid_adj_wet"
DEFAULT_LEN_OVER_WIDTH_THRESHOLD = 1.5
CORE_WIDTH_FIELDS = ("wid", "wid_adj", "wid_med")
DEFAULT_REVIEW_CSV_NAME = "width_review_decisions.csv"
DEFAULT_REVIEW_SUMMARY_NAME = "width_review_apply_summary.json"
QC_TRIMMED_FIELD = "qc_mean_opposition_cos_trimmed"
QC_TRIMMED_FALLBACK_FIELD = "qc_mean_opposition_cos_trimmed_from_samples"
DISPLAY_METRIC_FIELDS = (
    "len_over_width",
    "us_dev",
    "ds_dev",
    "node_dev_max",
    QC_TRIMMED_FALLBACK_FIELD,
    "qc_mean_opposition_cos",
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
    frame["us_dev"] = (frame["us_node_out_over_in"] - 1.0).abs()
    frame["ds_dev"] = (frame["ds_node_out_over_in"] - 1.0).abs()
    frame["node_dev_max"] = frame[["us_dev", "ds_dev"]].max(axis=1)

    frame["candidate_flag"] = frame["n_units"].gt(0) & frame["len_over_width"].lt(float(len_over_width_threshold))
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
    link_id = int(candidate_row["id_link"])
    us_node = int(candidate_row["id_us_node"])
    ds_node = int(candidate_row["id_ds_node"])
    current_width = float(candidate_row[width_field])

    us_in = float(frame.loc[frame["id_ds_node"] == us_node, width_field].sum(min_count=1))
    us_out_other = float(frame.loc[(frame["id_us_node"] == us_node) & (frame["id_link"] != link_id), width_field].sum(min_count=1))
    ds_in_other = float(frame.loc[(frame["id_ds_node"] == ds_node) & (frame["id_link"] != link_id), width_field].sum(min_count=1))
    ds_out = float(frame.loc[frame["id_us_node"] == ds_node, width_field].sum(min_count=1))

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
        raise FileExistsError(f"Backup already exists: {destination}")
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
        else (variant_output_dir / "width_review").resolve()
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

        if decision != "bad":
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
            current_value = pd.to_numeric(pd.Series([source_links.at[row_index, field]]), errors="coerce").iloc[0]
            if not np.isfinite(current_value):
                continue
            source_links.at[row_index, orig_field] = float(current_value)
            source_links.at[row_index, field] = float(current_value) * float(scale_factor)

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
        width_field: str,
    ) -> None:
        self.root = root
        self.paths = paths
        self.width_field = width_field
        self.root.title(f"Width review {paths.example_id} / {paths.variant_id}")
        self.collapsed_mask_path = paths.collapsed_mask_path

        self.metric_frame = metric_frame.copy()
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
        self.current_pos = 0 if not self.candidate_frame.empty else None
        self.saved = False

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
        self._draw()

    def _neighbor_table_text(self, local_links: pd.DataFrame, focal_link_id: int) -> str:
        rows = []
        for row in local_links.sort_values("id_link", kind="mergesort").itertuples(index=False):
            role = "FOCAL" if int(row.id_link) == focal_link_id else "nbr"
            wid = pd.to_numeric(pd.Series([getattr(row, "wid", np.nan)]), errors="coerce").iloc[0]
            wid_adj = pd.to_numeric(pd.Series([getattr(row, "wid_adj", np.nan)]), errors="coerce").iloc[0]
            wid_med = pd.to_numeric(pd.Series([getattr(row, "wid_med", np.nan)]), errors="coerce").iloc[0]
            len_over_width = pd.to_numeric(pd.Series([getattr(row, "len_over_width", np.nan)]), errors="coerce").iloc[0]
            us_dev = pd.to_numeric(pd.Series([getattr(row, "us_dev", np.nan)]), errors="coerce").iloc[0]
            ds_dev = pd.to_numeric(pd.Series([getattr(row, "ds_dev", np.nan)]), errors="coerce").iloc[0]
            node_dev_max = pd.to_numeric(pd.Series([getattr(row, "node_dev_max", np.nan)]), errors="coerce").iloc[0]
            qc_trimmed = pd.to_numeric(
                pd.Series([getattr(row, QC_TRIMMED_FALLBACK_FIELD, np.nan)]),
                errors="coerce",
            ).iloc[0]
            qc_mean = pd.to_numeric(pd.Series([getattr(row, "qc_mean_opposition_cos", np.nan)]), errors="coerce").iloc[0]
            rows.append(
                f"{role:5s} "
                f"{'link=':<7s}{int(row.id_link):<7d}  "
                f"{'dir=':<9s}{int(row.id_us_node):<3d}->{int(row.id_ds_node):<3d}  "
            )
            rows.append(
                "      "
                f"{'wid=':<7s}{wid:<7.1f}  " 
                f"{'wid_adj=':<9s}{wid_adj:<7.1f}  " 
                f"{'wid_med=':<7s}{wid_med:<7.1f}"

            )
            rows.append(
                "      "
                f"{'len/w=':<7s}{len_over_width:<7.3f}  "
                f"{'us_dev=':<9s}{us_dev:<7.3f}  "
                f"{'ds_dev=':<7s}{ds_dev:<7.3f}  "
                f"{'dev_max=':<7s}{node_dev_max:<7.3f}  "
            )
            rows.append(
                "      "
                f"{'qc_tr=':<7s}{qc_trimmed:<7.3f}  "
                f"{'qc=':<9s}{qc_mean:<7.3f}"
            )
            rows.append("")
        return "\n".join(rows)

    def _plot_direction_arrow(self, geometry: Any, *, color: str, linewidth: float, zorder: int) -> None:
        if geometry is None or geometry.is_empty:
            return
        try:
            midpoint = geometry.interpolate(0.5, normalized=True)
            half_step = 0.03
            start = geometry.interpolate(max(0.0, 0.3 - half_step), normalized=True)
            end = geometry.interpolate(min(1.0, 0.3 + half_step), normalized=True)
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

    def _local_island_geometries(self, local_links: pd.DataFrame) -> list[Any]:
        if local_links.empty:
            return []
        minx, miny, maxx, maxy = local_links.total_bounds
        span = max(maxx - minx, maxy - miny, 1.0)
        pad = span * 0.1
        bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        bbox = box(*bounds)

        with rasterio.open(self.collapsed_mask_path) as ds:
            window = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                return []
            data = ds.read(1, window=window, boundless=True, fill_value=0)
            transform = ds.window_transform(window)

        land_mask = (data == 0).astype(np.uint8)
        if not land_mask.any():
            return []

        islands: list[Any] = []
        eps = span * 1e-6 + 1e-9
        for geom_mapping, value in shapes(land_mask, mask=land_mask.astype(bool), transform=transform):
            if int(value) != 1:
                continue
            geom = shape(geom_mapping).intersection(bbox)
            if geom.is_empty:
                continue
            gx0, gy0, gx1, gy1 = geom.bounds
            touches_edge = (
                abs(gx0 - bounds[0]) <= eps or
                abs(gy0 - bounds[1]) <= eps or
                abs(gx1 - bounds[2]) <= eps or
                abs(gy1 - bounds[3]) <= eps
            )
            if touches_edge:
                continue
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
        island_geoms = self._local_island_geometries(local_links)

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
            focal_frame.plot(ax=self.ax_map, color="#d62728", linewidth=4, zorder=3)
        for feature in neighbor_frame.itertuples(index=False):
            self._plot_direction_arrow(feature.geometry, color="#3182bd", linewidth=1.5, zorder=2)
        # for feature in focal_frame.itertuples(index=False):
        #     self._plot_direction_arrow(feature.geometry, color="#7f0000", linewidth=2.2, zorder=4)

        for feature in local_links.itertuples(index=False):
            point = feature.geometry.representative_point()
            width_value = pd.to_numeric(pd.Series([getattr(feature, self.width_field, np.nan)]), errors="coerce").iloc[0]
            self.ax_map.text(
                point.x,
                point.y,
                f"{int(feature.id_link)}\n{width_value:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "0.3", "alpha": 0.8, "boxstyle": "round,pad=0.2"},
                zorder=4,
            )

        self.ax_map.set_aspect("equal", adjustable="box")
        self.ax_map.set_title("Local link neighborhood")

        metrics_lines = [
            f"link {link_id} | review {self.current_pos + 1}/{len(self.candidate_frame)} | decision={decision}",
            f"len_over_width={float(row['len_over_width']):.3f} | us_dev={float(row['us_dev']):.3f} | ds_dev={float(row['ds_dev']):.3f}",
            f"node_dev_max={float(row['node_dev_max']):.3f} | qc_trimmed={float(row[QC_TRIMMED_FIELD]):.3f} | qc_mean={float(row['qc_mean_opposition_cos']):.3f}",
            f"context width field={self.width_field} value={float(row[self.width_field]):.3f} | n_units={int(row['n_units'])}",
        ]
        self.info_var.set("\n".join(metrics_lines))

        self.ax_table.axis("off")
        self.ax_table.set_title("Local widths and metrics")
        self.ax_table.text(
            0.0,
            1.0,
            self._neighbor_table_text(local_links, link_id),
            fontsize=12,
            family="monospace",
            va="top",
            linespacing=1.5,
        )
        self.canvas.draw_idle()

    def _save_and_close(self) -> None:
        undecided = self.decisions["review_decision"].isna().sum()
        if undecided:
            if not messagebox.askyesno(
                "Save partial review",
                f"{undecided} candidate link(s) are still unreviewed. Save anyway and apply only the reviewed decisions?",
            ):
                return

        output = self.candidate_frame.merge(self.decisions, on="id_link", how="left", validate="1:1")
        self.paths.review_output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(self.paths.review_csv_path, index=False)
        self.saved = True
        self.root.destroy()


def load_review_context(
    paths: VariantReviewPaths,
    *,
    width_field: str,
    len_over_width_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, dict[int, int]]:
    families = pd.read_csv(paths.link_width_families_path)
    samples = pd.read_csv(paths.link_width_samples_path)
    _, sample_qc = compute_sample_qc_frame(samples)
    resolved_width_field = width_field if width_field in families.columns else "wid_adj"
    candidate_frame = build_candidate_frame(
        families,
        sample_qc,
        width_field=resolved_width_field,
        len_over_width_threshold=len_over_width_threshold,
    )
    directed_links = gpd.read_file(paths.directed_links_path)
    parent_link_mapping = build_parent_link_mapping(link_lineage_path=paths.link_lineage_path, families=families)
    return candidate_frame, sample_qc, directed_links, parent_link_mapping


def run_width_review_gui(
    paths: VariantReviewPaths,
    *,
    width_field: str,
    len_over_width_threshold: float,
    overwrite_backup: bool,
) -> dict[str, Any] | None:
    candidate_frame, _, directed_links, parent_link_mapping = load_review_context(
        paths,
        width_field=width_field,
        len_over_width_threshold=len_over_width_threshold,
    )
    resolved_width_field = width_field if width_field in candidate_frame.columns else "wid_adj"

    root = tk.Tk()
    gui = WidthReviewGui(
        root,
        paths=paths,
        candidate_frame=candidate_frame,
        metric_frame=candidate_frame,
        directed_links=directed_links,
        width_field=resolved_width_field,
    )
    root.mainloop()

    if not gui.saved:
        return None

    review_decisions = pd.read_csv(paths.review_csv_path)
    review_decisions["review_decision"] = review_decisions["review_decision"].astype("string")
    reviewed = review_decisions.loc[review_decisions["review_decision"].notna()].copy()
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
        "apply_summary": apply_summary,
    }
    paths.review_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.review_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
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
    parser.add_argument("--review-output-dir", default=None, help="Optional directory for review CSV and apply summary outputs.")
    parser.add_argument("--review-csv", default=None, help="Optional explicit review CSV output path.")
    parser.add_argument("--review-summary-json", default=None, help="Optional explicit apply-summary JSON output path.")
    parser.add_argument("--overwrite-backup", action="store_true", help="Overwrite an existing *_width_review_old.gpkg backup.")
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
    )
    if summary is None:
        print("Review closed without saving.")
        return
    print(f"Wrote review CSV to {paths.review_csv_path}")
    print(f"Wrote apply summary to {paths.review_summary_path}")
    print(f"Updated source links at {paths.source_links_path}")


if __name__ == "__main__":
    main()

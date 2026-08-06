"""Windowed raster operations used by panelized collapse states."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_fill_holes
from shapely.geometry import LineString, Point, box

from network_variants.width_two_bank import (
    DEFAULT_MIN_SEARCH_PIXELS,
    DEFAULT_SEARCH_RADIUS_MULTIPLIER,
    ORIGIN_WET_LABEL,
    _compute_family_stats,
    _decompose_two_bank_section,
    _distances_along_link,
    _fill_invalid_width_series,
    _find_local_controlling_pair,
    _local_tangent_normal,
    _trim_slice,
    _trim_valid_series,
    build_boundary_component_index_from_label_mask,
    build_origin_mask,
    build_state_bank_labels,
)


def _serialize_width_values(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.6f}" for value in np.asarray(values, dtype=float))


def compute_panelized_base_width_families(
    links: Any,
    source_samples: pd.DataFrame,
    *,
    bank_labels_path: str | Path,
    origin_mask_path: str | Path,
    bank_candidate_neighbors_max: int = 2048,
) -> tuple[Any, pd.DataFrame]:
    """Compute regular two-bank base widths from bounded link-local raster windows."""
    required = {"id_link", "sample_index", "width", "x", "y"}
    missing = sorted(required.difference(source_samples.columns))
    if missing:
        raise ValueError(f"Panelized base width samples are missing required columns: {missing}")

    source_samples = source_samples.copy()
    for column in ("id_link", "sample_index"):
        source_samples[column] = pd.to_numeric(source_samples[column], errors="raise").astype(int)
    for column in ("width", "x", "y"):
        source_samples[column] = pd.to_numeric(source_samples[column], errors="coerce")

    samples_by_link = {
        int(link_id): frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
        for link_id, frame in source_samples.groupby("id_link", sort=False)
    }
    sample_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    n_links = int(len(links))
    print(
        f"Computing panelized two-bank widths for {n_links} links "
        f"and {len(source_samples)} samples...",
        flush=True,
    )
    with rasterio.open(bank_labels_path) as bank_source, rasterio.open(origin_mask_path) as origin_source:
        if bank_source.shape != origin_source.shape or bank_source.transform != origin_source.transform:
            raise ValueError("Bank labels and origin mask must share shape and transform.")
        pixel_length = float(max(abs(bank_source.transform.a), abs(bank_source.transform.e)))
        raster_extent = Window(0, 0, bank_source.width, bank_source.height)

        for link_number, link in enumerate(links.itertuples(index=False), start=1):
            link_id = int(link.id_link)
            selected = samples_by_link.get(link_id)
            if selected is None or selected.empty:
                raise ValueError(f"Panelized base link {link_id} has no source width samples.")
            raw_widths = selected["width"].to_numpy(dtype=float)
            xs = selected["x"].to_numpy(dtype=float)
            ys = selected["y"].to_numpy(dtype=float)
            if not np.isfinite(raw_widths).any():
                raise ValueError(f"Panelized base link {link_id} has no finite source widths.")
            fallback_width = float(np.nanmedian(raw_widths[np.isfinite(raw_widths)]))
            raw_widths = np.where(np.isfinite(raw_widths) & (raw_widths > 0), raw_widths, fallback_width)

            search_buffer = max(
                float(np.max(raw_widths)) * DEFAULT_SEARCH_RADIUS_MULTIPLIER,
                pixel_length * DEFAULT_MIN_SEARCH_PIXELS,
            )
            search_geometry = link.geometry.buffer(search_buffer)
            window = (
                from_bounds(*search_geometry.bounds, transform=bank_source.transform)
                .round_offsets()
                .round_lengths()
                .intersection(raster_extent)
            )
            local_transform = bank_source.window_transform(window)
            bank_labels = bank_source.read(1, window=window).astype(np.int32)
            origin_mask = origin_source.read(1, window=window).astype(np.uint16)
            boundary_index = build_boundary_component_index_from_label_mask(bank_labels, local_transform)

            distances = _distances_along_link(xs, ys)
            centerline = LineString(np.column_stack([xs, ys])) if len(xs) >= 2 else None
            total_values: list[float] = []
            wet_values: list[float] = []
            dry_values: list[float] = []
            channel_values: list[int] = []
            invalid_values: list[bool] = []
            qc_dt_values: list[float] = []
            qc_opposition_values: list[float] = []
            qc_chord_values: list[float] = []

            for index, sample in selected.iterrows():
                center_xy = np.asarray([float(sample["x"]), float(sample["y"])], dtype=float)
                _, normal = _local_tangent_normal(xs, ys, int(index))
                raw_width = float(raw_widths[int(index)])
                centerline_without_center = None
                if centerline is not None:
                    centerline_without_center = centerline.difference(
                        Point(float(center_xy[0]), float(center_xy[1])).buffer(pixel_length * 0.51)
                    )
                pair = _find_local_controlling_pair(
                    center_xy,
                    normal,
                    boundary_index=boundary_index,
                    search_radius=max(
                        raw_width * DEFAULT_SEARCH_RADIUS_MULTIPLIER,
                        pixel_length * DEFAULT_MIN_SEARCH_PIXELS,
                    ),
                    candidate_neighbors_max=bank_candidate_neighbors_max,
                    centerline_without_center=centerline_without_center,
                )
                if pair is None:
                    total_values.append(float("nan"))
                    wet_values.append(float("nan"))
                    dry_values.append(float("nan"))
                    channel_values.append(0)
                    invalid_values.append(True)
                    qc_dt_values.append(float("nan"))
                    qc_opposition_values.append(float("nan"))
                    qc_chord_values.append(float("nan"))
                    sample_records.append(
                        {
                            "id_link": link_id,
                            "sample_index": int(sample["sample_index"]),
                            "x": float(center_xy[0]),
                            "y": float(center_xy[1]),
                            "width_total": float("nan"),
                            "width_wet": float("nan"),
                            "width_dry": float("nan"),
                            "wet_fraction": float("nan"),
                            "dry_fraction": float("nan"),
                            "n_wet_threads": 0,
                            "n_channels": 0,
                            "left_component_id": pd.NA,
                            "right_component_id": pd.NA,
                            "left_bank_x": float("nan"),
                            "left_bank_y": float("nan"),
                            "right_bank_x": float("nan"),
                            "right_bank_y": float("nan"),
                            "dt_width_global": raw_width,
                            "width_chord": float("nan"),
                            "qc_relative_dt_diff": float("nan"),
                            "qc_opposition_cos": float("nan"),
                            "qc_width_chord_diff": float("nan"),
                            "width_method": "two_bank_panelized_windowed",
                            "width_invalid": True,
                        }
                    )
                    continue

                left_point = np.asarray(pair["left_point"], dtype=float)
                right_point = np.asarray(pair["right_point"], dtype=float)
                width_total = float(pair["left_distance"] + pair["right_distance"])
                width_chord = float(np.linalg.norm(right_point - left_point))
                decomposition = _decompose_two_bank_section(
                    center_xy,
                    left_point,
                    right_point,
                    origin_mask=origin_mask,
                    transform=local_transform,
                    pixel_length=pixel_length,
                )
                width_wet = float(decomposition["width_wet"])
                width_dry = float(decomposition["width_dry"])
                n_channels = int(decomposition["n_channels"])
                left_vector, right_vector = left_point - center_xy, right_point - center_xy
                left_norm, right_norm = float(np.linalg.norm(left_vector)), float(np.linalg.norm(right_vector))
                opposition = (
                    float(np.dot(left_vector / left_norm, right_vector / right_norm))
                    if left_norm > 0 and right_norm > 0
                    else float("nan")
                )
                relative_dt_diff = abs(width_total - raw_width) / raw_width if raw_width > 0 else float("nan")
                chord_diff = abs(width_total - width_chord) / width_total if width_total > 0 else float("nan")
                total_values.append(width_total)
                wet_values.append(width_wet)
                dry_values.append(width_dry)
                channel_values.append(n_channels)
                invalid_values.append(False)
                qc_dt_values.append(relative_dt_diff)
                qc_opposition_values.append(opposition)
                qc_chord_values.append(chord_diff)
                sample_records.append(
                    {
                        "id_link": link_id,
                        "sample_index": int(sample["sample_index"]),
                        "x": float(center_xy[0]),
                        "y": float(center_xy[1]),
                        "width_total": width_total,
                        "width_wet": width_wet,
                        "width_dry": width_dry,
                        "wet_fraction": width_wet / width_total if width_total > 0 else float("nan"),
                        "dry_fraction": width_dry / width_total if width_total > 0 else float("nan"),
                        "n_wet_threads": n_channels,
                        "n_channels": n_channels,
                        "left_component_id": int(pair["left_component_id"]),
                        "right_component_id": int(pair["right_component_id"]),
                        "left_bank_x": float(left_point[0]),
                        "left_bank_y": float(left_point[1]),
                        "right_bank_x": float(right_point[0]),
                        "right_bank_y": float(right_point[1]),
                        "dt_width_global": raw_width,
                        "width_chord": width_chord,
                        "qc_relative_dt_diff": relative_dt_diff,
                        "qc_opposition_cos": opposition,
                        "qc_width_chord_diff": chord_diff,
                        "width_method": "two_bank_panelized_windowed",
                        "width_invalid": False,
                    }
                )

            total_array = _fill_invalid_width_series(np.asarray(total_values, dtype=float), raw_widths)
            wet_array = _fill_invalid_width_series(np.asarray(wet_values, dtype=float), total_array)
            dry_array = _fill_invalid_width_series(
                np.asarray(dry_values, dtype=float),
                np.maximum(total_array - wet_array, 0.0),
            )
            channels_array = _fill_invalid_width_series(
                np.asarray(channel_values, dtype=float),
                np.ones(len(selected), dtype=float),
            )
            trim = _trim_slice(total_array, distances)
            total_stats = _compute_family_stats(total_array, trim)
            wet_stats = _compute_family_stats(wet_array, trim)
            dry_stats = _compute_family_stats(dry_array, trim)
            trimmed_channels = channels_array[trim]
            if trimmed_channels.size == 0:
                trimmed_channels = channels_array
            qc_dt = np.asarray(qc_dt_values, dtype=float)
            qc_opposition = np.asarray(qc_opposition_values, dtype=float)
            qc_chord = np.asarray(qc_chord_values, dtype=float)
            qc_opposition_trimmed = _trim_valid_series(qc_opposition, trim)
            summary_records.append(
                {
                    "id_link": link_id,
                    "wid_pix_total": _serialize_width_values(total_array),
                    "wid_pix_wet": _serialize_width_values(wet_array),
                    "wid_pix_dry": _serialize_width_values(dry_array),
                    "wid_adj_total": total_stats["adj"],
                    "wid_med_total": total_stats["med"],
                    "wid_p05_total": total_stats["p05"],
                    "wid_p50_total": total_stats["p50"],
                    "wid_p95_total": total_stats["p95"],
                    "wid_adj_wet": wet_stats["adj"],
                    "wid_med_wet": wet_stats["med"],
                    "wid_p05_wet": wet_stats["p05"],
                    "wid_p50_wet": wet_stats["p50"],
                    "wid_p95_wet": wet_stats["p95"],
                    "wid_adj_dry": dry_stats["adj"],
                    "wid_med_dry": dry_stats["med"],
                    "wid_p05_dry": dry_stats["p05"],
                    "wid_p50_dry": dry_stats["p50"],
                    "wid_p95_dry": dry_stats["p95"],
                    "wet_fraction_adj": wet_stats["adj"] / total_stats["adj"] if total_stats["adj"] > 0 else float("nan"),
                    "dry_fraction_adj": dry_stats["adj"] / total_stats["adj"] if total_stats["adj"] > 0 else float("nan"),
                    "n_wet_threads_mean": float(np.mean(trimmed_channels)),
                    "n_wet_threads_max": int(np.max(trimmed_channels)),
                    "n_channels_mean": float(np.mean(trimmed_channels)),
                    "n_channels_max": int(np.max(trimmed_channels)),
                    "n_invalid_width_samples": int(np.count_nonzero(invalid_values)),
                    "qc_mean_relative_dt_diff": float(np.nanmean(qc_dt)) if np.isfinite(qc_dt).any() else float("nan"),
                    "qc_max_relative_dt_diff": float(np.nanmax(qc_dt)) if np.isfinite(qc_dt).any() else float("nan"),
                    "qc_mean_width_chord_diff": float(np.nanmean(qc_chord)) if np.isfinite(qc_chord).any() else float("nan"),
                    "qc_mean_opposition_cos_trimmed": (
                        float(np.mean(qc_opposition_trimmed)) if qc_opposition_trimmed.size else float("nan")
                    ),
                    "qc_min_opposition_cos": (
                        float(np.nanmin(qc_opposition)) if np.isfinite(qc_opposition).any() else float("nan")
                    ),
                    "width_method": "two_bank_panelized_windowed",
                }
            )
            if link_number % 25 == 0 or link_number == n_links:
                print(f"  panelized widths: {link_number}/{n_links} links", flush=True)

    enriched = links.merge(pd.DataFrame.from_records(summary_records), on="id_link", how="left", validate="1:1")
    return enriched, pd.DataFrame.from_records(sample_records)


def classify_panels_for_footprint(
    *,
    panel_plan_path: str | Path,
    footprint: object,
    halo_distance: float,
) -> dict[str, list[str]]:
    """Classify panel cores as affected, halo-only, or reusable for an edit."""
    import geopandas as gpd

    cores = gpd.read_file(panel_plan_path, layer="panel_cores")
    if footprint is None or footprint.is_empty:
        return {"affected": [], "halo_only": [], "reusable": sorted(cores["panel_id"].astype(str).tolist())}
    affected = cores.loc[cores.geometry.intersects(footprint), "panel_id"].astype(str).tolist()
    halo = footprint.buffer(float(halo_distance))
    halo_only = cores.loc[cores.geometry.intersects(halo) & ~cores["panel_id"].astype(str).isin(affected), "panel_id"].astype(str).tolist()
    reusable = sorted(set(cores["panel_id"].astype(str)).difference(affected).difference(halo_only))
    return {"affected": sorted(affected), "halo_only": sorted(halo_only), "reusable": reusable}


def write_state_bank_labels_windowed(
    *,
    boundary_component_labels_path: str | Path,
    origin_mask_path: str | Path,
    output_path: str | Path,
) -> list[int]:
    """Write bank labels block-by-block, never loading a run-scale raster."""
    removed_labels: set[int] = set()
    with rasterio.open(boundary_component_labels_path) as boundary, rasterio.open(origin_mask_path) as origin:
        if boundary.shape != origin.shape or boundary.transform != origin.transform:
            raise ValueError("Boundary labels and origin mask must share shape and transform.")
        profile = boundary.profile.copy()
        profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            for _, window in boundary.block_windows(1):
                labels = boundary.read(1, window=window).astype(np.int32)
                origins = origin.read(1, window=window).astype(np.uint16)
                bank_labels, removed = build_state_bank_labels(labels, origins)
                destination.write(bank_labels, 1, window=window)
                removed_labels.update(removed)
    return sorted(removed_labels)


def write_state_origin_mask_windowed(
    *,
    base_mask_path: str | Path,
    collapsed_mask_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write a bounded-memory origin mask for a panelized child state."""
    with rasterio.open(base_mask_path) as base, rasterio.open(collapsed_mask_path) as collapsed:
        if base.shape != collapsed.shape or base.transform != collapsed.transform:
            raise ValueError("Base and collapsed masks must share shape and transform.")
        profile = collapsed.profile.copy()
        profile.update(dtype="uint16", count=1, nodata=0, compress="lzw")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            for _, window in collapsed.block_windows(1):
                base_water = base.read(1, window=window) > 0
                child_water = collapsed.read(1, window=window) > 0
                origin = np.zeros(base_water.shape, dtype=np.uint16)
                origin[base_water] = ORIGIN_WET_LABEL
                origin[child_water & ~base_water] = 101
                destination.write(origin, 1, window=window)
    return output_path


def compute_panelized_child_width_patch(
    *,
    child_links: Any,
    child_source_samples: pd.DataFrame,
    child_merged_links: Any,
    parent_merged_links: Any,
    parent_width_links: Any,
    parent_width_samples: pd.DataFrame,
    affected_panel_ids: list[str],
    bank_labels_path: str | Path,
    origin_mask_path: str | Path,
    bank_candidate_neighbors_max: int = 2048,
) -> tuple[Any, pd.DataFrame, dict[str, object]]:
    """Recompute affected child links and inherit all exact unchanged parent widths."""
    affected_prefixes = tuple(f"{panel_id}:" for panel_id in affected_panel_ids)
    child_sources = {
        int(row.global_link_id): str(row.source_segments)
        for row in child_merged_links.itertuples(index=False)
    }
    parent_by_sources = {
        str(row.source_segments): int(row.global_link_id)
        for row in parent_merged_links.itertuples(index=False)
    }
    changed_child_ids: set[int] = set()
    reused_parent_by_child: dict[int, int] = {}
    for child_id, source_segments in child_sources.items():
        parts = tuple(part.strip() for part in source_segments.split(";") if part.strip())
        touches_affected = any(part.startswith(affected_prefixes) for part in parts)
        parent_id = parent_by_sources.get(source_segments)
        if touches_affected or parent_id is None:
            changed_child_ids.add(int(child_id))
        else:
            reused_parent_by_child[int(child_id)] = int(parent_id)

    child_ids = set(pd.to_numeric(child_links["id_link"], errors="raise").astype(int))
    classified_ids = changed_child_ids.union(reused_parent_by_child)
    if child_ids != classified_ids:
        raise ValueError(
            "Every child link must be classified exactly once for width patching; "
            f"missing={sorted(child_ids - classified_ids)}, extra={sorted(classified_ids - child_ids)}"
        )

    changed_links = child_links.loc[
        pd.to_numeric(child_links["id_link"], errors="raise").astype(int).isin(changed_child_ids)
    ].copy()
    changed_samples = child_source_samples.loc[
        pd.to_numeric(child_source_samples["id_link"], errors="raise").astype(int).isin(changed_child_ids)
    ].copy()
    if changed_child_ids:
        changed_enriched, changed_width_samples = compute_panelized_base_width_families(
            changed_links,
            changed_samples,
            bank_labels_path=bank_labels_path,
            origin_mask_path=origin_mask_path,
            bank_candidate_neighbors_max=bank_candidate_neighbors_max,
        )
    else:
        changed_enriched = changed_links.copy()
        changed_width_samples = pd.DataFrame()

    width_columns = [
        column
        for column in changed_enriched.columns
        if column not in child_links.columns and column != changed_enriched.geometry.name
    ]
    missing_parent_width_columns = sorted(set(width_columns).difference(parent_width_links.columns))
    if missing_parent_width_columns:
        raise ValueError(
            f"Parent width links are missing patch fields: {missing_parent_width_columns}"
        )
    changed_summary = pd.DataFrame(changed_enriched.drop(columns=changed_enriched.geometry.name))[
        ["id_link", *width_columns]
    ]
    parent_width_index = parent_width_links.set_index("id_link", drop=False)
    reused_records: list[dict[str, object]] = []
    for child_id, parent_id in sorted(reused_parent_by_child.items()):
        if parent_id not in parent_width_index.index:
            raise ValueError(f"Parent width link {parent_id} is unavailable for child link {child_id}.")
        parent_row = parent_width_index.loc[parent_id]
        record = {"id_link": int(child_id)}
        record.update({column: parent_row[column] for column in width_columns})
        reused_records.append(record)
    summary = pd.concat(
        [pd.DataFrame.from_records(reused_records, columns=["id_link", *width_columns]), changed_summary],
        ignore_index=True,
    )
    if summary["id_link"].duplicated().any() or set(summary["id_link"].astype(int)) != child_ids:
        raise ValueError("Width patch summary does not cover every child link exactly once.")
    enriched = child_links.merge(summary, on="id_link", how="left", validate="1:1")

    reused_samples: list[pd.DataFrame] = []
    parent_sample_ids = pd.to_numeric(parent_width_samples["id_link"], errors="raise").astype(int)
    for child_id, parent_id in sorted(reused_parent_by_child.items()):
        selected = parent_width_samples.loc[parent_sample_ids == int(parent_id)].copy()
        if selected.empty:
            raise ValueError(f"Parent width samples are unavailable for link {parent_id}.")
        selected["id_link"] = int(child_id)
        reused_samples.append(selected)
    sample_frames = reused_samples + ([changed_width_samples] if not changed_width_samples.empty else [])
    samples = pd.concat(sample_frames, ignore_index=True)
    samples["id_link"] = pd.to_numeric(samples["id_link"], errors="raise").astype(int)
    samples = samples.sort_values(["id_link", "sample_index"], kind="mergesort").reset_index(drop=True)
    if set(samples["id_link"].unique()) != child_ids:
        raise ValueError("Width patch samples do not cover every child link.")
    return enriched, samples, {
        "affected_panel_ids": sorted(affected_panel_ids),
        "n_child_links": int(len(child_ids)),
        "n_recomputed_links": int(len(changed_child_ids)),
        "n_reused_links": int(len(reused_parent_by_child)),
        "recomputed_link_ids": sorted(changed_child_ids),
    }


def compute_panelized_segmented_width_patch(
    *,
    segmented_links: Any,
    source_links: Any,
    source_width_links: Any,
    source_width_samples: pd.DataFrame,
    bank_labels_path: str | Path,
    origin_mask_path: str | Path,
    bank_candidate_neighbors_max: int = 2048,
) -> tuple[Any, pd.DataFrame, dict[str, object]]:
    """Patch widths after links are split without changing the underlying mask.

    Links retaining both their identifier and geometry inherit their complete
    source width products. New or shortened links receive the source link's raw
    distance-transform samples partitioned by nearest segmented geometry and
    are recomputed with bounded raster windows.
    """
    import geopandas as gpd

    required_sample_columns = {"id_link", "x", "y"}
    missing = sorted(required_sample_columns.difference(source_width_samples.columns))
    if missing:
        raise ValueError(f"Source width samples are missing required columns: {missing}")

    raw_width_field = next(
        (
            field
            for field in ("dt_width_global", "width_total", "width_wet")
            if field in source_width_samples.columns
        ),
        None,
    )
    if raw_width_field is None:
        raise ValueError("Source width samples do not contain a usable raw width field.")

    segmented = segmented_links.copy()
    source = source_links.copy()
    segmented["id_link"] = pd.to_numeric(segmented["id_link"], errors="raise").astype(int)
    source["id_link"] = pd.to_numeric(source["id_link"], errors="raise").astype(int)
    source_width_links = source_width_links.copy()
    source_width_links["id_link"] = pd.to_numeric(
        source_width_links["id_link"], errors="raise"
    ).astype(int)

    source_by_id = source.set_index("id_link", drop=False)
    reused_source_by_segmented: dict[int, int] = {}
    changed_ids: set[int] = set()
    for row in segmented.itertuples(index=False):
        link_id = int(row.id_link)
        if link_id in source_by_id.index and row.geometry.equals_exact(
            source_by_id.loc[link_id].geometry, tolerance=1e-8
        ):
            reused_source_by_segmented[link_id] = link_id
        else:
            changed_ids.add(link_id)

    with rasterio.open(origin_mask_path) as origin:
        pixel_length = float(max(abs(origin.transform.a), abs(origin.transform.e)))
    coverage_tolerance = max(pixel_length * 0.75, 1e-8)

    changed = segmented.loc[segmented["id_link"].isin(changed_ids)].copy()
    changed_to_source: dict[int, int] = {}
    for row in changed.itertuples(index=False):
        best_source_id: int | None = None
        best_score: tuple[float, float] | None = None
        for source_row in source.itertuples(index=False):
            overlap_length = float(
                row.geometry.intersection(
                    source_row.geometry.buffer(coverage_tolerance)
                ).length
            )
            distance = float(row.geometry.distance(source_row.geometry))
            score = (overlap_length, -distance)
            if best_score is None or score > best_score:
                best_score = score
                best_source_id = int(source_row.id_link)
        if best_source_id is None or best_score is None or best_score[0] <= 0:
            raise ValueError(f"Could not identify the source link for segmented link {row.id_link}.")
        changed_to_source[int(row.id_link)] = best_source_id

    source_sample_ids = pd.to_numeric(
        source_width_samples["id_link"], errors="raise"
    ).astype(int)
    raw_records: list[dict[str, object]] = []
    changed_by_id = changed.set_index("id_link", drop=False)
    for source_id in sorted(set(changed_to_source.values())):
        child_ids = sorted(
            child_id
            for child_id, mapped_source_id in changed_to_source.items()
            if mapped_source_id == source_id
        )
        samples = source_width_samples.loc[source_sample_ids == source_id].copy()
        if samples.empty:
            raise ValueError(f"Source width samples are unavailable for link {source_id}.")

        child_geometries = {
            child_id: changed_by_id.loc[child_id].geometry for child_id in child_ids
        }
        assigned: dict[int, list[pd.Series]] = {child_id: [] for child_id in child_ids}
        for _, sample in samples.iterrows():
            point = Point(float(sample["x"]), float(sample["y"]))
            child_id = min(
                child_ids,
                key=lambda candidate: (
                    float(point.distance(child_geometries[candidate])),
                    candidate,
                ),
            )
            assigned[child_id].append(sample)

        finite_source_widths = pd.to_numeric(samples[raw_width_field], errors="coerce")
        finite_source_widths = finite_source_widths.loc[
            np.isfinite(finite_source_widths) & (finite_source_widths > 0)
        ]
        if finite_source_widths.empty:
            raise ValueError(f"Source link {source_id} has no finite positive raw widths.")
        fallback_width = float(finite_source_widths.median())

        for child_id in child_ids:
            geometry = child_geometries[child_id]
            child_samples = assigned[child_id]
            if len(child_samples) < 2:
                source_points = [
                    (
                        Point(float(sample["x"]), float(sample["y"])),
                        float(
                            pd.to_numeric(
                                pd.Series([sample[raw_width_field]]), errors="coerce"
                            ).iloc[0]
                        ),
                    )
                    for _, sample in samples.iterrows()
                ]
                target_points = [
                    geometry.interpolate(0.0),
                    geometry.interpolate(float(geometry.length)),
                ]
                child_samples = []
                for target in target_points:
                    nearest_point, nearest_width = min(
                        source_points,
                        key=lambda item: float(target.distance(item[0])),
                    )
                    child_samples.append(
                        pd.Series(
                            {
                                "x": float(target.x),
                                "y": float(target.y),
                                raw_width_field: (
                                    nearest_width
                                    if np.isfinite(nearest_width) and nearest_width > 0
                                    else fallback_width
                                ),
                            }
                        )
                    )

            ordered = sorted(
                child_samples,
                key=lambda sample: float(
                    geometry.project(Point(float(sample["x"]), float(sample["y"])))
                ),
            )
            for sample_index, sample in enumerate(ordered):
                raw_width = float(
                    pd.to_numeric(
                        pd.Series([sample[raw_width_field]]), errors="coerce"
                    ).iloc[0]
                )
                raw_records.append(
                    {
                        "id_link": child_id,
                        "sample_index": sample_index,
                        "width": (
                            raw_width
                            if np.isfinite(raw_width) and raw_width > 0
                            else fallback_width
                        ),
                        "x": float(sample["x"]),
                        "y": float(sample["y"]),
                    }
                )

    changed_raw_samples = pd.DataFrame.from_records(
        raw_records,
        columns=["id_link", "sample_index", "width", "x", "y"],
    )
    if changed_ids:
        changed_enriched, changed_width_samples = compute_panelized_base_width_families(
            changed,
            changed_raw_samples,
            bank_labels_path=bank_labels_path,
            origin_mask_path=origin_mask_path,
            bank_candidate_neighbors_max=bank_candidate_neighbors_max,
        )
    else:
        changed_enriched = changed.copy()
        changed_width_samples = pd.DataFrame()

    width_columns = [
        column
        for column in changed_enriched.columns
        if column not in segmented.columns and column != changed_enriched.geometry.name
    ]
    missing_width_columns = sorted(set(width_columns).difference(source_width_links.columns))
    if missing_width_columns:
        raise ValueError(f"Source width links are missing patch fields: {missing_width_columns}")

    summary_frames: list[pd.DataFrame] = []
    if changed_ids:
        summary_frames.append(
            pd.DataFrame(changed_enriched.drop(columns=changed_enriched.geometry.name))[
                ["id_link", *width_columns]
            ]
        )
    source_width_index = source_width_links.set_index("id_link", drop=False)
    reused_records: list[dict[str, object]] = []
    for child_id, source_id in sorted(reused_source_by_segmented.items()):
        if source_id not in source_width_index.index:
            raise ValueError(f"Source width link {source_id} is unavailable.")
        source_row = source_width_index.loc[source_id]
        reused_records.append(
            {
                "id_link": child_id,
                **{column: source_row[column] for column in width_columns},
            }
        )
    if reused_records:
        summary_frames.append(
            pd.DataFrame.from_records(
                reused_records, columns=["id_link", *width_columns]
            )
        )
    summary = pd.concat(summary_frames, ignore_index=True)
    segmented_ids = set(segmented["id_link"].astype(int))
    if summary["id_link"].duplicated().any() or set(summary["id_link"].astype(int)) != segmented_ids:
        raise ValueError("Segmented width patch does not cover every link exactly once.")
    enriched = segmented.merge(summary, on="id_link", how="left", validate="1:1")
    if not isinstance(enriched, gpd.GeoDataFrame):
        enriched = gpd.GeoDataFrame(enriched, geometry=segmented.geometry.name, crs=segmented.crs)

    reused_samples: list[pd.DataFrame] = []
    for child_id, source_id in sorted(reused_source_by_segmented.items()):
        selected = source_width_samples.loc[source_sample_ids == source_id].copy()
        if selected.empty:
            raise ValueError(f"Source width samples are unavailable for link {source_id}.")
        selected["id_link"] = child_id
        reused_samples.append(selected)
    sample_frames = reused_samples + (
        [changed_width_samples] if not changed_width_samples.empty else []
    )
    patched_samples = pd.concat(sample_frames, ignore_index=True)
    patched_samples["id_link"] = pd.to_numeric(
        patched_samples["id_link"], errors="raise"
    ).astype(int)
    patched_samples = patched_samples.sort_values(
        ["id_link", "sample_index"], kind="mergesort"
    ).reset_index(drop=True)
    if set(patched_samples["id_link"].unique()) != segmented_ids:
        raise ValueError("Segmented width samples do not cover every link.")

    return enriched, patched_samples, {
        "n_source_links": int(len(source)),
        "n_segmented_links": int(len(segmented)),
        "n_recomputed_links": int(len(changed_ids)),
        "n_reused_links": int(len(reused_source_by_segmented)),
        "recomputed_link_ids": sorted(changed_ids),
        "source_link_by_recomputed_link": {
            str(child_id): int(source_id)
            for child_id, source_id in sorted(changed_to_source.items())
        },
    }


def write_panel_origin_and_bank_labels(
    *,
    base_mask_path: str | Path,
    collapsed_mask_path: str | Path,
    boundary_component_labels_path: str | Path,
    origin_output_path: str | Path,
    bank_output_path: str | Path,
) -> list[int]:
    """Create local state labels for one panel from a global label raster window."""
    with rasterio.open(base_mask_path) as base, rasterio.open(collapsed_mask_path) as collapsed:
        if base.shape != collapsed.shape or base.transform != collapsed.transform:
            raise ValueError("Base and collapsed panel masks must share shape and transform.")
        origin = build_origin_mask(collapsed.read(1) > 0, base.read(1) > 0)
        profile = collapsed.profile.copy()
        profile.update(dtype="uint16", count=1, compress="lzw")
        Path(origin_output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(origin_output_path, "w", **profile) as dst:
            dst.write(origin, 1)
        bounds = collapsed.bounds
    with rasterio.open(boundary_component_labels_path) as labels:
        window = from_bounds(*bounds, transform=labels.transform).round_offsets().round_lengths()
        boundary = labels.read(1, window=window, out_shape=origin.shape).astype(np.int32)
    bank, removed = build_state_bank_labels(boundary, origin)
    profile.update(dtype="int32", count=1, compress="lzw")
    with rasterio.open(bank_output_path, "w", **profile) as dst:
        dst.write(bank, 1)
    return removed


def apply_panel_collapse_edit(
    *,
    source_mask_path: str | Path,
    output_mask_path: str | Path,
    footprint: object,
    all_touched: bool = True,
) -> dict[str, int]:
    """Fill enclosed dry holes in one already-overlapped panel mask."""
    with rasterio.open(source_mask_path) as source:
        water = source.read(1) > 0
        profile = source.profile.copy()
        footprint_mask = rasterize(
            [(footprint, 1)],
            out_shape=source.shape,
            transform=source.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)
    component_water = water & footprint_mask
    filled = binary_fill_holes(component_water)
    added = filled & ~component_water
    collapsed = water | added
    profile.update(dtype="uint8", count=1, compress="lzw")
    Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_mask_path, "w", **profile) as destination:
        destination.write(collapsed.astype("uint8"), 1)
    return {"base_water_pixels": int(water.sum()), "added_pixels": int(added.sum()), "collapsed_water_pixels": int(collapsed.sum())}


def materialize_affected_panel_masks(
    *,
    panel_root: str | Path,
    output_root: str | Path,
    panel_ids: list[str],
    footprint: object,
    all_touched: bool = True,
) -> dict[str, dict[str, int]]:
    """Write edited masks for affected panels; callers reuse all other panels."""
    panel_root, output_root = Path(panel_root), Path(output_root)
    results: dict[str, dict[str, int]] = {}
    for panel_id in panel_ids:
        source = panel_root / panel_id / "mask.tif"
        if not source.exists():
            raise FileNotFoundError(f"Panel mask does not exist: {source}")
        results[panel_id] = apply_panel_collapse_edit(
            source_mask_path=source,
            output_mask_path=output_root / panel_id / "mask.tif",
            footprint=footprint,
            all_touched=all_touched,
        )
    return results


def prepare_panel_state_workspace(
    *,
    source_panel_root: str | Path,
    state_panel_root: str | Path,
    affected_panel_ids: list[str],
) -> dict[str, str]:
    """Create a state panel root, symlinking reusable panels to their parent outputs."""
    source_panel_root, state_panel_root = Path(source_panel_root), Path(state_panel_root)
    affected = set(affected_panel_ids)
    result: dict[str, str] = {}
    for source in sorted(path for path in source_panel_root.iterdir() if path.is_dir() and path.name.startswith("P")):
        destination = state_panel_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.name in affected:
            destination.mkdir(exist_ok=True)
            result[source.name] = "affected"
        else:
            os.symlink(source.resolve(), destination, target_is_directory=True)
            result[source.name] = "reused"
    return result


def regenerate_affected_panels(
    *,
    state_panel_root: str | Path,
    affected_panel_ids: list[str],
    run_name: str,
    exit_sides_by_panel: dict[str, str],
    panel_runner: Callable[..., Any],
) -> None:
    """Run RivGraph only for edited panels in a prepared state workspace."""
    state_panel_root = Path(state_panel_root)
    for panel_id in affected_panel_ids:
        panel_dir = state_panel_root / panel_id
        panel_runner(
            name=f"{run_name}_{panel_id}",
            run_dir=panel_dir,
            mask=panel_dir / "mask.tif",
            mask_stage="explicit",
            exit_sides=exit_sides_by_panel[panel_id],
            assign_flow=True,
        )


def merge_panelized_state(
    *,
    panel_plan_path: str | Path,
    panel_breaks_path: str | Path,
    boundary_match_plan_path: str | Path,
    state_panel_root: str | Path,
    run_name: str,
    output_root: str | Path,
    parent_directed_links_path: str | Path | None = None,
) -> dict[str, str]:
    """Merge a mixed reused/regenerated panel workspace into a directed state."""
    from rivgraph_centerline.panelization import (
        assign_core_network_direction,
        build_core_boundary_network,
        publish_panelized_rivgraph_handoff,
        rebuild_core_network_nodes,
    )

    panel_plan_path = Path(panel_plan_path)
    state_panel_root = Path(state_panel_root)
    output_root = Path(output_root)
    core_dir = output_root / "core_boundary"
    build_core_boundary_network(
        plan_path=panel_plan_path,
        breaks_path=Path(panel_breaks_path),
        panel_root=state_panel_root,
        run_name=run_name,
        output_dir=core_dir,
        fallback_directed_links_path=(
            Path(parent_directed_links_path) if parent_directed_links_path is not None else None
        ),
        overwrite=True,
        boundary_match_plan_path=Path(boundary_match_plan_path),
    )
    rebuild_core_network_nodes(
        merged_links_path=core_dir / "merged_core_links.gpkg",
        output_dir=core_dir,
        overwrite=True,
    )
    assign_core_network_direction(
        merged_links_path=core_dir / "merged_links.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        panel_root=state_panel_root,
        run_name=run_name,
        output_dir=core_dir,
        overwrite=True,
    )
    rivgraph_dir = output_root / "rivgraph"
    publish_panelized_rivgraph_handoff(
        directed_links_path=core_dir / "merged_links_directed.gpkg",
        nodes_path=core_dir / "merged_nodes.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        plan_path=panel_plan_path,
        panel_root=state_panel_root,
        run_name=run_name,
        rivgraph_dir=rivgraph_dir,
        overwrite=True,
    )
    return {
        "directed_links": str((rivgraph_dir / f"{run_name}_links.gpkg").resolve()),
        "directed_nodes": str((rivgraph_dir / f"{run_name}_nodes.gpkg").resolve()),
        "width_samples": str((rivgraph_dir / f"{run_name}_link_width_samples.csv").resolve()),
        "core_dir": str(core_dir.resolve()),
    }


def compute_panel_two_bank_widths(
    *,
    base_panel_root: str | Path,
    state_panel_root: str | Path,
    panel_ids: list[str],
    run_name: str,
    boundary_component_labels_path: str | Path,
    output_root: str | Path,
    width_runner: Callable[..., Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Compute standard two-bank width artifacts independently for each panel."""
    if width_runner is None:
        from network_variants.variant_generation import compute_width_families
        width_runner = compute_width_families
    base_panel_root, state_panel_root, output_root = Path(base_panel_root), Path(state_panel_root), Path(output_root)
    outputs: dict[str, dict[str, str]] = {}
    for panel_id in panel_ids:
        base_mask = base_panel_root / panel_id / "mask.tif"
        state_dir = state_panel_root / panel_id
        collapsed_mask = state_dir / "mask.tif"
        links_path = state_dir / "rivgraph" / f"{run_name}_{panel_id}_links.gpkg"
        width_dir = output_root / panel_id / "widths"
        width_dir.mkdir(parents=True, exist_ok=True)
        origin_path, bank_path = width_dir / "origin_mask.tif", width_dir / "bank_labels.tif"
        write_panel_origin_and_bank_labels(
            base_mask_path=base_mask,
            collapsed_mask_path=collapsed_mask,
            boundary_component_labels_path=boundary_component_labels_path,
            origin_output_path=origin_path,
            bank_output_path=bank_path,
        )
        import geopandas as gpd
        links = gpd.read_file(links_path)
        enriched, samples = width_runner(
            links,
            collapsed_mask_path=collapsed_mask,
            wet_reference_mask_path=base_mask,
            origin_mask_path=origin_path,
            bank_labels_path=bank_path,
        )
        enriched["panel_id"] = panel_id
        samples["panel_id"] = panel_id
        enriched.to_file(width_dir / "links_with_width_families.gpkg", driver="GPKG")
        enriched.drop(columns=enriched.geometry.name).to_csv(width_dir / "link_width_families.csv", index=False)
        samples.to_csv(width_dir / "link_width_samples.csv", index=False)
        outputs[panel_id] = {"links": str((width_dir / "links_with_width_families.gpkg").resolve()), "samples": str((width_dir / "link_width_samples.csv").resolve())}
    return outputs


def merge_panel_two_bank_widths(
    *,
    merged_links_path: str | Path,
    core_segments_path: str | Path,
    panel_width_root: str | Path,
    output_dir: str | Path,
    ownership_tolerance: float = 1.0,
) -> dict[str, str]:
    """Merge core-owned panel two-bank samples into global link summaries."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    from network_variants.width_two_bank import _compute_family_stats, _distances_along_link, _trim_slice

    links = gpd.read_file(merged_links_path, layer="merged_links")
    segments = gpd.read_file(core_segments_path, layer="core_segments").set_index("segment_id")
    samples_by_panel: dict[str, pd.DataFrame] = {}
    for panel in segments["panel_id"].astype(str).unique():
        path = Path(panel_width_root) / panel / "widths" / "link_width_samples.csv"
        if not path.exists():
            raise FileNotFoundError(f"Panel width samples do not exist: {path}")
        samples_by_panel[panel] = pd.read_csv(path)
    merged_samples: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for row in links.itertuples(index=False):
        parts: list[pd.DataFrame] = []
        for segment_id in str(row.source_segments).split(";"):
            segment = segments.loc[segment_id]
            panel = str(segment.panel_id)
            local = samples_by_panel[panel]
            selected = local.loc[local["id_link"].astype(int) == int(segment.source_link_id)].copy()
            if selected.empty:
                continue
            selected = selected.loc[selected.apply(lambda value: Point(value.x, value.y).distance(segment.geometry) <= ownership_tolerance, axis=1)].copy()
            if selected.empty:
                continue
            selected["source_segment_id"] = segment_id
            selected["global_link_id"] = int(row.global_link_id)
            selected["_projection"] = selected.apply(lambda value: row.geometry.project(Point(value.x, value.y)), axis=1)
            parts.append(selected)
        if not parts:
            raise ValueError(f"Merged link {row.global_link_id} has no core-owned two-bank samples.")
        combined = pd.concat(parts, ignore_index=True).sort_values("_projection", kind="mergesort").reset_index(drop=True)
        combined["sample_index"] = range(len(combined))
        xs, ys = combined["x"].to_numpy(float), combined["y"].to_numpy(float)
        distances = _distances_along_link(xs, ys)
        trim = _trim_slice(combined["width_total"].to_numpy(float), distances)
        summary: dict[str, object] = {"global_link_id": int(row.global_link_id)}
        for family in ("total", "wet", "dry"):
            stats = _compute_family_stats(combined[f"width_{family}"].to_numpy(float), trim)
            summary.update({f"wid_{key}_{family}": value for key, value in stats.items()})
        summary["n_width_samples"] = int(len(combined))
        summaries.append(summary)
        merged_samples.append(combined.drop(columns="_projection"))
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    enriched = links.merge(summary_frame, on="global_link_id", how="left", validate="1:1")
    samples = pd.concat(merged_samples, ignore_index=True)
    enriched.to_file(output_dir / "links_with_width_families.gpkg", driver="GPKG")
    enriched.drop(columns=enriched.geometry.name).to_csv(output_dir / "link_width_families.csv", index=False)
    samples.to_csv(output_dir / "link_width_samples.csv", index=False)
    return {"links": str((output_dir / "links_with_width_families.gpkg").resolve()), "samples": str((output_dir / "link_width_samples.csv").resolve())}


def materialize_panelized_child_state(
    *,
    parent_mask_path: str | Path,
    source_panel_root: str | Path,
    state_root: str | Path,
    panel_plan_path: str | Path,
    panel_breaks_path: str | Path,
    boundary_match_plan_path: str | Path,
    boundary_component_labels_path: str | Path,
    footprint: object,
    halo_distance: float,
    run_name: str,
    exit_sides_by_panel: dict[str, str],
    panel_runner: Callable[..., Any],
    parent_directed_links_path: str | Path | None = None,
    materialize_full_mask: bool = True,
    all_touched: bool = True,
) -> dict[str, object]:
    """Materialize one child state from a footprint using only affected panels."""
    source_panel_root, state_root = Path(source_panel_root), Path(state_root)
    plan = classify_panels_for_footprint(panel_plan_path=panel_plan_path, footprint=footprint, halo_distance=halo_distance)
    state_panels = state_root / "panels"
    workspace = prepare_panel_state_workspace(source_panel_root=source_panel_root, state_panel_root=state_panels, affected_panel_ids=plan["affected"])
    edits = materialize_affected_panel_masks(
        panel_root=source_panel_root,
        output_root=state_panels,
        panel_ids=plan["affected"],
        footprint=footprint,
        all_touched=all_touched,
    )
    regenerate_affected_panels(state_panel_root=state_panels, affected_panel_ids=plan["affected"], run_name=run_name, exit_sides_by_panel=exit_sides_by_panel, panel_runner=panel_runner)
    merged = merge_panelized_state(
        panel_plan_path=panel_plan_path,
        panel_breaks_path=panel_breaks_path,
        boundary_match_plan_path=boundary_match_plan_path,
        state_panel_root=state_panels,
        run_name=run_name,
        output_root=state_root / "merged",
        parent_directed_links_path=parent_directed_links_path,
    )
    if not materialize_full_mask:
        raise NotImplementedError(
            "Virtual panelized states are not yet accepted by the shared child-state contract."
        )
    collapsed_mask = write_panelized_state_mosaic(parent_mask_path=parent_mask_path, panel_plan_path=panel_plan_path, state_panel_root=state_panels, affected_panel_ids=plan["affected"], output_mask_path=state_root / "mask" / "collapsed_mask.tif")
    return {
        "panel_plan": plan,
        "workspace": workspace,
        "edits": edits,
        "merged": merged,
        "collapsed_mask": str(collapsed_mask.resolve()),
    }


def write_panelized_state_mosaic(
    *,
    parent_mask_path: str | Path,
    panel_plan_path: str | Path,
    state_panel_root: str | Path,
    affected_panel_ids: list[str],
    output_mask_path: str | Path,
) -> Path:
    """Stream a full state mask while replacing only affected panel core pixels."""
    import geopandas as gpd

    parent_mask_path, state_panel_root, output_mask_path = Path(parent_mask_path), Path(state_panel_root), Path(output_mask_path)
    cores = gpd.read_file(panel_plan_path, layer="panel_cores").set_index("panel_id")
    with rasterio.open(parent_mask_path) as parent:
        profile = parent.profile.copy(); profile.update(dtype="uint8", count=1, compress="lzw")
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_mask_path, "w+", **profile) as output:
            for _, window in parent.block_windows(1):
                output.write(parent.read(1, window=window), 1, window=window)
            for panel_id in affected_panel_ids:
                core = cores.loc[panel_id].geometry
                with rasterio.open(state_panel_root / panel_id / "mask.tif") as panel:
                    patch_geometry = core.intersection(box(*panel.bounds))
                    if patch_geometry.is_empty:
                        continue
                    window = (
                        from_bounds(*patch_geometry.bounds, transform=parent.transform)
                        .round_offsets()
                        .round_lengths()
                    )
                    bounds = rasterio.windows.bounds(window, parent.transform)
                    panel_window = (
                        from_bounds(*bounds, transform=panel.transform)
                        .round_offsets()
                        .round_lengths()
                    )
                    values = panel.read(1, window=panel_window)
                expected_shape = (int(window.height), int(window.width))
                if values.shape != expected_shape:
                    raise ValueError(
                        f"Panel {panel_id} patch window has shape {values.shape}; "
                        f"expected {expected_shape} in the parent grid."
                    )
                core_mask = rasterize(
                    [(patch_geometry, 1)],
                    out_shape=expected_shape,
                    transform=rasterio.windows.transform(window, parent.transform),
                    dtype="uint8",
                ).astype(bool)
                existing = output.read(1, window=window)
                existing[core_mask] = values[core_mask]
                output.write(existing, 1, window=window)
    return output_mask_path

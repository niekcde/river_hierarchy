from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.transform import xy as raster_xy
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from scipy.spatial import cKDTree

from hierarchy_level_definition.metrics.unit_metrics import edge_width_samples_from_attrs

DEFAULT_WIDTH_QUANTILES = (0.05, 0.5, 0.95)
RIVGRAPH_WIDTH_TRIM_MULTIPLIER = 1.1
DEFAULT_SEARCH_RADIUS_MULTIPLIER = 4.0
DEFAULT_MIN_SEARCH_PIXELS = 6.0
DEFAULT_MAX_SEARCH_MULTIPLIER = 16.0
DEFAULT_CANDIDATE_NEIGHBORS = 96
DEFAULT_LINE_SAMPLE_SPACING_FACTOR = 0.5
ORIGIN_WET_LABEL = 1
ORIGIN_FILL_LABEL_START = 101


@dataclass
class BoundaryComponentIndex:
    tree: cKDTree
    xs: np.ndarray
    ys: np.ndarray
    component_ids: np.ndarray
    n_components: int


def _serialize_float_array(values: Sequence[float]) -> str:
    if not values:
        return ""
    return ",".join(f"{float(value):.6f}" for value in values)


def _serialize_int_list(values: Sequence[int]) -> str:
    if not values:
        return ""
    return ",".join(str(int(value)) for value in values)


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    text = str(value).strip()
    if not text or text.lower() in {"<na>", "nan", "none"}:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _compute_family_stats(values: np.ndarray, trim_slice: slice) -> dict[str, float]:
    trimmed = values[trim_slice]
    if trimmed.size == 0:
        trimmed = values
    if trimmed.size == 0:
        return {
            "adj": float("nan"),
            "med": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "mean": float("nan"),
        }
    quantiles = np.quantile(trimmed, DEFAULT_WIDTH_QUANTILES)
    return {
        "adj": float(np.mean(trimmed)),
        "med": float(np.median(trimmed)),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "mean": float(np.mean(trimmed)),
    }


def _trim_slice(total_widths: np.ndarray, distances: np.ndarray) -> slice:
    if total_widths.size == 0 or distances.size == 0:
        return slice(0, 0)

    revdistances = np.insert(np.cumsum(np.flipud(np.diff(distances))), 0, 0.0)
    start_index = int(np.argmin(np.abs(distances - total_widths[0] / 2 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER)))
    end_index = int(
        len(distances)
        - np.argmin(np.abs(revdistances - total_widths[-1] / 2 * RIVGRAPH_WIDTH_TRIM_MULTIPLIER))
        - 1
    )
    if start_index >= end_index:
        return slice(0, len(total_widths))
    return slice(start_index, end_index)


def _unravel_link_pixels(
    flat_indices: Sequence[int],
    shape: tuple[int, int],
    transform: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.unravel_index(np.asarray(flat_indices, dtype=int), shape)
    xs, ys = raster_xy(transform, rows, cols, offset="center")
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
    )


def _distances_along_link(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(xs) <= 1:
        return np.asarray([0.0], dtype=float)
    step = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return np.insert(np.cumsum(step), 0, 0.0)


def _pixel_length_from_transform(transform: Any) -> float:
    return float(max(abs(float(transform.a)), abs(float(transform.e))))


def extract_obstacle_boundary_mask(water_mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    obstacle_mask = ~water_mask.astype(bool)
    water_touch = binary_dilation(water_mask.astype(bool), structure=structure)
    obstacle_boundary = obstacle_mask & water_touch
    return obstacle_boundary


def remove_open_end_boundary_pixels(boundary_mask: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    border_water = np.zeros_like(water_mask, dtype=bool)
    border_water[0, :] = water_mask[0, :]
    border_water[-1, :] = water_mask[-1, :]
    border_water[:, 0] = water_mask[:, 0]
    border_water[:, -1] = water_mask[:, -1]
    if not border_water.any():
        return boundary_mask
    structure = np.ones((3, 3), dtype=bool)
    open_zone = binary_dilation(border_water, structure=structure)
    return boundary_mask & (~open_zone)


def label_obstacle_boundary_components(water_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundary_mask = extract_obstacle_boundary_mask(water_mask)
    filtered_boundary = remove_open_end_boundary_pixels(boundary_mask, water_mask)
    boundary_labels, _ = label(filtered_boundary.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    return filtered_boundary, boundary_labels.astype(np.int32)


def build_boundary_component_index(boundary_labels: np.ndarray, transform: Any) -> BoundaryComponentIndex | None:
    rows, cols = np.nonzero(boundary_labels > 0)
    if len(rows) == 0:
        return None
    xs, ys = raster_xy(transform, rows, cols, offset="center")
    xs_array = np.asarray(xs, dtype=float)
    ys_array = np.asarray(ys, dtype=float)
    component_ids = boundary_labels[rows, cols].astype(int)
    tree = cKDTree(np.column_stack([xs_array, ys_array]))
    return BoundaryComponentIndex(
        tree=tree,
        xs=xs_array,
        ys=ys_array,
        component_ids=component_ids,
        n_components=int(np.unique(component_ids).size),
    )


def build_origin_mask(
    collapsed_mask: np.ndarray,
    wet_mask: np.ndarray,
) -> np.ndarray:
    origin_mask = np.where(wet_mask.astype(bool), ORIGIN_WET_LABEL, 0).astype(np.uint16)
    added_mask = collapsed_mask.astype(bool) & (~wet_mask.astype(bool))
    if not added_mask.any():
        return origin_mask
    added_labels, n_added = label(added_mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    for index in range(1, int(n_added) + 1):
        origin_mask[added_labels == index] = int(ORIGIN_FILL_LABEL_START + index - 1)
    return origin_mask


def _compute_state_distance_transform(
    collapsed_mask: np.ndarray,
    *,
    pixel_length: float,
    rivgraph_distance_path: str | Path | None = None,
) -> np.ndarray:
    if rivgraph_distance_path is not None and Path(rivgraph_distance_path).exists():
        with rasterio.open(rivgraph_distance_path) as src:
            dt = src.read(1).astype(float)
        # RivGraph Idist can be treated as pixel-space distance; convert to map units.
        return np.asarray(dt, dtype=float) * float(pixel_length)
    return np.asarray(distance_transform_edt(collapsed_mask.astype(bool)), dtype=float) * float(pixel_length)


def _local_tangent_normal(xs: np.ndarray, ys: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) == 1:
        tangent = np.asarray([1.0, 0.0], dtype=float)
    else:
        prev_index = max(index - 1, 0)
        next_index = min(index + 1, len(xs) - 1)
        dx = float(xs[next_index] - xs[prev_index])
        dy = float(ys[next_index] - ys[prev_index])
        if dx == 0.0 and dy == 0.0:
            tangent = np.asarray([1.0, 0.0], dtype=float)
        else:
            norm = math.hypot(dx, dy)
            tangent = np.asarray([dx / norm, dy / norm], dtype=float)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    return tangent, normal


def _query_candidate_components(
    center_xy: np.ndarray,
    *,
    boundary_index: BoundaryComponentIndex,
    radius: float,
    max_neighbors: int,
) -> dict[int, tuple[float, np.ndarray]]:
    if max_neighbors <= 0:
        return {}
    distances, indices = boundary_index.tree.query(
        center_xy,
        k=max_neighbors,
        distance_upper_bound=float(radius),
    )
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)
    candidates: dict[int, tuple[float, np.ndarray]] = {}
    for distance, raw_index in zip(distances, indices):
        if not np.isfinite(distance):
            continue
        point_index = int(raw_index)
        if point_index < 0 or point_index >= len(boundary_index.component_ids):
            continue
        component_id = int(boundary_index.component_ids[point_index])
        point = np.asarray([boundary_index.xs[point_index], boundary_index.ys[point_index]], dtype=float)
        current = candidates.get(component_id)
        if current is None or float(distance) < current[0]:
            candidates[component_id] = (float(distance), point)
    return candidates


def _select_local_controlling_pair(
    center_xy: np.ndarray,
    normal: np.ndarray,
    candidates: Mapping[int, tuple[float, np.ndarray]],
) -> dict[str, Any] | None:
    left_best: tuple[int, float, np.ndarray] | None = None
    right_best: tuple[int, float, np.ndarray] | None = None
    for component_id, (distance, point) in candidates.items():
        vector = point - center_xy
        signed = float(np.dot(vector, normal))
        if signed == 0.0:
            continue
        if signed < 0.0:
            if left_best is None or distance < left_best[1]:
                left_best = (int(component_id), float(distance), point)
        else:
            if right_best is None or distance < right_best[1]:
                right_best = (int(component_id), float(distance), point)
    if left_best is None or right_best is None:
        return None
    return {
        "left_component_id": int(left_best[0]),
        "left_distance": float(left_best[1]),
        "left_point": np.asarray(left_best[2], dtype=float),
        "right_component_id": int(right_best[0]),
        "right_distance": float(right_best[1]),
        "right_point": np.asarray(right_best[2], dtype=float),
    }


def _sample_segment_labels(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    origin_mask: np.ndarray,
    transform: Any,
    pixel_length: float,
) -> tuple[list[int], list[float]]:
    vector = end_xy - start_xy
    length = float(np.linalg.norm(vector))
    if length <= 0:
        return [], []
    step = max(float(pixel_length) * DEFAULT_LINE_SAMPLE_SPACING_FACTOR, float(pixel_length) * 0.25)
    n_steps = max(int(math.ceil(length / step)), 1)
    labels: list[int] = []
    lengths: list[float] = []
    for index in range(n_steps):
        t0 = float(index) / float(n_steps)
        t1 = float(index + 1) / float(n_steps)
        midpoint = start_xy + 0.5 * (t0 + t1) * vector
        row, col = rowcol(transform, float(midpoint[0]), float(midpoint[1]))
        if 0 <= int(row) < origin_mask.shape[0] and 0 <= int(col) < origin_mask.shape[1]:
            label_value = int(origin_mask[int(row), int(col)])
        else:
            label_value = 0
        labels.append(label_value)
        lengths.append(length / float(n_steps))
    return labels, lengths


def _decompose_two_bank_section(
    center_xy: np.ndarray,
    left_point: np.ndarray,
    right_point: np.ndarray,
    *,
    origin_mask: np.ndarray,
    transform: Any,
    pixel_length: float,
) -> dict[str, float | int]:
    left_labels, left_lengths = _sample_segment_labels(
        left_point,
        center_xy,
        origin_mask=origin_mask,
        transform=transform,
        pixel_length=pixel_length,
    )
    right_labels, right_lengths = _sample_segment_labels(
        center_xy,
        right_point,
        origin_mask=origin_mask,
        transform=transform,
        pixel_length=pixel_length,
    )
    labels = left_labels + right_labels
    lengths = left_lengths + right_lengths
    wet_width = 0.0
    dry_width = 0.0
    channel_runs = 0
    in_channel = False
    for label_value, segment_length in zip(labels, lengths):
        if int(label_value) == ORIGIN_WET_LABEL:
            wet_width += float(segment_length)
            if not in_channel:
                channel_runs += 1
                in_channel = True
        else:
            in_channel = False
            if int(label_value) > ORIGIN_WET_LABEL:
                dry_width += float(segment_length)
    return {
        "width_wet": float(wet_width),
        "width_dry": float(dry_width),
        "n_channels": int(channel_runs),
    }


def _fill_invalid_width_series(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    filled = pd.Series(values.astype(float)).replace([np.inf, -np.inf], np.nan)
    filled = filled.interpolate(limit_direction="both")
    if filled.isna().all():
        return np.asarray(fallback, dtype=float)
    fallback_series = pd.Series(np.asarray(fallback, dtype=float))
    filled = filled.fillna(fallback_series)
    return filled.to_numpy(dtype=float)


def compute_two_bank_width_families(
    regenerated_links: gpd.GeoDataFrame,
    *,
    collapsed_mask_path: str | Path,
    wet_reference_mask_path: str | Path,
    origin_mask_path: str | Path | None = None,
    rivgraph_distance_path: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    with rasterio.open(collapsed_mask_path) as collapsed_ds:
        collapsed_mask = collapsed_ds.read(1) > 0
        collapsed_transform = collapsed_ds.transform
        collapsed_shape = collapsed_ds.shape
    with rasterio.open(wet_reference_mask_path) as wet_ds:
        wet_mask = wet_ds.read(1) > 0
        wet_transform = wet_ds.transform
    if collapsed_shape != wet_mask.shape or collapsed_transform != wet_transform:
        raise ValueError("collapsed_mask_path and wet_reference_mask_path must share shape and transform.")
    if origin_mask_path is not None and Path(origin_mask_path).exists():
        with rasterio.open(origin_mask_path) as origin_ds:
            origin_mask = origin_ds.read(1).astype(np.uint16)
        if origin_mask.shape != collapsed_shape:
            raise ValueError("origin_mask_path must share shape with collapsed_mask_path.")
    else:
        origin_mask = build_origin_mask(collapsed_mask.astype(np.uint8), wet_mask.astype(np.uint8))

    pixel_length = _pixel_length_from_transform(collapsed_transform)
    dt_map = _compute_state_distance_transform(
        collapsed_mask,
        pixel_length=pixel_length,
        rivgraph_distance_path=rivgraph_distance_path,
    )
    _, boundary_labels = label_obstacle_boundary_components(collapsed_mask)
    boundary_index = build_boundary_component_index(boundary_labels, collapsed_transform)

    enriched = regenerated_links.copy()
    sample_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for row in regenerated_links.itertuples(index=False):
        attrs = row._asdict()
        link_id = int(attrs["id_link"])
        flat_indices = _parse_int_list(attrs.get("idx_link"))
        if not flat_indices:
            continue

        total_widths_seed = edge_width_samples_from_attrs({"wid_pix": attrs.get("wid_pix")}, pixel_width_fields=["wid_pix"])
        rows, cols, xs, ys = _unravel_link_pixels(flat_indices, collapsed_shape, collapsed_transform)
        distances = _distances_along_link(xs, ys)
        if total_widths_seed.size == 0:
            total_widths_seed = 2.0 * dt_map[rows, cols]

        width_total_values: list[float] = []
        width_wet_values: list[float] = []
        width_dry_values: list[float] = []
        n_channels_values: list[int] = []
        qc_relative_dt_diff: list[float] = []
        qc_opposition_cos: list[float] = []
        qc_width_chord_diff: list[float] = []
        invalid_flags: list[bool] = []

        for index in range(len(flat_indices)):
            center_xy = np.asarray([float(xs[index]), float(ys[index])], dtype=float)
            _, normal = _local_tangent_normal(xs, ys, index)
            dt_width = max(float(2.0 * dt_map[rows[index], cols[index]]), float(pixel_length))

            pair: dict[str, Any] | None = None
            search_radius = max(dt_width * DEFAULT_SEARCH_RADIUS_MULTIPLIER, pixel_length * DEFAULT_MIN_SEARCH_PIXELS)
            if boundary_index is not None:
                for factor in (1.0, 2.0, 4.0, DEFAULT_MAX_SEARCH_MULTIPLIER / DEFAULT_SEARCH_RADIUS_MULTIPLIER):
                    candidates = _query_candidate_components(
                        center_xy,
                        boundary_index=boundary_index,
                        radius=search_radius * float(factor),
                        max_neighbors=max(DEFAULT_CANDIDATE_NEIGHBORS, 16),
                    )
                    pair = _select_local_controlling_pair(center_xy, normal, candidates)
                    if pair is not None:
                        break

            if pair is None:
                width_total_values.append(float("nan"))
                width_wet_values.append(float("nan"))
                width_dry_values.append(float("nan"))
                n_channels_values.append(0)
                qc_relative_dt_diff.append(float("nan"))
                qc_opposition_cos.append(float("nan"))
                qc_width_chord_diff.append(float("nan"))
                invalid_flags.append(True)
                sample_records.append(
                    {
                        "id_link": link_id,
                        "sample_index": index,
                        "x": float(xs[index]),
                        "y": float(ys[index]),
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
                        "dt_width_global": float(dt_width),
                        "width_chord": float("nan"),
                        "qc_relative_dt_diff": float("nan"),
                        "qc_opposition_cos": float("nan"),
                        "qc_width_chord_diff": float("nan"),
                        "width_method": "two_bank_dt",
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
                transform=collapsed_transform,
                pixel_length=pixel_length,
            )
            width_wet = float(decomposition["width_wet"])
            width_dry = float(decomposition["width_dry"])
            n_channels = int(decomposition["n_channels"])

            left_vector = left_point - center_xy
            right_vector = right_point - center_xy
            left_norm = float(np.linalg.norm(left_vector))
            right_norm = float(np.linalg.norm(right_vector))
            if left_norm > 0 and right_norm > 0:
                opposition_cos = float(np.dot(left_vector / left_norm, right_vector / right_norm))
            else:
                opposition_cos = float("nan")
            relative_dt_diff = abs(width_total - dt_width) / dt_width if dt_width > 0 else float("nan")
            width_chord_diff = abs(width_total - width_chord) / width_total if width_total > 0 else float("nan")

            width_total_values.append(width_total)
            width_wet_values.append(width_wet)
            width_dry_values.append(width_dry)
            n_channels_values.append(n_channels)
            qc_relative_dt_diff.append(float(relative_dt_diff))
            qc_opposition_cos.append(float(opposition_cos))
            qc_width_chord_diff.append(float(width_chord_diff))
            invalid_flags.append(False)

            sample_records.append(
                {
                    "id_link": link_id,
                    "sample_index": index,
                    "x": float(xs[index]),
                    "y": float(ys[index]),
                    "width_total": width_total,
                    "width_wet": width_wet,
                    "width_dry": width_dry,
                    "wet_fraction": float(width_wet / width_total) if width_total > 0 else float("nan"),
                    "dry_fraction": float(width_dry / width_total) if width_total > 0 else float("nan"),
                    "n_wet_threads": int(n_channels),
                    "n_channels": int(n_channels),
                    "left_component_id": int(pair["left_component_id"]),
                    "right_component_id": int(pair["right_component_id"]),
                    "left_bank_x": float(left_point[0]),
                    "left_bank_y": float(left_point[1]),
                    "right_bank_x": float(right_point[0]),
                    "right_bank_y": float(right_point[1]),
                    "dt_width_global": float(dt_width),
                    "width_chord": float(width_chord),
                    "qc_relative_dt_diff": float(relative_dt_diff),
                    "qc_opposition_cos": float(opposition_cos),
                    "qc_width_chord_diff": float(width_chord_diff),
                    "width_method": "two_bank_dt",
                    "width_invalid": False,
                }
            )

        dt_seed_fallback = np.asarray([max(float(2.0 * dt_map[row_, col_]), pixel_length) for row_, col_ in zip(rows, cols)], dtype=float)
        total_array = _fill_invalid_width_series(np.asarray(width_total_values, dtype=float), dt_seed_fallback)
        wet_array = _fill_invalid_width_series(np.asarray(width_wet_values, dtype=float), total_array)
        dry_array = _fill_invalid_width_series(np.asarray(width_dry_values, dtype=float), np.maximum(total_array - wet_array, 0.0))
        channels_array = _fill_invalid_width_series(np.asarray(n_channels_values, dtype=float), np.ones(len(flat_indices), dtype=float))

        trim_slice = _trim_slice(total_array, distances)
        total_stats = _compute_family_stats(total_array, trim_slice)
        wet_stats = _compute_family_stats(wet_array, trim_slice)
        dry_stats = _compute_family_stats(dry_array, trim_slice)
        trimmed_channels = channels_array[trim_slice]
        if trimmed_channels.size == 0:
            trimmed_channels = channels_array
        qc_dt_array = np.asarray(qc_relative_dt_diff, dtype=float)
        qc_opp_array = np.asarray(qc_opposition_cos, dtype=float)
        qc_chord_array = np.asarray(qc_width_chord_diff, dtype=float)

        summary_records.append(
            {
                "id_link": link_id,
                "wid_pix_total": _serialize_float_array(total_array.tolist()),
                "wid_pix_wet": _serialize_float_array(wet_array.tolist()),
                "wid_pix_dry": _serialize_float_array(dry_array.tolist()),
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
                "wet_fraction_adj": float(wet_stats["adj"] / total_stats["adj"]) if total_stats["adj"] > 0 else float("nan"),
                "dry_fraction_adj": float(dry_stats["adj"] / total_stats["adj"]) if total_stats["adj"] > 0 else float("nan"),
                "n_wet_threads_mean": float(np.mean(trimmed_channels)) if trimmed_channels.size else float("nan"),
                "n_wet_threads_max": int(np.max(trimmed_channels)) if trimmed_channels.size else 0,
                "n_channels_mean": float(np.mean(trimmed_channels)) if trimmed_channels.size else float("nan"),
                "n_channels_max": int(np.max(trimmed_channels)) if trimmed_channels.size else 0,
                "n_invalid_width_samples": int(np.count_nonzero(np.asarray(invalid_flags, dtype=bool))),
                "qc_mean_relative_dt_diff": float(np.nanmean(qc_dt_array)) if np.isfinite(qc_dt_array).any() else float("nan"),
                "qc_max_relative_dt_diff": float(np.nanmax(qc_dt_array)) if np.isfinite(qc_dt_array).any() else float("nan"),
                "qc_mean_width_chord_diff": float(np.nanmean(qc_chord_array)) if np.isfinite(qc_chord_array).any() else float("nan"),
                "qc_min_opposition_cos": float(np.nanmin(qc_opp_array)) if np.isfinite(qc_opp_array).any() else float("nan"),
                "width_method": "two_bank_dt",
            }
        )

    if not summary_records:
        return enriched, pd.DataFrame()

    summary_frame = pd.DataFrame.from_records(summary_records)
    sample_frame = pd.DataFrame.from_records(sample_records)
    enriched_lookup = summary_frame.copy()
    enriched = enriched.merge(enriched_lookup, on="id_link", how="left", validate="1:1")
    return enriched, sample_frame

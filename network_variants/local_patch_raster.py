"""Adaptive cropped-raster planning for topology-defined local patches.

This phase is deliberately upstream of collapse editing and RivGraph.  It reads
bounded windows from the existing base mask and global land-component labels,
selects water connected to the planned network context, and accepts a crop only
when its perimeter has one inlet and one outlet with consistent outer banks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_dilation, label
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge

from network_variants.local_patch import LocalPatchContract


LOCAL_PATCH_RASTER_PLANNER_VERSION = "topology-local-patch-raster-v2"


@dataclass(frozen=True, slots=True)
class MainBankCut:
    role: Literal["upstream", "downstream"]
    anchor_node_id: int
    anchor_x: float
    anchor_y: float
    first_bank_component_id: int
    second_bank_component_id: int
    first_endpoint_x: float
    first_endpoint_y: float
    second_endpoint_x: float
    second_endpoint_y: float
    first_bank_distance_m: float
    second_bank_distance_m: float
    bank_overlap_m: float
    inner_boundary_node_id: int
    outer_anchor_node_id: int
    context_distance_from_boundary_m: float
    extended_beyond_outer_anchor: bool

    @property
    def geometry(self) -> LineString:
        return LineString(
            [
                (self.first_endpoint_x, self.first_endpoint_y),
                (self.anchor_x, self.anchor_y),
                (self.second_endpoint_x, self.second_endpoint_y),
            ]
        )

    @property
    def bank_component_ids(self) -> tuple[int, int]:
        return tuple(
            sorted((int(self.first_bank_component_id), int(self.second_bank_component_id)))
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bank_component_ids"] = list(self.bank_component_ids)
        return payload


@dataclass(frozen=True, slots=True)
class BoundaryCrossing:
    crossing_id: int
    pixel_count: int
    center_x: float
    center_y: float
    preceding_land_component_id: int
    following_land_component_id: int
    preceding_row: int
    preceding_col: int
    following_row: int
    following_col: int
    assigned_role: str = ""
    anchor_distance_m: float = float("nan")

    @property
    def bank_component_ids(self) -> tuple[int, int]:
        return tuple(
            sorted(
                (
                    int(self.preceding_land_component_id),
                    int(self.following_land_component_id),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class LocalPatchRasterAttempt:
    halo_m: float
    window_col_off: int
    window_row_off: int
    window_width: int
    window_height: int
    bounds: tuple[float, float, float, float]
    full_raster_fraction: float
    seeded_water_component_ids: tuple[int, ...]
    relevant_water_pixel_count: int
    crossings: tuple[BoundaryCrossing, ...]
    bank_cuts: tuple[MainBankCut, ...]
    unauthorized_boundary_section_count: int
    cut_contact_pixel_counts: tuple[int, ...]
    cuts_contained_in_crop: tuple[bool, ...]
    status: Literal["safe", "unsafe"]
    reason_codes: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crossings"] = [
            {
                **asdict(crossing),
                "bank_component_ids": list(crossing.bank_component_ids),
            }
            for crossing in self.crossings
        ]
        payload["bank_cuts"] = [cut.to_manifest() for cut in self.bank_cuts]
        return payload


@dataclass(frozen=True, slots=True)
class LocalPatchRasterResult:
    status: Literal["safe", "fallback_full_raster"]
    attempts: tuple[LocalPatchRasterAttempt, ...]
    selected_attempt_index: int | None
    output_paths: dict[str, str]
    fallback_reason: str = ""
    planner_version: str = LOCAL_PATCH_RASTER_PLANNER_VERSION

    def __post_init__(self) -> None:
        if self.status == "safe":
            if self.selected_attempt_index is None:
                raise ValueError("A safe raster result requires a selected attempt.")
            selected = self.attempts[int(self.selected_attempt_index)]
            if selected.status != "safe":
                raise ValueError("The selected raster attempt is not safe.")
        elif self.status == "fallback_full_raster":
            if self.selected_attempt_index is not None:
                raise ValueError("A fallback raster result cannot select an attempt.")
        else:
            raise ValueError(f"Unsupported raster result status: {self.status!r}.")

    @property
    def is_safe(self) -> bool:
        return self.status == "safe"

    @property
    def selected_attempt(self) -> LocalPatchRasterAttempt | None:
        if self.selected_attempt_index is None:
            return None
        return self.attempts[int(self.selected_attempt_index)]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "planner_version": self.planner_version,
            "status": self.status,
            "selected_attempt_index": self.selected_attempt_index,
            "fallback_reason": self.fallback_reason,
            "output_paths": dict(self.output_paths),
            "attempts": [attempt.to_manifest() for attempt in self.attempts],
        }


def _outward_window(
    bounds: Sequence[float],
    *,
    transform: Any,
    raster_width: int,
    raster_height: int,
) -> Window:
    raw = from_bounds(*bounds, transform=transform)
    tolerance = 1e-6
    col_start = int(np.floor(float(raw.col_off) + tolerance))
    row_start = int(np.floor(float(raw.row_off) + tolerance))
    col_stop = int(np.ceil(float(raw.col_off + raw.width) - tolerance))
    row_stop = int(np.ceil(float(raw.row_off + raw.height) - tolerance))
    return Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    ).intersection(Window(0, 0, int(raster_width), int(raster_height)))


def _perimeter_indices(height: int, width: int) -> list[tuple[int, int]]:
    if height < 2 or width < 2:
        raise ValueError("A local-patch raster window must be at least 2 by 2 pixels.")
    result = [(0, col) for col in range(width)]
    result.extend((row, width - 1) for row in range(1, height))
    result.extend((height - 1, col) for col in range(width - 2, -1, -1))
    result.extend((row, 0) for row in range(height - 2, 0, -1))
    return result


def _circular_true_runs(values: np.ndarray) -> list[list[int]]:
    values = np.asarray(values, dtype=bool)
    if values.size == 0 or not values.any():
        return []
    if values.all():
        return [list(range(len(values)))]
    starts = np.flatnonzero(values & ~np.roll(values, 1))
    runs: list[list[int]] = []
    for start in starts:
        run: list[int] = []
        index = int(start)
        while values[index]:
            run.append(index)
            index = (index + 1) % len(values)
        runs.append(run)
    return runs


def _seeded_water(
    water: np.ndarray,
    *,
    transform: Any,
    context_geometry: Any,
    pixel_size: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    structure8 = np.ones((3, 3), dtype=np.uint8)
    components, _ = label(water.astype(np.uint8), structure=structure8)
    seed_geometry = context_geometry.buffer(max(float(pixel_size) * 0.75, 1e-9))
    seeds = rasterize(
        [(seed_geometry, 1)],
        out_shape=water.shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    component_ids = tuple(
        sorted(int(value) for value in np.unique(components[seeds & water]) if int(value) > 0)
    )
    if not component_ids:
        return np.zeros(water.shape, dtype=bool), ()
    return np.isin(components, component_ids), component_ids


def _crossings_from_perimeter(
    relevant_water: np.ndarray,
    land_components: np.ndarray,
    *,
    transform: Any,
) -> tuple[BoundaryCrossing, ...]:
    perimeter = _perimeter_indices(*relevant_water.shape)
    perimeter_water = np.asarray(
        [relevant_water[row, col] for row, col in perimeter], dtype=bool
    )
    runs = _circular_true_runs(perimeter_water)
    crossings: list[BoundaryCrossing] = []
    for crossing_id, run in enumerate(runs, start=1):
        coordinates = []
        for perimeter_index in run:
            row, col = perimeter[perimeter_index]
            x, y = rasterio.transform.xy(transform, row, col, offset="center")
            coordinates.append((float(x), float(y)))
        preceding_index = (run[0] - 1) % len(perimeter)
        following_index = (run[-1] + 1) % len(perimeter)
        preceding_row, preceding_col = perimeter[preceding_index]
        following_row, following_col = perimeter[following_index]
        crossings.append(
            BoundaryCrossing(
                crossing_id=crossing_id,
                pixel_count=len(run),
                center_x=float(np.mean([value[0] for value in coordinates])),
                center_y=float(np.mean([value[1] for value in coordinates])),
                preceding_land_component_id=int(
                    land_components[preceding_row, preceding_col]
                ),
                following_land_component_id=int(
                    land_components[following_row, following_col]
                ),
                preceding_row=int(preceding_row),
                preceding_col=int(preceding_col),
                following_row=int(following_row),
                following_col=int(following_col),
            )
        )
    return tuple(crossings)


def _anchor_points(
    nodes: gpd.GeoDataFrame,
    contract: LocalPatchContract,
) -> tuple[Point, Point]:
    if "id_node" not in nodes.columns:
        raise ValueError("full_nodes is missing required column 'id_node'.")
    ids = pd.to_numeric(nodes["id_node"], errors="coerce")
    points: list[Point] = []
    for node_id in (
        contract.source_splice_anchor_node_id,
        contract.sink_splice_anchor_node_id,
    ):
        selected = nodes.loc[ids.eq(int(node_id))]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one geometry for splice-anchor node {node_id}; "
                f"found {len(selected)}."
            )
        geometry = selected.iloc[0].geometry
        if geometry is None or geometry.is_empty:
            raise ValueError(f"Splice-anchor node {node_id} has no geometry.")
        points.append(Point(geometry.x, geometry.y))
    return points[0], points[1]


def _node_point(nodes: gpd.GeoDataFrame, node_id: int) -> Point:
    ids = pd.to_numeric(nodes["id_node"], errors="coerce")
    selected = nodes.loc[ids.eq(int(node_id))]
    if len(selected) != 1 or selected.iloc[0].geometry is None or selected.iloc[0].geometry.is_empty:
        raise ValueError(f"Expected one non-empty geometry for node {node_id}.")
    geometry = selected.iloc[0].geometry
    return Point(float(geometry.x), float(geometry.y))


def _load_reference_centerline(
    reference_centerline_path: str | Path,
    *,
    target_crs: Any,
) -> LineString:
    centerlines = gpd.read_file(reference_centerline_path)
    if centerlines.empty or centerlines.crs is None:
        raise ValueError("The reference centerline requires one geometry and a CRS.")
    centerlines = centerlines.to_crs(target_crs)
    geometry = centerlines.geometry.union_all()
    if isinstance(geometry, MultiLineString):
        geometry = linemerge(geometry)
    if not isinstance(geometry, LineString) or geometry.is_empty:
        raise ValueError("The reference centerline must resolve to one continuous LineString.")
    return geometry


def _cut_center_points(
    *,
    nodes: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    contract: LocalPatchContract,
    reference_centerline: LineString,
) -> tuple[tuple[Point, float, bool], tuple[Point, float, bool]]:
    """Return computational cut centers with minimum along-centerline context."""
    results: list[tuple[Point, float, bool]] = []
    for collar in (contract.upstream_collar, contract.downstream_collar):
        inner = _node_point(nodes, collar.domain_boundary_node_id)
        outer = _node_point(nodes, collar.outer_splice_anchor_node_id)
        selected = links.loc[
            pd.to_numeric(links["id_link"], errors="coerce").eq(int(collar.base_link_id))
        ]
        if len(selected) != 1:
            raise ValueError(f"Could not resolve collar link {collar.base_link_id}.")
        stored_length = pd.to_numeric(
            pd.Series([selected.iloc[0].get("len")]), errors="coerce"
        ).iloc[0]
        collar_length = (
            float(stored_length)
            if not pd.isna(stored_length) and float(stored_length) > 0
            else float(selected.iloc[0].geometry.length)
        )
        inner_measure = float(reference_centerline.project(inner))
        outer_measure = float(reference_centerline.project(outer))
        direction = float(np.sign(outer_measure - inner_measure))
        if direction == 0:
            raise ValueError(
                f"Inner and outer nodes of collar {collar.base_link_id} have the same "
                "reference-centerline projection."
            )
        minimum_context = float(contract.raster_policy.minimum_collar_context_m)
        if collar_length >= minimum_context:
            cut_measure = outer_measure
            extended = False
        else:
            cut_measure = inner_measure + direction * minimum_context
            extended = True
        if cut_measure < 0 or cut_measure > float(reference_centerline.length):
            raise ValueError(
                f"A {minimum_context:.1f} m context cut for collar {collar.base_link_id} "
                "falls beyond the reference centerline."
            )
        cut_point = reference_centerline.interpolate(cut_measure)
        results.append((Point(cut_point.x, cut_point.y), abs(cut_measure - inner_measure), extended))
    return results[0], results[1]


def _resolve_main_bank_component_ids(
    labels_source: Any,
    mask_source: Any,
) -> tuple[int, int]:
    """Resolve the two largest global land components without a full allocation."""
    summary_paths = sorted(
        path
        for path in Path(labels_source.name).parent.glob("*land_component_summary*.csv")
        if not path.name.startswith("._")
    )
    for summary_path in summary_paths:
        summary = pd.read_csv(summary_path)
        if {"raster_value", "n_pixels"}.issubset(summary.columns):
            values = (
                summary.assign(
                    raster_value=pd.to_numeric(summary["raster_value"], errors="coerce"),
                    n_pixels=pd.to_numeric(summary["n_pixels"], errors="coerce"),
                )
                .dropna(subset=["raster_value", "n_pixels"])
                .nlargest(2, "n_pixels")["raster_value"]
                .astype(int)
                .tolist()
            )
            if len(values) == 2 and values[0] != values[1]:
                return tuple(sorted(values))

    counts: dict[int, int] = {}
    for _, window in labels_source.block_windows(1):
        labels = labels_source.read(1, window=window)
        water = mask_source.read(1, window=window) > 0
        values, frequencies = np.unique(labels[(~water) & (labels > 0)], return_counts=True)
        for value, frequency in zip(values, frequencies):
            component_id = int(value)
            counts[component_id] = counts.get(component_id, 0) + int(frequency)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2:
        raise ValueError("Could not resolve two distinct global main-bank components.")
    return tuple(sorted((int(ranked[0][0]), int(ranked[1][0]))))


def _nearest_component_point(
    anchor: Point,
    *,
    component_id: int,
    labels_source: Any,
    pixel_size: float,
) -> tuple[np.ndarray, float]:
    """Find the globally nearest labelled pixel using bounded expanding windows."""
    raster_extent = float(
        np.hypot(
            labels_source.width * float(pixel_size),
            labels_source.height * float(pixel_size),
        )
    )
    search_radius = min(
        raster_extent,
        max(500.0, 16.0 * float(pixel_size)),
    )
    while True:
        raw = from_bounds(
            anchor.x - search_radius,
            anchor.y - search_radius,
            anchor.x + search_radius,
            anchor.y + search_radius,
            transform=labels_source.transform,
        )
        col_start = max(0, int(np.floor(raw.col_off)))
        row_start = max(0, int(np.floor(raw.row_off)))
        col_stop = min(labels_source.width, int(np.ceil(raw.col_off + raw.width)))
        row_stop = min(labels_source.height, int(np.ceil(raw.row_off + raw.height)))
        if col_stop <= col_start or row_stop <= row_start:
            search_radius *= 2.0
            continue
        window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )
        labels = labels_source.read(1, window=window)
        rows, cols = np.nonzero(labels == int(component_id))
        if len(rows):
            transform = labels_source.window_transform(window)
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            distances = np.hypot(xs - anchor.x, ys - anchor.y)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= search_radius:
                return (
                    np.asarray([xs[nearest], ys[nearest]], dtype=float),
                    float(distances[nearest]),
                )
        if search_radius >= raster_extent:
            break
        search_radius = min(search_radius * 2.0, raster_extent)
    raise ValueError(
        f"Could not find main-bank component {component_id} near anchor "
        f"({anchor.x}, {anchor.y})."
    )


def _build_main_bank_cuts(
    *,
    source_anchor: Point,
    sink_anchor: Point,
    cut_context: tuple[tuple[float, bool], tuple[float, bool]],
    contract: LocalPatchContract,
    main_bank_component_ids: tuple[int, int],
    labels_source: Any,
    pixel_size: float,
) -> tuple[MainBankCut, MainBankCut]:
    overlap = float(contract.raster_policy.bank_overlap_pixels) * float(pixel_size)
    cuts: list[MainBankCut] = []
    for role, anchor, collar, context in (
        ("upstream", source_anchor, contract.upstream_collar, cut_context[0]),
        ("downstream", sink_anchor, contract.downstream_collar, cut_context[1]),
    ):
        node_id = int(collar.outer_splice_anchor_node_id)
        endpoints: list[np.ndarray] = []
        distances: list[float] = []
        anchor_xy = np.asarray([anchor.x, anchor.y], dtype=float)
        for component_id in main_bank_component_ids:
            bank_point, distance = _nearest_component_point(
                anchor,
                component_id=int(component_id),
                labels_source=labels_source,
                pixel_size=pixel_size,
            )
            direction = bank_point - anchor_xy
            norm = float(np.linalg.norm(direction))
            if norm <= 0:
                raise ValueError(
                    f"Splice anchor {node_id} lies inside main-bank component {component_id}."
                )
            endpoints.append(bank_point + direction / norm * overlap)
            distances.append(distance)
        cut = MainBankCut(
            role=role,
            anchor_node_id=int(node_id),
            anchor_x=float(anchor.x),
            anchor_y=float(anchor.y),
            first_bank_component_id=int(main_bank_component_ids[0]),
            second_bank_component_id=int(main_bank_component_ids[1]),
            first_endpoint_x=float(endpoints[0][0]),
            first_endpoint_y=float(endpoints[0][1]),
            second_endpoint_x=float(endpoints[1][0]),
            second_endpoint_y=float(endpoints[1][1]),
            first_bank_distance_m=float(distances[0]),
            second_bank_distance_m=float(distances[1]),
            bank_overlap_m=float(overlap),
            inner_boundary_node_id=int(collar.domain_boundary_node_id),
            outer_anchor_node_id=int(collar.outer_splice_anchor_node_id),
            context_distance_from_boundary_m=float(context[0]),
            extended_beyond_outer_anchor=bool(context[1]),
        )
        if cut.geometry.distance(anchor) > 1e-9:
            raise ValueError(f"Constructed {role} bank cut does not pass through its anchor.")
        cuts.append(cut)
    return cuts[0], cuts[1]


def _rasterize_bank_cuts(
    cuts: Sequence[MainBankCut],
    *,
    out_shape: tuple[int, int],
    transform: Any,
    half_width_m: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    masks = tuple(
        rasterize(
            [(cut.geometry.buffer(float(half_width_m)), 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        for cut in cuts
    )
    return np.logical_or.reduce(masks), masks


def _assign_crossing_roles(
    crossings: tuple[BoundaryCrossing, ...],
    *,
    source_anchor: Point,
    sink_anchor: Point,
) -> tuple[BoundaryCrossing, ...]:
    if len(crossings) != 2:
        return crossings
    options = [
        (
            source_anchor.distance(Point(crossings[0].center_x, crossings[0].center_y))
            + sink_anchor.distance(Point(crossings[1].center_x, crossings[1].center_y)),
            (0, 1),
        ),
        (
            source_anchor.distance(Point(crossings[1].center_x, crossings[1].center_y))
            + sink_anchor.distance(Point(crossings[0].center_x, crossings[0].center_y)),
            (1, 0),
        ),
    ]
    _, (source_index, sink_index) = min(options, key=lambda item: item[0])
    assigned: list[BoundaryCrossing] = []
    for index, crossing in enumerate(crossings):
        if index == source_index:
            role, anchor = "upstream", source_anchor
        else:
            role, anchor = "downstream", sink_anchor
        assigned.append(
            BoundaryCrossing(
                crossing_id=crossing.crossing_id,
                pixel_count=crossing.pixel_count,
                center_x=crossing.center_x,
                center_y=crossing.center_y,
                preceding_land_component_id=crossing.preceding_land_component_id,
                following_land_component_id=crossing.following_land_component_id,
                preceding_row=crossing.preceding_row,
                preceding_col=crossing.preceding_col,
                following_row=crossing.following_row,
                following_col=crossing.following_col,
                assigned_role=role,
                anchor_distance_m=float(
                    anchor.distance(Point(crossing.center_x, crossing.center_y))
                ),
            )
        )
    return tuple(assigned)


def _validate_attempt(
    *,
    seeded_component_ids: tuple[int, ...],
    crossings: tuple[BoundaryCrossing, ...],
    land_components: np.ndarray,
    halo_m: float,
    contract: LocalPatchContract,
) -> tuple[str, ...]:
    policy = contract.raster_policy
    reasons: list[str] = []
    if not seeded_component_ids:
        reasons.append("no_seeded_water_component")
    if len(crossings) != int(policy.expected_main_water_boundary_crossings):
        reasons.append("unexpected_main_water_boundary_crossing_count")
        return tuple(reasons)

    if policy.require_crossings_near_expected_continuations:
        maximum_distance = max(
            float(policy.minimum_halo_m),
            float(halo_m) * float(policy.crossing_anchor_max_distance_halo_factor),
        )
        if any(
            not np.isfinite(crossing.anchor_distance_m)
            or float(crossing.anchor_distance_m) > maximum_distance
            for crossing in crossings
        ):
            reasons.append("crossing_not_near_expected_splice_continuation")

    bank_pairs = [crossing.bank_component_ids for crossing in crossings]
    if policy.require_distinct_controlling_outer_banks and any(
        pair[0] <= 0 or pair[1] <= 0 or pair[0] == pair[1] for pair in bank_pairs
    ):
        reasons.append("invalid_or_nondistinct_crossing_bank_components")
    if (
        policy.require_same_outer_bank_pair_at_both_crossings
        and len(bank_pairs) == 2
        and bank_pairs[0] != bank_pairs[1]
    ):
        reasons.append("inconsistent_outer_bank_components_between_crossings")
    if (
        policy.require_untruncated_controlling_outer_banks
        and len(bank_pairs) == 2
        and bank_pairs[0] == bank_pairs[1]
        and all(value > 0 for value in bank_pairs[0])
    ):
        structure4 = np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )
        for bank_id in bank_pairs[0]:
            local_bank_components, _ = label(
                np.asarray(land_components == int(bank_id), dtype=np.uint8),
                structure=structure4,
            )
            crossing_component_ids: list[int] = []
            for crossing in crossings:
                candidates = []
                if crossing.preceding_land_component_id == bank_id:
                    candidates.append(
                        int(
                            local_bank_components[
                                crossing.preceding_row, crossing.preceding_col
                            ]
                        )
                    )
                if crossing.following_land_component_id == bank_id:
                    candidates.append(
                        int(
                            local_bank_components[
                                crossing.following_row, crossing.following_col
                            ]
                        )
                    )
                crossing_component_ids.extend(value for value in candidates if value > 0)
            if (
                len(crossing_component_ids) != 2
                or crossing_component_ids[0] != crossing_component_ids[1]
            ):
                reasons.append("controlling_outer_bank_not_connected_within_crop")
                break
    return tuple(reasons)


def _cut_crossing_records(
    cuts: Sequence[MainBankCut],
    contact_pixel_counts: Sequence[int],
) -> tuple[BoundaryCrossing, ...]:
    records: list[BoundaryCrossing] = []
    for crossing_id, (cut, contact_count) in enumerate(
        zip(cuts, contact_pixel_counts), start=1
    ):
        records.append(
            BoundaryCrossing(
                crossing_id=int(crossing_id),
                pixel_count=int(contact_count),
                center_x=float(cut.anchor_x),
                center_y=float(cut.anchor_y),
                preceding_land_component_id=int(cut.first_bank_component_id),
                following_land_component_id=int(cut.second_bank_component_id),
                preceding_row=-1,
                preceding_col=-1,
                following_row=-1,
                following_col=-1,
                assigned_role=cut.role,
                anchor_distance_m=0.0,
            )
        )
    return tuple(records)


def _validate_cut_attempt(
    *,
    seeded_component_ids: tuple[int, ...],
    unauthorized_boundary_section_count: int,
    cuts_contained_in_crop: Sequence[bool],
    cut_contact_pixel_counts: Sequence[int],
    contract: LocalPatchContract,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not seeded_component_ids:
        reasons.append("no_seeded_water_component")
    if (
        contract.raster_policy.require_complete_cuts_inside_crop
        and not all(bool(value) for value in cuts_contained_in_crop)
    ):
        reasons.append("main_bank_cut_not_contained_in_crop")
    if len(cut_contact_pixel_counts) != 2 or any(
        int(value) <= 0 for value in cut_contact_pixel_counts
    ):
        reasons.append("main_water_does_not_contact_both_bank_cuts")
    if int(unauthorized_boundary_section_count) != int(
        contract.raster_policy.expected_unauthorized_boundary_sections
    ):
        reasons.append("unauthorized_main_water_crop_edge_contact")
    return tuple(reasons)


def _write_raster(
    output_path: Path,
    values: np.ndarray,
    *,
    source_profile: dict[str, Any],
    transform: Any,
    dtype: str,
    nodata: int,
) -> Path:
    profile = source_profile.copy()
    profile.update(
        height=int(values.shape[0]),
        width=int(values.shape[1]),
        transform=transform,
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
    )
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(values.astype(dtype, copy=False), 1)
    return output_path


def build_local_patch_base_raster(
    *,
    contract: LocalPatchContract,
    full_links: gpd.GeoDataFrame,
    full_nodes: gpd.GeoDataFrame,
    cleaned_mask_path: str | Path,
    boundary_component_labels_path: str | Path,
    output_dir: str | Path,
    reference_centerline_path: str | Path | None = None,
) -> LocalPatchRasterResult:
    """Build a validated base crop or request full-raster fallback."""
    if full_links.crs is None or full_nodes.crs is None:
        raise ValueError("Full links and nodes require a projected CRS.")
    if not full_links.crs.is_projected or not full_nodes.crs.is_projected:
        raise ValueError("Full links and nodes require a projected CRS.")
    context_ids = set(contract.local_context_link_ids)
    link_ids = pd.to_numeric(full_links.get("id_link"), errors="coerce")
    context_links = full_links.loc[link_ids.isin(context_ids)].copy()
    found_ids = set(pd.to_numeric(context_links["id_link"], errors="coerce").dropna().astype(int))
    if found_ids != context_ids:
        raise ValueError(
            f"Local context links are unavailable: {sorted(context_ids - found_ids)}."
        )
    if context_links.geometry.isna().any() or context_links.geometry.is_empty.any():
        raise ValueError("Every local context link requires a non-empty geometry.")

    cleaned_mask_path = Path(cleaned_mask_path)
    boundary_component_labels_path = Path(boundary_component_labels_path)
    output_dir = Path(output_dir)
    attempts: list[LocalPatchRasterAttempt] = []
    selected_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, dict[str, Any]
    ] | None = None
    selected_index: int | None = None

    with rasterio.open(cleaned_mask_path) as mask_source, rasterio.open(
        boundary_component_labels_path
    ) as labels_source:
        if (
            mask_source.shape != labels_source.shape
            or mask_source.transform != labels_source.transform
            or mask_source.crs != labels_source.crs
        ):
            raise ValueError(
                "The cleaned mask and boundary-component labels must be exactly aligned."
            )
        if mask_source.crs is None or not mask_source.crs.is_projected:
            raise ValueError("The cleaned mask requires a projected CRS.")
        links = context_links.to_crs(mask_source.crs)
        nodes = full_nodes.to_crs(mask_source.crs)
        if reference_centerline_path is None:
            reference_centerline = linemerge(links.geometry.union_all())
            if not isinstance(reference_centerline, LineString):
                raise ValueError(
                    "A reference_centerline_path is required when local context links "
                    "do not merge to one LineString."
                )
        else:
            reference_centerline = _load_reference_centerline(
                reference_centerline_path, target_crs=mask_source.crs
            )
        source_cut, sink_cut = _cut_center_points(
            nodes=nodes,
            links=links,
            contract=contract,
            reference_centerline=reference_centerline,
        )
        source_anchor, source_context_m, source_extended = source_cut
        sink_anchor, sink_context_m, sink_extended = sink_cut
        context_geometry = links.geometry.union_all()
        core_links = links.loc[
            pd.to_numeric(links["id_link"], errors="coerce").isin(contract.core_link_ids)
        ]
        if core_links.empty:
            raise ValueError("The local core has no geometry in the raster CRS.")
        core_geometry = core_links.geometry.union_all()
        base_bounds = context_geometry.bounds
        pixel_size = float(max(abs(mask_source.transform.a), abs(mask_source.transform.e)))
        main_bank_component_ids = _resolve_main_bank_component_ids(
            labels_source, mask_source
        )
        bank_cuts = _build_main_bank_cuts(
            source_anchor=source_anchor,
            sink_anchor=sink_anchor,
            cut_context=(
                (source_context_m, source_extended),
                (sink_context_m, sink_extended),
            ),
            contract=contract,
            main_bank_component_ids=main_bank_component_ids,
            labels_source=labels_source,
            pixel_size=pixel_size,
        )
        base_bounds = links.geometry.union_all().union(
            gpd.GeoSeries([cut.geometry for cut in bank_cuts], crs=links.crs).union_all()
        ).bounds
        halo_values = np.arange(
            float(contract.raster_policy.minimum_halo_m),
            float(contract.raster_policy.maximum_halo_m)
            + float(contract.raster_policy.halo_expansion_step_m) * 0.5,
            float(contract.raster_policy.halo_expansion_step_m),
        )
        for halo_m in halo_values:
            candidate_bounds = box(*base_bounds).buffer(float(halo_m)).bounds
            window = _outward_window(
                candidate_bounds,
                transform=mask_source.transform,
                raster_width=mask_source.width,
                raster_height=mask_source.height,
            )
            transform = mask_source.window_transform(window)
            source_values = mask_source.read(1, window=window)
            water = source_values > 0
            land_components = labels_source.read(1, window=window).astype(np.int32)
            barrier, cut_masks = _rasterize_bank_cuts(
                bank_cuts,
                out_shape=water.shape,
                transform=transform,
                half_width_m=(
                    float(contract.raster_policy.cut_barrier_half_width_pixels)
                    * pixel_size
                ),
            )
            relevant_water, seeded_ids = _seeded_water(
                water & (~barrier),
                transform=transform,
                context_geometry=core_geometry,
                pixel_size=pixel_size,
            )
            perimeter = _perimeter_indices(*relevant_water.shape)
            perimeter_water = np.asarray(
                [relevant_water[row, col] for row, col in perimeter], dtype=bool
            )
            unauthorized_sections = len(_circular_true_runs(perimeter_water))
            cut_contact_counts = tuple(
                int(
                    np.count_nonzero(
                        binary_dilation(relevant_water) & cut_mask & water
                    )
                )
                for cut_mask in cut_masks
            )
            crop_polygon = box(
                *rasterio.windows.bounds(window, mask_source.transform)
            )
            cuts_contained = tuple(
                bool(crop_polygon.covers(cut.geometry)) for cut in bank_cuts
            )
            crossings = _cut_crossing_records(bank_cuts, cut_contact_counts)
            reasons = _validate_cut_attempt(
                seeded_component_ids=seeded_ids,
                unauthorized_boundary_section_count=unauthorized_sections,
                cuts_contained_in_crop=cuts_contained,
                cut_contact_pixel_counts=cut_contact_counts,
                contract=contract,
            )
            attempt = LocalPatchRasterAttempt(
                halo_m=float(halo_m),
                window_col_off=int(window.col_off),
                window_row_off=int(window.row_off),
                window_width=int(window.width),
                window_height=int(window.height),
                bounds=tuple(float(value) for value in rasterio.windows.bounds(window, mask_source.transform)),
                full_raster_fraction=float(
                    (window.width * window.height) / (mask_source.width * mask_source.height)
                ),
                seeded_water_component_ids=seeded_ids,
                relevant_water_pixel_count=int(np.count_nonzero(relevant_water)),
                crossings=crossings,
                bank_cuts=bank_cuts,
                unauthorized_boundary_section_count=int(unauthorized_sections),
                cut_contact_pixel_counts=cut_contact_counts,
                cuts_contained_in_crop=cuts_contained,
                status="safe" if not reasons else "unsafe",
                reason_codes=reasons,
            )
            attempts.append(attempt)
            if not reasons:
                selected_index = len(attempts) - 1
                selected_arrays = (
                    source_values,
                    relevant_water.astype(np.uint8),
                    land_components,
                    barrier.astype(np.uint8),
                    transform,
                    mask_source.profile.copy(),
                )
                break

    if selected_index is None or selected_arrays is None:
        reason = (
            ",".join(attempts[-1].reason_codes)
            if attempts
            else "no_raster_attempt_was_created"
        )
        result = LocalPatchRasterResult(
            status="fallback_full_raster",
            attempts=tuple(attempts),
            selected_attempt_index=None,
            output_paths={},
            fallback_reason=reason,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_local_patch_raster_result(result, output_dir / "local_patch_raster_plan.json")
        return result

    (
        source_values,
        relevant_water,
        land_components,
        cut_barrier,
        transform,
        source_profile,
    ) = selected_arrays
    source_window_path = _write_raster(
        output_dir / "source_window_mask.tif",
        source_values,
        source_profile=source_profile,
        transform=transform,
        dtype=str(source_values.dtype),
        nodata=int(source_profile.get("nodata") or 0),
    )
    local_base_path = _write_raster(
        output_dir / "local_base_mask.tif",
        relevant_water,
        source_profile=source_profile,
        transform=transform,
        dtype="uint8",
        nodata=0,
    )
    local_labels_path = _write_raster(
        output_dir / "boundary_component_labels.tif",
        land_components,
        source_profile=source_profile,
        transform=transform,
        dtype="int32",
        nodata=0,
    )
    cut_barrier_path = _write_raster(
        output_dir / "main_bank_cut_barrier.tif",
        cut_barrier,
        source_profile=source_profile,
        transform=transform,
        dtype="uint8",
        nodata=0,
    )
    selected = attempts[selected_index]
    crop_path = output_dir / "local_patch_crop.gpkg"
    gpd.GeoDataFrame(
        [
            {
                "example_id": contract.example_id,
                "halo_m": selected.halo_m,
                "full_raster_fraction": selected.full_raster_fraction,
                "geometry": box(*selected.bounds),
            }
        ],
        crs=source_profile.get("crs"),
    ).to_file(crop_path, driver="GPKG")
    context_path = output_dir / "local_patch_context.gpkg"
    links.assign(
        context_role=links["id_link"].map(
            lambda value: (
                "core"
                if int(value) in set(contract.core_link_ids)
                else "upstream_connector"
                if int(value) in set(contract.upstream_connector_link_ids)
                else "downstream_connector"
                if int(value) in set(contract.downstream_connector_link_ids)
                else "upstream_collar"
                if int(value) == int(contract.upstream_collar.base_link_id)
                else "downstream_collar"
            )
        )
    ).to_file(context_path, driver="GPKG")
    crossing_path = output_dir / "boundary_crossings.gpkg"
    gpd.GeoDataFrame(
        [
            {
                "crossing_id": crossing.crossing_id,
                "assigned_role": crossing.assigned_role,
                "pixel_count": crossing.pixel_count,
                "anchor_distance_m": crossing.anchor_distance_m,
                "bank_component_1": crossing.bank_component_ids[0],
                "bank_component_2": crossing.bank_component_ids[1],
                "geometry": Point(crossing.center_x, crossing.center_y),
            }
            for crossing in selected.crossings
        ],
        crs=source_profile.get("crs"),
    ).to_file(crossing_path, driver="GPKG")
    cuts_path = output_dir / "main_bank_cuts.gpkg"
    gpd.GeoDataFrame(
        [
            {
                "role": cut.role,
                "anchor_node_id": cut.anchor_node_id,
                "inner_boundary_node_id": cut.inner_boundary_node_id,
                "outer_anchor_node_id": cut.outer_anchor_node_id,
                "context_distance_from_boundary_m": cut.context_distance_from_boundary_m,
                "extended_beyond_outer_anchor": cut.extended_beyond_outer_anchor,
                "bank_component_1": cut.bank_component_ids[0],
                "bank_component_2": cut.bank_component_ids[1],
                "bank_distance_1_m": cut.first_bank_distance_m,
                "bank_distance_2_m": cut.second_bank_distance_m,
                "bank_overlap_m": cut.bank_overlap_m,
                "geometry": cut.geometry,
            }
            for cut in selected.bank_cuts
        ],
        crs=source_profile.get("crs"),
    ).to_file(cuts_path, driver="GPKG")
    manifest_path = output_dir / "local_patch_raster_plan.json"
    output_paths = {
        "source_window_mask": str(source_window_path.resolve()),
        "local_base_mask": str(local_base_path.resolve()),
        "boundary_component_labels": str(local_labels_path.resolve()),
        "main_bank_cut_barrier": str(cut_barrier_path.resolve()),
        "crop_geometry": str(crop_path.resolve()),
        "context_links": str(context_path.resolve()),
        "boundary_crossings": str(crossing_path.resolve()),
        "main_bank_cuts": str(cuts_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    result = LocalPatchRasterResult(
        status="safe",
        attempts=tuple(attempts),
        selected_attempt_index=selected_index,
        output_paths=output_paths,
    )
    write_local_patch_raster_result(result, manifest_path)
    return result


def write_local_patch_raster_result(
    result: LocalPatchRasterResult,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "BoundaryCrossing",
    "LOCAL_PATCH_RASTER_PLANNER_VERSION",
    "LocalPatchRasterAttempt",
    "LocalPatchRasterResult",
    "MainBankCut",
    "build_local_patch_base_raster",
    "write_local_patch_raster_result",
]

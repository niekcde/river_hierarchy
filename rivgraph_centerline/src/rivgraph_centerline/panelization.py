"""Create and validate editable panel layouts for large RivGraph masks."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import LineString, MultiLineString, box
from shapely.geometry import Point
from shapely.ops import linemerge, polygonize, unary_union

from rivgraph_centerline.raster_mask import crop_binary_mask


PANEL_BREAK_LAYER = "panel_breaks"
STANDARD_LINK_COLUMNS = [
    "id_link", "idx_link", "id_nodes", "n_nodes", "id_us_node", "id_ds_node",
    "is_inlet", "is_outlet", "type_io", "schema_rg", "link_conn", "wid_pix",
    "len", "wid", "wid_adj", "wid_med", "len_adj", "sinuosity", "certain",
    "certain_order", "certain_alg", "guess", "guess_alg", "maxang", "cldists",
    "clangs", "wid_pctdiff", "geometry",
]
STANDARD_NODE_COLUMNS = [
    "id_node", "idx_node", "id_links", "n_links", "is_inlet", "is_outlet",
    "type_io", "schema_rg", "geometry",
]


def _line_parts(geometry: object) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if part.length > 0]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _concatenate_two_lines(first: LineString, second: LineString) -> LineString:
    """Orient two segments at their nearest endpoints and return one line."""
    first_coords = list(first.coords)
    second_coords = list(second.coords)
    candidates = [
        (first_coords, second_coords),
        (first_coords, second_coords[::-1]),
        (first_coords[::-1], second_coords),
        (first_coords[::-1], second_coords[::-1]),
    ]
    left, right = min(candidates, key=lambda pair: Point(pair[0][-1]).distance(Point(pair[1][0])))
    join_x = (left[-1][0] + right[0][0]) / 2
    join_y = (left[-1][1] + right[0][1]) / 2
    join = (join_x, join_y)
    return LineString(left[:-1] + [join] + right[1:])


def _parse_numeric_values(value: object, *, dtype: type[float] | type[int]) -> list[float] | list[int]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    return [dtype(item.strip()) for item in str(value).split(",") if item.strip()]


def _rivgraph_width_metrics(widths: list[float], coordinates: list[tuple[float, float]]) -> dict[str, float]:
    """Apply RivGraph's endpoint-width trimming to an ordered sample series."""
    if len(widths) != len(coordinates) or not widths:
        raise ValueError("Width values and sample coordinates must be non-empty and have equal length.")
    values = np.asarray(widths, dtype=float)
    xy = np.asarray(coordinates, dtype=float)
    dists = np.insert(np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))), 0, 0.0)
    revdists = np.insert(np.cumsum(np.hypot(np.diff(xy[::-1, 0]), np.diff(xy[::-1, 1]))), 0, 0.0)
    width_mult = 1.1
    start = int(np.argmin(np.abs(dists - values[0] / 2 * width_mult)))
    end = len(dists) - int(np.argmin(np.abs(revdists - values[-1] / 2 * width_mult))) - 1
    if start >= end:
        adjusted = values
        adjusted_length = float(dists[-1])
        endpoints = (0, len(values) - 1)
    else:
        adjusted = values[start:end]
        adjusted_length = float(dists[end] - dists[start])
        endpoints = (start, end)
    straight_length = float(np.hypot(*(xy[endpoints[0]] - xy[endpoints[1]])))
    return {
        "wid": max(1.0, float(np.mean(values))),
        "wid_adj": max(1.0, float(np.mean(adjusted))),
        "wid_med": float(np.median(adjusted)),
        "len": max(1.0, float(dists[-1])),
        "len_adj": max(1.0, adjusted_length),
        "sinuosity": adjusted_length / straight_length if straight_length > 0 else 1.0,
    }


def _core_owned_width_samples(
    *,
    merged_geometry: LineString,
    source_segments: list[object],
    source_rows: dict[tuple[str, int], object],
    panel_root: Path,
    excluded_sample_keys: set[tuple[str, int, int]] | None = None,
) -> tuple[list[float], list[tuple[float, float]], list[dict[str, object]]]:
    """Return ordered, core-owned RivGraph width samples for one merged link."""
    excluded_sample_keys = excluded_sample_keys or set()
    selected_sample_keys: set[tuple[str, int, int]] = set()
    pieces: list[tuple[float, list[float], list[tuple[float, float]], list[dict[str, object]]]] = []
    for segment in source_segments:
        panel = str(segment.panel_id)
        source_link_id = int(segment.source_link_id)
        source = source_rows[(panel, source_link_id)]
        widths = _parse_numeric_values(source["wid_pix"], dtype=float)
        indices = _parse_numeric_values(source["idx_link"], dtype=int)
        if len(widths) != len(indices):
            raise ValueError(f"Panel {panel} link {source_link_id} has mismatched wid_pix and idx_link lengths.")
        with rasterio.open(panel_root / panel / "mask.tif") as mask:
            rows = np.asarray(indices, dtype=np.int64) // mask.width
            cols = np.asarray(indices, dtype=np.int64) % mask.width
            xs, ys = rasterio.transform.xy(mask.transform, rows, cols, offset="center")
            pixel_size = max(abs(float(mask.transform.a)), abs(float(mask.transform.e)))
        coordinates = list(zip(map(float, xs), map(float, ys)))
        ownership_tolerance = pixel_size * 0.75
        # Restrict the relatively short source segment to this merged part
        # once.  Testing every point against a very long merged reference
        # line makes a normalized single-link reference section needlessly
        # quadratic in its coordinate count.
        owned_segment = segment.geometry.intersection(
            merged_geometry.buffer(max(1e-7, pixel_size * 1e-6))
        )
        selected = []
        for width, coordinate, index in zip(widths, coordinates, indices):
            point = Point(coordinate)
            sample_key = (panel, source_link_id, int(index))
            owned = (
                not owned_segment.is_empty
                and point.distance(owned_segment) <= ownership_tolerance
                and sample_key not in excluded_sample_keys
                and sample_key not in selected_sample_keys
            )
            if owned:
                selected.append((width, coordinate, index))
                selected_sample_keys.add(sample_key)
        if not selected:
            continue
        selected.sort(
            key=lambda item: segment.geometry.project(Point(item[1]))
        )
        segment_start = Point(segment.geometry.coords[0])
        segment_end = Point(segment.geometry.coords[-1])
        start_projection = merged_geometry.project(segment_start)
        end_projection = merged_geometry.project(segment_end)
        if start_projection > end_projection:
            selected.reverse()
        selected_widths = [float(item[0]) for item in selected]
        selected_coordinates = [item[1] for item in selected]
        first_projection = merged_geometry.project(Point(selected_coordinates[0]))
        last_projection = merged_geometry.project(Point(selected_coordinates[-1]))
        provenance = [
            {"panel_id": panel, "source_link_id": source_link_id, "source_segment_id": str(segment["segment_id"]), "source_pixel_index": int(item[2])}
            for item in selected
        ]
        pieces.append(
            (
                float((first_projection + last_projection) / 2.0),
                selected_widths,
                selected_coordinates,
                provenance,
            )
        )
    if not pieces:
        raise ValueError("No core-owned width samples were found for a merged link.")
    pieces.sort(key=lambda item: item[0])
    return (
        [width for _, widths, _, _ in pieces for width in widths],
        [coordinate for _, _, coordinates, _ in pieces for coordinate in coordinates],
        [record for _, _, _, records in pieces for record in records],
    )


def _material_source_segment_ids(
    merged_part: LineString,
    source_segments: gpd.GeoDataFrame,
) -> list[str]:
    """Return source segments with linear overlap, excluding endpoint touches."""
    selected: list[str] = []
    for row in source_segments.itertuples(index=False):
        overlap_length = float(merged_part.intersection(row.geometry).length)
        numerical_floor = max(
            1e-7,
            min(float(merged_part.length), float(row.geometry.length)) * 1e-10,
        )
        if overlap_length > numerical_floor:
            selected.append(str(row.segment_id))
    return selected


def _build_panelized_reference_centerline(
    *,
    plan_path: Path,
    panel_root: Path,
    run_name: str,
    endpoint_tolerance_m: float = 10,
) -> gpd.GeoDataFrame:
    """Stitch core-owned per-panel reference centerlines into one LineString."""
    cores = gpd.read_file(plan_path, layer="panel_cores")
    parts: list[LineString] = []
    for core in cores.itertuples(index=False):
        panel = str(core.panel_id)
        centerline_path = panel_root / panel / "rivgraph" / f"{run_name}_{panel}_centerline.gpkg"
        if not centerline_path.exists():
            # Panels without core water do not have a RivGraph output, just as
            # they do not contribute source links to the assembled network.
            continue
        panel_centerline = gpd.read_file(centerline_path)
        if len(panel_centerline) != 1:
            raise ValueError(
                f"Expected exactly one reference centerline feature for panel {panel}, "
                f"found {len(panel_centerline)} in {centerline_path}."
            )
        core_parts = _line_parts(panel_centerline.geometry.iloc[0].intersection(core.geometry))
        if len(core_parts) != 1:
            raise ValueError(
                f"Expected one core-owned reference centerline part for panel {panel}, "
                f"found {len(core_parts)} in {centerline_path}."
            )
        parts.append(core_parts[0])

    if not parts:
        raise ValueError("No panel reference centerlines were found for the panelized handoff.")

    endpoint_groups: list[list[Point]] = []
    part_endpoints: list[tuple[int, int]] = []
    for part in parts:
        endpoint_ids: list[int] = []
        for coordinate in (part.coords[0], part.coords[-1]):
            point = Point(coordinate)
            matching = next(
                (
                    index
                    for index, group in enumerate(endpoint_groups)
                    if any(point.distance(existing) <= endpoint_tolerance_m for existing in group)
                ),
                None,
            )
            if matching is None:
                matching = len(endpoint_groups)
                endpoint_groups.append([])
            endpoint_groups[matching].append(point)
            endpoint_ids.append(matching)
        if endpoint_ids[0] == endpoint_ids[1]:
            raise ValueError("A panel reference centerline collapses to the same endpoint cluster at both ends.")
        part_endpoints.append((endpoint_ids[0], endpoint_ids[1]))

    incident: dict[int, list[int]] = {index: [] for index in range(len(endpoint_groups))}
    for part_index, (start, end) in enumerate(part_endpoints):
        incident[start].append(part_index)
        incident[end].append(part_index)
    endpoint_nodes = sorted(node for node, linked_parts in incident.items() if len(linked_parts) == 1)
    if len(endpoint_nodes) != 2 or any(len(linked_parts) > 2 for linked_parts in incident.values()):
        raise ValueError(
            "Core-owned panel reference centerlines must form one unbranched path; "
            f"found endpoint degrees {[len(incident[index]) for index in sorted(incident)]}."
        )

    current_node = endpoint_nodes[0]
    used_parts: set[int] = set()
    stitched_coordinates: list[tuple[float, float]] = []
    while len(used_parts) < len(parts):
        candidates = [part_index for part_index in incident[current_node] if part_index not in used_parts]
        if len(candidates) != 1:
            raise ValueError("Panel reference centerlines do not form one connected path.")
        part_index = candidates[0]
        start_node, end_node = part_endpoints[part_index]
        coordinates = list(parts[part_index].coords)
        if start_node != current_node:
            coordinates.reverse()
            start_node, end_node = end_node, start_node
        if stitched_coordinates:
            previous = stitched_coordinates[-1]
            join = ((previous[0] + coordinates[0][0]) / 2, (previous[1] + coordinates[0][1]) / 2)
            stitched_coordinates[-1] = join
            stitched_coordinates.extend(coordinates[1:])
        else:
            stitched_coordinates.extend(coordinates)
        used_parts.add(part_index)
        current_node = end_node

    return gpd.GeoDataFrame(
        {"schema_rg": ["panelized-reference-centerline-v1"]},
        geometry=[LineString(stitched_coordinates)],
        crs=cores.crs,
    )


def build_core_boundary_network(*, plan_path: Path, breaks_path: Path, panel_root: Path, run_name: str, output_dir: Path, tolerance_m: float = 10, overwrite: bool = False, boundary_match_plan_path: Path | None = None) -> dict[str, object]:
    """Build a core-owned network and merge only degree-two matched boundary nodes."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    cores = gpd.read_file(plan_path, layer="panel_cores")
    breaks = gpd.read_file(breaks_path, layer=PANEL_BREAK_LAYER).to_crs(cores.crs)
    segments: list[dict[str, object]] = []
    boundary_nodes: list[dict[str, object]] = []
    for core_row in cores.itertuples(index=False):
        panel = core_row.panel_id
        link_path = panel_root / panel / "rivgraph" / f"{run_name}_{panel}_links.gpkg"
        if not link_path.exists():
            continue
        links = gpd.read_file(link_path)
        for link in links.itertuples(index=False):
            for part_index, part in enumerate(_line_parts(link.geometry.intersection(core_row.geometry))):
                if part.length <= tolerance_m:
                    continue
                segment_id = f"{panel}:{int(link.id_link)}:{part_index}"
                segments.append({"segment_id": segment_id, "panel_id": panel, "source_link_id": int(link.id_link), "geometry": part})
                for endpoint_role, coord in (("start", part.coords[0]), ("end", part.coords[-1])):
                    point = Point(coord)
                    hit = breaks.loc[breaks.geometry.distance(point) <= tolerance_m]
                    if not hit.empty:
                        boundary_nodes.append({"boundary_node_id": f"{segment_id}:{endpoint_role}", "segment_id": segment_id, "panel_id": panel, "break_id": hit.iloc[0].break_id, "endpoint_role": endpoint_role, "geometry": point})
    seg = gpd.GeoDataFrame(segments, geometry="geometry", crs=cores.crs)
    nodes = gpd.GeoDataFrame(boundary_nodes, geometry="geometry", crs=cores.crs)
    match_plan_path = boundary_match_plan_path or (plan_path.parent.parent / "panel_matching" / "boundary_match_plan.json")
    pairs = json.loads(match_plan_path.read_text()).get("pairs", [])
    match_rows: list[dict[str, object]] = []
    for pair in pairs:
        left = nodes.loc[(nodes.break_id == pair["break_id"]) & (nodes.panel_id == pair["left_panel"])]
        right = nodes.loc[(nodes.break_id == pair["break_id"]) & (nodes.panel_id == pair["right_panel"])]
        for a in left.itertuples(index=False):
            for b in right.itertuples(index=False):
                distance = a.geometry.distance(b.geometry)
                if distance <= tolerance_m:
                    match_rows.append({"break_id": pair["break_id"], "left_segment_id": a.segment_id, "right_segment_id": b.segment_id, "distance_m": distance, "geometry": LineString([a.geometry, b.geometry])})
    matches = gpd.GeoDataFrame(match_rows, geometry="geometry", crs=cores.crs) if match_rows else gpd.GeoDataFrame({"break_id": [], "left_segment_id": [], "right_segment_id": [], "distance_m": []}, geometry=gpd.GeoSeries([], crs=cores.crs), crs=cores.crs)
    # Only unique one-to-one boundary-node matches are degree-two continuations.
    if not matches.empty:
        left_counts = matches.left_segment_id.value_counts()
        right_counts = matches.right_segment_id.value_counts()
        matches["accepted"] = matches.left_segment_id.map(left_counts).eq(1) & matches.right_segment_id.map(right_counts).eq(1)
    else:
        matches["accepted"] = pd.Series(dtype=bool)
    parent = {sid: sid for sid in seg.segment_id}
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for row in matches.loc[matches.accepted].itertuples(index=False):
        union(row.left_segment_id, row.right_segment_id)
    seg["merge_group"] = [find(sid) for sid in seg.segment_id]
    merged_rows = []
    for group, frame in seg.groupby("merge_group", sort=True):
        lines = frame.geometry.tolist()
        if len(lines) == 2:
            geom = _concatenate_two_lines(lines[0], lines[1])
        else:
            unioned = unary_union(lines)
            geom = linemerge(unioned) if isinstance(unioned, MultiLineString) else unioned
        assigned_segment_ids: set[str] = set()
        for part_index, part in enumerate(_line_parts(geom)):
            source_segment_ids = _material_source_segment_ids(part, frame)
            if not source_segment_ids:
                raise ValueError(
                    f"Merged part {part_index} in group {group!r} has no materially "
                    "overlapping source segment."
                )
            assigned_segment_ids.update(source_segment_ids)
            merged_rows.append(
                {
                    "global_link_id": len(merged_rows),
                    "source_segments": ";".join(source_segment_ids),
                    "geometry": part,
                }
            )
        missing_segment_ids = sorted(set(frame.segment_id.astype(str)).difference(assigned_segment_ids))
        if missing_segment_ids:
            raise ValueError(
                f"Merge group {group!r} lost source segments while assigning split-link "
                f"provenance: {missing_segment_ids}."
            )
    merged = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=cores.crs)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg.to_file(output_dir / "core_segments.gpkg", layer="core_segments", driver="GPKG")
    nodes.to_file(output_dir / "boundary_nodes.gpkg", layer="boundary_nodes", driver="GPKG")
    matches.to_file(output_dir / "boundary_node_matches.gpkg", layer="boundary_node_matches", driver="GPKG")
    merged.to_file(output_dir / "merged_core_links.gpkg", layer="merged_core_links", driver="GPKG")
    report = {"core_segment_count": len(seg), "boundary_node_count": len(nodes), "boundary_match_count": len(matches), "merged_link_count": len(merged)}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), **report}


def validate_pilot_topology(*, links_path: Path, nodes_path: Path, output_path: Path) -> dict[str, object]:
    links = gpd.read_file(links_path, layer="merged_links")
    nodes = gpd.read_file(nodes_path, layer="merged_nodes")
    node_ids = set(nodes["id_node"].astype(int))
    referenced = set(links["id_us_node"].astype(int)) | set(links["id_ds_node"].astype(int))
    invalid = sorted(referenced.difference(node_ids))
    duplicate_nodes = int(nodes.geometry.duplicated().sum())
    self_links = int((links["id_us_node"].astype(int) == links["id_ds_node"].astype(int)).sum())
    degree = pd.concat([links["id_us_node"], links["id_ds_node"]]).value_counts()
    report = {"is_valid": not invalid and duplicate_nodes == 0 and self_links == 0, "link_count": len(links), "node_count": len(nodes), "invalid_node_ids": invalid, "duplicate_node_count": duplicate_nodes, "self_link_count": self_links, "orphan_node_count": int(len(node_ids.difference(referenced))), "max_degree": int(degree.max()) if not degree.empty else 0}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def rebuild_core_network_nodes(
    *,
    merged_links_path: Path,
    output_dir: Path,
    endpoint_tolerance_m: float = 0.01,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create global, undirected nodes for the core-merged panel network.

    Link geometry orientation is not a reliable flow-direction indicator after
    clipping and boundary concatenation.  This stage therefore records neutral
    endpoint IDs (``id_node_a`` and ``id_node_b``); assigning ``id_us_node`` and
    ``id_ds_node`` belongs to the subsequent flow-direction reconciliation.
    """
    if endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be non-negative.")
    links = gpd.read_file(merged_links_path, layer="merged_core_links")
    if links.empty:
        raise ValueError(f"No merged links found in {merged_links_path}.")
    if not links.geometry.geom_type.isin(["LineString"]).all():
        invalid = sorted(set(links.geometry.geom_type).difference({"LineString"}))
        raise ValueError(f"Merged links must all be LineStrings; found {invalid}.")

    links_output = output_dir / "merged_links.gpkg"
    nodes_output = output_dir / "merged_nodes.gpkg"
    validation_output = output_dir / "topology_validation.json"
    if not overwrite and any(path.exists() for path in (links_output, nodes_output, validation_output)):
        raise FileExistsError(f"Reconciled outputs already exist in {output_dir}; use --overwrite to replace them.")

    # Cluster only coincident (or numerically near-coincident) endpoints.  A
    # strict tolerance avoids accidentally collapsing nearby, distinct channels.
    node_points: list[Point] = []
    node_counts: list[int] = []

    def node_for(point: Point) -> int:
        for node_id, existing in enumerate(node_points):
            if point.distance(existing) <= endpoint_tolerance_m:
                count = node_counts[node_id]
                node_points[node_id] = Point(
                    ((existing.x * count) + point.x) / (count + 1),
                    ((existing.y * count) + point.y) / (count + 1),
                )
                node_counts[node_id] = count + 1
                return node_id
        node_points.append(point)
        node_counts.append(1)
        return len(node_points) - 1

    node_a: list[int] = []
    node_b: list[int] = []
    for geometry in links.geometry:
        coordinates = list(geometry.coords)
        node_a.append(node_for(Point(coordinates[0])))
        node_b.append(node_for(Point(coordinates[-1])))

    reconciled = links.copy()
    reconciled["id_node_a"] = node_a
    reconciled["id_node_b"] = node_b
    reconciled["id_nodes"] = [f"{a}, {b}" for a, b in zip(node_a, node_b)]
    reconciled["direction_status"] = "unassigned"
    degree = pd.concat([reconciled["id_node_a"], reconciled["id_node_b"]]).value_counts().sort_index()
    nodes = gpd.GeoDataFrame(
        {
            "id_node": range(len(node_points)),
            "degree": [int(degree.get(node_id, 0)) for node_id in range(len(node_points))],
        },
        geometry=node_points,
        crs=links.crs,
    )

    node_ids = set(nodes["id_node"].astype(int))
    referenced = set(reconciled["id_node_a"].astype(int)) | set(reconciled["id_node_b"].astype(int))
    duplicate_nodes = int(nodes.geometry.duplicated().sum())
    self_links = int((reconciled["id_node_a"] == reconciled["id_node_b"]).sum())
    component_count = _undirected_component_count(reconciled, node_ids)
    report = {
        "is_valid": duplicate_nodes == 0 and self_links == 0 and referenced == node_ids,
        "link_count": len(reconciled),
        "node_count": len(nodes),
        "invalid_node_ids": sorted(referenced.difference(node_ids)),
        "duplicate_node_count": duplicate_nodes,
        "self_link_count": self_links,
        "orphan_node_count": int(len(node_ids.difference(referenced))),
        "max_degree": int(degree.max()) if not degree.empty else 0,
        "connected_component_count": component_count,
        "endpoint_tolerance_m": endpoint_tolerance_m,
        "direction_status": "unassigned",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    reconciled.to_file(links_output, layer="merged_links", driver="GPKG")
    nodes.to_file(nodes_output, layer="merged_nodes", driver="GPKG")
    validation_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), **report}


def _undirected_component_count(links: gpd.GeoDataFrame, node_ids: set[int]) -> int:
    """Return the number of connected node components in an undirected link graph."""
    neighbours = {node_id: set() for node_id in node_ids}
    for row in links.itertuples(index=False):
        first, second = int(row.id_node_a), int(row.id_node_b)
        neighbours[first].add(second)
        neighbours[second].add(first)
    remaining = set(node_ids)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            unseen = neighbours[node].intersection(remaining)
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return count


def assign_core_network_direction(
    *,
    merged_links_path: Path,
    core_segments_path: Path,
    panel_root: Path,
    run_name: str,
    output_dir: Path,
    fallback_directed_links_path: Path | None = None,
    require_single_source_sink_path: bool = False,
    endpoint_tolerance_m: float = 0.01,
    overwrite: bool = False,
) -> dict[str, object]:
    """Orient reconciled global links using native flow assignments from panels.

    Each clipped core segment inherits an upstream/downstream direction from its
    source RivGraph link.  Segments that reach a global-link endpoint vote for
    which endpoint is upstream; a unanimous (or majority) vote orients the
    global link. Ambiguous links are exported unchanged and reported instead
    of being silently guessed. If a directed parent network is supplied, a
    tied vote may inherit direction from one uniquely matching parent link.
    """
    links = gpd.read_file(merged_links_path, layer="merged_links")
    segments = gpd.read_file(core_segments_path, layer="core_segments")
    required = {"global_link_id", "source_segments", "id_node_a", "id_node_b"}
    missing = required.difference(links.columns)
    if missing:
        raise ValueError(f"{merged_links_path} is missing required columns: {sorted(missing)}")
    directed_output = output_dir / "merged_links_directed.gpkg"
    summary_output = output_dir / "flow_direction_summary.json"
    if not overwrite and (directed_output.exists() or summary_output.exists()):
        raise FileExistsError(f"Directed outputs already exist in {output_dir}; use --overwrite to replace them.")

    source_data: dict[tuple[str, int], tuple[LineString, Point, Point]] = {}
    for panel in sorted(segments["panel_id"].unique()):
        links_path = panel_root / panel / "rivgraph" / f"{run_name}_{panel}_links.gpkg"
        nodes_path = panel_root / panel / "rivgraph" / f"{run_name}_{panel}_nodes.gpkg"
        local_links = gpd.read_file(links_path)
        if local_links["id_us_node"].isna().any() or local_links["id_ds_node"].isna().any():
            raise ValueError(
                f"Panel {panel} has no complete native flow assignment. Rerun run-planned-panels with --assign-flow first."
            )
        local_nodes = gpd.read_file(nodes_path).set_index("id_node")
        for row in local_links.itertuples(index=False):
            source_data[(panel, int(row.id_link))] = (
                row.geometry,
                local_nodes.loc[int(row.id_us_node)].geometry,
                local_nodes.loc[int(row.id_ds_node)].geometry,
            )

    fallback_links: gpd.GeoDataFrame | None = None
    if fallback_directed_links_path is not None:
        fallback_links = gpd.read_file(fallback_directed_links_path)
        fallback_required = {"id_us_node", "id_ds_node"}
        fallback_missing = fallback_required.difference(fallback_links.columns)
        if fallback_missing:
            raise ValueError(
                f"{fallback_directed_links_path} is missing required columns: {sorted(fallback_missing)}"
            )
        if fallback_links["id_us_node"].isna().any() or fallback_links["id_ds_node"].isna().any():
            raise ValueError(f"Fallback directed links are not fully directed: {fallback_directed_links_path}")
        if fallback_links.crs != links.crs:
            fallback_links = fallback_links.to_crs(links.crs)

    segment_by_id = segments.set_index("segment_id")
    result = links.copy()
    upstream_ids: list[int | None] = []
    downstream_ids: list[int | None] = []
    status: list[str] = []
    geometries: list[LineString] = []
    ambiguous: list[int] = []
    inherited: list[int] = []
    for row in result.itertuples(index=False):
        endpoints = (Point(row.geometry.coords[0]), Point(row.geometry.coords[-1]))
        votes = [0, 0]  # votes that endpoint 0 / 1 is upstream
        for segment_id in str(row.source_segments).split(";"):
            segment = segment_by_id.loc[segment_id]
            source_geometry, source_us, source_ds = source_data[(segment.panel_id, int(segment.source_link_id))]
            segment_endpoints = (Point(segment.geometry.coords[0]), Point(segment.geometry.coords[-1]))
            # Determine which clipped-segment endpoint is nearest the source
            # upstream node along the original source-link path.
            us_position = source_geometry.project(source_us)
            ds_position = source_geometry.project(source_ds)
            positions = [source_geometry.project(point) for point in segment_endpoints]
            upstream_segment_endpoint = 0 if abs(positions[0] - us_position) <= abs(positions[1] - us_position) else 1
            # Reject a malformed local direction rather than use it as evidence.
            if abs(positions[upstream_segment_endpoint] - ds_position) < abs(positions[upstream_segment_endpoint] - us_position):
                continue
            for segment_endpoint_id, point in enumerate(segment_endpoints):
                for global_endpoint_id, global_point in enumerate(endpoints):
                    if point.distance(global_point) <= endpoint_tolerance_m:
                        if segment_endpoint_id == upstream_segment_endpoint:
                            votes[global_endpoint_id] += 1
                        else:
                            votes[1 - global_endpoint_id] += 1
        if votes[0] == votes[1] and fallback_links is not None:
            matches = fallback_links.loc[
                fallback_links.geometry.apply(
                    lambda geometry: (
                        abs(float(geometry.length) - float(row.geometry.length)) <= endpoint_tolerance_m
                        and geometry.hausdorff_distance(row.geometry) <= endpoint_tolerance_m
                    )
                )
            ]
            if len(matches) == 1:
                parent_upstream = Point(matches.iloc[0].geometry.coords[0])
                distances = [parent_upstream.distance(endpoint) for endpoint in endpoints]
                upstream_endpoint = 0 if distances[0] <= distances[1] else 1
                downstream_endpoint = 1 - upstream_endpoint
                upstream_ids.append(int(row.id_node_a) if upstream_endpoint == 0 else int(row.id_node_b))
                downstream_ids.append(int(row.id_node_a) if downstream_endpoint == 0 else int(row.id_node_b))
                status.append("inherited_parent")
                coordinates = list(row.geometry.coords)
                geometries.append(row.geometry if upstream_endpoint == 0 else LineString(coordinates[::-1]))
                inherited.append(int(row.global_link_id))
                continue
        if votes[0] == votes[1]:
            upstream_ids.append(None); downstream_ids.append(None); status.append("ambiguous")
            geometries.append(row.geometry); ambiguous.append(int(row.global_link_id))
            continue
        upstream_endpoint = 0 if votes[0] > votes[1] else 1
        downstream_endpoint = 1 - upstream_endpoint
        upstream_ids.append(int(row.id_node_a) if upstream_endpoint == 0 else int(row.id_node_b))
        downstream_ids.append(int(row.id_node_a) if downstream_endpoint == 0 else int(row.id_node_b))
        status.append("assigned")
        coordinates = list(row.geometry.coords)
        geometries.append(row.geometry if upstream_endpoint == 0 else LineString(coordinates[::-1]))

    result["id_us_node"] = pd.array(upstream_ids, dtype="Int64")
    result["id_ds_node"] = pd.array(downstream_ids, dtype="Int64")
    result["direction_status"] = status
    result = gpd.GeoDataFrame(result, geometry=geometries, crs=links.crs)

    global_path_assigned: list[int] = []
    global_path_overridden: list[int] = []
    if require_single_source_sink_path:
        if fallback_links is None:
            raise ValueError(
                "Single-path panel direction completion requires a directed parent network."
            )
        parent_sources = sorted(
            set(fallback_links["id_us_node"].astype(int)).difference(
                fallback_links["id_ds_node"].astype(int)
            )
        )
        parent_sinks = sorted(
            set(fallback_links["id_ds_node"].astype(int)).difference(
                fallback_links["id_us_node"].astype(int)
            )
        )
        if len(parent_sources) != 1 or len(parent_sinks) != 1:
            raise ValueError(
                "Single-path panel direction completion requires exactly one parent "
                f"source and sink; found sources={parent_sources}, sinks={parent_sinks}."
            )

        node_points: dict[int, Point] = {}
        for row in links.itertuples(index=False):
            node_points.setdefault(int(row.id_node_a), Point(row.geometry.coords[0]))
            node_points.setdefault(int(row.id_node_b), Point(row.geometry.coords[-1]))
        degree_graph = nx.MultiGraph()
        for row in links.itertuples(index=False):
            degree_graph.add_edge(
                int(row.id_node_a),
                int(row.id_node_b),
                key=int(row.global_link_id),
            )
        if not nx.is_connected(degree_graph):
            raise ValueError("Collapsed reference-section merge is not connected.")
        endpoints = sorted(node for node in degree_graph if degree_graph.degree(node) == 1)
        internal_bad = sorted(
            node for node in degree_graph
            if degree_graph.degree(node) not in {1, 2}
        )
        if (
            len(endpoints) != 2
            or internal_bad
            or degree_graph.number_of_edges() != degree_graph.number_of_nodes() - 1
        ):
            raise ValueError(
                "Collapsed reference-section merge is not a single undirected path: "
                f"endpoints={endpoints}, non_path_degree_nodes={internal_bad}, "
                f"nodes={degree_graph.number_of_nodes()}, edges={degree_graph.number_of_edges()}."
            )

        def parent_boundary_point(node_id: int, *, upstream: bool) -> Point:
            field = "id_us_node" if upstream else "id_ds_node"
            matches = fallback_links.loc[fallback_links[field].astype(int).eq(node_id)]
            row = matches.iloc[0]
            coordinates = list(row.geometry.coords)
            return Point(coordinates[0] if upstream else coordinates[-1])

        parent_source_point = parent_boundary_point(parent_sources[0], upstream=True)
        parent_sink_point = parent_boundary_point(parent_sinks[0], upstream=False)
        source_node = min(endpoints, key=lambda node: node_points[node].distance(parent_source_point))
        sink_node = min(endpoints, key=lambda node: node_points[node].distance(parent_sink_point))
        if source_node == sink_node:
            raise ValueError("Parent source and sink mapped to the same merged endpoint.")
        boundary_tolerance = max(float(endpoint_tolerance_m), 0.01)
        if (
            node_points[source_node].distance(parent_source_point) > boundary_tolerance
            or node_points[sink_node].distance(parent_sink_point) > boundary_tolerance
        ):
            raise ValueError(
                "Collapsed reference-section endpoints do not match the directed parent "
                "source and sink within the merge tolerance."
            )

        node_path = nx.shortest_path(nx.Graph(degree_graph), source_node, sink_node)
        order = {node: index for index, node in enumerate(node_path)}
        original_geometry = links.set_index("global_link_id").geometry
        for index, row in result.iterrows():
            first, second = int(row.id_node_a), int(row.id_node_b)
            if abs(order[first] - order[second]) != 1:
                raise ValueError(
                    f"Merged link {row.global_link_id} does not connect adjacent nodes "
                    "on the sole source-to-sink path."
                )
            expected_us, expected_ds = (
                (first, second) if order[first] < order[second] else (second, first)
            )
            old_us = row.id_us_node
            if pd.isna(old_us):
                result.at[index, "direction_status"] = "global_single_path"
                global_path_assigned.append(int(row.global_link_id))
            elif int(old_us) != expected_us:
                result.at[index, "direction_status"] = "global_single_path_override"
                global_path_overridden.append(int(row.global_link_id))
            result.at[index, "id_us_node"] = expected_us
            result.at[index, "id_ds_node"] = expected_ds
            geometry = original_geometry.loc[int(row.global_link_id)]
            coordinates = list(geometry.coords)
            result.at[index, "geometry"] = (
                geometry if expected_us == first else LineString(coordinates[::-1])
            )
        ambiguous = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_file(directed_output, layer="merged_links", driver="GPKG")
    report = {
        "link_count": len(result),
        "assigned_link_count": int(result["id_us_node"].notna().sum()),
        "panel_vote_assigned_link_count": int((result.direction_status == "assigned").sum()),
        "parent_inherited_link_count": len(inherited),
        "parent_inherited_global_link_ids": inherited,
        "global_single_path_assigned_link_count": len(global_path_assigned),
        "global_single_path_assigned_global_link_ids": global_path_assigned,
        "global_single_path_override_link_count": len(global_path_overridden),
        "global_single_path_override_global_link_ids": global_path_overridden,
        "single_source_sink_path_required": bool(require_single_source_sink_path),
        "ambiguous_link_count": len(ambiguous),
        "ambiguous_global_link_ids": ambiguous,
        "output": str(directed_output),
    }
    summary_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def publish_panelized_rivgraph_handoff(
    *,
    directed_links_path: Path,
    nodes_path: Path,
    core_segments_path: Path,
    plan_path: Path,
    panel_root: Path,
    run_name: str,
    rivgraph_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Publish panelized vectors under the normal ``<run>/rivgraph`` contract.

    This deliberately publishes vectors only. A full-mask skeleton, distance
    transform, and direction raster cannot be reconstructed without attempting
    the memory-prohibitive whole-mask RivGraph run.
    """
    links = gpd.read_file(directed_links_path, layer="merged_links")
    nodes = gpd.read_file(nodes_path, layer="merged_nodes")
    segments = gpd.read_file(core_segments_path, layer="core_segments")
    if links["id_us_node"].isna().any() or links["id_ds_node"].isna().any():
        raise ValueError("Cannot publish panelized handoff while directed links remain ambiguous.")
    output_links = rivgraph_dir / f"{run_name}_links.gpkg"
    output_nodes = rivgraph_dir / f"{run_name}_nodes.gpkg"
    output_centerline = rivgraph_dir / f"{run_name}_centerline.gpkg"
    output_network_centerlines = rivgraph_dir / f"{run_name}_network_centerlines.gpkg"
    if not overwrite and any(path.exists() for path in (output_links, output_nodes, output_centerline, output_network_centerlines)):
        raise FileExistsError(f"Panelized RivGraph handoff already exists in {rivgraph_dir}; use --overwrite to replace it.")

    source_rows: dict[tuple[str, int], object] = {}
    for panel in sorted(segments["panel_id"].unique()):
        panel_links = gpd.read_file(panel_root / panel / "rivgraph" / f"{run_name}_{panel}_links.gpkg").set_index("id_link")
        for source_link_id, source in panel_links.iterrows():
            source_rows[(panel, int(source_link_id))] = source
    segment_by_id = segments.set_index("segment_id")
    enriched_records: list[dict[str, object]] = []
    width_sample_records: list[dict[str, object]] = []
    assigned_sample_keys: set[tuple[str, int, int]] = set()
    for row in links.itertuples(index=False):
        source_segments = []
        for segment_id in str(row.source_segments).split(";"):
            segment = segment_by_id.loc[segment_id].copy()
            segment["segment_id"] = segment_id
            source_segments.append(segment)
        widths, coordinates, provenance = _core_owned_width_samples(
            merged_geometry=row.geometry,
            source_segments=source_segments,
            source_rows=source_rows,
            panel_root=panel_root,
            excluded_sample_keys=assigned_sample_keys,
        )
        sample_keys = {
            (
                str(source["panel_id"]),
                int(source["source_link_id"]),
                int(source["source_pixel_index"]),
            )
            for source in provenance
        }
        if sample_keys.intersection(assigned_sample_keys):
            raise ValueError(
                f"Merged link {row.global_link_id} reuses width samples already assigned "
                "to another merged link."
            )
        assigned_sample_keys.update(sample_keys)
        metrics = _rivgraph_width_metrics(widths, coordinates)
        record = {
            "id_link": int(row.global_link_id),
            # Global raster indices do not exist for a panelized handoff.
            # Keep RivGraph's string field type and mark the value unavailable.
            "idx_link": "",
            "id_nodes": f"{int(row.id_us_node)}, {int(row.id_ds_node)}",
            "n_nodes": 2,
            "id_us_node": int(row.id_us_node),
            "id_ds_node": int(row.id_ds_node),
            "_node_a": int(row.id_node_a),
            "_node_b": int(row.id_node_b),
            "schema_rg": "rg-v1",
            "wid_pix": ", ".join(str(value) for value in widths),
            "len": metrics["len"],
            "len_adj": metrics["len_adj"],
            "certain": 1.0,
            "certain_order": 0.0,
            "certain_alg": 0.0,
            "guess": "",
            "guess_alg": "panelized-global",
            "maxang": float("nan"),
            "cldists": "",
            "clangs": "",
            "wid_pctdiff": None,
            "geometry": row.geometry,
        }
        for field in ("wid", "wid_adj", "wid_med", "sinuosity"):
            record[field] = metrics[field]
        enriched_records.append(record)
        for sample_index, (width, coordinate, source) in enumerate(zip(widths, coordinates, provenance)):
            width_sample_records.append({"id_link": int(row.global_link_id), "sample_index": sample_index, "width": width, "x": coordinate[0], "y": coordinate[1], **source})
    published_links = gpd.GeoDataFrame(enriched_records, geometry="geometry", crs=links.crs)
    incoming = published_links["id_ds_node"].astype(int).value_counts()
    outgoing = published_links["id_us_node"].astype(int).value_counts()
    node_ids = nodes["id_node"].astype(int)
    inlet_ids = set(node_ids.loc[~node_ids.isin(incoming.index) & node_ids.isin(outgoing.index)])
    outlet_ids = set(node_ids.loc[~node_ids.isin(outgoing.index) & node_ids.isin(incoming.index)])
    published_links["is_inlet"] = published_links["id_us_node"].astype(int).isin(inlet_ids)
    published_links["is_outlet"] = published_links["id_ds_node"].astype(int).isin(outlet_ids)
    published_links["type_io"] = "neither"
    published_links.loc[published_links.is_inlet, "type_io"] = "inlet"
    published_links.loc[published_links.is_outlet, "type_io"] = "outlet"
    node_links: dict[int, list[int]] = {int(node_id): [] for node_id in node_ids}
    for row in published_links.itertuples(index=False):
        node_links[int(row.id_us_node)].append(int(row.id_link))
        node_links[int(row.id_ds_node)].append(int(row.id_link))
    published_nodes = nodes.copy()
    published_nodes["idx_node"] = published_nodes["id_node"].astype(int)
    published_nodes["id_links"] = published_nodes["id_node"].astype(int).map(lambda node_id: ", ".join(map(str, node_links[node_id])))
    published_nodes["n_links"] = published_nodes["id_node"].astype(int).map(lambda node_id: len(node_links[node_id]))
    published_nodes["is_inlet"] = published_nodes["id_node"].astype(int).isin(inlet_ids)
    published_nodes["is_outlet"] = published_nodes["id_node"].astype(int).isin(outlet_ids)
    published_nodes["type_io"] = "neither"
    published_nodes.loc[published_nodes.is_inlet, "type_io"] = "inlet"
    published_nodes.loc[published_nodes.is_outlet, "type_io"] = "outlet"
    published_nodes["schema_rg"] = "panelized-rivgraph-v1"

    # The standard run/rivgraph files intentionally contain only RivGraph's
    # normal fields. Panel provenance remains available in merged_network.
    published_links["link_conn"] = published_links.apply(
        lambda row: ", ".join(
            str(link_id) for node_id in (int(row["_node_a"]), int(row["_node_b"]))
            for link_id in node_links[node_id] if link_id != int(row.id_link)
        ),
        axis=1,
    )
    published_links = published_links.reindex(columns=STANDARD_LINK_COLUMNS)
    published_nodes["schema_rg"] = "rg-v1"
    published_nodes = published_nodes.reindex(columns=STANDARD_NODE_COLUMNS)

    reference_centerline = _build_panelized_reference_centerline(
        plan_path=plan_path,
        panel_root=panel_root,
        run_name=run_name,
    )
    network_centerlines = gpd.GeoDataFrame(
        {"schema_rg": ["panelized-network-centerlines-v1"]},
        geometry=[unary_union(published_links.geometry.tolist())],
        crs=links.crs,
    )
    rivgraph_dir.mkdir(parents=True, exist_ok=True)
    published_links.to_file(output_links, driver="GPKG")
    published_nodes.to_file(output_nodes, driver="GPKG")
    reference_centerline.to_file(output_centerline, driver="GPKG")
    network_centerlines.to_file(output_network_centerlines, driver="GPKG")
    width_samples = pd.DataFrame.from_records(width_sample_records)
    source_key_columns = ["panel_id", "source_link_id", "source_pixel_index"]
    if width_samples.duplicated(source_key_columns).any():
        raise ValueError("Panelized handoff contains duplicate source width-sample assignments.")
    width_samples.to_csv(rivgraph_dir / f"{run_name}_link_width_samples.csv", index=False)
    manifest = {
        "mode": "panelized",
        "links": str(output_links),
        "nodes": str(output_nodes),
        "centerline": str(output_centerline),
        "network_centerlines": str(output_network_centerlines),
        "link_width_samples": str(rivgraph_dir / f"{run_name}_link_width_samples.csv"),
        "width_sample_count": int(len(width_samples)),
        "unique_source_width_sample_count": int(
            width_samples[source_key_columns].drop_duplicates().shape[0]
        ),
        "unavailable_full_mask_products": ["skeleton", "distance_transform", "link_directions_raster"],
    }
    (rivgraph_dir / f"{run_name}_panelized_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _ordered_panel_exit_config(
    *,
    cores: gpd.GeoDataFrame,
    pairs: list[dict[str, object]],
    active_panels: set[str],
    global_exit_sides: str,
) -> dict[str, dict[str, object]]:
    """Order local panel sides by traversing from the global inlet to outlet."""
    ordered_global_sides = global_exit_sides.upper()
    if (
        len(ordered_global_sides) != 2
        or len(set(ordered_global_sides)) != 2
        or any(side not in "NSEW" for side in ordered_global_sides)
    ):
        raise ValueError(
            "global_exit_sides must contain exactly two distinct sides from N, S, E, and W, "
            "ordered upstream first."
        )
    if not active_panels:
        return {}

    cores_by_id = cores.set_index("panel_id").geometry
    unknown_panels = active_panels.difference(cores_by_id.index.astype(str))
    if unknown_panels:
        raise ValueError(f"Active panels are missing from the panel plan: {sorted(unknown_panels)}")

    adjacency: dict[str, dict[str, str]] = {panel: {} for panel in active_panels}
    for pair in pairs:
        left = str(pair["left_panel"])
        right = str(pair["right_panel"])
        if left not in active_panels or right not in active_panels:
            continue
        a = cores_by_id[left].centroid
        b = cores_by_id[right].centroid
        if str(pair["break_id"]).upper().startswith("V"):
            left_side, right_side = ("E", "W") if a.x < b.x else ("W", "E")
        else:
            left_side, right_side = ("N", "S") if a.y < b.y else ("S", "N")
        for panel, neighbor, side in (
            (left, right, left_side),
            (right, left, right_side),
        ):
            previous = adjacency[panel].get(neighbor)
            if previous is not None and previous != side:
                raise ValueError(
                    f"Panel pair {panel}/{neighbor} was assigned conflicting boundary sides "
                    f"{previous} and {side}."
                )
            adjacency[panel][neighbor] = side

    extent = cores.geometry.union_all().bounds

    def touches_global_side(panel: str, side: str) -> bool:
        minx, miny, maxx, maxy = cores_by_id[panel].bounds
        return {
            "W": abs(minx - extent[0]) < 2,
            "E": abs(maxx - extent[2]) < 2,
            "S": abs(miny - extent[1]) < 2,
            "N": abs(maxy - extent[3]) < 2,
        }[side]

    upstream_side, downstream_side = ordered_global_sides
    start_candidates = sorted(
        panel for panel in active_panels if touches_global_side(panel, upstream_side)
    )
    end_candidates = sorted(
        panel for panel in active_panels if touches_global_side(panel, downstream_side)
    )
    if len(start_candidates) != 1 or len(end_candidates) != 1:
        raise ValueError(
            "The active panel chain must touch each requested global side exactly once; "
            f"{upstream_side} candidates={start_candidates}, "
            f"{downstream_side} candidates={end_candidates}."
        )
    start = start_candidates[0]
    end = end_candidates[0]

    if len(active_panels) == 1:
        if start != end:
            raise ValueError("A single active panel must touch both requested global sides.")
        ordered_exits = {start: ordered_global_sides}
    else:
        if start == end:
            raise ValueError(
                "The same panel touches both requested global sides, but additional active panels exist."
            )
        invalid_degrees = {
            panel: len(neighbors)
            for panel, neighbors in adjacency.items()
            if len(neighbors) != (1 if panel in {start, end} else 2)
        }
        if invalid_degrees:
            raise ValueError(
                "Active panels must form one unbranched inlet-to-outlet chain; "
                f"invalid panel degrees={invalid_degrees}."
            )

        opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
        ordered_exits: dict[str, str] = {}
        previous_panel: str | None = None
        current_panel = start
        entry_side = upstream_side
        visited: set[str] = set()
        while current_panel != end:
            if current_panel in visited:
                raise ValueError("The active panel adjacency contains a cycle.")
            visited.add(current_panel)
            next_candidates = [
                panel for panel in adjacency[current_panel] if panel != previous_panel
            ]
            if len(next_candidates) != 1:
                raise ValueError(
                    f"Expected one downstream neighbor for {current_panel}, found {next_candidates}."
                )
            next_panel = next_candidates[0]
            exit_side = adjacency[current_panel][next_panel]
            next_entry_side = adjacency[next_panel][current_panel]
            if next_entry_side != opposite[exit_side]:
                raise ValueError(
                    f"Panel boundary {current_panel}/{next_panel} has inconsistent sides "
                    f"{exit_side}/{next_entry_side}."
                )
            if entry_side == exit_side:
                raise ValueError(
                    f"Panel {current_panel} enters and exits through the same side {entry_side}."
                )
            ordered_exits[current_panel] = entry_side + exit_side
            previous_panel, current_panel = current_panel, next_panel
            entry_side = next_entry_side

        if entry_side == downstream_side:
            raise ValueError(f"Panel {end} enters and exits through the same side {entry_side}.")
        ordered_exits[end] = entry_side + downstream_side
        visited.add(end)
        if visited != active_panels:
            raise ValueError(
                "The active panel adjacency is disconnected from the requested global path; "
                f"unvisited panels={sorted(active_panels.difference(visited))}."
            )

    exit_config: dict[str, dict[str, object]] = {}
    for panel in sorted(active_panels):
        global_contacts = [
            side for side in ordered_global_sides if touches_global_side(panel, side)
        ]
        exit_config[panel] = {
            "boundary_sides": "".join(sorted(set(adjacency[panel].values()))),
            "global_sides": "".join(global_contacts),
            "exit_sides": ordered_exits[panel],
            "review_required": False,
        }
    return exit_config


def plan_boundary_matches(*, plan_path: Path, breaks_path: Path, mask_path: Path, output_path: Path, global_exit_sides: str = "SE") -> dict[str, object]:
    cores = gpd.read_file(plan_path, layer="panel_cores")
    breaks = gpd.read_file(breaks_path, layer=PANEL_BREAK_LAYER).to_crs(cores.crs)
    water_panels = set()
    for panel_id in cores["panel_id"]:
        summary_path = plan_path.parent / panel_id / "summary.json"
        if summary_path.exists():
            panel_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            core_water_pixels = panel_summary.get(
                "core_water_pixels",
                panel_summary.get("water_pixels", 0),
            )
            if core_water_pixels > 0:
                water_panels.add(str(panel_id))
    pairs = []
    with rasterio.open(mask_path) as src:
        step = abs(src.transform.a) / 2
        for br in breaks.itertuples(index=False):
            if not bool(br.active): continue
            n = max(2, int(br.geometry.length / step) + 1)
            points = [br.geometry.interpolate(i * br.geometry.length / (n - 1)) for i in range(n)]
            water = [value[0] > 0 for value in src.sample([(p.x, p.y) for p in points])]
            starts = [i for i, value in enumerate(water) if value and (i == 0 or not water[i - 1])]
            ends = [i for i, value in enumerate(water) if value and (i == len(water) - 1 or not water[i + 1])]
            for start, end in zip(starts, ends):
                point = points[(start + end) // 2]
                offset = 100.0
                if str(br.break_id).upper().startswith("V"):
                    probes = [Point(point.x - offset, point.y), Point(point.x + offset, point.y)]
                else:
                    probes = [Point(point.x, point.y - offset), Point(point.x, point.y + offset)]
                ids = []
                for probe in probes:
                    found = cores.loc[cores.geometry.covers(probe), "panel_id"].tolist()
                    ids.append(found[0] if len(found) == 1 else None)
                if None not in ids and ids[0] != ids[1] and set(ids).issubset(water_panels):
                    pairs.append({"break_id": br.break_id, "left_panel": ids[0], "right_panel": ids[1]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
    exit_config = _ordered_panel_exit_config(
        cores=cores,
        pairs=pairs,
        active_panels=water_panels,
        global_exit_sides=global_exit_sides,
    )
    output_path.with_name("panel_exit_sides.json").write_text(json.dumps(exit_config, indent=2), encoding="utf-8")
    return {"output": str(output_path), "pair_count": len(pairs)}


def create_panel_breaks(
    *,
    mask_path: Path,
    output_path: Path,
    rows: int | None = None,
    columns: int | None = None,
    max_panel_pixels: int = 160_000_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create evenly spaced editable break transects covering a mask extent."""
    if max_panel_pixels <= 0:
        raise ValueError("max_panel_pixels must be positive.")
    with rasterio.open(mask_path) as src:
        bounds, crs = src.bounds, src.crs
        if rows is None or columns is None:
            auto_rows, auto_columns = _provisional_panel_grid(src.height, src.width, max_panel_pixels)
            rows = auto_rows if rows is None else rows
            columns = auto_columns if columns is None else columns
    if rows < 2 or columns < 2:
        raise ValueError("rows and columns must both be at least 2.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Panel-break file already exists: {output_path}")
    records: list[dict[str, object]] = []
    for index in range(1, columns):
        fraction = index / columns
        x = bounds.left + (bounds.right - bounds.left) * fraction
        records.append({"break_id": f"V{index}", "orientation": "vertical", "active": True,
                        "notes": "Provisional divider; move to a simple single-link reach.",
                        "geometry": LineString([(x, bounds.bottom), (x, bounds.top)])})
    for index in range(1, rows):
        fraction = index / rows
        y = bounds.bottom + (bounds.top - bounds.bottom) * fraction
        records.append({"break_id": f"H{index}", "orientation": "horizontal", "active": True,
                        "notes": "Provisional divider; move to a simple single-link reach.",
                        "geometry": LineString([(bounds.left, y), (bounds.right, y)])})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_file(output_path, layer=PANEL_BREAK_LAYER, driver="GPKG")
    return {"output": str(output_path), "layer": PANEL_BREAK_LAYER, "break_count": len(records), "rows": rows, "columns": columns, "max_panel_pixels": max_panel_pixels}


def _provisional_panel_grid(height: int, width: int, max_panel_pixels: int) -> tuple[int, int]:
    """Choose the smallest aspect-ratio-aware grid satisfying the pixel budget."""
    import math

    required = max(1, math.ceil((height * width) / max_panel_pixels))
    aspect = width / height
    columns = max(1, math.ceil(math.sqrt(required * aspect)))
    rows = max(1, math.ceil(required / columns))
    while math.ceil(height / rows) * math.ceil(width / columns) > max_panel_pixels:
        if (width / columns) >= (height / rows):
            columns += 1
        else:
            rows += 1
    # Panelization is invoked only after a full-mask failure. Keep a two-axis
    # provisional layout so the user receives movable V and H lines.
    return max(2, rows), max(2, columns)


def plan_panels(
    *, mask_path: Path, breaks_path: Path, output_path: Path, overlap_m: float = 5000, max_panel_pixels: int = 160_000_000, overwrite: bool = False
) -> dict[str, object]:
    """Derive irregular core and buffered processing polygons from edited breaks."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Panel plan already exists: {output_path}")
    with rasterio.open(mask_path) as src:
        bounds, crs, transform = src.bounds, src.crs, src.transform
    extent = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    breaks = gpd.read_file(breaks_path, layer=PANEL_BREAK_LAYER).to_crs(crs)
    active = breaks.loc[breaks["active"].astype(bool)].copy()
    lines = [geom.intersection(extent) for geom in active.geometry if geom is not None and not geom.is_empty]
    polygons = list(polygonize(unary_union([extent.boundary, *lines])))
    if len(polygons) < 2:
        raise ValueError("Active break lines do not partition the mask extent into multiple panels.")
    polygons.sort(key=lambda geom: (-geom.centroid.y, geom.centroid.x))
    pixel_area = abs(transform.a * transform.e)
    records = []
    for index, core in enumerate(polygons, start=1):
        processing = core.buffer(overlap_m).intersection(extent)
        estimated_pixels = int(round(processing.area / pixel_area))
        records.append({"panel_id": f"P{index:03d}", "estimated_processing_pixels": estimated_pixels,
                        "within_pixel_budget": estimated_pixels <= max_panel_pixels, "geometry": core})
    cores = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    processing = cores.copy()
    processing["geometry"] = [geom.buffer(overlap_m).intersection(extent) for geom in cores.geometry]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cores.to_file(output_path, layer="panel_cores", driver="GPKG")
    processing.to_file(output_path, layer="processing_areas", driver="GPKG")
    summary = {"mask": str(mask_path), "breaks": str(breaks_path), "output": str(output_path), "panel_count": len(cores), "overlap_m": overlap_m,
               "max_panel_pixels": max_panel_pixels, "over_budget_panel_ids": cores.loc[~cores["within_pixel_budget"], "panel_id"].tolist()}
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_panel_masks(*, mask_path: Path, plan_path: Path, output_dir: Path, panel_ids: set[str] | None = None, overwrite: bool = False, crop: bool = True) -> dict[str, object]:
    """Clip buffered processing areas to independently runnable binary masks."""
    cores = gpd.read_file(plan_path, layer="panel_cores")
    processing = gpd.read_file(plan_path, layer="processing_areas")
    selected = processing if panel_ids is None else processing.loc[processing["panel_id"].isin(panel_ids)]
    missing = set() if panel_ids is None else panel_ids.difference(set(processing["panel_id"]))
    if missing:
        raise ValueError(f"Unknown panel_id(s): {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with rasterio.open(mask_path) as src:
        for record in selected.itertuples(index=False):
            panel_dir = output_dir / record.panel_id
            panel_mask = panel_dir / "mask.tif"
            if panel_mask.exists() and not overwrite:
                raise FileExistsError(f"Panel mask already exists: {panel_mask}")
            geometry = record.geometry
            window = from_bounds(*geometry.bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(1, window=window)
            local_transform = src.window_transform(window)
            keep = rasterize([(geometry, 1)], out_shape=data.shape, transform=local_transform, fill=0, dtype="uint8").astype(bool)
            clipped = ((data > 0) & keep).astype("uint8")
            core = cores.loc[cores["panel_id"] == record.panel_id].geometry.iloc[0]
            core_keep = rasterize([(core, 1)], out_shape=data.shape, transform=local_transform, fill=0, dtype="uint8").astype(bool)
            core_water_pixels = int(((data > 0) & core_keep).sum())
            cropped, cropped_transform, crop_window = crop_binary_mask(clipped, local_transform) if crop else (clipped, local_transform, {"row_min": 0, "row_max": int(clipped.shape[0]), "col_min": 0, "col_max": int(clipped.shape[1])})
            profile = src.profile.copy()
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", height=cropped.shape[0], width=cropped.shape[1], transform=cropped_transform)
            panel_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(panel_mask, "w", **profile) as dst:
                dst.write(cropped, 1)
            summary = {"panel_id": record.panel_id, "mask": str(panel_mask), "source_mask": str(mask_path), "water_pixels": int(cropped.sum()),
                       "core_water_pixels": core_water_pixels,
                       "shape": [int(cropped.shape[0]), int(cropped.shape[1])], "crop_window": crop_window,
                       "core_bounds": list(core.bounds), "processing_bounds": list(geometry.bounds)}
            (panel_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            manifest.append(summary)
    manifest_path = output_dir / "panel_mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "manifest": str(manifest_path), "prepared_panel_ids": [item["panel_id"] for item in manifest]}


def write_panelized_mask_mosaic(
    *,
    template_mask_path: Path,
    plan_path: Path,
    panel_root: Path,
    output_mask_path: Path,
    overwrite: bool = False,
    chunk_pixels: int = 2048,
) -> Path:
    """Write a core-owned full mask from cropped, overlapping panel masks.

    Only modest chunks are held in memory. Overlap is resolved by assigning
    each output pixel to its panel core, so it never duplicates water from two
    buffered panel masks.
    """
    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive.")
    template_mask_path, plan_path, panel_root, output_mask_path = map(Path, (template_mask_path, plan_path, panel_root, output_mask_path))
    if output_mask_path.exists() and not overwrite:
        raise FileExistsError(f"Output mask already exists: {output_mask_path}")
    cores = gpd.read_file(plan_path, layer="panel_cores")
    with rasterio.open(template_mask_path) as template:
        profile = template.profile.copy()
        profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw")
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_mask_path, "w+", **profile) as output:
            for core_row in cores.itertuples(index=False):
                panel_path = panel_root / str(core_row.panel_id) / "mask.tif"
                if not panel_path.exists():
                    raise FileNotFoundError(f"Cleaned panel mask does not exist: {panel_path}")
                core_window = from_bounds(*core_row.geometry.bounds, transform=template.transform).round_offsets().round_lengths()
                core_window = core_window.intersection(rasterio.windows.Window(0, 0, template.width, template.height))
                with rasterio.open(panel_path) as panel:
                    for row in range(int(core_window.row_off), int(core_window.row_off + core_window.height), chunk_pixels):
                        height = min(chunk_pixels, int(core_window.row_off + core_window.height) - row)
                        for col in range(int(core_window.col_off), int(core_window.col_off + core_window.width), chunk_pixels):
                            width = min(chunk_pixels, int(core_window.col_off + core_window.width) - col)
                            window = rasterio.windows.Window(col, row, width, height)
                            bounds = rasterio.windows.bounds(window, template.transform)
                            panel_window = from_bounds(*bounds, transform=panel.transform).round_offsets().round_lengths()
                            values = panel.read(1, window=panel_window, out_shape=(height, width), boundless=True, fill_value=0)
                            core_mask = rasterize(
                                [(core_row.geometry, 1)], out_shape=(height, width),
                                transform=rasterio.windows.transform(window, template.transform), fill=0, dtype="uint8",
                            ).astype(bool)
                            existing = output.read(1, window=window)
                            existing[core_mask] = values[core_mask]
                            output.write(existing, 1, window=window)
    return output_mask_path


def match_panel_boundary_links(*, breaks_path: Path, left_links_path: Path, right_links_path: Path, break_id: str, left_panel: str, right_panel: str, output_path: Path, tolerance_m: float = 10, overwrite: bool = False) -> dict[str, object]:
    """Write conservative geometric link matches across one named panel break."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Match output already exists: {output_path}")
    breaks = gpd.read_file(breaks_path, layer=PANEL_BREAK_LAYER)
    boundary = breaks.loc[breaks["break_id"] == break_id, "geometry"]
    if len(boundary) != 1:
        raise ValueError(f"Expected exactly one break_id {break_id!r}.")
    boundary = boundary.iloc[0]
    left, right = gpd.read_file(left_links_path), gpd.read_file(right_links_path)
    left_id = "id_link" if "id_link" in left.columns else "id"
    right_id = "id_link" if "id_link" in right.columns else "id"
    left = left.loc[left.geometry.intersects(boundary.buffer(tolerance_m))]
    right = right.loc[right.geometry.intersects(boundary.buffer(tolerance_m))]
    records = []
    for l in left.itertuples(index=False):
        for r in right.itertuples(index=False):
            distance = float(l.geometry.hausdorff_distance(r.geometry))
            if distance > tolerance_m:
                continue
            records.append({"break_id": break_id, "left_panel": left_panel, "right_panel": right_panel,
                            "left_link_id": int(getattr(l, left_id)), "right_link_id": int(getattr(r, right_id)),
                            "hausdorff_m": distance, "status": "candidate", "geometry": l.geometry})
    matches = gpd.GeoDataFrame(records, geometry="geometry", crs=left.crs) if records else gpd.GeoDataFrame(
        {"break_id": [], "left_panel": [], "right_panel": [], "left_link_id": [], "right_link_id": [], "hausdorff_m": [], "status": []},
        geometry=gpd.GeoSeries([], crs=left.crs), crs=left.crs,
    )
    if not matches.empty:
        left_counts = matches.groupby("left_link_id").size()
        right_counts = matches.groupby("right_link_id").size()
        unique = matches["left_link_id"].map(left_counts).eq(1) & matches["right_link_id"].map(right_counts).eq(1)
        matches.loc[unique, "status"] = "accepted"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_file(output_path, layer="boundary_matches", driver="GPKG")
    return {"output": str(output_path), "candidate_count": len(matches), "accepted_count": int((matches["status"] == "accepted").sum()) if not matches.empty else 0}


def concatenate_matched_panel_links(*, left_links_path: Path, right_links_path: Path, matches_path: Path, plan_path: Path, output_path: Path, left_panel: str, right_panel: str, overwrite: bool = False) -> dict[str, object]:
    if output_path.exists() and not overwrite: raise FileExistsError(f"Merged links already exist: {output_path}")
    left, right = gpd.read_file(left_links_path), gpd.read_file(right_links_path)
    cores = gpd.read_file(plan_path, layer="panel_cores")
    matches = gpd.read_file(matches_path, layer="boundary_matches")
    accepted = matches.loc[matches["status"] == "accepted"]
    drop_right = set(accepted["right_link_id"].astype(int)); keep_left = set(accepted["left_link_id"].astype(int))
    left_core = cores.loc[cores["panel_id"] == left_panel, "geometry"].iloc[0]
    right_core = cores.loc[cores["panel_id"] == right_panel, "geometry"].iloc[0]
    left = left.loc[left.geometry.representative_point().apply(left_core.covers) | left["id_link"].astype(int).isin(keep_left)].copy()
    right = right.loc[right.geometry.representative_point().apply(right_core.covers) & ~right["id_link"].astype(int).isin(drop_right)].copy()
    left["source_panel"], right["source_panel"] = left_panel, right_panel
    left["source_link_id"], right["source_link_id"] = left["id_link"].astype(int), right["id_link"].astype(int)
    merged = gpd.GeoDataFrame(pd.concat([left, right], ignore_index=True), geometry="geometry", crs=left.crs)
    merged["global_link_id"] = range(len(merged))
    output_path.parent.mkdir(parents=True, exist_ok=True); merged.to_file(output_path, layer="merged_links", driver="GPKG")
    return {"output": str(output_path), "link_count": len(merged), "duplicates_removed": len(drop_right)}


def rebuild_pilot_nodes(*, merged_links_path: Path, output_dir: Path, overwrite: bool = False) -> dict[str, object]:
    """Rebuild global endpoint nodes and remap a concatenated pilot link layer."""
    links = gpd.read_file(merged_links_path, layer="merged_links")
    endpoints: list[Point] = []
    for geom in links.geometry:
        coords = list(geom.coords)
        endpoints.extend([Point(coords[0]), Point(coords[-1])])
    unique: list[Point] = []
    ids: list[int] = []
    for point in endpoints:
        match = next((i for i, existing in enumerate(unique) if point.distance(existing) <= 1e-6), None)
        if match is None:
            unique.append(point); match = len(unique) - 1
        ids.append(match)
    links = links.copy(); links["id_us_node"] = ids[0::2]; links["id_ds_node"] = ids[1::2]; links["id_nodes"] = [f"{a}, {b}" for a,b in zip(links.id_us_node, links.id_ds_node)]
    nodes = gpd.GeoDataFrame({"id_node": range(len(unique))}, geometry=unique, crs=links.crs)
    output_dir.mkdir(parents=True, exist_ok=True)
    links.to_file(output_dir / "merged_links.gpkg", layer="merged_links", driver="GPKG")
    nodes.to_file(output_dir / "merged_nodes.gpkg", layer="merged_nodes", driver="GPKG")
    degree = pd.concat([links.id_us_node, links.id_ds_node]).value_counts()
    report={"link_count":len(links),"node_count":len(nodes),"isolated_nodes":int((degree==0).sum()),"max_degree":int(degree.max())}
    (output_dir / "validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return {"output_dir":str(output_dir),**report}

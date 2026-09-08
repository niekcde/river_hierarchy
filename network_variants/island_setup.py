"""Create audited, reviewed-input island candidate sets for one experiment.

The island-only runner deliberately consumes a frozen candidate set rather
than detecting islands while it is creating network variants.  This module is
the inexpensive, reproducible preflight that creates that set: it retains the
largest 8-connected water body, finds enclosed 8-connected dry components,
crosswalks them to realised independent-unit fills, and records a deterministic
anchor selection for every runnable candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from scipy.ndimage import find_objects, label as nd_label
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union


@dataclass(frozen=True)
class IslandSetupResult:
    """Products and audit counts from one island-candidate preflight."""

    setup_dir: Path
    candidate_registry_path: Path | None
    candidate_geometries_path: Path | None
    manifest_path: Path | None
    equivalence_registry_path: Path | None
    candidate_registry: pd.DataFrame
    summary: dict[str, object]


def _one_path(directory: Path, pattern: str, description: str) -> Path:
    paths = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {description} matching {pattern!r} in {directory}, found {len(paths)}."
        )
    return paths[0]


def _unit_id_from_state(path: Path) -> int:
    match = re.fullmatch(r"S\d+_unit_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not an independent-unit state directory: {path}")
    return int(match.group(1))


def _parse_unit_ids(value: object) -> tuple[int, ...]:
    if pd.isna(value):
        return ()
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _candidate_anchor(unit_ids: tuple[int, ...], units: pd.DataFrame) -> int | None:
    if not unit_ids:
        return None
    candidates = units.loc[list(unit_ids)].reset_index(drop=True).sort_values(
        ["is_compound", "depth_from_root", "max_path_length", "unit_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return int(candidates.iloc[0]["unit_id"])


def _collect_current_unit_fills(states_dir: Path, crs: object) -> gpd.GeoDataFrame:
    records: list[dict[str, object]] = []
    # The regular independent writer stores edits under mask/, while older
    # panelized state handoffs publish the same canonical product under
    # summary/. Accept both locations, preferring mask/ when a state has both.
    candidates = list(states_dir.glob("S*_unit_*/variant/mask/collapse_edit_geometries.gpkg"))
    candidates.extend(states_dir.glob("S*_unit_*/variant/summary/collapse_edit_geometries.gpkg"))
    fill_paths_by_state: dict[Path, Path] = {}
    for candidate in candidates:
        state_dir = candidate.parents[2]
        existing = fill_paths_by_state.get(state_dir)
        if existing is None or candidate.parent.name == "mask":
            fill_paths_by_state[state_dir] = candidate
    for state_dir, fill_path in sorted(fill_paths_by_state.items(), key=lambda item: item[0].name):
        edits = gpd.read_file(fill_path)
        if "action" not in edits.columns:
            continue
        fills = edits.loc[
            edits["action"].astype(str).str.lower().isin(["add", "fill"])
            & edits.geometry.notna()
            & ~edits.geometry.is_empty
        ]
        for index, row in fills.iterrows():
            records.append(
                {
                    "unit_id": _unit_id_from_state(state_dir),
                    "state_id": state_dir.name,
                    "edit_index": int(index),
                    "geometry": row.geometry,
                }
            )
    if not records:
        return gpd.GeoDataFrame(
            {"unit_id": pd.Series(dtype="int64"), "state_id": pd.Series(dtype="str"), "edit_index": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


class _ComponentUnionFind:
    """Small union-find metadata table for window-labelled raster components."""

    def __init__(self) -> None:
        self.parent = [0]
        self.size = [0]
        self.edge = [False]

    def add(self, size: int, *, edge: bool = False) -> int:
        value = len(self.parent)
        self.parent.append(value)
        self.size.append(int(size))
        self.edge.append(bool(edge))
        return value

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> int:
        first, second = self.find(int(first)), self.find(int(second))
        if first == second:
            return first
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        self.edge[first] = self.edge[first] or self.edge[second]
        return first


def _union_touching_labels(
    components: _ComponentUnionFind,
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    """Union positive IDs that touch across one 8-connected tile seam."""
    valid = (first > 0) & (second > 0)
    for left, right in zip(first[valid].tolist(), second[valid].tolist(), strict=False):
        components.union(int(left), int(right))


def _tile_windows(source: rasterio.io.DatasetReader, tile_size: int = 1024):
    for row_off in range(0, source.height, tile_size):
        height = min(tile_size, source.height - row_off)
        for col_off in range(0, source.width, tile_size):
            width = min(tile_size, source.width - col_off)
            yield int(row_off), int(col_off), Window(col_off, row_off, width, height)


def _window_component_labels(
    mask_path: Path,
    *,
    wet: bool,
    water_lookup: dict[tuple[int, int], np.ndarray] | None = None,
    water_components: _ComponentUnionFind | None = None,
    main_water_root: int | None = None,
    retain_lookup: bool = True,
) -> tuple[_ComponentUnionFind, dict[tuple[int, int], np.ndarray], int]:
    """Label a raster predicate with bounded arrays and 8-connected seams."""
    structure = np.ones((3, 3), dtype=np.uint8)
    components = _ComponentUnionFind()
    lookup: dict[tuple[int, int], np.ndarray] = {}
    n_local_components = 0
    with rasterio.open(mask_path) as source:
        previous_bottom = np.zeros(source.width, dtype=np.int64)
        for row_off, col_off, window in _tile_windows(source):
            water = source.read(1, window=window) > 0
            if wet:
                target = water
            else:
                if water_lookup is None or water_components is None or main_water_root is None:
                    raise ValueError("Dry component pass requires water-component lookups.")
                water_labels, _ = nd_label(water, structure=structure)
                local_to_global = water_lookup[(row_off, col_off)]
                water_ids = local_to_global[water_labels]
                roots = np.zeros_like(water_ids, dtype=np.int64)
                for value in np.unique(water_ids):
                    if value:
                        roots[water_ids == value] = water_components.find(int(value))
                target = roots != int(main_water_root)
            labels, n_labels = nd_label(target, structure=structure)
            n_local_components += int(n_labels)
            counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
            local_to_global = np.zeros(n_labels + 1, dtype=np.int64)
            for label_value in range(1, n_labels + 1):
                touches_edge = bool(
                    (row_off == 0 and np.any(labels[0] == label_value))
                    or (col_off == 0 and np.any(labels[:, 0] == label_value))
                    or (row_off + window.height == source.height and np.any(labels[-1] == label_value))
                    or (col_off + window.width == source.width and np.any(labels[:, -1] == label_value))
                )
                local_to_global[label_value] = components.add(int(counts[label_value]), edge=touches_edge)
            ids = local_to_global[labels]
            if col_off:
                left = previous_right
                current = ids[:, 0]
                _union_touching_labels(components, left, current)
                _union_touching_labels(components, left[:-1], current[1:])
                _union_touching_labels(components, left[1:], current[:-1])
            if row_off:
                above = previous_bottom[col_off : col_off + int(window.width)]
                top = ids[0]
                _union_touching_labels(components, above, top)
                _union_touching_labels(components, above[:-1], top[1:])
                _union_touching_labels(components, above[1:], top[:-1])
            previous_bottom[col_off : col_off + int(window.width)] = ids[-1]
            previous_right = ids[:, -1]
            if retain_lookup:
                lookup[(row_off, col_off)] = local_to_global
    return components, lookup, n_local_components


def _windowed_enclosed_islands(mask_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Extract enclosed islands without ever allocating a full-extent array.

    This is used for panelized parents, whose masks are deliberately too large
    for the otherwise convenient in-memory connected-component calculation.
    Component connectivity remains exactly 8-connected for both water and dry
    land, including diagonal joins at tile boundaries.
    """
    water_components, water_lookup, n_water_local = _window_component_labels(mask_path, wet=True)
    roots = [value for value in range(1, len(water_components.parent)) if water_components.find(value) == value]
    if not roots:
        raise ValueError(f"Base mask contains no wet pixels: {mask_path}")
    main_water_root = max(roots, key=lambda value: water_components.size[value])
    dry_components, dry_lookup, n_dry_local = _window_component_labels(
        mask_path,
        wet=False,
        water_lookup=water_lookup,
        water_components=water_components,
        main_water_root=main_water_root,
    )
    dry_roots = [value for value in range(1, len(dry_components.parent)) if dry_components.find(value) == value]
    enclosed_roots = {value for value in dry_roots if not dry_components.edge[value]}
    with rasterio.open(mask_path) as source:
        crs, transform = source.crs, source.transform
        pixel_area_m2 = abs(transform.a * transform.e)
        fragments: dict[int, list[object]] = {value: [] for value in enclosed_roots}
        structure = np.ones((3, 3), dtype=np.uint8)
        for row_off, col_off, window in _tile_windows(source):
            water = source.read(1, window=window) > 0
            water_labels, _ = nd_label(water, structure=structure)
            water_ids = water_lookup[(row_off, col_off)][water_labels]
            roots_array = np.zeros_like(water_ids, dtype=np.int64)
            for value in np.unique(water_ids):
                if value:
                    roots_array[water_ids == value] = water_components.find(int(value))
            dry_labels, _ = nd_label(roots_array != main_water_root, structure=structure)
            dry_ids = dry_lookup[(row_off, col_off)][dry_labels]
            root_labels = np.zeros_like(dry_ids, dtype=np.int32)
            for value in np.unique(dry_ids):
                if value:
                    root = dry_components.find(int(value))
                    if root in enclosed_roots:
                        root_labels[dry_ids == value] = root
            mask = root_labels > 0
            if not mask.any():
                continue
            for mapping, label_value in shapes(
                root_labels,
                mask=mask,
                transform=source.window_transform(window),
                connectivity=8,
            ):
                fragments[int(label_value)].append(shapely_shape(mapping))
    rows: list[dict[str, object]] = []
    for component_id, root in enumerate(sorted(enclosed_roots), start=1):
        geometry = unary_union(fragments.get(root, []))
        if geometry.is_empty:
            continue
        minx, miny, maxx, maxy = geometry.bounds
        col_min_f, row_max_f = ~transform * (minx, miny)
        col_max_f, row_min_f = ~transform * (maxx, maxy)
        rows.append({
            "island_component_id": component_id,
            "n_pixels": int(dry_components.size[root]),
            "area_m2": float(dry_components.size[root] * pixel_area_m2),
            "area_km2": float(dry_components.size[root] * pixel_area_m2 / 1e6),
            "touches_raster_edge": False,
            "is_provisional_island": True,
            "row_min": int(np.floor(row_min_f)),
            "row_max": int(np.ceil(row_max_f) - 1),
            "col_min": int(np.floor(col_min_f)),
            "col_max": int(np.ceil(col_max_f) - 1),
            "geometry": geometry,
        })
    islands = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return islands, {
        "n_water_components": int(len(roots)),
        "main_water_pixels": int(water_components.size[main_water_root]),
        "n_dry_components": int(len(dry_roots)),
        "n_water_local_components": int(n_water_local),
        "n_dry_local_components": int(n_dry_local),
    }


def _in_memory_enclosed_islands(mask_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Original compact implementation, retained for normal-sized masks."""
    with rasterio.open(mask_path) as source:
        water = source.read(1) > 0
        transform, crs = source.transform, source.crs
        pixel_area_m2 = abs(transform.a * transform.e)
    if not water.any():
        raise ValueError(f"Base mask contains no wet pixels: {mask_path}")
    water_labels, n_water_components = nd_label(water, structure=np.ones((3, 3), dtype=np.uint8))
    water_pixels = np.bincount(water_labels.ravel())
    main_water_label = int(np.argmax(water_pixels[1:]) + 1)
    main_water = water_labels == main_water_label
    island_labels, n_dry_components = nd_label(~main_water, structure=np.ones((3, 3), dtype=np.uint8))
    island_pixels = np.bincount(island_labels.ravel())
    edge_labels = {int(value) for value in np.unique(np.concatenate(
        [island_labels[0], island_labels[-1], island_labels[:, 0], island_labels[:, -1]]
    )) if value > 0}
    component_rows: list[dict[str, object]] = []
    for component_id, component_slice in enumerate(find_objects(island_labels), start=1):
        if component_slice is None:
            continue
        n_pixels = int(island_pixels[component_id])
        component_rows.append({
            "island_component_id": component_id, "n_pixels": n_pixels,
            "area_m2": n_pixels * pixel_area_m2, "area_km2": n_pixels * pixel_area_m2 / 1e6,
            "touches_raster_edge": component_id in edge_labels,
            "is_provisional_island": component_id not in edge_labels,
            "row_min": component_slice[0].start, "row_max": component_slice[0].stop - 1,
            "col_min": component_slice[1].start, "col_max": component_slice[1].stop - 1,
        })
    components = pd.DataFrame(component_rows)
    provisional = components.loc[components["is_provisional_island"]].copy()
    provisional_mask = (island_labels > 0) & ~np.isin(island_labels, list(edge_labels))
    features = [
        {"island_component_id": int(label_value), "geometry": shapely_shape(mapping)}
        for mapping, label_value in shapes(island_labels.astype(np.int32), mask=provisional_mask,
                                           transform=transform, connectivity=8)
    ]
    island_geometries = gpd.GeoDataFrame(features, geometry="geometry", crs=crs) if features else gpd.GeoDataFrame(
        {"island_component_id": pd.Series(dtype="int64")}, geometry=gpd.GeoSeries([], crs=crs), crs=crs
    )
    islands = island_geometries.merge(provisional.drop(columns=["is_provisional_island"]),
                                      on="island_component_id", how="left", validate="1:1")
    return islands, {
        "n_water_components": int(n_water_components), "main_water_pixels": int(main_water.sum()),
        "n_dry_components": int(n_dry_components),
    }


def _complete_existing_setup_equivalence(
    destination: Path,
    experiment: Path,
    *,
    operation: str = "completed_existing_candidate_setup",
) -> IslandSetupResult:
    """Append the exact equivalence audit to an existing candidate-only setup.

    This is intentionally narrow: it preserves the reviewed candidate registry
    and geometries byte-for-byte, writes only the missing small equivalence
    table, and annotates the existing manifest with its result.
    """
    registry_path = destination / "island_candidate_registry.csv"
    geometries_path = destination / "island_candidate_geometries.gpkg"
    manifest_path = destination / "island_experiment_manifest.json"
    equivalence_path = destination / "island_equivalence_registry.csv"
    required = [registry_path, geometries_path, manifest_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(
            "Island setup destination is non-empty but not a complete candidate setup; "
            f"refusing to modify it: {destination}. Missing: {missing}"
        )
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_experiment = summary.get("experiment_dir")
    if recorded_experiment and Path(str(recorded_experiment)).expanduser().resolve() != experiment:
        raise ValueError(
            "Existing island setup belongs to a different independent experiment: "
            f"{recorded_experiment} != {experiment}"
        )
    if equivalence_path.exists():
        reused_summary = dict(summary)
        reused_summary["setup_operation"] = "reused_existing_setup_and_equivalence"
        return IslandSetupResult(
            destination,
            registry_path,
            geometries_path,
            manifest_path,
            equivalence_path,
            pd.read_csv(registry_path, low_memory=False),
            reused_summary,
        )
    from network_variants.island_experiment import build_island_equivalence_registry

    equivalence = build_island_equivalence_registry(
        setup_dir=destination,
        experiment_dir=experiment,
        output_path=equivalence_path,
    )
    summary.update(
        {
            "equivalence_check": "completed",
            "equivalence_registry": str(equivalence_path),
            "n_alias_existing_unit_state": int(
                equivalence["equivalence_status"].eq("alias_existing_unit_state").sum()
            ),
            "n_requires_island_state": int(
                equivalence["equivalence_status"].eq("requires_full_island_state").sum()
            ),
            "setup_operation": operation,
        }
    )
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return IslandSetupResult(
        destination,
        registry_path,
        geometries_path,
        manifest_path,
        equivalence_path,
        pd.read_csv(registry_path, low_memory=False),
        summary,
    )


def create_reviewed_island_setup(
    experiment_dir: str | Path,
    *,
    setup_dir: str | Path | None = None,
    tiny_island_pixels: int = 9,
    nearby_link_buffer_m: float = 500.0,
    dry_run: bool = False,
) -> IslandSetupResult:
    """Create the reviewable 8-connected island setup for an independent run.

    The operation is intentionally limited to small setup artefacts; it does
    not make masks, call RivGraph, or create island network states.  Components
    at or below ``tiny_island_pixels`` are retained in the audit but excluded
    from automatic approval. A non-tiny component without a realised
    independent-unit fill is also retained, marked ``requires_anchor_review``,
    and never added to the approved list.
    """
    experiment = Path(experiment_dir).expanduser().resolve()
    states_dir = experiment / "states"
    base = states_dir / "S000_base"
    variant = base / "variant"
    if not base.is_dir():
        raise FileNotFoundError(f"Independent base state is missing: {base}")
    destination = (
        Path(setup_dir).expanduser().resolve()
        if setup_dir is not None
        else experiment / "island_only_setup_8connected"
    )
    if destination.exists() and any(destination.iterdir()) and not dry_run:
        return _complete_existing_setup_equivalence(destination, experiment)
    mask_path = _one_path(variant / "mask", "*_collapsed.tif", "base collapsed mask")
    links_path = _one_path(variant / "directed", "*directed_links.gpkg", "base directed links")
    units_path = base / "hierarchy" / "unit_summary.csv"
    if not units_path.is_file():
        raise FileNotFoundError(f"Base unit summary is missing: {units_path}")

    parent_manifest_path = experiment / "experiment_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text()) if parent_manifest_path.exists() else {}
    is_panelized_parent = (
        str(dict(parent_manifest.get("execution_backends") or {}).get("base_backend", ""))
        == "panelized"
    )
    islands, component_summary = (
        _windowed_enclosed_islands(mask_path)
        if is_panelized_parent else _in_memory_enclosed_islands(mask_path)
    )
    crs = islands.crs
    example_id = experiment.name.split("_")[0]
    islands["island_id"] = [
        f"island_{example_id}_{int(component):03d}"
        for component in islands["island_component_id"]
    ]
    islands = islands.sort_values("area_m2", ascending=False, kind="mergesort").reset_index(drop=True)

    links = gpd.read_file(links_path).to_crs(crs)
    if "id_link" not in links.columns:
        raise ValueError(f"Base directed links lacks id_link: {links_path}")
    fills = _collect_current_unit_fills(states_dir, crs)
    units = pd.read_csv(units_path).set_index("unit_id", drop=False)
    required_unit_columns = {
        "unit_id", "bifurcation", "confluence", "is_compound", "depth_from_root", "max_path_length"
    }
    missing_unit_columns = sorted(required_unit_columns - set(units.columns))
    if missing_unit_columns:
        raise ValueError(f"Base unit summary is missing columns: {missing_unit_columns}")

    crosswalk_rows: list[dict[str, object]] = []
    for island in islands.itertuples(index=False):
        matching = fills.loc[fills.intersects(island.geometry)].copy() if not fills.empty else fills.copy()
        if not matching.empty:
            matching["overlap_m2"] = matching.geometry.intersection(island.geometry).area
            matching = matching.loc[matching["overlap_m2"] > 0].copy()
            matching["overlap_fraction_of_island"] = matching["overlap_m2"] / island.geometry.area
        nearby = links.loc[links.intersects(island.geometry.buffer(nearby_link_buffer_m))]
        unit_ids = tuple(sorted(set(matching["unit_id"].astype(int)))) if not matching.empty else ()
        crosswalk_rows.append(
            {
                "island_id": island.island_id,
                "island_component_id": int(island.island_component_id),
                "area_km2": float(island.area_km2),
                "n_pixels": int(island.n_pixels),
                "current_unit_ids": ", ".join(map(str, unit_ids)),
                "n_matching_unit_fills": int(len(matching)),
                "max_fill_overlap_fraction": (
                    float(matching["overlap_fraction_of_island"].max()) if not matching.empty else 0.0
                ),
                "nearby_link_ids": ", ".join(map(str, sorted(nearby["id_link"].astype(int).tolist()))),
                "n_nearby_links": int(len(nearby)),
            }
        )
    crosswalk = pd.DataFrame(crosswalk_rows)
    registry = islands.drop(columns="geometry").merge(
        crosswalk, on=["island_id", "island_component_id", "area_km2", "n_pixels"], how="left", validate="1:1"
    )
    registry["source_unit_ids"] = registry["current_unit_ids"].map(_parse_unit_ids)
    registry["proposed_anchor_unit_id"] = pd.array(
        registry["source_unit_ids"].map(lambda value: _candidate_anchor(value, units)),
        dtype="Int64",
    )
    anchor_columns = units[["unit_id", "bifurcation", "confluence", "is_compound", "depth_from_root"]].rename(
        columns={
            "unit_id": "proposed_anchor_unit_id",
            "bifurcation": "anchor_bifurcation",
            "confluence": "anchor_confluence",
            "is_compound": "anchor_unit_is_compound",
            "depth_from_root": "anchor_unit_depth",
        }
    )
    anchor_columns["proposed_anchor_unit_id"] = pd.array(
        anchor_columns["proposed_anchor_unit_id"], dtype="Int64"
    )
    registry = registry.merge(anchor_columns, on="proposed_anchor_unit_id", how="left", validate="m:1")
    registry["selection_status"] = np.select(
        [
            registry["n_pixels"].le(tiny_island_pixels),
            registry["proposed_anchor_unit_id"].isna(),
        ],
        ["exclude_raster_artifact", "requires_anchor_review"],
        default="approved_pilot",
    )
    registry["anchor_rule"] = "noncompound_then_deeper_then_shorter_max_path"
    registry = registry.drop(columns="source_unit_ids").sort_values(
        "area_km2", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    geometries = islands[["island_id", "island_component_id", "n_pixels", "area_km2", "geometry"]].merge(
        registry.drop(columns=["area_km2", "n_pixels"]),
        on=["island_id", "island_component_id"],
        how="left",
        validate="1:1",
    )
    geometries = gpd.GeoDataFrame(geometries, geometry="geometry", crs=crs)

    approved_ids = registry.loc[registry["selection_status"].eq("approved_pilot"), "island_id"].astype(str).tolist()
    summary: dict[str, object] = {
        "schema_version": "island-reviewed-setup-v1",
        "example_id": example_id,
        "experiment_dir": str(experiment),
        "setup_dir": str(destination),
        "base_mask": str(mask_path),
        "base_directed_links": str(links_path),
        "candidate_definition": "enclosed dry 8-connected component within largest 8-connected water body",
        "tiny_island_pixels": int(tiny_island_pixels),
        "nearby_link_buffer_m": float(nearby_link_buffer_m),
        "component_detection_backend": "windowed_8connected" if is_panelized_parent else "in_memory_8connected",
        **component_summary,
        "n_enclosed_islands": int(len(registry)),
        "n_tiny_excluded": int(registry["selection_status"].eq("exclude_raster_artifact").sum()),
        "n_requires_anchor_review": int(registry["selection_status"].eq("requires_anchor_review").sum()),
        "n_approved_islands": int(len(approved_ids)),
        "approved_island_ids": approved_ids,
        "excluded_island_ids": registry.loc[registry["selection_status"].eq("exclude_raster_artifact"), "island_id"].astype(str).tolist(),
        "requires_anchor_review_island_ids": registry.loc[registry["selection_status"].eq("requires_anchor_review"), "island_id"].astype(str).tolist(),
        "anchor_rule": "noncompound_then_deeper_then_shorter_max_path",
    }

    if dry_run:
        summary["equivalence_check"] = "not_computed_dry_run"
        return IslandSetupResult(destination, None, None, None, None, registry, summary)
    destination.mkdir(parents=True, exist_ok=True)
    registry_path = destination / "island_candidate_registry.csv"
    geometries_path = destination / "island_candidate_geometries.gpkg"
    manifest_path = destination / "island_experiment_manifest.json"
    equivalence_path = destination / "island_equivalence_registry.csv"
    registry.to_csv(registry_path, index=False)
    geometries.to_file(geometries_path, driver="GPKG")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _complete_existing_setup_equivalence(
        destination,
        experiment,
        operation="created_candidate_setup_and_equivalence",
    )

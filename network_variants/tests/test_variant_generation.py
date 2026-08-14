from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import StructuralUnit
from network_variants.variant_generation import (
    _apply_collapsed_selection_metadata,
    _apply_collapse_components_to_mask,
    _apply_segmented_stub_width_inheritance,
    _assign_variant_directions,
    _boundary_overlap_cut_targets,
    _build_oriented_child_graph,
    _componentize_units,
    _compute_family_stats,
    _compute_changed_link_width_products,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _normalize_width_boolean_fields,
    _repair_invalid_directionality,
    _repair_unresolved_link_components,
    _resolve_link_by_progression,
    _resolve_link_lineage,
    _resolve_single_remaining_link,
    _run_rivgraph,
    _segment_collapsed_links_to_anchor_nodes,
    _trim_slice,
    compute_width_families,
)
from network_variants.width_two_bank import (
    BoundaryComponentIndex,
    _find_local_controlling_pair,
    _local_tangent_normal,
    _select_local_controlling_pair,
    build_state_bank_labels,
)
from hierarchy_level_definition.graph_building.directed_network_checks import (
    validate_single_inlet_single_outlet,
)


def _unit(unit_id: int, node_ids: set[int], edge_keys: set[tuple[int, int, int]]) -> StructuralUnit:
    return StructuralUnit(
        unit_id=unit_id,
        bifurcation=min(node_ids),
        confluence=max(node_ids),
        n_paths=2,
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=1.0,
        max_path_length=1.0,
        node_set=node_ids,
        edge_set=edge_keys,
    )


def test_variant_rivgraph_centerline_uses_metadata_compatibility(tmp_path, monkeypatch) -> None:
    import rivgraph.ln_utils as lnu
    import rivgraph.rivers.river_utils as ru
    import network_variants.variant_generation as variant_generation

    original_width_runner = lnu.link_widths_and_lengths
    observed = {"compat_active": False}

    class FakeRiver:
        def __init__(self, *_args, **_kwargs) -> None:
            self.links = {"id": [0], "idx": [[1]], "conn": [[0, 1]]}
            self.nodes = {"id": [0, 1], "idx": [1, 2], "conn": [[0], [0]]}
            self.exit_sides = "we"
            self.Iskel = np.ones((2, 2), dtype=bool)
            self.paths = {}

        def compute_network(self) -> None:
            pass

        def finalize_ids(self) -> None:
            pass

        def compute_distance_transform(self) -> None:
            pass

        def compute_link_width_and_length(self) -> None:
            pass

        def compute_centerline(self) -> None:
            observed["compat_active"] = lnu.link_widths_and_lengths is not original_width_runner

        def to_geotiff(self, *_args, **_kwargs) -> None:
            pass

        def to_geovectors(self, *_args, **_kwargs) -> None:
            pass

        def save_network(self) -> None:
            pass

    monkeypatch.setattr(variant_generation, "river", FakeRiver)
    monkeypatch.setattr(ru, "prune_river", lambda links, nodes, *_args: (links, nodes))

    _run_rivgraph(
        run_name="test",
        collapsed_mask_path=tmp_path / "mask.tif",
        rivgraph_output_dir=tmp_path / "rivgraph",
        exit_sides="WE",
        export_sword=False,
    )

    assert observed["compat_active"] is True
    assert lnu.link_widths_and_lengths is original_width_runner


def test_componentize_units_groups_by_shared_nodes() -> None:
    units_by_id = {
        1: _unit(1, {1, 2, 3}, {(1, 3, 10)}),
        2: _unit(2, {3, 4, 5}, {(3, 5, 11)}),
        3: _unit(3, {9, 10}, {(9, 10, 12)}),
    }

    components = _componentize_units([1, 2, 3], units_by_id)

    assert [component["unit_ids"] for component in components] == [[1, 2], [3]]
    assert components[0]["link_ids"] == [10, 11]
    assert components[1]["link_ids"] == [12]


def test_normalize_width_boolean_fields_preserves_text_identifiers() -> None:
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2],
            "collapsed_unit_ids": ["1", "1"],
            "width_correction_applied": pd.Series([True, False], dtype=object),
            "dry_corridor_meaningful": ["False", "True"],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
        },
        crs="EPSG:3857",
    )

    normalized = _normalize_width_boolean_fields(links)

    assert normalized["width_correction_applied"].dtype == bool
    assert normalized["dry_corridor_meaningful"].dtype == bool
    assert normalized["width_correction_applied"].tolist() == [True, False]
    assert normalized["dry_corridor_meaningful"].tolist() == [False, True]
    assert normalized["collapsed_unit_ids"].tolist() == ["1", "1"]


def test_changed_link_width_products_measure_only_non_inherited_links() -> None:
    parent_links = gpd.GeoDataFrame(
        {
            "id_link": [10, 20],
            "wid_adj_wet": [100.0, 200.0],
            "wid_adj_routing": [110.0, 210.0],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 1), (1, 1)]),
            ],
        },
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        {
            "id_link": [110, 220],
            "geometry": [
                LineString([(1, 0), (0, 0)]),
                LineString([(0, 1), (2, 1)]),
            ],
        },
        crs=parent_links.crs,
    )
    lineage = pd.DataFrame(
        {
            "id_link": [110, 220],
            "lineage_type": ["unchanged_1to1", "changed_merged"],
            "dominant_parent_link_id": [10, 20],
        }
    )
    parent_samples = pd.DataFrame(
        {
            "id_link": [10, 10, 20],
            "sample_index": [0, 1, 0],
            "width_wet": [95.0, 105.0, 200.0],
        }
    )
    measured_ids: list[int] = []

    def fake_width_runner(links: gpd.GeoDataFrame, **_: object):
        measured_ids.extend(links["id_link"].astype(int).tolist())
        enriched = links.copy()
        enriched["wid_adj_wet"] = 250.0
        enriched["wid_adj_routing"] = np.nan
        samples = pd.DataFrame(
            {"id_link": [220], "sample_index": [0], "width_wet": [250.0]}
        )
        return enriched, samples

    links, samples, changed_ids, summary = _compute_changed_link_width_products(
        child_links,
        parent_links=parent_links,
        parent_width_samples=parent_samples,
        link_lineage=lineage,
        width_runner=fake_width_runner,
    )

    assert measured_ids == [220]
    assert changed_ids == [220]
    by_id = links.set_index("id_link")
    assert by_id.loc[110, "wid_adj_wet"] == 100.0
    assert by_id.loc[110, "wid_adj_routing"] == 110.0
    assert by_id.loc[110, "width_inheritance_status"] == "inherited_unchanged_1to1"
    assert by_id.loc[220, "wid_adj_wet"] == 250.0
    assert by_id.loc[220, "width_inheritance_status"] == "changed_requires_evaluation"
    assert samples.loc[samples["id_link"].eq(110), "width_wet"].tolist() == [95.0, 105.0]
    assert samples.loc[samples["id_link"].eq(220), "width_wet"].tolist() == [250.0]
    assert summary["measurement_scope"] == "changed_links_only_full_raster_context"
    assert summary["measured_link_count"] == 1
    assert summary["skipped_measurement_link_count"] == 1


def test_trim_slice_matches_rivgraph_style_end_trimming() -> None:
    widths = np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=float)
    distances = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0, 15.0], dtype=float)

    trim = _trim_slice(widths, distances)

    assert trim.start == 1
    assert trim.stop == 4


def test_compute_family_stats_uses_trimmed_values() -> None:
    values = np.asarray([1.0, 2.0, 10.0, 20.0], dtype=float)
    trim = slice(1, 3)

    stats = _compute_family_stats(values, trim)

    assert np.isclose(stats["adj"], 6.0)
    assert np.isclose(stats["med"], 6.0)
    assert np.isclose(stats["p50"], 6.0)


def test_compute_width_families_base_channel_preserves_single_thread_width(tmp_path: Path) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 6:14] = 1

    mask_path = tmp_path / "mask.tif"
    transform = from_origin(0, 20, 1, 1)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    rows = np.arange(2, 18, dtype=int)
    cols = np.full_like(rows, 9)
    idx_link = ",".join(str(int(index)) for index in np.ravel_multi_index((rows, cols), mask.shape))
    wid_pix = ",".join("8.0" for _ in rows)
    geometry = LineString([(9.5, 17.5), (9.5, 2.5)])

    links = gpd.GeoDataFrame(
        [{"id_link": 1, "idx_link": idx_link, "wid_pix": wid_pix, "wid_adj": 8.0, "geometry": geometry}],
        geometry="geometry",
        crs="EPSG:3857",
    )

    enriched, samples = compute_width_families(
        links,
        collapsed_mask_path=mask_path,
        wet_reference_mask_path=mask_path,
    )

    assert enriched.loc[0, "width_method"] == "two_bank_dt"
    assert np.isclose(float(enriched.loc[0, "wid_adj_total"]), float(enriched.loc[0, "wid_adj_wet"]), rtol=0.15)
    assert float(enriched.loc[0, "wid_adj_dry"]) < 1.0
    assert int(enriched.loc[0, "n_wet_threads_max"]) == 1
    assert int(enriched.loc[0, "n_channels_max"]) == 1
    assert np.isclose(float(enriched.loc[0, "qc_mean_opposition_cos_trimmed"]), -1.0, atol=1e-6)
    assert np.isclose(float(enriched.loc[0, "qc_min_opposition_cos"]), -1.0, atol=1e-6)
    assert set(samples["width_method"]) == {"two_bank_dt"}
    assert int(samples["n_channels"].max()) == 1


def test_compute_width_families_collapsed_fill_separates_wet_and_dry_width(tmp_path: Path) -> None:
    collapsed_mask = np.zeros((20, 20), dtype=np.uint8)
    collapsed_mask[:, 6:14] = 1
    wet_mask = np.zeros((20, 20), dtype=np.uint8)
    wet_mask[:, 6:8] = 1
    wet_mask[:, 12:14] = 1

    transform = from_origin(0, 20, 1, 1)
    collapsed_mask_path = tmp_path / "collapsed_mask.tif"
    wet_mask_path = tmp_path / "wet_mask.tif"
    for path, array in ((collapsed_mask_path, collapsed_mask), (wet_mask_path, wet_mask)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            dst.write(array, 1)

    rows = np.arange(2, 18, dtype=int)
    cols = np.full_like(rows, 9)
    idx_link = ",".join(str(int(index)) for index in np.ravel_multi_index((rows, cols), collapsed_mask.shape))
    wid_pix = ",".join("8.0" for _ in rows)
    geometry = LineString([(9.5, 17.5), (9.5, 2.5)])

    links = gpd.GeoDataFrame(
        [{"id_link": 1, "idx_link": idx_link, "wid_pix": wid_pix, "wid_adj": 8.0, "geometry": geometry}],
        geometry="geometry",
        crs="EPSG:3857",
    )

    enriched, samples = compute_width_families(
        links,
        collapsed_mask_path=collapsed_mask_path,
        wet_reference_mask_path=wet_mask_path,
    )

    assert float(enriched.loc[0, "wid_adj_total"]) > float(enriched.loc[0, "wid_adj_wet"])
    assert float(enriched.loc[0, "wid_adj_dry"]) > 0
    assert int(enriched.loc[0, "n_wet_threads_max"]) == 2
    assert int(enriched.loc[0, "n_channels_max"]) == 2
    assert samples["left_component_id"].notna().any()
    assert samples["right_component_id"].notna().any()
    assert (samples["width_dry"] > 0).any()


def test_build_state_bank_labels_turns_selected_fill_land_into_water() -> None:
    boundary_labels = np.full((8, 10), 2, dtype=np.int32)
    boundary_labels[:, 5:] = 3
    boundary_labels[:, 3:7] = 1
    boundary_labels[:, 4:6] = 12

    origin_mask = np.zeros_like(boundary_labels, dtype=np.uint16)
    origin_mask[:, 3] = 1
    origin_mask[:, 6] = 1
    origin_mask[:, 4:6] = 101

    bank_labels, removed_labels = build_state_bank_labels(boundary_labels, origin_mask)

    assert removed_labels == [12]
    assert np.all(bank_labels[:, 4:6] == 1)
    assert int(bank_labels[0, 0]) == 2
    assert int(bank_labels[0, -1]) == 3


def test_find_local_controlling_pair_expands_candidate_neighbors() -> None:
    left_xs = np.asarray([-float(i) for i in range(1, 151)], dtype=float)
    left_ys = np.zeros_like(left_xs)
    right_xs = np.asarray([120.0], dtype=float)
    right_ys = np.asarray([0.0], dtype=float)
    xs = np.concatenate([left_xs, right_xs])
    ys = np.concatenate([left_ys, right_ys])
    component_ids = np.concatenate([np.full(len(left_xs), 3, dtype=int), np.asarray([2], dtype=int)])
    index = BoundaryComponentIndex(
        tree=cKDTree(np.column_stack([xs, ys])),
        xs=xs,
        ys=ys,
        component_ids=component_ids,
        n_components=2,
    )
    center_xy = np.asarray([0.0, 0.0], dtype=float)
    normal = np.asarray([1.0, 0.0], dtype=float)

    no_pair = _find_local_controlling_pair(
        center_xy,
        normal,
        boundary_index=index,
        search_radius=500.0,
        candidate_neighbors_max=96,
    )
    assert no_pair is None

    pair = _find_local_controlling_pair(
        center_xy,
        normal,
        boundary_index=index,
        search_radius=500.0,
        candidate_neighbors_max=256,
    )
    assert pair is not None
    assert int(pair["left_component_id"]) == 3
    assert int(pair["right_component_id"]) == 2


def test_local_tangent_normal_uses_window_to_reduce_single_pixel_kink() -> None:
    xs = np.asarray([0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

    tangent, normal = _local_tangent_normal(xs, ys, 3)

    assert tangent[0] > 0.9
    assert abs(tangent[1]) < 0.2
    assert abs(float(np.dot(tangent, normal))) < 1e-9


def test_select_local_controlling_pair_skips_crossing_bank_ray() -> None:
    center_xy = np.asarray([0.0, 0.0], dtype=float)
    normal = np.asarray([0.0, 1.0], dtype=float)
    centerline = LineString([(-2.0, 0.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    centerline_without_center = centerline.difference(Point(0.0, 0.0).buffer(0.51))
    candidates = [
        (9, float(np.linalg.norm(np.asarray([-1.2, 1.2]))), np.asarray([-1.2, 1.2], dtype=float)),
        (9, float(np.linalg.norm(np.asarray([0.2, 1.8]))), np.asarray([0.2, 1.8], dtype=float)),
        (3, float(np.linalg.norm(np.asarray([0.0, -2.0]))), np.asarray([0.0, -2.0], dtype=float)),
    ]

    pair = _select_local_controlling_pair(
        center_xy,
        normal,
        candidates,
        centerline_without_center=centerline_without_center,
    )

    assert pair is not None
    assert int(pair["left_component_id"]) == 3
    assert int(pair["right_component_id"]) == 9
    assert np.allclose(pair["right_point"], np.asarray([0.2, 1.8], dtype=float))


def test_select_local_controlling_pair_requires_distinct_components() -> None:
    center_xy = np.asarray([0.0, 0.0], dtype=float)
    normal = np.asarray([0.0, 1.0], dtype=float)
    candidates = [
        (3, 5.0, np.asarray([-1.0, -5.0], dtype=float)),
        (3, 5.5, np.asarray([1.0, 5.4], dtype=float)),
        (9, 6.0, np.asarray([0.5, 6.0], dtype=float)),
    ]

    pair = _select_local_controlling_pair(center_xy, normal, candidates)

    assert pair is not None
    assert int(pair["left_component_id"]) == 3
    assert int(pair["right_component_id"]) == 9


def test_apply_segmented_stub_width_inheritance_uses_clean_family_representative() -> None:
    directed_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 19,
                "id_us_node": 14,
                "id_ds_node": 31,
                "lineage_type": "touch_fallback_1to1",
                "dominant_parent_core_overlap_fraction": 0.009,
                "matched_parent_node_path": "",
                "geometry": LineString([(0.0, 0.0), (1.0, 0.0)]),
            },
            {
                "id_link": 44,
                "id_us_node": 31,
                "id_ds_node": 13,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_core_overlap_fraction": 1.0,
                "matched_parent_node_path": "15,13",
                "geometry": LineString([(1.0, 0.0), (2.0, 0.0)]),
            },
            {
                "id_link": 21,
                "id_us_node": 32,
                "id_ds_node": 14,
                "lineage_type": "touch_fallback_1to1",
                "dominant_parent_core_overlap_fraction": 0.006,
                "matched_parent_node_path": "",
                "geometry": LineString([(0.0, 1.0), (1.0, 1.0)]),
            },
            {
                "id_link": 45,
                "id_us_node": 32,
                "id_ds_node": 16,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_core_overlap_fraction": 1.0,
                "matched_parent_node_path": "17,16",
                "geometry": LineString([(0.0, 2.0), (1.0, 2.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    enriched_links = directed_links.copy()
    enriched_links["wid_adj_wet"] = [635.8, 44.8, 782.5, 966.1]
    enriched_links["wid_adj_total"] = [671.1, 51.8, 1035.1, 973.3]
    enriched_links["wid_adj_dry"] = [28.1, 0.0, 245.3, 0.0]
    for prefix in ("wid_med", "wid_p05", "wid_p50", "wid_p95"):
        enriched_links[f"{prefix}_wet"] = enriched_links["wid_adj_wet"]
        enriched_links[f"{prefix}_total"] = enriched_links["wid_adj_total"]
        enriched_links[f"{prefix}_dry"] = enriched_links["wid_adj_dry"]
    enriched_links["wet_fraction_adj"] = [0.95, 0.86, 0.76, 0.99]
    enriched_links["dry_fraction_adj"] = [0.04, 0.0, 0.24, 0.0]
    enriched_links["n_wet_threads_mean"] = [5.0, 1.0, 2.0, 1.0]
    enriched_links["n_wet_threads_max"] = [5, 1, 2, 1]
    enriched_links["n_channels_mean"] = [5.0, 1.0, 2.0, 1.0]
    enriched_links["n_channels_max"] = [5, 1, 2, 1]

    samples = pd.DataFrame(
        [
            *[
                {"id_link": 19, "qc_opposition_cos": value, "width_invalid": False}
                for value in (-0.2, -0.1, 0.0, 0.1, 0.2, -0.3)
            ],
            *[
                {"id_link": 44, "qc_opposition_cos": value, "width_invalid": False}
                for value in (-1.0, -0.9, -0.8, -0.7, -0.6, -0.4)
            ],
            *[
                {"id_link": 21, "qc_opposition_cos": value, "width_invalid": False}
                for value in (-1.0, -0.9, -0.8, -0.7)
            ],
            *[
                {"id_link": 45, "qc_opposition_cos": value, "width_invalid": False}
                for value in (-0.9, -0.8, -0.7, -0.6)
            ],
        ]
    )

    adjusted_links, adjusted_enriched, summary = _apply_segmented_stub_width_inheritance(
        directed_links=directed_links,
        enriched_links=enriched_links,
        link_width_samples=samples,
    )

    widths_by_link = adjusted_enriched.set_index("id_link")
    metadata_by_link = adjusted_links.set_index("id_link")

    assert np.isclose(float(widths_by_link.loc[19, "wid_adj_wet"]), float(widths_by_link.loc[44, "wid_adj_wet"]))
    assert np.isclose(float(widths_by_link.loc[19, "wid_adj_total"]), float(widths_by_link.loc[44, "wid_adj_total"]))
    assert bool(metadata_by_link.loc[19, "family_width_inheritance_applied"]) is True
    assert int(metadata_by_link.loc[19, "family_width_inherited_from_link"]) == 44
    assert bool(metadata_by_link.loc[21, "family_width_inheritance_applied"]) is False
    assert summary["applied_link_ids"] == [19]


def test_apply_collapsed_selection_metadata_adds_collapsed_unit_columns() -> None:
    frame = pd.DataFrame({"id_link": [10, 11]})

    result = _apply_collapsed_selection_metadata(frame, unit_ids=[5], group_label=None)

    assert result["collapsed_selection_label"].tolist() == ["collapsed_unit_5", "collapsed_unit_5"]
    assert result["collapsed_unit_ids"].tolist() == ["5", "5"]
    assert result["collapsed_selection_type"].tolist() == ["unit", "unit"]


def test_apply_collapse_components_to_mask_labels_only_added_pixels(tmp_path: Path) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4, 4:16] = 1
    mask[15, 4:16] = 1
    mask[4:16, 4] = 1
    mask[4:16, 15] = 1

    mask_path = tmp_path / "parent_mask.tif"
    transform = from_origin(0, 20, 1, 1)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    reviewed_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "wid_adj": 1.0, "geometry": LineString([(4.5, 15.5), (15.5, 15.5)])},
            {"id_link": 2, "wid_adj": 1.0, "geometry": LineString([(4.5, 4.5), (15.5, 4.5)])},
            {"id_link": 3, "wid_adj": 1.0, "geometry": LineString([(4.5, 4.5), (4.5, 15.5)])},
            {"id_link": 4, "wid_adj": 1.0, "geometry": LineString([(15.5, 4.5), (15.5, 15.5)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    components = [
            {
                "component_id": "C1",
                "unit_ids": [1],
                "link_ids": [1, 2, 3, 4],
                "node_ids": [10, 11],
                "compound_bubble_ids": [],
                "n_units": 1,
                "n_links": 4,
                "n_nodes": 2,
            }
        ]

    collapsed_mask, origin_mask, summary, component_frame, edit_geometries = _apply_collapse_components_to_mask(
        mask_path,
        components,
        reviewed_links,
        preferred_width_field="wid_adj",
        buffer_scale=0.5,
    )

    assert summary["pixels_added"] > 0
    assert int(collapsed_mask.sum()) > int(mask.sum())
    assert int(component_frame.loc[0, "added_pixels"]) == summary["pixels_added"]
    assert int(component_frame.loc[0, "origin_label"]) == 101
    assert int(component_frame.loc[0, "width_scope_label"]) == 101
    assert 101 in np.unique(origin_mask)
    assert int(origin_mask[10, 10]) == 101
    assert int(origin_mask[4, 4]) == 1
    assert set(edit_geometries["geometry_role"]) == {"footprint", "edit", "collapse_domain"}


def test_resolve_link_by_progression_prefers_upstream_source_distance() -> None:
    resolved = _resolve_link_by_progression(
        node_a=7,
        node_b=6,
        source_distance={6: 3283.5, 7: 3093.6},
        sink_distance={6: 63615.7, 7: 63657.4},
        geometry_order=(6, 7),
        allow_geometry_fallback=True,
    )

    assert resolved == ((7, 6), "global_progression_rule")


def test_resolve_single_remaining_link_uses_progression_when_both_directions_validate() -> None:
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 10)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 10)},
            {"id_node": 5, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 100, "len": 10.0, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 101, "len": 14.0, "geometry": LineString([(0, 0), (10, 10)])},
            {"id_link": 102, "len": 20.0, "geometry": LineString([(10, 0), (30, 0)])},
            {"id_link": 103, "len": 14.0, "geometry": LineString([(10, 10), (20, 10)])},
            {"id_link": 104, "len": 14.0, "geometry": LineString([(20, 10), (30, 0)])},
            {"id_link": 105, "len": 14.0, "geometry": LineString([(20, 10), (10, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    orientation_by_link = {
        100: (0, 2),
        101: (0, 1),
        102: (2, 5),
        103: (1, 3),
        104: (3, 5),
    }

    resolved = _resolve_single_remaining_link(
        link_id=105,
        edge_nodes={105: (2, 3)},
        geometry_order_by_link={105: (3, 2)},
        orientation_by_link=orientation_by_link,
        child_nodes=child_nodes,
        child_inlet_nodes={0},
        child_outlet_nodes={5},
        child_links=child_links,
    )

    assert resolved == ((2, 3), "global_progression_rule")


def test_matching_and_direction_assignment_populates_upstream_downstream_nodes() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2"]
    parent_nodes["n_links"] = [1, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False]
    parent_links["is_outlet"] = [False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 12, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_nodes["idx_node"] = np.arange(len(child_nodes))
    child_nodes["id_links"] = ["10", "10,11", "11,12", "12"]
    child_nodes["n_links"] = [1, 2, 2, 1]
    child_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    child_nodes["schema_rg"] = "rg-v1"

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 10, "id_nodes": "11, 10", "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 11, "id_nodes": "12, 11", "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 12, "id_nodes": "13, 12", "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links["idx_link"] = ""
    child_links["n_nodes"] = 2
    child_links["id_us_node"] = pd.NA
    child_links["id_ds_node"] = pd.NA
    child_links["is_inlet"] = [True, False, False]
    child_links["is_outlet"] = [False, False, True]
    child_links["type_io"] = ["inlet", "neither", "outlet"]
    child_links["schema_rg"] = "rg-v1"
    child_links["link_conn"] = ""
    child_links["wid_pix"] = ""
    child_links["len"] = [10.0, 10.0, 10.0]
    child_links["wid"] = [4.0, 4.0, 4.0]
    child_links["wid_adj"] = [4.0, 4.0, 4.0]
    child_links["wid_med"] = [4.0, 4.0, 4.0]
    child_links["len_adj"] = [10.0, 10.0, 10.0]
    child_links["sinuosity"] = [1.0, 1.0, 1.0]

    parent_order, _, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=1.0,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=1.0,
    )
    directed_links, directed_nodes, summary = _assign_variant_directions(
        child_links=child_links,
        child_nodes=child_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    _, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=1.0,
    )

    oriented_pairs = {
        int(row.id_link): (int(row.id_us_node), int(row.id_ds_node))
        for row in directed_links.itertuples(index=False)
    }
    assert oriented_pairs == {
        10: (10, 11),
        11: (11, 12),
        12: (12, 13),
    }
    assert summary["source_node"] == 10
    assert summary["sink_node"] == 13
    assert directed_nodes.loc[directed_nodes["id_node"] == 10, "is_inlet"].iloc[0]
    assert directed_nodes.loc[directed_nodes["id_node"] == 13, "is_outlet"].iloc[0]
    assert link_match["child_id_link"].nunique() == 3
    coords = list(directed_links.loc[directed_links["id_link"] == 10].iloc[0].geometry.coords)
    assert coords[0] == (0.0, 0.0)
    assert coords[-1] == (10.0, 0.0)
    lineage_row = link_lineage.loc[link_lineage["id_link"] == 10].iloc[0]
    assert lineage_row["lineage_type"] == "unchanged_1to1"
    assert lineage_row["matched_parent_link_ids"] == "0"
    assert lineage_row["touch_parent_link_ids"] == "1"
    assert lineage_row["secondary_parent_link_ids"] == ""
    assert np.isclose(lineage_row["matched_parent_overlap_fraction"], 1.0)


def test_resolved_lineage_uses_parent_node_path_and_keeps_parallel_edges() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1,2", "1,2,3", "3"]
    parent_nodes["n_links"] = [1, 3, 3, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 3, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False, False]
    parent_links["is_outlet"] = [False, False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_nodes["idx_node"] = np.arange(len(child_nodes))
    child_nodes["id_links"] = ["20", "20"]
    child_nodes["n_links"] = [1, 1]
    child_nodes["type_io"] = ["inlet", "outlet"]
    child_nodes["schema_rg"] = "rg-v1"

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 20, "id_nodes": "11, 10", "geometry": LineString([(0, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links["idx_link"] = ""
    child_links["n_nodes"] = 2
    child_links["id_us_node"] = pd.NA
    child_links["id_ds_node"] = pd.NA
    child_links["is_inlet"] = [True]
    child_links["is_outlet"] = [True]
    child_links["type_io"] = ["both"]
    child_links["schema_rg"] = "rg-v1"
    child_links["link_conn"] = ""
    child_links["wid_pix"] = ""
    child_links["len"] = [30.0]
    child_links["wid"] = [4.0]
    child_links["wid_adj"] = [4.0]
    child_links["wid_med"] = [4.0]
    child_links["len_adj"] = [30.0]
    child_links["sinuosity"] = [1.0]

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=1.0,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=1.0,
    )
    directed_links, _, _ = _assign_variant_directions(
        child_links=child_links,
        child_nodes=child_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=1.0,
    )

    lineage_row = link_lineage.loc[link_lineage["id_link"] == 20].iloc[0]
    assert lineage_row["lineage_method"] == "matched_parent_node_path"
    assert lineage_row["matched_parent_link_ids"] == "0,1,2,3"
    assert lineage_row["primary_parent_link_ids"] == "0,1,2,3"
    assert lineage_row["secondary_parent_link_ids"] == ""
    assert lineage_row["lineage_type"] == "collapsed_many_to_one"
    assert np.isclose(lineage_row["matched_parent_overlap_fraction"], 1.0)


def test_exact_geometry_overrides_ambiguous_parallel_parent_node_path() -> None:
    upper = LineString([(0, 0), (5, 2), (10, 0)])
    lower = LineString([(0, 0), (5, -2), (10, 0)])
    parent_graph = nx.MultiDiGraph()
    parent_graph.add_edge(0, 1, key=9, id_link=9, geometry=upper)
    parent_graph.add_edge(0, 1, key=10, id_link=10, geometry=lower)
    child_links = gpd.GeoDataFrame(
        {
            "id_link": [100],
            "id_us_node": [20],
            "id_ds_node": [21],
        },
        geometry=[upper],
        crs="EPSG:3857",
    )
    node_match = pd.DataFrame(
        {
            "child_id_node": [20, 21],
            "matched_parent_node_id": [0, 1],
        }
    )
    link_match = pd.DataFrame(
        {
            "child_id_link": [100, 100],
            "parent_id_link": [9, 10],
            "candidate_rank": [1, 2],
            "candidate_class": ["core_overlap", "touch_only"],
            "child_overlap_fraction": [1.0, 0.0],
            "child_overlap_length": [upper.length, 0.0],
                "child_core_overlap_fraction": [1.0, 0.0],
                "child_core_overlap_length": [upper.length, 0.0],
                "parent_overlap_fraction": [1.0, 0.0],
                "parent_overlap_length": [upper.length, 0.0],
                "parent_core_overlap_fraction": [1.0, 0.0],
                "parent_core_overlap_length": [upper.length, 0.0],
            }
        )

    lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=child_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=1.0,
    ).iloc[0]

    assert lineage["lineage_type"] == "unchanged_1to1"
    assert lineage["dominant_parent_link_id"] == 9
    assert lineage["matched_parent_link_ids"] == "9"
    assert lineage["touch_parent_link_ids"] == "10"
    assert lineage["lineage_method"] == "exact_geometry_override_parent_node_path"


def test_assign_variant_directions_preserves_canonical_parent_outlet() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2"]
    parent_nodes["n_links"] = [1, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False]
    parent_links["is_outlet"] = [False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": Point(20, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": False, "geometry": Point(20, -10)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_nodes["idx_node"] = np.arange(len(child_nodes))
    child_nodes["id_links"] = ["20", "20,21,22", "21", "22"]
    child_nodes["n_links"] = [1, 3, 1, 1]
    child_nodes["type_io"] = ["inlet", "neither", "outlet", "neither"]
    child_nodes["schema_rg"] = "rg-v1"

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 20, "id_nodes": "10, 11", "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 21, "id_nodes": "11, 12", "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 22, "id_nodes": "11, 13", "geometry": LineString([(10, 0), (20, -10)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links["idx_link"] = ""
    child_links["n_nodes"] = 2
    child_links["id_us_node"] = pd.NA
    child_links["id_ds_node"] = pd.NA
    child_links["is_inlet"] = [True, False, False]
    child_links["is_outlet"] = [False, True, False]
    child_links["type_io"] = ["inlet", "outlet", "neither"]
    child_links["schema_rg"] = "rg-v1"
    child_links["link_conn"] = ""
    child_links["wid_pix"] = ""
    child_links["len"] = [10.0, 10.0, 14.0]
    child_links["wid"] = [4.0, 4.0, 4.0]
    child_links["wid_adj"] = [4.0, 4.0, 4.0]
    child_links["wid_med"] = [4.0, 4.0, 4.0]
    child_links["len_adj"] = [10.0, 10.0, 14.0]
    child_links["sinuosity"] = [1.0, 1.0, 1.0]

    parent_order, _, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=1.0,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=1.0,
    )
    directed_links, directed_nodes, summary = _assign_variant_directions(
        child_links=child_links,
        child_nodes=child_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
    )

    outlet_links = directed_links.loc[directed_links["is_outlet"].fillna(False), "id_link"].astype(int).tolist()
    outlet_nodes = directed_nodes.loc[directed_nodes["is_outlet"].fillna(False), "id_node"].astype(int).tolist()
    assert outlet_links == [21]
    assert outlet_nodes == [12]
    assert summary["sink_nodes"] == [12]
    assert summary["canonical_sink_nodes"] == [12]
    assert sorted(summary["topology_sink_nodes"]) == [12, 13]


def test_segment_collapsed_links_to_anchor_nodes_splits_link_at_nominal_parent_nodes() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2"]
    parent_nodes["n_links"] = [1, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False]
    parent_links["is_outlet"] = [False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "idx_node": 0, "id_links": "20", "n_links": 1, "type_io": "inlet", "schema_rg": "rg-v1", "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "idx_node": 1, "id_links": "20", "n_links": 1, "type_io": "outlet", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "idx_link": "0,1,2,3",
                "id_nodes": "10, 11",
                "raw_id_nodes": "10, 11",
                "id_us_node": 10,
                "id_ds_node": 11,
                "n_nodes": 2,
                "is_inlet": True,
                "is_outlet": True,
                "type_io": "both",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "10,20,30,40",
                "len": 30.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 30.0,
                "sinuosity": 1.0,
                "geometry": LineString([(0, 0), (30, 0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 10,
                "matched_parent_node_id": 0,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 0,
                "child_is_inlet_raw": True,
                "child_is_outlet_raw": False,
                "parent_is_inlet": True,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 11,
                "matched_parent_node_id": 3,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 3,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": True,
                "parent_is_inlet": False,
                "parent_is_outlet": True,
            },
        ]
    )
    link_lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "matched_parent_node_path": "0,1,2,3",
            }
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    updated_links, updated_nodes, updated_node_match, summary = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_order,
        directed_links=child_links,
        directed_nodes=child_nodes,
        node_match=node_match,
        link_match=pd.DataFrame(),
        link_lineage=link_lineage,
        selection_edge_pairs={(1, 2)},
        nominal_upstream_node=1,
        nominal_downstream_node=2,
    )

    assert summary["mode_applied"] is True
    assert summary["n_anchor_splits"] == 1
    assert summary["n_new_nodes"] == 2
    assert summary["n_new_links"] == 2
    assert len(updated_links) == 3
    assert len(updated_nodes) == 4
    assert set(updated_links["id_link"].astype(int)) == {20, 21, 22}
    idx_link_by_id = dict(zip(updated_links["id_link"].astype(int), updated_links["idx_link"]))
    wid_pix_by_id = dict(zip(updated_links["id_link"].astype(int), updated_links["wid_pix"]))
    assert set(idx_link_by_id.values()) == {"0,1", "1,2", "2,3"}
    assert set(wid_pix_by_id.values()) == {
        "10.000000,20.000000",
        "20.000000,30.000000",
        "30.000000,40.000000",
    }
    legacy_widths_by_samples = {
        row.wid_pix: (
            float(row.wid),
            float(row.wid_adj),
            float(row.wid_med),
            float(row.len_adj),
            float(row.sinuosity),
        )
        for row in updated_links.itertuples(index=False)
    }
    assert legacy_widths_by_samples == {
        "10.000000,20.000000": (15.0, 15.0, 15.0, 10.0, 1.0),
        "20.000000,30.000000": (25.0, 25.0, 25.0, 10.0, 1.0),
        "30.000000,40.000000": (35.0, 35.0, 35.0, 10.0, 1.0),
    }

    kept = updated_links.loc[updated_links["id_link"] == 20].iloc[0]
    assert np.isclose(float(kept["len"]), 10.0)
    assert np.isclose(float(kept.geometry.length), 10.0)
    assert kept["idx_link"] == "1,2"
    assert kept["wid_pix"] == "20.000000,30.000000"
    assert int(kept["id_us_node"]) in set(updated_nodes["id_node"].astype(int))
    assert int(kept["id_ds_node"]) in set(updated_nodes["id_node"].astype(int))
    outlet_flags = dict(zip(updated_links["id_link"].astype(int), updated_links["is_outlet"].astype(bool)))
    inlet_flags = dict(zip(updated_links["id_link"].astype(int), updated_links["is_inlet"].astype(bool)))
    assert sum(outlet_flags.values()) == 1
    assert sum(inlet_flags.values()) == 1

    matched_pairs = {
        (int(row.child_id_node), int(row.matched_parent_node_id))
        for row in updated_node_match.itertuples(index=False)
        if pd.notna(row.matched_parent_node_id)
    }
    assert (10, 0) in matched_pairs
    assert (11, 3) in matched_pairs
    assert any(parent_id == 1 for _, parent_id in matched_pairs)
    assert any(parent_id == 2 for _, parent_id in matched_pairs)


def test_segment_collapsed_links_to_anchor_nodes_clips_target_corridor_across_two_links() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(30, 0)},
            {"id_node": 4, "is_inlet": False, "is_outlet": True, "geometry": Point(40, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2,3", "3"]
    parent_nodes["n_links"] = [1, 2, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(20, 0), (30, 0)])},
            {"id_link": 3, "id_nodes": "3, 4", "id_us_node": 3, "id_ds_node": 4, "geometry": LineString([(30, 0), (40, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False, False]
    parent_links["is_outlet"] = [False, False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["wid"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "idx_node": 0, "id_links": "20", "n_links": 1, "type_io": "inlet", "schema_rg": "rg-v1", "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "idx_node": 1, "id_links": "20,21", "n_links": 2, "type_io": "neither", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 12, "idx_node": 2, "id_links": "21", "n_links": 1, "type_io": "outlet", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": True, "geometry": Point(40, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "idx_link": "0,1,2",
                "id_nodes": "10, 11",
                "raw_id_nodes": "10, 11",
                "id_us_node": 10,
                "id_ds_node": 11,
                "n_nodes": 2,
                "is_inlet": True,
                "is_outlet": False,
                "type_io": "inlet",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "10,20,30",
                "len": 20.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 20.0,
                "sinuosity": 1.0,
                "geometry": LineString([(0, 0), (20, 0)]),
            },
            {
                "id_link": 21,
                "idx_link": "3,4,5",
                "id_nodes": "11, 12",
                "raw_id_nodes": "11, 12",
                "id_us_node": 11,
                "id_ds_node": 12,
                "n_nodes": 2,
                "is_inlet": False,
                "is_outlet": True,
                "type_io": "outlet",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "40,50,60",
                "len": 20.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 20.0,
                "sinuosity": 1.0,
                "geometry": LineString([(20, 0), (40, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 10,
                "matched_parent_node_id": 0,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 0,
                "child_is_inlet_raw": True,
                "child_is_outlet_raw": False,
                "parent_is_inlet": True,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 11,
                "matched_parent_node_id": 2,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 2,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": False,
                "parent_is_inlet": False,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 12,
                "matched_parent_node_id": 4,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 4,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": True,
                "parent_is_inlet": False,
                "parent_is_outlet": True,
            },
        ]
    )
    link_lineage = pd.DataFrame(
        [
            {"id_link": 20, "matched_parent_node_path": "0,1,2"},
            {"id_link": 21, "matched_parent_node_path": "2,3,4"},
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    updated_links, updated_nodes, updated_node_match, summary = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_order,
        directed_links=child_links,
        directed_nodes=child_nodes,
        node_match=node_match,
        link_match=pd.DataFrame(),
        link_lineage=link_lineage,
        selection_edge_pairs={(1, 2), (2, 3)},
        nominal_upstream_node=1,
        nominal_downstream_node=3,
    )

    assert summary["mode_applied"] is True
    assert summary["n_anchor_splits"] == 2
    assert summary["n_new_nodes"] == 2
    assert summary["n_new_links"] == 2
    assert len(updated_links) == 4
    assert len(updated_nodes) == 5

    kept_upstream = updated_links.loc[updated_links["id_link"] == 20].iloc[0]
    kept_downstream = updated_links.loc[updated_links["id_link"] == 21].iloc[0]
    assert np.isclose(float(kept_upstream["len"]), 10.0)
    assert np.isclose(float(kept_downstream["len"]), 10.0)
    assert kept_upstream["idx_link"] == "1,2"
    assert kept_upstream["wid_pix"] == "20.000000,30.000000"
    assert kept_downstream["idx_link"] == "3,4"
    assert kept_downstream["wid_pix"] == "40.000000,50.000000"

    matched_pairs = {
        (int(row.child_id_node), int(row.matched_parent_node_id))
        for row in updated_node_match.itertuples(index=False)
        if pd.notna(row.matched_parent_node_id)
    }
    assert (10, 0) in matched_pairs
    assert (11, 2) in matched_pairs
    assert (12, 4) in matched_pairs
    assert any(parent_id == 1 for _, parent_id in matched_pairs)
    assert any(parent_id == 3 for _, parent_id in matched_pairs)


def test_segment_collapsed_links_to_anchor_nodes_projects_parent_nodes_instead_of_fractional_lengths() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(12, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(26, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["0", "0,1", "1,2", "2"]
    parent_nodes["n_links"] = [1, 2, 2, 1]
    parent_nodes["type_io"] = ["inlet", "neither", "neither", "outlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "id_us_node": 0, "id_ds_node": 1, "geometry": LineString([(0, 0), (12, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(12, 0), (26, 0)])},
            {"id_link": 2, "id_nodes": "2, 3", "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(26, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [True, False, False]
    parent_links["is_outlet"] = [False, False, True]
    parent_links["type_io"] = ["inlet", "neither", "outlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [12.0, 14.0, 4.0]
    parent_links["wid"] = [4.0, 4.0, 4.0]
    parent_links["wid_adj"] = [4.0, 4.0, 4.0]
    parent_links["wid_med"] = [4.0, 4.0, 4.0]
    parent_links["len_adj"] = [10.0, 10.0, 10.0]
    parent_links["sinuosity"] = [1.0, 1.0, 1.0]

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "idx_node": 0, "id_links": "20", "n_links": 1, "type_io": "inlet", "schema_rg": "rg-v1", "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 11, "idx_node": 1, "id_links": "20", "n_links": 1, "type_io": "outlet", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "idx_link": "0,1,2,3",
                "id_nodes": "10, 11",
                "raw_id_nodes": "10, 11",
                "id_us_node": 10,
                "id_ds_node": 11,
                "n_nodes": 2,
                "is_inlet": True,
                "is_outlet": True,
                "type_io": "both",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "10,20,30,40",
                "len": 30.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 30.0,
                "sinuosity": 1.0,
                "geometry": LineString([(0, 0), (30, 0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 10,
                "matched_parent_node_id": 0,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 0,
                "child_is_inlet_raw": True,
                "child_is_outlet_raw": False,
                "parent_is_inlet": True,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 11,
                "matched_parent_node_id": 3,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 3,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": True,
                "parent_is_inlet": False,
                "parent_is_outlet": True,
            },
        ]
    )
    link_lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "matched_parent_node_path": "0,1,2,3",
            }
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    updated_links, updated_nodes, updated_node_match, summary = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_order,
        directed_links=child_links,
        directed_nodes=child_nodes,
        node_match=node_match,
        link_match=pd.DataFrame(),
        link_lineage=link_lineage,
        selection_edge_pairs={(1, 2)},
        nominal_upstream_node=1,
        nominal_downstream_node=2,
    )

    assert summary["mode_applied"] is True
    assert summary["n_anchor_splits"] == 1
    kept = updated_links.loc[updated_links["id_link"] == 20].iloc[0]
    assert np.isclose(float(kept["len"]), 14.0)
    assert np.isclose(float(kept.geometry.length), 14.0)

    updated_nodes = updated_nodes.set_index("id_node")
    updated_node_match = updated_node_match.dropna(subset=["matched_parent_node_id"]).copy()
    updated_node_match["matched_parent_node_id"] = updated_node_match["matched_parent_node_id"].astype(int)
    for parent_id in [1, 2]:
        child_id = int(
            updated_node_match.loc[
                updated_node_match["matched_parent_node_id"] == parent_id,
                "child_id_node",
            ].iloc[0]
        )
        assert updated_nodes.loc[child_id].geometry.distance(parent_nodes.set_index("id_node").loc[parent_id].geometry) == 0.0


def test_segment_collapsed_links_to_anchor_nodes_uses_high_parent_overlap_for_compound_boundaries() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": Point(0, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 14, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 15, "is_inlet": False, "is_outlet": False, "geometry": Point(30, 0)},
            {"id_node": 16, "is_inlet": True, "is_outlet": False, "geometry": Point(40, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["18", "18,19,20", "20,21,22", "19,21,23", "22,23"]
    parent_nodes["n_links"] = [1, 3, 3, 3, 2]
    parent_nodes["type_io"] = ["outlet", "neither", "neither", "neither", "inlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 18, "id_nodes": "13, 12", "id_us_node": 13, "id_ds_node": 12, "geometry": LineString([(10, 0), (0, 0)])},
            {"id_link": 19, "id_nodes": "15, 13", "id_us_node": 15, "id_ds_node": 13, "geometry": LineString([(30, 0), (10, 0)])},
            {"id_link": 20, "id_nodes": "14, 13", "id_us_node": 14, "id_ds_node": 13, "geometry": LineString([(20, 0), (10, 0)])},
            {"id_link": 21, "id_nodes": "15, 14", "id_us_node": 15, "id_ds_node": 14, "geometry": LineString([(30, 0), (20, 0)])},
            {"id_link": 22, "id_nodes": "16, 14", "id_us_node": 16, "id_ds_node": 14, "geometry": LineString([(40, 0), (20, 0)])},
            {"id_link": 23, "id_nodes": "16, 15", "id_us_node": 16, "id_ds_node": 15, "geometry": LineString([(40, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [False, False, False, False, True, True]
    parent_links["is_outlet"] = [True, False, False, False, False, False]
    parent_links["type_io"] = ["outlet", "neither", "neither", "neither", "inlet", "inlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 20.0, 10.0, 10.0, 20.0, 10.0]
    parent_links["wid"] = [4.0] * len(parent_links)
    parent_links["wid_adj"] = [4.0] * len(parent_links)
    parent_links["wid_med"] = [4.0] * len(parent_links)
    parent_links["len_adj"] = [10.0] * len(parent_links)
    parent_links["sinuosity"] = [1.0] * len(parent_links)

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 100, "idx_node": 0, "id_links": "30", "n_links": 1, "type_io": "inlet", "schema_rg": "rg-v1", "is_inlet": True, "is_outlet": False, "geometry": Point(40, 0)},
            {"id_node": 101, "idx_node": 1, "id_links": "30,31", "n_links": 2, "type_io": "neither", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 102, "idx_node": 2, "id_links": "31", "n_links": 1, "type_io": "outlet", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": True, "geometry": Point(0, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 30,
                "idx_link": "0,1,2,3",
                "id_nodes": "100, 101",
                "raw_id_nodes": "100, 101",
                "id_us_node": 100,
                "id_ds_node": 101,
                "n_nodes": 2,
                "is_inlet": True,
                "is_outlet": False,
                "type_io": "inlet",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "10,20,30,40",
                "len": 20.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 20.0,
                "sinuosity": 1.0,
                "geometry": LineString([(40, 0), (20, 0)]),
            },
            {
                "id_link": 31,
                "idx_link": "4,5,6,7",
                "id_nodes": "101, 102",
                "raw_id_nodes": "101, 102",
                "id_us_node": 101,
                "id_ds_node": 102,
                "n_nodes": 2,
                "is_inlet": False,
                "is_outlet": True,
                "type_io": "outlet",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "50,60,70,80",
                "len": 20.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 20.0,
                "sinuosity": 1.0,
                "geometry": LineString([(20, 0), (0, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 100,
                "matched_parent_node_id": 16,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 4,
                "child_is_inlet_raw": True,
                "child_is_outlet_raw": False,
                "parent_is_inlet": True,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 101,
                "matched_parent_node_id": 14,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 2,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": False,
                "parent_is_inlet": False,
                "parent_is_outlet": False,
            },
            {
                "child_id_node": 102,
                "matched_parent_node_id": 12,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 0,
                "child_is_inlet_raw": False,
                "child_is_outlet_raw": True,
                "parent_is_inlet": False,
                "parent_is_outlet": True,
            },
        ]
    )
    link_lineage = pd.DataFrame(
        [
            {"id_link": 30, "matched_parent_node_path": pd.NA},
            {"id_link": 31, "matched_parent_node_path": pd.NA},
        ]
    )
    link_match = pd.DataFrame(
        [
            {
                "child_id_link": 30,
                "parent_id_link": 23,
                "parent_overlap_fraction": 1.0,
                "child_overlap_fraction": 0.80,
                "candidate_rank": 1,
            },
            {
                "child_id_link": 31,
                "parent_id_link": 18,
                "parent_overlap_fraction": 1.0,
                "child_overlap_fraction": 0.75,
                "candidate_rank": 1,
            },
            {
                "child_id_link": 30,
                "parent_id_link": 21,
                "parent_overlap_fraction": 0.02,
                "child_overlap_fraction": 0.01,
                "candidate_rank": 3,
            },
            {
                "child_id_link": 31,
                "parent_id_link": 20,
                "parent_overlap_fraction": 0.04,
                "child_overlap_fraction": 0.01,
                "candidate_rank": 2,
            },
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    updated_links, updated_nodes, updated_node_match, summary = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_order,
        directed_links=child_links,
        directed_nodes=child_nodes,
        node_match=node_match,
        link_match=link_match,
        link_lineage=link_lineage,
        selection_edge_pairs={(15, 13), (15, 14), (14, 13)},
        nominal_upstream_node=15,
        nominal_downstream_node=13,
    )

    assert summary["mode_applied"] is True
    assert summary["n_anchor_splits"] == 2
    kept_upstream = updated_links.loc[updated_links["id_link"] == 30].iloc[0]
    kept_downstream = updated_links.loc[updated_links["id_link"] == 31].iloc[0]
    assert np.isclose(float(kept_upstream["len"]), 10.0)
    assert np.isclose(float(kept_downstream["len"]), 10.0)

    updated_nodes = updated_nodes.set_index("id_node")
    updated_node_match = updated_node_match.dropna(subset=["matched_parent_node_id"]).copy()
    updated_node_match["matched_parent_node_id"] = updated_node_match["matched_parent_node_id"].astype(int)
    for parent_id in [13, 15]:
        child_id = int(
            updated_node_match.loc[
                updated_node_match["matched_parent_node_id"] == parent_id,
                "child_id_node",
            ].iloc[0]
        )
        assert updated_nodes.loc[child_id].geometry.distance(parent_nodes.set_index("id_node").loc[parent_id].geometry) == 0.0


def test_boundary_overlap_cut_targets_accepts_parent_overlap_above_95pct_only() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": Point(0, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 14, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 15, "is_inlet": False, "is_outlet": False, "geometry": Point(30, 0)},
            {"id_node": 16, "is_inlet": True, "is_outlet": False, "geometry": Point(40, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 18, "id_nodes": "13, 12", "id_us_node": 13, "id_ds_node": 12, "geometry": LineString([(10, 0), (0, 0)])},
            {"id_link": 19, "id_nodes": "15, 13", "id_us_node": 15, "id_ds_node": 13, "geometry": LineString([(30, 0), (10, 0)])},
            {"id_link": 20, "id_nodes": "14, 13", "id_us_node": 14, "id_ds_node": 13, "geometry": LineString([(20, 0), (10, 0)])},
            {"id_link": 21, "id_nodes": "15, 14", "id_us_node": 15, "id_ds_node": 14, "geometry": LineString([(30, 0), (20, 0)])},
            {"id_link": 22, "id_nodes": "16, 14", "id_us_node": 16, "id_ds_node": 14, "geometry": LineString([(40, 0), (20, 0)])},
            {"id_link": 23, "id_nodes": "16, 15", "id_us_node": 16, "id_ds_node": 15, "geometry": LineString([(40, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 30, "id_us_node": 100, "id_ds_node": 101, "geometry": LineString([(40, 0), (20, 0)])},
            {"id_link": 31, "id_us_node": 101, "id_ds_node": 102, "geometry": LineString([(20, 0), (0, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    node_match = pd.DataFrame(
        [
            {"child_id_node": 100, "matched_parent_node_id": 16},
            {"child_id_node": 101, "matched_parent_node_id": 14},
            {"child_id_node": 102, "matched_parent_node_id": 12},
        ]
    )
    link_match = pd.DataFrame(
        [
            {
                "child_id_link": 30,
                "parent_id_link": 23,
                "parent_overlap_fraction": 0.96,
                "child_overlap_fraction": 0.80,
                "candidate_rank": 1,
            },
            {
                "child_id_link": 31,
                "parent_id_link": 18,
                "parent_overlap_fraction": 0.94,
                "child_overlap_fraction": 0.75,
                "candidate_rank": 1,
            },
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    targets = _boundary_overlap_cut_targets(
        parent_graph=parent_graph,
        parent_node_geometries=parent_nodes.set_index("id_node")["geometry"].to_dict(),
        directed_links=child_links,
        node_match=node_match,
        link_match=link_match,
        selection_edge_pairs={(15, 13), (15, 14), (14, 13)},
        nominal_upstream_node=15,
        nominal_downstream_node=13,
    )

    assert 30 in targets
    assert [spec["parent_node_id"] for spec in targets[30]] == [15]
    assert 31 not in targets


def test_segment_collapsed_links_to_anchor_nodes_cuts_internal_compound_entry_at_selected_node() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": Point(0, 0)},
            {"id_node": 13, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 14, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 15, "is_inlet": False, "is_outlet": False, "geometry": Point(30, 0)},
            {"id_node": 16, "is_inlet": True, "is_outlet": False, "geometry": Point(40, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_nodes["idx_node"] = np.arange(len(parent_nodes))
    parent_nodes["id_links"] = ["18", "18,19,20", "20,21,22", "19,21,23", "22,23"]
    parent_nodes["n_links"] = [1, 3, 3, 3, 2]
    parent_nodes["type_io"] = ["outlet", "neither", "neither", "neither", "inlet"]
    parent_nodes["schema_rg"] = "rg-v1"

    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 18, "id_nodes": "13, 12", "id_us_node": 13, "id_ds_node": 12, "geometry": LineString([(10, 0), (0, 0)])},
            {"id_link": 19, "id_nodes": "15, 13", "id_us_node": 15, "id_ds_node": 13, "geometry": LineString([(30, 0), (10, 0)])},
            {"id_link": 20, "id_nodes": "14, 13", "id_us_node": 14, "id_ds_node": 13, "geometry": LineString([(20, 0), (10, 0)])},
            {"id_link": 21, "id_nodes": "15, 14", "id_us_node": 15, "id_ds_node": 14, "geometry": LineString([(30, 0), (20, 0)])},
            {"id_link": 22, "id_nodes": "16, 14", "id_us_node": 16, "id_ds_node": 14, "geometry": LineString([(40, 0), (20, 0)])},
            {"id_link": 23, "id_nodes": "16, 15", "id_us_node": 16, "id_ds_node": 15, "geometry": LineString([(40, 0), (30, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    parent_links["idx_link"] = ""
    parent_links["n_nodes"] = 2
    parent_links["is_inlet"] = [False, False, False, False, True, True]
    parent_links["is_outlet"] = [True, False, False, False, False, False]
    parent_links["type_io"] = ["outlet", "neither", "neither", "neither", "inlet", "inlet"]
    parent_links["schema_rg"] = "rg-v1"
    parent_links["link_conn"] = ""
    parent_links["wid_pix"] = ""
    parent_links["len"] = [10.0, 20.0, 10.0, 10.0, 20.0, 10.0]
    parent_links["wid"] = [4.0] * len(parent_links)
    parent_links["wid_adj"] = [4.0] * len(parent_links)
    parent_links["wid_med"] = [4.0] * len(parent_links)
    parent_links["len_adj"] = [10.0] * len(parent_links)
    parent_links["sinuosity"] = [1.0] * len(parent_links)

    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 100, "idx_node": 0, "id_links": "30", "n_links": 1, "type_io": "inlet", "schema_rg": "rg-v1", "is_inlet": True, "is_outlet": False, "geometry": Point(40, 0)},
            {"id_node": 101, "idx_node": 1, "id_links": "30", "n_links": 1, "type_io": "outlet", "schema_rg": "rg-v1", "is_inlet": False, "is_outlet": True, "geometry": Point(10, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 30,
                "idx_link": "0,1,2,3",
                "id_nodes": "100, 101",
                "raw_id_nodes": "100, 101",
                "id_us_node": 100,
                "id_ds_node": 101,
                "n_nodes": 2,
                "is_inlet": True,
                "is_outlet": True,
                "type_io": "both",
                "schema_rg": "rg-v1",
                "link_conn": "",
                "wid_pix": "10,20,30,40",
                "len": 30.0,
                "wid": 4.0,
                "wid_adj": 4.0,
                "wid_med": 4.0,
                "len_adj": 30.0,
                "sinuosity": 1.0,
                "geometry": LineString([(40, 0), (10, 0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 100,
                "matched_parent_node_id": 16,
                "match_distance": 0.0,
                "match_within_tolerance": True,
                "parent_node_order": 4,
                "child_is_inlet_raw": True,
                "child_is_outlet_raw": False,
                "parent_is_inlet": True,
                "parent_is_outlet": False,
            },
        ]
    )
    link_lineage = pd.DataFrame(
        [
            {"id_link": 30, "matched_parent_node_path": pd.NA},
        ]
    )
    link_match = pd.DataFrame(
        [
            {
                "child_id_link": 30,
                "parent_id_link": 22,
                "parent_overlap_fraction": 1.0,
                "child_overlap_fraction": 0.84,
                "candidate_rank": 1,
            },
            {
                "child_id_link": 30,
                "parent_id_link": 21,
                "parent_overlap_fraction": 0.02,
                "child_overlap_fraction": 0.02,
                "candidate_rank": 2,
            },
            {
                "child_id_link": 30,
                "parent_id_link": 20,
                "parent_overlap_fraction": 0.25,
                "child_overlap_fraction": 0.25,
                "candidate_rank": 3,
            },
        ]
    )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    updated_links, updated_nodes, updated_node_match, summary = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=parent_graph,
        parent_nodes=parent_nodes,
        parent_node_order=parent_order,
        directed_links=child_links,
        directed_nodes=child_nodes,
        node_match=node_match,
        link_match=link_match,
        link_lineage=link_lineage,
        selection_edge_pairs={(15, 13), (15, 14), (14, 13)},
        nominal_upstream_node=15,
        nominal_downstream_node=13,
    )

    assert summary["mode_applied"] is True
    assert summary["n_anchor_splits"] == 1
    kept = updated_links.loc[updated_links["id_link"] == 30].iloc[0]
    assert np.isclose(float(kept["len"]), 10.0)
    assert np.isclose(float(kept.geometry.length), 10.0)

    updated_nodes = updated_nodes.set_index("id_node")
    updated_node_match = updated_node_match.dropna(subset=["matched_parent_node_id"]).copy()
    updated_node_match["matched_parent_node_id"] = updated_node_match["matched_parent_node_id"].astype(int)
    child_id = int(
        updated_node_match.loc[
            updated_node_match["matched_parent_node_id"] == 14,
            "child_id_node",
        ].iloc[0]
    )
    assert updated_nodes.loc[child_id].geometry.distance(parent_nodes.set_index("id_node").loc[14].geometry) == 0.0


def test_repair_invalid_directionality_prefers_weaker_evidence_flip() -> None:
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 10)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 4, "is_inlet": False, "is_outlet": False, "geometry": Point(20, -10)},
            {"id_node": 5, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "len": 10.0, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "len": 10.0, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "len": 14.0, "geometry": LineString([(10, 0), (20, -10)])},
            {"id_link": 3, "len": 14.0, "geometry": LineString([(20, 0), (30, 0)])},
            {"id_link": 4, "len": 14.0, "geometry": LineString([(20, -10), (30, 0)])},
            {"id_link": 10, "len": 14.0, "geometry": LineString([(20, 10), (20, 0)])},
            {"id_link": 11, "len": 20.0, "geometry": LineString([(20, 10), (20, -10)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    orientation_by_link = {
        0: (0, 1),
        1: (1, 3),
        2: (1, 4),
        3: (3, 5),
        4: (4, 5),
        10: (2, 3),
        11: (2, 4),
    }
    edge_nodes = {
        0: (0, 1),
        1: (1, 3),
        2: (1, 4),
        3: (3, 5),
        4: (4, 5),
        10: (2, 3),
        11: (2, 4),
    }
    geometry_order_by_link = {link_id: nodes for link_id, nodes in edge_nodes.items()}
    incident_links = {
        0: [0],
        1: [0, 1, 2],
        2: [10, 11],
        3: [1, 3, 10],
        4: [2, 4, 11],
        5: [3, 4],
    }

    repaired = _repair_invalid_directionality(
        orientation_by_link=orientation_by_link,
        edge_nodes=edge_nodes,
        incident_links=incident_links,
        child_links=child_links,
        child_nodes=child_nodes,
        geometry_order_by_link=geometry_order_by_link,
        orientation_score_forward={10: 100.0, 11: 10.0},
        orientation_score_reverse={10: 1000.0, 11: 20.0},
        parent_order_lookup={},
        child_inlet_nodes={0},
        child_outlet_nodes={5},
    )

    assert repaired[10] == (2, 3)
    assert repaired[11] == (4, 2)

    graph = _build_oriented_child_graph(child_nodes=child_nodes, orientation_by_link=repaired)
    report = validate_single_inlet_single_outlet(graph, require_flag_match=False)
    assert report.is_valid
    assert report.source_nodes == [0]
    assert report.sink_nodes == [5]


def test_repair_unresolved_link_components_solves_small_local_motif() -> None:
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(10, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 10)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 4, "is_inlet": False, "is_outlet": False, "geometry": Point(20, -10)},
            {"id_node": 5, "is_inlet": False, "is_outlet": True, "geometry": Point(30, 0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "len": 10.0, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 1, "len": 10.0, "geometry": LineString([(10, 0), (20, 0)])},
            {"id_link": 2, "len": 14.0, "geometry": LineString([(10, 0), (20, -10)])},
            {"id_link": 3, "len": 14.0, "geometry": LineString([(20, 0), (30, 0)])},
            {"id_link": 4, "len": 14.0, "geometry": LineString([(20, -10), (30, 0)])},
            {"id_link": 10, "len": 14.0, "geometry": LineString([(20, 10), (20, 0)])},
            {"id_link": 11, "len": 20.0, "geometry": LineString([(20, 10), (20, -10)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    orientation_by_link = {
        0: (0, 1),
        1: (1, 3),
        2: (1, 4),
        3: (3, 5),
        4: (4, 5),
    }
    edge_nodes = {
        0: (0, 1),
        1: (1, 3),
        2: (1, 4),
        3: (3, 5),
        4: (4, 5),
        10: (2, 3),
        11: (2, 4),
    }
    geometry_order_by_link = {link_id: nodes for link_id, nodes in edge_nodes.items()}
    incident_links = {
        0: [0],
        1: [0, 1, 2],
        2: [10, 11],
        3: [1, 3, 10],
        4: [2, 4, 11],
        5: [3, 4],
    }

    repaired = _repair_unresolved_link_components(
        unresolved_links=[10, 11],
        edge_nodes=edge_nodes,
        incident_links=incident_links,
        orientation_by_link=orientation_by_link,
        child_links=child_links,
        child_nodes=child_nodes,
        geometry_order_by_link=geometry_order_by_link,
        orientation_score_forward={10: 100.0, 11: 10.0},
        orientation_score_reverse={10: 1000.0, 11: 20.0},
        parent_order_lookup={},
        child_inlet_nodes={0},
        child_outlet_nodes={5},
    )

    assert repaired == {10: (2, 3), 11: (4, 2)}

    graph = _build_oriented_child_graph(
        child_nodes=child_nodes,
        orientation_by_link={**orientation_by_link, **repaired},
    )
    report = validate_single_inlet_single_outlet(graph, require_flag_match=False)
    assert report.is_valid

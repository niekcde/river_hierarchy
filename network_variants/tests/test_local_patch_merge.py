from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from network_variants.local_patch import LocalPatchCollar, LocalPatchContract, LocalPatchRasterPolicy
from network_variants.local_patch_merge import (
    _boundary_bindings,
    _trim_to_anchored_boundaries,
    merge_local_patch_variant,
)
from network_variants.local_patch_products import (
    _local_link_lineage,
    materialize_local_patch_width_length_products,
)
from network_variants.local_patch_raster import build_local_patch_base_raster
from network_variants.local_patch_variant import LocalPatchVariantResult


def _raster(path: Path, values: np.ndarray, dtype: str) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=dtype, crs="EPSG:3857", transform=from_origin(0, 100, 1, 1), nodata=0,
    ) as destination:
        destination.write(values.astype(dtype), 1)
    return path


def test_local_lineage_allows_removed_internal_parent_nodes() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(1, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(2, 0)},
        ],
        crs="EPSG:3857",
    )
    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 2, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "geometry": LineString([(1, 0), (2, 0)])},
        ],
        crs=parent_nodes.crs,
    )
    # Node 2 is legitimately absent after both parent links collapse into one.
    child_nodes = parent_nodes.loc[parent_nodes["id_node"].isin([1, 3])].copy()
    child_links = gpd.GeoDataFrame(
        [{"id_link": 3, "id_us_node": 1, "id_ds_node": 3, "id_nodes": "1, 3", "geometry": LineString([(0, 0), (2, 0)])}],
        crs=parent_nodes.crs,
    )

    lineage = _local_link_lineage(
        child_links,
        child_nodes,
        parent_links,
        parent_nodes,
        match_tolerance=0.1,
    )

    assert lineage["id_link"].astype(int).tolist() == [3]
    assert lineage.loc[0, "lineage_type"] == "collapsed_many_to_one"


def test_conservative_merge_binds_cuts_and_retains_outside_parent_links(tmp_path) -> None:
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 48:52] = 1
    labels = np.zeros_like(water, dtype=np.int32)
    labels[:, :48], labels[:, 52:] = 1, 2
    mask_path = _raster(tmp_path / "mask.tif", water, "uint8")
    labels_path = _raster(tmp_path / "labels.tif", labels, "int32")

    node_values = [(9, 90), (10, 80), (11, 65), (12, 35), (13, 20), (14, 10)]
    nodes = gpd.GeoDataFrame(
        [{
            "id_node": node_id, "is_inlet": node_id == 9, "is_outlet": node_id == 14,
            "id_links": "", "n_links": 0, "type_io": "internal", "geometry": Point(50, y),
        } for node_id, y in node_values], crs="EPSG:3857",
    )
    link_values = [(4, 9, 10), (1, 10, 11), (2, 11, 12), (3, 12, 13), (5, 13, 14)]
    node_y = dict(node_values)
    links = gpd.GeoDataFrame(
        [{
            "id_link": link_id, "id_us_node": us, "id_ds_node": ds, "id_nodes": f"{us}, {ds}",
            "is_inlet": us == 9, "is_outlet": ds == 14, "type_io": "internal",
            "link_conn": "", "idx_link": "1, 2", "scientific_value": link_id * 100.0,
            "len": float(abs(node_y[us] - node_y[ds])),
            "len_adj": float(abs(node_y[us] - node_y[ds])),
            "wid": 4.0,
            "wid_adj": 4.0,
            "wid_med": 4.0,
            "wid_adj_wet": link_id * 10.0,
            "wid_adj_routing": link_id * 11.0,
            "audit_flag": link_id % 2 == 0,
            "geometry": LineString([(50, node_y[us]), (50, node_y[ds])]),
        } for link_id, us, ds in link_values], crs="EPSG:3857",
    )
    parent_links_path, parent_nodes_path = tmp_path / "parent_links.gpkg", tmp_path / "parent_nodes.gpkg"
    links.to_file(parent_links_path, driver="GPKG")
    nodes.to_file(parent_nodes_path, driver="GPKG")

    contract = LocalPatchContract(
        example_id="synthetic", selection_type="unit", selected_unit_ids=(1,),
        domain_type="closed_unit", core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10, halo_expansion_step_m=10, maximum_halo_m=10,
            minimum_collar_context_m=10,
        ),
    )
    crop = build_local_patch_base_raster(
        contract=contract, full_links=links, full_nodes=nodes,
        cleaned_mask_path=mask_path, boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "crop",
    )

    raw_nodes = gpd.GeoDataFrame([
        {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(50, 80)},
        {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(50, 20)},
    ], crs="EPSG:3857")
    raw_links = gpd.GeoDataFrame([{
        "id_link": 0, "id_nodes": "0, 1", "is_inlet": True, "is_outlet": True,
        "idx_link": "10, 11", "geometry": LineString([(50, 80), (60, 50), (50, 20)]),
    }], crs="EPSG:3857")
    raw_links_path, raw_nodes_path = tmp_path / "raw_links.gpkg", tmp_path / "raw_nodes.gpkg"
    raw_links.to_file(raw_links_path, driver="GPKG")
    raw_nodes.to_file(raw_nodes_path, driver="GPKG")
    variant = LocalPatchVariantResult(
        status="complete_raw_local_variant", selected_unit_ids=(1,),
        output_paths={
            "raw_rivgraph_links": str(raw_links_path), "raw_rivgraph_nodes": str(raw_nodes_path),
            "local_collapsed_mask": crop.output_paths["local_base_mask"],
            "local_base_mask": crop.output_paths["local_base_mask"],
            "local_origin_mask": crop.output_paths["local_base_mask"],
            "boundary_component_labels": crop.output_paths["boundary_component_labels"],
        }, collapse_summary={},
    )

    result = merge_local_patch_variant(
        contract=contract, raster_result=crop, variant_result=variant,
        parent_links_path=parent_links_path, parent_nodes_path=parent_nodes_path,
        output_dir=tmp_path / "merge",
    )
    merged_links = gpd.read_file(result.output_paths["merged_links"])
    merged_nodes = gpd.read_file(result.output_paths["merged_nodes"])
    assert set(merged_links["id_link"].astype(int)) == {1, 3, 4, 5, 6}
    assert merged_links.set_index("id_link").loc[4, "scientific_value"] == 400.0
    assert merged_links.set_index("id_link").loc[5, "scientific_value"] == 500.0
    assert merged_links.set_index("id_link").loc[4, "audit_flag"] == np.bool_(True)
    assert merged_links.set_index("id_link").loc[5, "audit_flag"] == np.bool_(False)
    assert merged_links["audit_flag"].dtype == bool
    inserted = merged_links.loc[merged_links["id_link"].astype(int) == 6].iloc[0]
    assert (int(inserted.id_us_node), int(inserted.id_ds_node)) == (11, 12)
    assert inserted.idx_link is None
    assert set(merged_nodes["id_node"].astype(int)) == {9, 10, 11, 12, 13, 14}
    assert result.summary["validation_report"]["is_valid"] is True
    node_audit = pd.read_csv(result.output_paths["node_merge_audit"])
    assert node_audit.loc[
        node_audit["match_method"].eq("forced_anchored_boundary"),
        "raw_boundary_role_matches",
    ].all()
    assert set(
        node_audit.loc[
            node_audit["match_method"].eq("forced_anchored_boundary"), "binding_method"
        ]
    ) == {"split_local_link_at_anchored_boundary"}


def test_anchored_trim_prunes_context_and_splits_only_missing_boundary() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 11, "geometry": Point(50, 65)},
            {"id_node": 12, "geometry": Point(50, 35)},
        ],
        crs="EPSG:3857",
    )
    local_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(50, 80)},
            {"id_node": 1, "geometry": Point(50, 65)},
            {"id_node": 2, "geometry": Point(50, 50)},
            {"id_node": 3, "geometry": Point(50, 20)},
        ],
        crs="EPSG:3857",
    )
    local_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_us_node": 0, "id_ds_node": 1, "id_nodes": "0, 1", "geometry": LineString([(50, 80), (50, 65)])},
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(50, 65), (48, 50)])},
            {"id_link": 2, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(50, 65), (52, 50)])},
            {"id_link": 3, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "geometry": LineString([(50, 50), (50, 20)])},
        ],
        crs="EPSG:3857",
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(1, 2),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=10, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=11, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    trimmed_links, _, bindings = _trim_to_anchored_boundaries(
        local_links, local_nodes, parent_nodes, contract, tolerance=1.0
    )
    assert 0 not in set(trimmed_links["id_link"].astype(int))
    assert len(trimmed_links) == 3
    assert set(bindings["binding_method"]) == {
        "existing_local_node_at_anchored_boundary",
        "split_local_link_at_anchored_boundary",
    }
    source = int(bindings.loc[bindings["cut_role"].eq("upstream"), "child_id_node"].iloc[0])
    sink = int(bindings.loc[bindings["cut_role"].eq("downstream"), "child_id_node"].iloc[0])
    assert source == 1
    assert set(trimmed_links.loc[trimmed_links["id_us_node"].astype(int).eq(source), "id_link"]) == {1, 2}
    assert set(trimmed_links.loc[trimmed_links["id_ds_node"].astype(int).eq(sink), "id_link"]) == {3}


def test_global_boundary_binding_accepts_unique_role_within_one_cut_width() -> None:
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 200)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(0, -195)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 1, "id_nodes": "0, 1", "geometry": LineString([(0, 200), (0, -195)])}],
        crs=nodes.crs,
    )
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(-200, 0), (200, 0)])},
            {"role": "downstream", "geometry": LineString([(-20, -200), (20, -200)])},
        ],
        crs=nodes.crs,
    )

    bindings = _boundary_bindings(
        links,
        nodes,
        cuts,
        contract,
        tolerance=50.0,
        global_boundary_roles={"upstream": 10, "downstream": 13},
    ).set_index("cut_role")

    assert bindings.loc["upstream", "cut_distance_m"] == 200.0
    assert bindings.loc["upstream", "binding_method"] == (
        "unique_global_boundary_role_extended_tolerance"
    )
    assert bindings.loc["downstream", "binding_method"] == (
        "cut_proximity_and_raw_boundary_role"
    )


def test_internal_boundary_does_not_use_extended_role_tolerance() -> None:
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 200)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(0, -195)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 1, "id_nodes": "0, 1", "geometry": LineString([(0, 200), (0, -195)])}],
        crs=nodes.crs,
    )
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(-200, 0), (200, 0)])},
            {"role": "downstream", "geometry": LineString([(-20, -200), (20, -200)])},
        ],
        crs=nodes.crs,
    )

    with pytest.raises(ValueError, match="Upstream cut has 0 local terminal candidates"):
        _boundary_bindings(
            links,
            nodes,
            cuts,
            contract,
            tolerance=50.0,
            global_boundary_roles={"upstream": 999, "downstream": 13},
        )


def test_phase6_inherits_unchanged_and_measures_only_inserted_link(tmp_path) -> None:
    # Reuse the complete Phase-5 fixture above by running it as a compact local
    # setup with the same deterministic straight-channel inputs.
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 35:65] = 1
    labels = np.zeros_like(water, dtype=np.int32)
    labels[:, :35], labels[:, 65:] = 1, 2
    mask_path = _raster(tmp_path / "mask.tif", water, "uint8")
    labels_path = _raster(tmp_path / "labels.tif", labels, "int32")
    origin_path = _raster(tmp_path / "origin.tif", water, "uint16")

    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(50, 90)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(50, 65)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(50, 35)},
            {"id_node": 4, "is_inlet": False, "is_outlet": True, "geometry": Point(50, 10)},
        ],
        crs="EPSG:3857",
    )
    parent = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "len": 25.0, "len_adj": 24.0, "wid": 10.0, "wid_adj": 10.0, "wid_med": 10.0, "wid_adj_wet": 10.0, "wid_adj_routing": 11.0, "geometry": LineString([(50, 90), (50, 65)])},
            {"id_link": 2, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "len": 30.0, "len_adj": 29.0, "wid": 20.0, "wid_adj": 20.0, "wid_med": 20.0, "wid_adj_wet": 20.0, "wid_adj_routing": 22.0, "geometry": LineString([(50, 65), (50, 35)])},
            {"id_link": 3, "id_us_node": 3, "id_ds_node": 4, "id_nodes": "3, 4", "len": 25.0, "len_adj": 24.0, "wid": 30.0, "wid_adj": 30.0, "wid_med": 30.0, "wid_adj_wet": 30.0, "wid_adj_routing": 33.0, "geometry": LineString([(50, 35), (50, 10)])},
        ],
        crs="EPSG:3857",
    )
    child = parent.loc[parent["id_link"].isin([1, 3])].copy()
    inserted = parent.loc[parent["id_link"].eq(2)].copy()
    inserted["id_link"] = 4
    inserted["geometry"] = [LineString([(50, 65), (60, 50), (50, 35)])]
    inserted["len"] = np.nan
    inserted["len_adj"] = np.nan
    inserted["wid_adj_wet"] = np.nan
    inserted["wid_adj_routing"] = np.nan
    child = gpd.GeoDataFrame(pd.concat([child, inserted], ignore_index=True), crs=parent.crs)

    merged_links_path = tmp_path / "merged_links.gpkg"
    merged_nodes_path = tmp_path / "merged_nodes.gpkg"
    parent_path = tmp_path / "parent_width_links.gpkg"
    child.to_file(merged_links_path, driver="GPKG")
    nodes.to_file(merged_nodes_path, driver="GPKG")
    parent.to_file(parent_path, driver="GPKG")
    parent_samples_path = tmp_path / "parent_samples.csv"
    pd.DataFrame(
        [{"id_link": link_id, "sample_index": 0, "width_total": link_id * 10.0}
         for link_id in (1, 2, 3)]
    ).to_csv(parent_samples_path, index=False)

    merge_result = type("Merge", (), {
        "status": "complete_conservative_merge",
        "output_paths": {"merged_links": str(merged_links_path), "merged_nodes": str(merged_nodes_path)},
        "summary": {"match_tolerance_m": 1.0},
    })()
    variant_result = LocalPatchVariantResult(
        status="complete_raw_local_variant",
        selected_unit_ids=(1,),
        output_paths={
            "local_collapsed_mask": str(mask_path),
            "local_base_mask": str(mask_path),
            "local_origin_mask": str(origin_path),
            "boundary_component_labels": str(labels_path),
        },
        collapse_summary={},
    )
    measured_calls = []

    def fake_width_runner(changed_links, **kwargs):
        measured_calls.append(sorted(changed_links["id_link"].astype(int).tolist()))
        enriched = changed_links.copy()
        enriched["wid_adj_total"] = 44.0
        enriched["wid_adj_wet"] = 40.0
        enriched["wid_adj_dry"] = 4.0
        samples = pd.DataFrame([
            {"id_link": 4, "sample_index": 0, "width_total": 44.0, "width_wet": 40.0},
            {"id_link": 4, "sample_index": 1, "width_total": 44.0, "width_wet": 40.0},
        ])
        return enriched, samples

    def fake_routing_runner(enriched, samples, child_nodes, **kwargs):
        assert kwargs["evaluation_link_ids"] == [4]
        routed = enriched.copy()
        routed.loc[routed["id_link"].astype(int).eq(4), "wid_adj_routing"] = 41.0
        return routed, pd.DataFrame({"id_link": routed["id_link"]}), {"status": "fake_complete"}

    products = materialize_local_patch_width_length_products(
        merge_result=merge_result,
        variant_result=variant_result,
        parent_width_links_path=parent_path,
        parent_nodes_path=merged_nodes_path,
        parent_width_samples_path=parent_samples_path,
        output_dir=tmp_path / "products",
        example_id="synthetic",
        variant_id="unit_1",
        width_runner=fake_width_runner,
        routing_width_runner=fake_routing_runner,
    )
    output = gpd.read_file(products.output_paths["links_with_width_families"]).set_index("id_link")
    assert measured_calls == [[4]]
    assert output.loc[1, "wid_adj_wet"] == 10.0
    assert output.loc[1, "wid_adj_routing"] == 11.0
    assert output.loc[3, "wid_adj_wet"] == 30.0
    assert output.loc[4, "wid_adj_wet"] == 40.0
    assert output.loc[4, "wid_adj_routing"] == 41.0
    assert output.loc[4, "len"] == output.loc[4].geometry.length
    assert output.loc[1, "length_product_status"] == "inherited_unchanged_1to1"
    assert output.loc[4, "length_product_status"] == "recalculated_local_geometry"
    assert products.summary["inherited_link_count"] == 2
    assert products.summary["recalculated_link_ids"] == [4]

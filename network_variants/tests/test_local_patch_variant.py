from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
)
from network_variants.local_patch import (
    LocalPatchCollar,
    LocalPatchContract,
    LocalPatchRasterPolicy,
)
from network_variants.local_patch_raster import build_local_patch_base_raster
from network_variants.local_patch_variant import (
    _local_rivgraph_run_name,
    _normalize_rivgraph_geometries_to_skeleton,
    derive_local_exit_side_candidates,
    derive_local_exit_sides,
    materialize_local_patch_variant,
    validate_cut_terminal_graph,
    validate_raw_cut_terminals,
)


def _write_raster(path: Path, values: np.ndarray, dtype: str) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:3857",
        transform=from_origin(0, 100, 1, 1),
        nodata=0,
    ) as destination:
        destination.write(values.astype(dtype), 1)
    return path


def test_raw_geometries_use_skeleton_shape_not_outer_local_mask(tmp_path) -> None:
    skeleton_path = _write_raster(
        tmp_path / "skeleton.tif", np.ones((4, 5), dtype=np.uint8), "uint8"
    )
    reference_path = _write_raster(
        tmp_path / "reference.tif", np.ones((4, 10), dtype=np.uint8), "uint8"
    )
    links_path = tmp_path / "raw_links.gpkg"
    nodes_path = tmp_path / "raw_nodes.gpkg"
    gpd.GeoDataFrame(
        [{
            "id_link": 7,
            "id_nodes": "1, 2",
            "idx_link": "0, 1, 6",
            "geometry": LineString([(0, 0), (1000, 1000)]),
        }],
        crs="EPSG:3857",
    ).to_file(links_path, driver="GPKG")
    gpd.GeoDataFrame(
        [
            {"id_node": 1, "idx_node": 0, "geometry": Point(0, 0)},
            {"id_node": 2, "idx_node": 6, "geometry": Point(1000, 1000)},
        ],
        crs="EPSG:3857",
    ).to_file(nodes_path, driver="GPKG")

    summary = _normalize_rivgraph_geometries_to_skeleton(
        links_path,
        nodes_path,
        skeleton_path,
        reference_mask_path=reference_path,
    )

    normalized_links = gpd.read_file(links_path)
    normalized_nodes = gpd.read_file(nodes_path).set_index("id_node")
    assert summary["skeleton_shape"] == [4, 5]
    assert summary["reference_mask_shape"] == [4, 10]
    assert summary["skeleton_differs_from_reference_mask"]
    assert normalized_links.iloc[0].geometry.length < 3.0
    assert normalized_nodes.loc[1].geometry.equals(Point(0.5, 99.5))
    assert normalized_nodes.loc[2].geometry.equals(Point(1.5, 98.5))


def _inputs(tmp_path: Path):
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 48:52] = 1
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:, :48] = 1
    labels[:, 52:] = 2
    mask_path = _write_raster(tmp_path / "mask.tif", water, "uint8")
    labels_path = _write_raster(tmp_path / "labels.tif", labels, "int32")
    links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 10, "id_ds_node": 11, "wid_adj_routing": 4.0, "geometry": LineString([(50, 80), (50, 65)])},
            {"id_link": 2, "id_us_node": 11, "id_ds_node": 12, "wid_adj_routing": 4.0, "geometry": LineString([(50, 65), (50, 35)])},
            {"id_link": 3, "id_us_node": 12, "id_ds_node": 13, "wid_adj_routing": 4.0, "geometry": LineString([(50, 35), (50, 20)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "geometry": Point(50, 80)},
            {"id_node": 11, "geometry": Point(50, 65)},
            {"id_node": 12, "geometry": Point(50, 35)},
            {"id_node": 13, "geometry": Point(50, 20)},
        ],
        crs="EPSG:3857",
    )
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")
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
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10, halo_expansion_step_m=10, maximum_halo_m=10,
            minimum_collar_context_m=10,
        ),
    )
    return mask_path, labels_path, links_path, nodes_path, links, nodes, contract


def test_materialize_local_patch_variant_preserves_cuts_and_raw_outputs(tmp_path) -> None:
    mask_path, labels_path, links_path, nodes_path, links, nodes, contract = _inputs(tmp_path)
    crop = build_local_patch_base_raster(
        contract=contract,
        full_links=links,
        full_nodes=nodes,
        cleaned_mask_path=mask_path,
        boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "crop",
    )
    unit = StructuralUnit(
        unit_id=1,
        bifurcation=11,
        confluence=12,
        n_paths=2,
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=30.0,
        max_path_length=30.0,
        node_set={11, 12},
        edge_set={(11, 12, 2)},
    )

    def fake_analyzer(*args, **kwargs):
        return None, [unit], None

    rivgraph_calls = []

    def fake_rivgraph_runner(**kwargs):
        rivgraph_calls.append(kwargs)
        output = Path(kwargs["rivgraph_output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        links_output = output / "raw_links.gpkg"
        nodes_output = output / "raw_nodes.gpkg"
        skeleton_output = output / "raw_skeleton.tif"
        upstream, downstream = crop.selected_attempt.bank_cuts
        with rasterio.open(kwargs["collapsed_mask_path"]) as source:
            skeleton = source.read(1)
            skeleton_profile = source.profile.copy()
            width = source.width
            upstream_row, upstream_col = source.index(
                upstream.anchor_x, upstream.anchor_y
            )
            downstream_row, downstream_col = source.index(
                downstream.anchor_x, downstream.anchor_y
            )
        with rasterio.open(skeleton_output, "w", **skeleton_profile) as destination:
            destination.write(skeleton, 1)
        upstream_index = int(upstream_row * width + upstream_col)
        downstream_index = int(downstream_row * width + downstream_col)
        raw_nodes = gpd.GeoDataFrame(
            [
                {
                    "id_node": 0, "is_inlet": True, "is_outlet": False,
                    "idx_node": upstream_index,
                    "geometry": Point(upstream.anchor_x, upstream.anchor_y),
                },
                {
                    "id_node": 1, "is_inlet": False, "is_outlet": True,
                    "idx_node": downstream_index,
                    "geometry": Point(downstream.anchor_x, downstream.anchor_y),
                },
            ],
            crs=links.crs,
        )
        raw_links = gpd.GeoDataFrame(
            [{
                "id_link": 0, "id_nodes": "0, 1",
                "idx_link": f"{upstream_index}, {downstream_index}",
                "geometry": LineString([
                    (upstream.anchor_x, upstream.anchor_y),
                    (downstream.anchor_x, downstream.anchor_y),
                ]),
            }],
            crs=links.crs,
        )
        raw_links.to_file(links_output, driver="GPKG")
        raw_nodes.to_file(nodes_output, driver="GPKG")
        return {
            "links": links_output,
            "nodes": nodes_output,
            "skeleton": skeleton_output,
        }

    result = materialize_local_patch_variant(
        contract=contract,
        raster_result=crop,
        reviewed_links_path=links_path,
        reviewed_nodes_path=nodes_path,
        # This is deliberately inconsistent with the north-to-south local
        # crop: local cut geometry must override the global hint.
        exit_sides="EW",
        output_dir=tmp_path / "variant",
        allow_noop=True,
        network_analyzer=fake_analyzer,
        rivgraph_runner=fake_rivgraph_runner,
    )

    assert result.status == "complete_raw_local_variant"
    assert len(rivgraph_calls) == 1
    assert rivgraph_calls[0]["exit_sides"] == "NS"
    assert rivgraph_calls[0]["metadata"]["global_exit_sides"] == "EW"
    assert rivgraph_calls[0]["metadata"]["local_exit_sides"] == "NS"
    assert result.collapse_summary["global_exit_sides"] == "EW"
    assert result.collapse_summary["local_exit_sides"] == "NS"
    assert result.collapse_summary["local_exit_side_assignment"]["method"] == (
        "dominant_cut_to_cut_reference_displacement"
    )
    assert result.collapse_summary["local_exit_side_assignment"]["source_role"] == "upstream"
    assert result.collapse_summary["local_exit_side_assignment"]["sink_role"] == "downstream"
    assert result.collapse_summary["raw_cut_terminal_validation"]["undirected_cut_to_cut_path"]
    with rasterio.open(result.output_paths["local_collapsed_mask"]) as collapsed_source:
        collapsed = collapsed_source.read(1) > 0
    with rasterio.open(result.output_paths["main_bank_cut_barrier"]) as barrier_source:
        barrier = barrier_source.read(1) > 0
    assert not np.any(collapsed & barrier)
    assert Path(result.output_paths["raw_rivgraph_links"]).exists()
    assert Path(result.output_paths["raw_rivgraph_nodes"]).exists()
    assert Path(result.output_paths["manifest"]).exists()


def test_local_exit_sides_follow_cut_to_cut_displacement_not_crop_edges(tmp_path) -> None:
    mask_path, labels_path, _, _, links, nodes, contract = _inputs(tmp_path)
    crop = build_local_patch_base_raster(
        contract=contract,
        full_links=links,
        full_nodes=nodes,
        cleaned_mask_path=mask_path,
        boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "crop",
    )
    attempt = crop.selected_attempt
    assert attempt is not None
    minx, _, maxx, maxy = attempt.bounds
    upstream, downstream = attempt.bank_cuts
    curved_attempt = replace(
        attempt,
        bank_cuts=(
            replace(upstream, anchor_x=minx + 10.0, anchor_y=maxy - 1.0),
            replace(downstream, anchor_x=maxx - 2.0, anchor_y=maxy - 1.0),
        ),
    )
    curved_crop = replace(crop, attempts=(curved_attempt,), selected_attempt_index=0)

    local_exit_sides, diagnostics = derive_local_exit_sides(curved_crop)

    assert local_exit_sides == "WE"
    assert diagnostics["dominant_axis"] == "horizontal"
    assert diagnostics["cut_to_cut_delta_x_m"] > 0
    assert diagnostics["cut_to_cut_delta_y_m"] == 0

    candidates = derive_local_exit_side_candidates(curved_crop)
    assert candidates[0] == "WE"
    assert len(candidates) == 4
    assert len(set(candidates)) == len(candidates)


def test_large_compound_uses_short_deterministic_rivgraph_name(tmp_path) -> None:
    _, _, _, _, _, _, base_contract = _inputs(tmp_path)
    contract = replace(
        base_contract,
        selection_type="compound_bubble",
        domain_type="selected_compound",
        selected_unit_ids=tuple(range(1, 140)),
        selected_compound_bubble_id=4,
    )

    first = _local_rivgraph_run_name(contract)
    second = _local_rivgraph_run_name(contract)

    assert first == second
    assert first.startswith("synthetic__local_compound-bubble_bubble-4_")
    assert len(first) < 100


def test_raw_cut_terminal_validation_rejects_both_terminals_at_one_cut(tmp_path) -> None:
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(90, 40), (90, 60)])},
            {"role": "downstream", "geometry": LineString([(10, 40), (10, 60)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(12, 49)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(12, 51)},
        ],
        crs=cuts.crs,
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 0, "id_nodes": "0, 1", "geometry": LineString([(12, 49), (12, 51)])}],
        crs=cuts.crs,
    )
    cuts_path = tmp_path / "cuts.gpkg"
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    cuts.to_file(cuts_path, driver="GPKG")
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    with pytest.raises(ValueError, match="no unambiguous inlet terminal.*upstream cut"):
        validate_raw_cut_terminals(
            links_path, nodes_path, cuts_path, tolerance_m=5.0
        )


def test_raw_cut_terminal_validation_accepts_clear_relative_assignment() -> None:
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(0, -10), (0, 10)])},
            {"role": "downstream", "geometry": LineString([(1000, -10), (1000, 10)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(105, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(770, 0)},
        ],
        crs=cuts.crs,
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 0, "id_nodes": "0, 1", "geometry": LineString([(105, 0), (770, 0)])}],
        crs=cuts.crs,
    )

    result = validate_cut_terminal_graph(
        links, nodes, cuts, tolerance_m=75.0, require_directed_path=False
    )

    assert result["upstream_terminal_node_id"] == 0
    assert result["downstream_terminal_node_id"] == 1
    assert result["upstream_terminal_selection_method"] == "relative_cut_proximity"
    assert result["downstream_terminal_selection_method"] == "relative_cut_proximity"


def test_raw_cut_terminal_validation_selects_nearest_of_multiple_soft_sink_terminals() -> None:
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(0, -10), (0, 10)])},
            {"role": "downstream", "geometry": LineString([(1000, -10), (1000, 10)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(20, 0)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(500, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": True, "geometry": Point(980, 5)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(975, -5)},
        ],
        crs=cuts.crs,
    )
    links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "geometry": LineString([(20, 0), (500, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "geometry": LineString([(500, 0), (980, 5)])},
            {"id_link": 2, "id_nodes": "1, 3", "geometry": LineString([(500, 0), (975, -5)])},
        ],
        crs=cuts.crs,
    )

    result = validate_cut_terminal_graph(
        links, nodes, cuts, tolerance_m=75.0, require_directed_path=False
    )

    assert result["downstream_terminal_node_id"] == 2
    assert len(result["terminal_candidates"]["downstream"]) == 2


def test_raw_cut_terminal_validation_treats_id_nodes_as_undirected(tmp_path) -> None:
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(90, 40), (90, 60)])},
            {"role": "downstream", "geometry": LineString([(10, 40), (10, 60)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 2, "is_inlet": True, "is_outlet": False, "geometry": Point(90, 50)},
            {"id_node": 1, "is_inlet": False, "is_outlet": False, "geometry": Point(50, 50)},
            {"id_node": 0, "is_inlet": False, "is_outlet": True, "geometry": Point(10, 50)},
        ],
        crs=cuts.crs,
    )
    # The second pair is deliberately stored opposite to flow. Raw id_nodes
    # records connectivity; direction is assigned later during the merge.
    links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_nodes": "2, 1", "geometry": LineString([(90, 50), (50, 50)])},
            {"id_link": 2, "id_nodes": "0, 1", "geometry": LineString([(10, 50), (50, 50)])},
        ],
        crs=cuts.crs,
    )
    cuts_path = tmp_path / "cuts.gpkg"
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    cuts.to_file(cuts_path, driver="GPKG")
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    result = validate_raw_cut_terminals(
        links_path, nodes_path, cuts_path, tolerance_m=5.0
    )

    assert result["undirected_cut_to_cut_path"] is True

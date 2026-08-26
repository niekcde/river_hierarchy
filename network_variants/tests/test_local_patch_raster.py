from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from network_variants.local_patch import (
    LocalPatchCollar,
    LocalPatchContract,
    LocalPatchRasterPolicy,
)
from network_variants.local_patch_raster import _cut_center_points
from network_variants.local_patch_raster import _cut_is_contained_or_clipped_at_global_edge
from network_variants.local_patch_raster import _resolve_main_bank_component_ids
from network_variants.local_patch_raster import MainBankCut
from network_variants.local_patch_raster import build_local_patch_base_raster


def _write_raster(path: Path, values: np.ndarray, *, dtype: str) -> Path:
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


def _contract() -> LocalPatchContract:
    return LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream",
            base_link_id=1,
            base_upstream_node_id=10,
            base_downstream_node_id=11,
            domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream",
            base_link_id=3,
            base_upstream_node_id=12,
            base_downstream_node_id=13,
            domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10,
            halo_expansion_step_m=10,
            maximum_halo_m=10,
            minimum_collar_context_m=10,
        ),
    )


def _network() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 10, "id_ds_node": 11, "geometry": LineString([(50, 80), (50, 65)])},
            {"id_link": 2, "id_us_node": 11, "id_ds_node": 12, "geometry": LineString([(50, 65), (50, 35)])},
            {"id_link": 3, "id_us_node": 12, "id_ds_node": 13, "geometry": LineString([(50, 35), (50, 20)])},
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
    return links, nodes


def test_short_collars_extend_to_minimum_reference_centerline_context() -> None:
    links, nodes = _network()
    contract = _contract()
    contract = LocalPatchContract(
        example_id=contract.example_id,
        selection_type=contract.selection_type,
        selected_unit_ids=contract.selected_unit_ids,
        domain_type=contract.domain_type,
        core_link_ids=contract.core_link_ids,
        upstream_collar=contract.upstream_collar,
        downstream_collar=contract.downstream_collar,
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10,
            halo_expansion_step_m=10,
            maximum_halo_m=10,
            minimum_collar_context_m=20,
        ),
    )
    source, sink = _cut_center_points(
        nodes=nodes,
        links=links,
        contract=contract,
        reference_centerline=LineString([(50, 0), (50, 100)]),
    )
    assert source[1:] == (20.0, True)
    assert sink[1:] == (20.0, True)
    assert source[0].equals_exact(Point(50, 85), 1e-9)
    assert sink[0].equals_exact(Point(50, 15), 1e-9)


def test_long_collar_is_cut_at_outer_parent_node() -> None:
    links, nodes = _network()
    source, sink = _cut_center_points(
        nodes=nodes,
        links=links,
        contract=_contract(),
        reference_centerline=LineString([(50, 0), (50, 100)]),
    )
    assert source[1:] == (15.0, False)
    assert sink[1:] == (15.0, False)
    assert source[0].equals_exact(Point(50, 80), 1e-9)
    assert sink[0].equals_exact(Point(50, 20), 1e-9)


def test_context_cut_clamps_to_global_reference_centerline_boundary() -> None:
    links, nodes = _network()
    contract = _contract()
    contract = LocalPatchContract(
        example_id=contract.example_id,
        selection_type=contract.selection_type,
        selected_unit_ids=contract.selected_unit_ids,
        domain_type=contract.domain_type,
        core_link_ids=contract.core_link_ids,
        upstream_collar=contract.upstream_collar,
        downstream_collar=contract.downstream_collar,
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10,
            halo_expansion_step_m=10,
            maximum_halo_m=10,
            minimum_collar_context_m=40,
        ),
    )

    source, sink = _cut_center_points(
        nodes=nodes,
        links=links,
        contract=contract,
        reference_centerline=LineString([(50, 10), (50, 90)]),
    )

    assert source[0].equals_exact(Point(50, 90), 1e-9)
    assert sink[0].equals_exact(Point(50, 10), 1e-9)
    assert source[1:] == (25.0, True)
    assert sink[1:] == (25.0, True)


def test_equal_centerline_projections_follow_unique_directed_parent_continuation() -> None:
    links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_us_node": 9, "id_ds_node": 10, "geometry": LineString([(-30, 70), (-10, 70)])},
            {"id_link": 1, "id_us_node": 10, "id_ds_node": 11, "geometry": LineString([(-10, 70), (0, 70)])},
            {"id_link": 2, "id_us_node": 11, "id_ds_node": 12, "geometry": LineString([(0, 70), (0, 30)])},
            {"id_link": 3, "id_us_node": 12, "id_ds_node": 13, "geometry": LineString([(0, 30), (10, 30)])},
            {"id_link": 4, "id_us_node": 13, "id_ds_node": 14, "geometry": LineString([(10, 30), (30, 30)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": node_id, "geometry": Point(x, y)}
            for node_id, x, y in (
                (9, -30, 70), (10, -10, 70), (11, 0, 70),
                (12, 0, 30), (13, 10, 30), (14, 30, 30),
            )
        ],
        crs=links.crs,
    )
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
            minimum_collar_context_m=20,
        ),
    )
    source, sink = _cut_center_points(
        nodes=nodes,
        links=links,
        contract=contract,
        reference_centerline=LineString([(0, 0), (0, 100)]),
    )
    assert source[0].equals_exact(Point(-20, 70), 1e-9)
    assert sink[0].equals_exact(Point(20, 30), 1e-9)
    assert source[1:] == (20.0, True)
    assert sink[1:] == (20.0, True)


def test_equal_centerline_projections_use_global_flow_when_continuation_branches() -> None:
    links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_us_node": 8, "id_ds_node": 14, "geometry": LineString([(0, 90), (-10, 80)])},
            {"id_link": 1, "id_us_node": 14, "id_ds_node": 9, "geometry": LineString([(-10, 80), (0, 70)])},
            {"id_link": 2, "id_us_node": 8, "id_ds_node": 15, "geometry": LineString([(0, 90), (10, 80)])},
            {"id_link": 3, "id_us_node": 15, "id_ds_node": 9, "geometry": LineString([(10, 80), (0, 70)])},
            {"id_link": 4, "id_us_node": 9, "id_ds_node": 10, "geometry": LineString([(0, 70), (10, 70)])},
            {"id_link": 5, "id_us_node": 10, "id_ds_node": 11, "geometry": LineString([(10, 70), (0, 30)])},
            {"id_link": 6, "id_us_node": 11, "id_ds_node": 12, "geometry": LineString([(0, 30), (10, 30)])},
            {"id_link": 7, "id_us_node": 12, "id_ds_node": 16, "geometry": LineString([(10, 30), (-10, 20)])},
            {"id_link": 8, "id_us_node": 16, "id_ds_node": 13, "geometry": LineString([(-10, 20), (0, 10)])},
            {"id_link": 9, "id_us_node": 12, "id_ds_node": 17, "geometry": LineString([(10, 30), (10, 20)])},
            {"id_link": 10, "id_us_node": 17, "id_ds_node": 13, "geometry": LineString([(10, 20), (0, 10)])},
            {"id_link": 11, "id_us_node": 13, "id_ds_node": 18, "geometry": LineString([(0, 10), (0, 0)])},
        ],
        crs="EPSG:3857",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": node_id, "geometry": Point(x, y)}
            for node_id, x, y in (
                (8, 0, 90), (14, -10, 80), (15, 10, 80),
                (9, 0, 70), (10, 10, 70), (11, 0, 30), (12, 10, 30),
                (16, -10, 20), (17, 10, 20), (13, 0, 10), (18, 0, 0),
            )
        ],
        crs=links.crs,
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(5,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=4, base_upstream_node_id=9,
            base_downstream_node_id=10, domain_boundary_node_id=10,
            outer_splice_anchor_node_id=9,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=6, base_upstream_node_id=11,
            base_downstream_node_id=12, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=12,
        ),
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10, halo_expansion_step_m=10, maximum_halo_m=10,
            minimum_collar_context_m=20,
        ),
    )

    source, sink = _cut_center_points(
        nodes=nodes,
        links=links,
        contract=contract,
        reference_centerline=LineString([(0, 0), (0, 100)]),
    )

    assert source[0].equals_exact(Point(0, 80), 1e-9)
    assert sink[0].equals_exact(Point(0, 20), 1e-9)
    assert source[1:] == (20.0, True)
    assert sink[1:] == (20.0, True)


def _edge_cut(*, outside_tail: float) -> MainBankCut:
    return MainBankCut(
        role="upstream",
        anchor_node_id=1,
        anchor_x=50.0,
        anchor_y=5.0,
        first_bank_component_id=1,
        second_bank_component_id=2,
        first_endpoint_x=-float(outside_tail),
        first_endpoint_y=5.0,
        second_endpoint_x=60.0,
        second_endpoint_y=5.0,
        first_bank_distance_m=50.0,
        second_bank_distance_m=10.0,
        bank_overlap_m=4.0,
        inner_boundary_node_id=2,
        outer_anchor_node_id=1,
        context_distance_from_boundary_m=10.0,
        extended_beyond_outer_anchor=False,
    )


def test_global_edge_accepts_only_short_terminal_bank_overlap_tail() -> None:
    raster = box(0, 0, 100, 100)
    assert _cut_is_contained_or_clipped_at_global_edge(
        _edge_cut(outside_tail=3.0),
        crop_polygon=raster,
        full_raster_polygon=raster,
        pixel_size=1.0,
    )
    assert not _cut_is_contained_or_clipped_at_global_edge(
        _edge_cut(outside_tail=20.0),
        crop_polygon=raster,
        full_raster_polygon=raster,
        pixel_size=1.0,
    )


def test_main_bank_summary_ignores_macos_sidecar(tmp_path) -> None:
    water, land = _base_arrays()
    mask_path = _write_raster(tmp_path / "mask.tif", water, dtype="uint8")
    labels_path = _write_raster(tmp_path / "labels.tif", land, dtype="int32")
    (tmp_path / "._land_component_summary.csv").write_bytes(b"\x00\x05\xb0invalid metadata")
    (tmp_path / "land_component_summary.csv").write_text(
        "raster_value,n_pixels\n3,4000\n2,3900\n4,20\n",
        encoding="utf-8",
    )

    with rasterio.open(labels_path) as labels_source, rasterio.open(mask_path) as mask_source:
        assert _resolve_main_bank_component_ids(labels_source, mask_source) == (2, 3)


def _base_arrays(*, lateral_branch: bool = False) -> tuple[np.ndarray, np.ndarray]:
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 48:52] = 1
    if lateral_branch:
        water[48:52, 50:] = 1
    land = np.zeros((100, 100), dtype=np.int32)
    land[:, :48] = 1
    land[:, 52:] = 2
    if lateral_branch:
        land[48:52, 52:] = 0
    return water, land


def test_build_local_patch_base_raster_accepts_two_crossings_and_stable_banks(
    tmp_path,
) -> None:
    water, land = _base_arrays()
    mask_path = _write_raster(tmp_path / "mask.tif", water, dtype="uint8")
    labels_path = _write_raster(tmp_path / "labels.tif", land, dtype="int32")
    links, nodes = _network()

    result = build_local_patch_base_raster(
        contract=_contract(),
        full_links=links,
        full_nodes=nodes,
        cleaned_mask_path=mask_path,
        boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "output",
    )

    assert result.is_safe
    selected = result.selected_attempt
    assert selected is not None
    assert len(selected.crossings) == 2
    assert {crossing.assigned_role for crossing in selected.crossings} == {
        "upstream",
        "downstream",
    }
    assert {crossing.bank_component_ids for crossing in selected.crossings} == {(1, 2)}
    assert selected.unauthorized_boundary_section_count == 0
    assert selected.cuts_contained_in_crop == (True, True)
    assert len(selected.bank_cuts) == 2
    assert all(
        cut.geometry.distance(Point(cut.anchor_x, cut.anchor_y)) <= 1e-9
        for cut in selected.bank_cuts
    )
    assert selected.full_raster_fraction < 0.3
    for output_path in result.output_paths.values():
        assert Path(output_path).exists()


def test_build_local_patch_base_raster_rejects_an_extra_connected_water_crossing(
    tmp_path,
) -> None:
    water, land = _base_arrays(lateral_branch=True)
    mask_path = _write_raster(tmp_path / "mask.tif", water, dtype="uint8")
    labels_path = _write_raster(tmp_path / "labels.tif", land, dtype="int32")
    links, nodes = _network()

    result = build_local_patch_base_raster(
        contract=_contract(),
        full_links=links,
        full_nodes=nodes,
        cleaned_mask_path=mask_path,
        boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "output",
    )

    assert not result.is_safe
    assert result.status == "fallback_full_raster"
    assert result.selected_attempt_index is None
    assert result.attempts[-1].reason_codes == (
        "unauthorized_main_water_crop_edge_contact",
    )
    assert (tmp_path / "output/local_patch_raster_plan.json").exists()

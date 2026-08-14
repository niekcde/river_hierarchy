from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
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
    derive_local_exit_sides,
    materialize_local_patch_variant,
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
        links_output.write_text("placeholder", encoding="utf-8")
        nodes_output.write_text("placeholder", encoding="utf-8")
        return {"links": links_output, "nodes": nodes_output}

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
    assert result.collapse_summary["local_exit_side_assignment"]["source_role"] == "upstream"
    assert result.collapse_summary["local_exit_side_assignment"]["sink_role"] == "downstream"
    with rasterio.open(result.output_paths["local_collapsed_mask"]) as collapsed_source:
        collapsed = collapsed_source.read(1) > 0
    with rasterio.open(result.output_paths["main_bank_cut_barrier"]) as barrier_source:
        barrier = barrier_source.read(1) > 0
    assert not np.any(collapsed & barrier)
    assert Path(result.output_paths["raw_rivgraph_links"]).exists()
    assert Path(result.output_paths["raw_rivgraph_nodes"]).exists()
    assert Path(result.output_paths["manifest"]).exists()


def test_local_exit_sides_jointly_resolve_two_cuts_nearest_the_same_side(tmp_path) -> None:
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

    assert local_exit_sides == "NE"
    assert len(set(local_exit_sides)) == 2
    assert diagnostics["source_side_distances_m"]["N"] == 1.0
    assert diagnostics["sink_side_distances_m"]["N"] == 1.0

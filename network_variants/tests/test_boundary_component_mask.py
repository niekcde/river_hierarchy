import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from network_variants.build_boundary_component_mask import (
    _ComponentUnion,
    _apply_resolved_ids,
    _label_panel_land,
    _resolved_component_ids,
    _union_overlapping_panel_labels,
    _write_panel_land_labels,
    build_boundary_component_mask,
    build_panelized_boundary_component_labels,
)


def test_component_union_merges_transitive_panel_contacts() -> None:
    components = _ComponentUnion()
    components.union(101, 205)
    components.union(205, 309)

    assert components.find(101) == components.find(309)


def test_label_panel_land_uses_four_neighbour_connectivity() -> None:
    water = np.array([[0, 1], [1, 0]], dtype=bool)

    labels, count = _label_panel_land(water)

    assert count == 2
    assert labels[0, 0] != labels[1, 1]


def test_overlap_contacts_union_matching_local_components() -> None:
    components = _ComponentUnion()
    _union_overlapping_panel_labels(
        components,
        np.array([[1, 0], [2, 2]], dtype=np.int32),
        np.array([[4, 0], [3, 3]], dtype=np.int32),
        left_offset=100,
        right_offset=200,
    )

    assert components.find(101) == components.find(204)
    assert components.find(102) == components.find(203)


def test_resolved_component_ids_are_stable_across_merged_locals() -> None:
    components = _ComponentUnion()
    components.union(10, 30)
    components.add(20)

    resolved = _resolved_component_ids(components)

    assert resolved[10] == resolved[30]
    assert resolved[10] != resolved[20]


def test_apply_resolved_ids_preserves_water_and_offsets_land_values() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    actual = _apply_resolved_ids(labels, offset=100, resolved_ids={101: 7, 102: 9})

    assert actual.tolist() == [[1, 8], [10, 1]]


def test_tiled_panel_labels_match_whole_array_across_tile_seams(tmp_path) -> None:
    water = np.zeros((8, 9), dtype=np.uint8)
    water[1:7, 4] = 1
    water[4, 4] = 0
    water[6, 0:3] = 1
    mask_path = tmp_path / "panel_mask.tif"
    labels_path = tmp_path / "panel_labels.tif"
    profile = {
        "driver": "GTiff",
        "height": water.shape[0],
        "width": water.shape[1],
        "count": 1,
        "dtype": "uint8",
        "transform": from_origin(0, water.shape[0], 1, 1),
        "crs": "EPSG:3857",
    }
    with rasterio.open(mask_path, "w", **profile) as destination:
        destination.write(water, 1)

    expected, expected_count = _label_panel_land(water > 0)
    actual_count = _write_panel_land_labels(mask_path, labels_path, tile_size=3)
    with rasterio.open(labels_path) as source:
        actual = source.read(1)

    assert actual_count == expected_count
    for expected_id in np.unique(expected[expected > 0]):
        assert len(np.unique(actual[expected == expected_id])) == 1
    for actual_id in np.unique(actual[actual > 0]):
        assert len(np.unique(expected[actual == actual_id])) == 1
    assert not labels_path.with_name("panel_labels_raw.tif").exists()


def test_panelized_builder_writes_global_label_artifacts(tmp_path) -> None:
    transform = from_origin(0, 4, 1, 1)
    full = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    profile = {"driver": "GTiff", "height": 4, "width": 4, "count": 1, "dtype": "uint8", "transform": transform, "crs": "EPSG:3857"}
    base = tmp_path / "toy_cleaned.tif"
    with rasterio.open(base, "w", **profile) as dst: dst.write(full, 1)
    panel_root = tmp_path / "panels"
    for panel_id, cols in (("P001", slice(0, 3)), ("P002", slice(1, 4))):
        path = panel_root / panel_id / "mask.tif"; path.parent.mkdir(parents=True)
        local_profile = dict(profile, width=3, transform=from_origin(cols.start, 4, 1, 1))
        with rasterio.open(path, "w", **local_profile): pass
        with rasterio.open(path, "r+") as dst: dst.write(full[:, cols], 1)
    plan = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame({"panel_id": ["P001", "P002"]}, geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)], crs="EPSG:3857").to_file(plan, layer="panel_cores", driver="GPKG")
    gpd.GeoDataFrame(
        {"panel_id": ["P001", "P002"]},
        geometry=[box(0, 0, 3, 4), box(1, 0, 4, 4)],
        crs="EPSG:3857",
    ).to_file(plan, layer="processing_areas", driver="GPKG")
    centerline = tmp_path / "centerline.gpkg"
    gpd.GeoDataFrame(geometry=[LineString([(1.5, 2.5), (2.5, 1.5)])], crs="EPSG:3857").to_file(centerline, driver="GPKG")
    outputs = build_panelized_boundary_component_labels(
        cleaned_mask_path=base,
        centerline_path=centerline,
        panel_plan_path=plan,
        panel_root=panel_root,
        output_dir=tmp_path / "boundary",
        tile_size=2,
    )
    legacy = build_boundary_component_mask(
        cleaned_mask_path=base,
        centerline_path=centerline,
        output_dir=tmp_path / "legacy_boundary",
    )
    assert all((tmp_path / "boundary" / name).exists() for name in ("toy_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif", "toy_centerline_border_extensions_snapped.gpkg", "toy_full_extent_land_component_summary_with_snapped_buffered_border_cut.csv"))
    with rasterio.open(outputs["labels"]) as result, rasterio.open(legacy["labels"]) as reference:
        actual = result.read(1)
        expected = reference.read(1)
    assert actual.shape == full.shape
    assert np.array_equal(actual == 1, expected == 1)
    for expected_id in np.unique(expected[expected > 1]):
        assert len(np.unique(actual[expected == expected_id])) == 1
    for actual_id in np.unique(actual[actual > 1]):
        assert len(np.unique(expected[actual == actual_id])) == 1

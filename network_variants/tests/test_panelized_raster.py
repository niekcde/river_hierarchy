from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

import geopandas as gpd

import network_variants.panelized_raster as panelized_raster_module
from network_variants.panelized_raster import apply_panel_collapse_edit, classify_panels_for_footprint, compute_panelized_base_width_families, compute_panelized_child_width_patch, compute_panelized_segmented_width_patch, compute_segmented_width_patch, materialize_affected_panel_masks, merge_panel_two_bank_widths, prepare_panel_state_workspace, regenerate_affected_panels, write_panel_origin_and_bank_labels, write_panelized_state_mosaic, write_panelized_state_virtual_mosaic, write_panelized_virtual_support_rasters, write_state_bank_labels_windowed


def _write(path: Path, values: np.ndarray) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1, dtype=values.dtype, transform=from_origin(0, values.shape[0], 1, 1), compress="lzw") as dst:
        dst.write(values, 1)


def test_windowed_bank_labels_and_panel_edit(tmp_path: Path) -> None:
    labels = np.array([[2, 2, 2], [2, 3, 2], [2, 2, 2]], dtype="int32")
    origin = np.array([[1, 1, 1], [1, 101, 1], [1, 1, 1]], dtype="uint16")
    labels_path, origin_path, banks_path = tmp_path / "labels.tif", tmp_path / "origin.tif", tmp_path / "banks.tif"
    _write(labels_path, labels); _write(origin_path, origin)
    assert write_state_bank_labels_windowed(boundary_component_labels_path=labels_path, origin_mask_path=origin_path, output_path=banks_path) == [3]
    with rasterio.open(banks_path) as src:
        assert src.read(1)[1, 1] == 1

    mask_path, collapsed_path = tmp_path / "mask.tif", tmp_path / "collapsed.tif"
    _write(mask_path, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="uint8"))
    summary = apply_panel_collapse_edit(source_mask_path=mask_path, output_mask_path=collapsed_path, footprint=box(0, 0, 3, 3))
    assert summary["added_pixels"] == 1


def test_classify_and_materialize_affected_panels(tmp_path: Path) -> None:
    plan = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame({"panel_id": ["P001", "P002", "P003"]}, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)], crs="EPSG:32611").to_file(plan, layer="panel_cores", driver="GPKG")
    classified = classify_panels_for_footprint(panel_plan_path=plan, footprint=box(8, 3, 9, 7), halo_distance=3)
    assert classified == {"affected": ["P001"], "halo_only": ["P002"], "reusable": ["P003"]}
    panel_root = tmp_path / "panels"
    for panel_id in ("P001", "P002"):
        (panel_root / panel_id).mkdir(parents=True)
        _write(panel_root / panel_id / "mask.tif", np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="uint8"))
    result = materialize_affected_panel_masks(panel_root=panel_root, output_root=tmp_path / "edited", panel_ids=classified["affected"], footprint=box(0, 0, 3, 3))
    assert result["P001"]["added_pixels"] == 1
    assert not (tmp_path / "edited" / "P002" / "mask.tif").exists()


def test_panel_state_workspace_reuses_only_unchanged_panels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for panel_id in ("P001", "P002"):
        (source / panel_id).mkdir(parents=True)
    state = tmp_path / "state"
    assert prepare_panel_state_workspace(source_panel_root=source, state_panel_root=state, affected_panel_ids=["P001"]) == {"P001": "affected", "P002": "reused"}
    assert not (state / "P001").is_symlink()
    assert (state / "P002").is_symlink()
    calls = []
    regenerate_affected_panels(state_panel_root=state, affected_panel_ids=["P001"], run_name="demo", exit_sides_by_panel={"P001": "SE"}, panel_runner=lambda **kwargs: calls.append(kwargs))
    assert calls[0]["name"] == "demo_P001"


def test_panel_origin_and_bank_labels(tmp_path: Path) -> None:
    base, collapsed, labels = tmp_path / "base.tif", tmp_path / "collapsed.tif", tmp_path / "labels.tif"
    _write(base, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="uint8")); _write(collapsed, np.ones((3, 3), dtype="uint8")); _write(labels, np.array([[2, 2, 2], [2, 3, 2], [2, 2, 2]], dtype="int32"))
    assert write_panel_origin_and_bank_labels(base_mask_path=base, collapsed_mask_path=collapsed, boundary_component_labels_path=labels, origin_output_path=tmp_path / "origin.tif", bank_output_path=tmp_path / "bank.tif") == [3]


def test_panelized_base_widths_use_regular_two_bank_schema(tmp_path: Path) -> None:
    bank_labels = np.ones((9, 9), dtype="int32")
    bank_labels[:, :3] = 2
    bank_labels[:, 6:] = 3
    origin = np.zeros((9, 9), dtype="uint16")
    origin[:, 3:6] = 1
    bank_path, origin_path = tmp_path / "banks.tif", tmp_path / "origin.tif"
    _write(bank_path, bank_labels)
    _write(origin_path, origin)
    links = gpd.GeoDataFrame(
        {"id_link": [7]},
        geometry=[LineString([(4.5, 7.5), (4.5, 1.5)])],
        crs="EPSG:32611",
    )
    source_samples = pd.DataFrame(
        {
            "id_link": [7, 7, 7],
            "sample_index": [10, 11, 12],
            "width": [3.0, 3.0, 3.0],
            "x": [4.5, 4.5, 4.5],
            "y": [7.5, 4.5, 1.5],
        }
    )

    enriched, samples = compute_panelized_base_width_families(
        links,
        source_samples,
        bank_labels_path=bank_path,
        origin_mask_path=origin_path,
    )

    assert enriched.loc[0, "width_method"] == "two_bank_panelized_windowed"
    assert enriched.loc[0, "n_invalid_width_samples"] == 0
    assert samples["sample_index"].tolist() == [10, 11, 12]
    assert {
        "width_total",
        "width_wet",
        "width_dry",
        "left_component_id",
        "right_component_id",
        "qc_relative_dt_diff",
        "width_invalid",
    }.issubset(samples.columns)
    assert set(samples["left_component_id"]).union(samples["right_component_id"]) == {2, 3}
    assert np.allclose(samples["width_total"], 4.0)
    assert np.allclose(samples["width_wet"], 3.0)
    assert np.allclose(samples["width_dry"], 0.0)


def test_child_width_patch_recomputes_only_affected_panel_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child_links = gpd.GeoDataFrame(
        {"id_link": [10, 11]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(2, 0), (3, 0)])],
        crs="EPSG:32611",
    )
    child_merged = gpd.GeoDataFrame(
        {
            "global_link_id": [10, 11],
            "source_segments": ["P001:1:0", "P002:2:0"],
        },
        geometry=child_links.geometry,
        crs=child_links.crs,
    )
    parent_merged = gpd.GeoDataFrame(
        {
            "global_link_id": [1, 2],
            "source_segments": ["P001:1:0", "P002:1:0"],
        },
        geometry=child_links.geometry,
        crs=child_links.crs,
    )
    parent_width_links = gpd.GeoDataFrame(
        {"id_link": [1, 2], "width_metric": [10.0, 20.0]},
        geometry=child_links.geometry,
        crs=child_links.crs,
    )
    parent_samples = pd.DataFrame(
        {
            "id_link": [1, 2],
            "sample_index": [0, 0],
            "width_metric_sample": [10.0, 20.0],
        }
    )
    child_source_samples = pd.DataFrame(
        {
            "id_link": [10, 11],
            "sample_index": [0, 0],
            "width": [1.0, 1.0],
            "x": [0.5, 2.5],
            "y": [0.0, 0.0],
        }
    )
    calls: list[list[int]] = []

    def fake_widths(links, samples, **kwargs):
        calls.append(links["id_link"].astype(int).tolist())
        enriched = links.copy()
        enriched["width_metric"] = 99.0
        output_samples = pd.DataFrame(
            {
                "id_link": links["id_link"].astype(int),
                "sample_index": 0,
                "width_metric_sample": 99.0,
            }
        )
        return enriched, output_samples

    monkeypatch.setattr(panelized_raster_module, "compute_panelized_base_width_families", fake_widths)
    enriched, samples, summary = compute_panelized_child_width_patch(
        child_links=child_links,
        child_source_samples=child_source_samples,
        child_merged_links=child_merged,
        parent_merged_links=parent_merged,
        parent_width_links=parent_width_links,
        parent_width_samples=parent_samples,
        affected_panel_ids=["P002"],
        bank_labels_path=tmp_path / "unused_banks.tif",
        origin_mask_path=tmp_path / "unused_origin.tif",
    )

    assert calls == [[11]]
    assert enriched.set_index("id_link")["width_metric"].to_dict() == {10: 10.0, 11: 99.0}
    assert samples.set_index("id_link")["width_metric_sample"].to_dict() == {10: 10.0, 11: 99.0}
    assert summary["n_recomputed_links"] == 1
    assert summary["n_reused_links"] == 1


def test_segmented_width_patch_recomputes_only_split_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_links = gpd.GeoDataFrame(
        {"id_link": [0, 1]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 5), (10, 5)]),
        ],
        crs="EPSG:32611",
    )
    segmented_links = gpd.GeoDataFrame(
        {"id_link": [0, 1, 2]},
        geometry=[
            LineString([(0, 0), (5, 0)]),
            LineString([(0, 5), (10, 5)]),
            LineString([(5, 0), (10, 0)]),
        ],
        crs=source_links.crs,
    )
    source_width_links = source_links.copy()
    source_width_links["width_metric"] = [10.0, 20.0]
    source_samples = pd.DataFrame(
        {
            "id_link": [0, 0, 0, 1, 1],
            "sample_index": [0, 1, 2, 0, 1],
            "x": [0.0, 5.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 0.0, 5.0, 5.0],
            "dt_width_global": [10.0, 10.0, 10.0, 20.0, 20.0],
            "width_metric_sample": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )
    origin_path = tmp_path / "origin.tif"
    _write(origin_path, np.ones((12, 12), dtype="uint16"))
    calls: list[list[int]] = []

    def fake_widths(links, samples, **kwargs):
        calls.append(sorted(links["id_link"].astype(int).tolist()))
        assert set(samples["id_link"].astype(int)) == {0, 2}
        enriched = links.copy()
        enriched["width_metric"] = 99.0
        output_samples = samples.rename(
            columns={"width": "width_metric_sample"}
        ).copy()
        return enriched, output_samples

    monkeypatch.setattr(
        panelized_raster_module,
        "compute_panelized_base_width_families",
        fake_widths,
    )
    enriched, samples, summary = compute_panelized_segmented_width_patch(
        segmented_links=segmented_links,
        source_links=source_links,
        source_width_links=source_width_links,
        source_width_samples=source_samples,
        bank_labels_path=tmp_path / "unused_banks.tif",
        origin_mask_path=origin_path,
    )

    assert calls == [[0, 2]]
    assert enriched.set_index("id_link")["width_metric"].to_dict() == {
        0: 99.0,
        1: 20.0,
        2: 99.0,
    }
    assert set(samples["id_link"].astype(int)) == {0, 1, 2}
    assert summary["n_recomputed_links"] == 2
    assert summary["n_reused_links"] == 1
    assert summary["source_link_by_recomputed_link"] == {"0": 0, "2": 0}


def test_segmented_width_patch_reuses_unique_exact_secondary_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_links = gpd.GeoDataFrame(
        {"id_link": [0, 1]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 5), (10, 5)]),
        ],
        crs="EPSG:32611",
    )
    segmented_links = gpd.GeoDataFrame(
        {"id_link": [0, 1, 2]},
        geometry=[
            LineString([(0, 0), (5, 0)]),
            LineString([(0, 5), (10, 5)]),
            LineString([(5, 0), (10, 0)]),
        ],
        crs=source_links.crs,
    )
    source_width_links = source_links.copy()
    source_width_links["width_metric"] = [10.0, 20.0]
    source_samples = pd.DataFrame(
        {
            "id_link": [0, 0, 0, 1, 1],
            "sample_index": [0, 1, 2, 0, 1],
            "x": [0.0, 5.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 0.0, 5.0, 5.0],
            "dt_width_global": [10.0, 10.0, 10.0, 20.0, 20.0],
            "width_metric_sample": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )
    secondary_links = gpd.GeoDataFrame(
        {"id_link": [9], "width_metric": [42.0]},
        geometry=[LineString([(10, 0), (5, 0)])],
        crs=source_links.crs,
    )
    secondary_samples = pd.DataFrame(
        {
            "id_link": [9, 9],
            "sample_index": [0, 1],
            "x": [5.0, 10.0],
            "y": [0.0, 0.0],
            "dt_width_global": [42.0, 42.0],
            "width_metric_sample": [42.0, 42.0],
        }
    )
    origin_path = tmp_path / "origin.tif"
    _write(origin_path, np.ones((12, 12), dtype="uint16"))
    calls: list[list[int]] = []

    def fake_widths(links, samples, **kwargs):
        calls.append(sorted(links["id_link"].astype(int).tolist()))
        enriched = links.copy()
        enriched["width_metric"] = 99.0
        return enriched, samples.rename(columns={"width": "width_metric_sample"})

    monkeypatch.setattr(
        panelized_raster_module,
        "compute_panelized_base_width_families",
        fake_widths,
    )
    enriched, samples, summary = compute_segmented_width_patch(
        segmented_links=segmented_links,
        source_links=source_links,
        source_width_links=source_width_links,
        source_width_samples=source_samples,
        secondary_source_links=secondary_links,
        secondary_source_width_links=secondary_links,
        secondary_source_width_samples=secondary_samples,
        bank_labels_path=tmp_path / "unused_banks.tif",
        origin_mask_path=origin_path,
    )

    assert calls == [[0]]
    assert enriched.set_index("id_link")["width_metric"].to_dict() == {
        0: 99.0,
        1: 20.0,
        2: 42.0,
    }
    assert set(samples["id_link"].astype(int)) == {0, 1, 2}
    assert summary["n_recomputed_links"] == 1
    assert summary["n_reused_primary_links"] == 1
    assert summary["n_reused_secondary_links"] == 1
    assert summary["secondary_source_link_by_reused_link"] == {"2": 9}


def test_state_mosaic_intersects_cropped_panel_bounds(tmp_path: Path) -> None:
    parent = tmp_path / "parent.tif"
    panel_root = tmp_path / "panels"
    (panel_root / "P001").mkdir(parents=True)
    parent_values = np.zeros((10, 10), dtype="uint8")
    panel_values = np.zeros((6, 6), dtype="uint8")
    panel_values[2, 2] = 1
    _write(parent, parent_values)
    with rasterio.open(
        panel_root / "P001" / "mask.tif",
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 10, 1, 1),
    ) as destination:
        destination.write(panel_values, 1)
    plan = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame(
        {"panel_id": ["P001"]},
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:32611",
    ).to_file(plan, layer="panel_cores", driver="GPKG")

    output = write_panelized_state_mosaic(
        parent_mask_path=parent,
        panel_plan_path=plan,
        state_panel_root=panel_root,
        affected_panel_ids=["P001"],
        output_mask_path=tmp_path / "child.tif",
    )

    with rasterio.open(output) as source:
        values = source.read(1)
    assert int(values.sum()) == 1
    assert values[2, 2] == 1


def test_virtual_state_mosaic_and_support_rasters_match_materialized_values(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.tif"
    labels = tmp_path / "labels.tif"
    panel_root = tmp_path / "panels"
    (panel_root / "P001").mkdir(parents=True)
    parent_values = np.zeros((10, 10), dtype="uint8")
    panel_values = np.zeros((6, 6), dtype="uint8")
    panel_values[2, 2] = 1
    _write(parent, parent_values)
    _write(labels, np.full((10, 10), 2, dtype="int32"))
    with rasterio.open(
        panel_root / "P001" / "mask.tif",
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 10, 1, 1),
    ) as destination:
        destination.write(panel_values, 1)
    plan = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame(
        {"panel_id": ["P001"]},
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:32611",
    ).to_file(plan, layer="panel_cores", driver="GPKG")
    materialized = write_panelized_state_mosaic(
        parent_mask_path=parent,
        panel_plan_path=plan,
        state_panel_root=panel_root,
        affected_panel_ids=["P001"],
        output_mask_path=tmp_path / "materialized.tif",
    )
    virtual, publication = write_panelized_state_virtual_mosaic(
        parent_mask_path=parent,
        panel_plan_path=plan,
        state_panel_root=panel_root,
        affected_panel_ids=["P001"],
        output_dir=tmp_path / "virtual",
    )
    with rasterio.open(materialized) as expected, rasterio.open(virtual) as actual:
        assert np.array_equal(actual.read(1), expected.read(1))
    origin, banks, removed, support = write_panelized_virtual_support_rasters(
        base_mask_path=parent,
        collapsed_patch_path=publication["overlay_path"],
        boundary_component_labels_path=labels,
        output_dir=tmp_path / "support",
    )
    with rasterio.open(origin) as source:
        values = source.read(1)
        assert values[2, 2] == 101
        assert int(np.count_nonzero(values)) == 1
    with rasterio.open(banks) as source:
        assert source.shape == (10, 10)
        # Component 2 is absorbed locally, and therefore remapped across its
        # complete global extent by the virtual raster.
        assert np.all(source.read(1) == 1)
    # The newly wetted pixel overlaps component 2, so the state-bank builder
    # correctly records that component as absorbed by the collapsed channel.
    assert removed == [2]
    assert support["origin"]["overlay_path"].endswith("origin_mask_patch.tif")


def test_merge_panel_two_bank_widths(tmp_path: Path) -> None:
    line = LineString([(0, 0), (2, 0)])
    merged_path, segments_path = tmp_path / "merged.gpkg", tmp_path / "segments.gpkg"
    gpd.GeoDataFrame({"global_link_id": [7], "source_segments": ["P001:3:0"]}, geometry=[line], crs="EPSG:32611").to_file(merged_path, layer="merged_links", driver="GPKG")
    gpd.GeoDataFrame({"segment_id": ["P001:3:0"], "panel_id": ["P001"], "source_link_id": [3]}, geometry=[line], crs="EPSG:32611").to_file(segments_path, layer="core_segments", driver="GPKG")
    sample_dir = tmp_path / "panel_widths" / "P001" / "widths"; sample_dir.mkdir(parents=True)
    pd.DataFrame({"id_link": [3, 3, 3], "x": [0., 1., 2.], "y": [0., 0., 0.], "width_total": [10., 12., 14.], "width_wet": [10., 12., 14.], "width_dry": [0., 0., 0.]}).to_csv(sample_dir / "link_width_samples.csv", index=False)
    result = merge_panel_two_bank_widths(merged_links_path=merged_path, core_segments_path=segments_path, panel_width_root=tmp_path / "panel_widths", output_dir=tmp_path / "merged_widths")
    output = gpd.read_file(result["links"])
    assert output.loc[0, "n_width_samples"] == 3
    assert output.loc[0, "wid_adj_total"] == 12.

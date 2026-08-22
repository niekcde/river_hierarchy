from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.panelization import (
    _build_panelized_reference_centerline,
    _core_owned_width_samples,
    _material_source_segment_ids,
    _ordered_panel_exit_config,
    _provisional_panel_grid,
    assign_core_network_direction,
    create_panel_breaks,
    plan_panels,
    prepare_panel_masks,
    rebuild_core_network_nodes,
    write_panelized_mask_mosaic,
)


def test_split_merged_parts_keep_only_materially_overlapping_source_segments() -> None:
    crs = "EPSG:32611"
    source_segments = gpd.GeoDataFrame(
        {"segment_id": ["P001:1:0", "P002:2:0", "P003:3:0"]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(20, 0), (30, 0)]),
        ],
        crs=crs,
    )

    assert _material_source_segment_ids(
        LineString([(0, 0), (10, 0)]), source_segments
    ) == ["P001:1:0"]
    assert _material_source_segment_ids(
        LineString([(10, 0), (30, 0)]), source_segments
    ) == ["P002:2:0", "P003:3:0"]


def test_core_owned_width_samples_do_not_spill_beyond_merged_part(tmp_path: Path) -> None:
    panel_root = tmp_path / "panels"
    panel_dir = panel_root / "P001"
    panel_dir.mkdir(parents=True)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 3,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 1, 1, 1),
    }
    with rasterio.open(panel_dir / "mask.tif", "w", **profile) as target:
        target.write(np.ones((1, 3), dtype="uint8"), 1)
    segment = gpd.GeoDataFrame(
        {
            "segment_id": ["P001:1:0"],
            "panel_id": ["P001"],
            "source_link_id": [1],
        },
        geometry=[LineString([(0.5, 0.5), (2.5, 0.5)])],
        crs="EPSG:32611",
    ).iloc[0]
    source_rows = {
        ("P001", 1): {
            "wid_pix": "10, 20, 30",
            "idx_link": "0, 1, 2",
        }
    }

    widths, coordinates, provenance = _core_owned_width_samples(
        merged_geometry=LineString([(0.5, 0.5), (1.5, 0.5)]),
        source_segments=[segment],
        source_rows=source_rows,
        panel_root=panel_root,
        excluded_sample_keys={("P001", 1, 0)},
    )

    assert widths == [20.0]
    assert coordinates == [(1.5, 0.5)]
    assert [row["source_pixel_index"] for row in provenance] == [1]


def test_panel_breaks_and_plan_create_four_irregular_core_outputs(tmp_path: Path) -> None:
    mask = tmp_path / "mask.tif"
    with rasterio.open(mask, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:32611", transform=from_origin(0, 100, 1, 1), nodata=0) as dst:
        dst.write(np.ones((100, 100), dtype="uint8"), 1)
    breaks = tmp_path / "breaks.gpkg"
    plan = tmp_path / "panel_plan.gpkg"
    assert create_panel_breaks(mask_path=mask, output_path=breaks, rows=2, columns=2)["break_count"] == 2
    summary = plan_panels(mask_path=mask, breaks_path=breaks, output_path=plan, overlap_m=5)
    assert summary["panel_count"] == 4
    assert summary["over_budget_panel_ids"] == []
    assert plan.exists()
    assert plan.with_suffix(".json").exists()
    prepared = prepare_panel_masks(mask_path=mask, plan_path=plan, output_dir=tmp_path / "panels", panel_ids={"P001"})
    assert prepared["prepared_panel_ids"] == ["P001"]
    assert (tmp_path / "panels" / "P001" / "mask.tif").exists()


def test_panelized_mask_mosaic_uses_core_owned_pixels(tmp_path: Path) -> None:
    template = tmp_path / "prepared.tif"
    profile = {"driver": "GTiff", "height": 6, "width": 6, "count": 1, "dtype": "uint8", "crs": "EPSG:32611", "transform": from_origin(0, 6, 1, 1), "nodata": 0}
    with rasterio.open(template, "w", **profile) as dst:
        dst.write(np.zeros((6, 6), dtype="uint8"), 1)
    plan = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame({"panel_id": ["P001", "P002"]}, geometry=[box(0, 0, 3, 6), box(3, 0, 6, 6)], crs="EPSG:32611").to_file(plan, layer="panel_cores", driver="GPKG")
    for panel_id, transform, value in (("P001", from_origin(0, 6, 1, 1), 1), ("P002", from_origin(3, 6, 1, 1), 0)):
        path = tmp_path / "panels" / panel_id / "mask.tif"; path.parent.mkdir(parents=True)
        with rasterio.open(path, "w", **{**profile, "width": 3, "transform": transform}) as dst:
            dst.write(np.full((6, 3), value, dtype="uint8"), 1)
    output = write_panelized_mask_mosaic(template_mask_path=template, plan_path=plan, panel_root=tmp_path / "panels", output_mask_path=tmp_path / "cleaned.tif")
    with rasterio.open(output) as mosaic:
        assert np.array_equal(mosaic.read(1), np.column_stack((np.ones((6, 3), dtype="uint8"), np.zeros((6, 3), dtype="uint8"))))


def test_rebuild_core_network_nodes_creates_global_endpoint_topology(tmp_path: Path) -> None:
    source = tmp_path / "merged_core_links.gpkg"
    links = gpd.GeoDataFrame(
        {"global_link_id": [0, 1, 2]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(10, 0), (10, 10)]),
        ],
        crs="EPSG:32611",
    )
    links.to_file(source, layer="merged_core_links", driver="GPKG")

    summary = rebuild_core_network_nodes(merged_links_path=source, output_dir=tmp_path / "reconciled")

    assert summary["is_valid"] is True
    assert summary["link_count"] == 3
    assert summary["node_count"] == 4
    assert summary["max_degree"] == 3
    assert summary["connected_component_count"] == 1
    reconciled = gpd.read_file(tmp_path / "reconciled" / "merged_links.gpkg", layer="merged_links")
    assert {"id_node_a", "id_node_b", "id_nodes", "direction_status"}.issubset(reconciled.columns)


def test_provisional_panel_grid_is_budget_based_and_aspect_aware() -> None:
    rows, columns = _provisional_panel_grid(height=48_000, width=40_500, max_panel_pixels=160_000_000)
    assert (rows, columns) == (4, 4)
    assert np.ceil(48_000 / rows) * np.ceil(40_500 / columns) <= 160_000_000


def test_ordered_panel_exit_config_preserves_run_46_south_to_east_direction() -> None:
    cores = gpd.GeoDataFrame(
        {
            "panel_id": [
                "P013",
                "P009",
                "P005",
                "P006",
                "P008",
                "P004",
                "P002",
                "P003",
            ]
        },
        geometry=[
            box(0, 0, 100, 100),
            box(0, 100, 100, 200),
            box(0, 200, 100, 300),
            box(100, 200, 200, 300),
            box(200, 200, 300, 300),
            box(200, 300, 300, 400),
            box(300, 300, 400, 400),
            box(100, 0, 200, 100),
        ],
        crs="EPSG:32611",
    )
    pairs = [
        {"break_id": "H1", "left_panel": "P013", "right_panel": "P009"},
        {"break_id": "H2", "left_panel": "P009", "right_panel": "P005"},
        {"break_id": "V1", "left_panel": "P005", "right_panel": "P006"},
        {"break_id": "V2", "left_panel": "P006", "right_panel": "P008"},
        {"break_id": "H3", "left_panel": "P008", "right_panel": "P004"},
        {"break_id": "V3", "left_panel": "P004", "right_panel": "P002"},
    ]
    active_panels = {"P013", "P009", "P005", "P006", "P008", "P004", "P002"}

    result = _ordered_panel_exit_config(
        cores=cores,
        pairs=pairs,
        active_panels=active_panels,
        global_exit_sides="SE",
    )

    assert {panel: values["exit_sides"] for panel, values in result.items()} == {
        "P002": "WE",
        "P004": "SE",
        "P005": "SE",
        "P006": "WE",
        "P008": "WN",
        "P009": "SN",
        "P013": "SN",
    }
    assert "P003" not in result


def test_ordered_panel_exit_config_preserves_run_41_east_to_west_direction() -> None:
    panel_boxes = {
        "P004": box(700, 100, 800, 200),
        "P008": box(600, 100, 700, 200),
        "P015": box(600, 0, 700, 100),
        "P016": box(500, 0, 600, 100),
        "P014": box(400, 0, 500, 100),
        "P017": box(300, 0, 400, 100),
        "P018": box(200, 0, 300, 100),
        "P019": box(100, 0, 200, 100),
        "P020": box(0, 0, 100, 100),
    }
    cores = gpd.GeoDataFrame(
        {"panel_id": list(panel_boxes)},
        geometry=list(panel_boxes.values()),
        crs="EPSG:32611",
    )
    pairs = [
        {"break_id": "V1", "left_panel": "P008", "right_panel": "P004"},
        {"break_id": "H1", "left_panel": "P015", "right_panel": "P008"},
        {"break_id": "V2", "left_panel": "P016", "right_panel": "P015"},
        {"break_id": "V3", "left_panel": "P014", "right_panel": "P016"},
        {"break_id": "V4", "left_panel": "P017", "right_panel": "P014"},
        {"break_id": "V5", "left_panel": "P018", "right_panel": "P017"},
        {"break_id": "V6", "left_panel": "P019", "right_panel": "P018"},
        {"break_id": "V7", "left_panel": "P020", "right_panel": "P019"},
    ]

    result = _ordered_panel_exit_config(
        cores=cores,
        pairs=pairs,
        active_panels=set(panel_boxes),
        global_exit_sides="EW",
    )

    assert {panel: values["exit_sides"] for panel, values in result.items()} == {
        "P004": "EW",
        "P008": "ES",
        "P014": "EW",
        "P015": "NW",
        "P016": "EW",
        "P017": "EW",
        "P018": "EW",
        "P019": "EW",
        "P020": "EW",
    }


def test_build_panelized_reference_centerline_stitches_core_owned_panel_lines(tmp_path: Path) -> None:
    plan_path = tmp_path / "panel_plan.gpkg"
    panel_root = tmp_path / "panels"
    crs = "EPSG:32611"
    gpd.GeoDataFrame(
        {"panel_id": ["P001", "P002"]},
        geometry=[box(-1, -1, 5, 1), box(5, -1, 11, 1)],
        crs=crs,
    ).to_file(plan_path, layer="panel_cores", driver="GPKG")
    for panel_id, geometry in (
        ("P001", LineString([(0, 0), (5, 0)])),
        ("P002", LineString([(5.2, 0), (10, 0)])),
    ):
        output_dir = panel_root / panel_id / "rivgraph"
        output_dir.mkdir(parents=True)
        gpd.GeoDataFrame(geometry=[geometry], crs=crs).to_file(
            output_dir / f"demo_{panel_id}_centerline.gpkg",
            driver="GPKG",
        )

    result = _build_panelized_reference_centerline(
        plan_path=plan_path,
        panel_root=panel_root,
        run_name="demo",
        endpoint_tolerance_m=1,
    )

    line = result.geometry.iloc[0]
    assert line.geom_type == "LineString"
    assert list(line.coords) == [(0.0, 0.0), (5.1, 0.0), (10.0, 0.0)]


def test_assign_core_network_direction_uses_distinct_downstream_node(tmp_path: Path) -> None:
    core_dir = tmp_path / "core"
    panel_rivgraph = tmp_path / "panels" / "P001" / "rivgraph"
    core_dir.mkdir(parents=True)
    panel_rivgraph.mkdir(parents=True)
    crs = "EPSG:32611"
    gpd.GeoDataFrame(
        {"global_link_id": [0], "source_segments": ["P001:1:0"], "id_node_a": [10], "id_node_b": [11]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(core_dir / "merged_links.gpkg", layer="merged_links", driver="GPKG")
    gpd.GeoDataFrame(
        {"segment_id": ["P001:1:0"], "panel_id": ["P001"], "source_link_id": [1]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(core_dir / "core_segments.gpkg", layer="core_segments", driver="GPKG")
    gpd.GeoDataFrame(
        {"id_link": [1], "id_us_node": [5], "id_ds_node": [6]},
        geometry=[LineString([(0, 0), (10, 0)])], crs=crs,
    ).to_file(panel_rivgraph / "x_P001_links.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"id_node": [5, 6]}, geometry=[Point(0, 0), Point(10, 0)], crs=crs).to_file(panel_rivgraph / "x_P001_nodes.gpkg", driver="GPKG")

    summary = assign_core_network_direction(
        merged_links_path=core_dir / "merged_links.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        panel_root=tmp_path / "panels",
        run_name="x",
        output_dir=core_dir,
    )

    assert summary["assigned_link_count"] == 1
    directed = gpd.read_file(core_dir / "merged_links_directed.gpkg", layer="merged_links")
    assert directed.loc[0, "id_us_node"] == 10
    assert directed.loc[0, "id_ds_node"] == 11


def test_assign_core_network_direction_inherits_exact_parent_when_panel_votes_tie(tmp_path: Path) -> None:
    core_dir = tmp_path / "core"
    core_dir.mkdir(parents=True)
    crs = "EPSG:32611"
    merged_geometry = LineString([(0, 0), (5, 0), (10, 0)])
    gpd.GeoDataFrame(
        {
            "global_link_id": [7],
            "source_segments": ["P001:1:0;P002:2:0"],
            "id_node_a": [20],
            "id_node_b": [21],
        },
        geometry=[merged_geometry],
        crs=crs,
    ).to_file(core_dir / "merged_links.gpkg", layer="merged_links", driver="GPKG")
    gpd.GeoDataFrame(
        {
            "segment_id": ["P001:1:0", "P002:2:0"],
            "panel_id": ["P001", "P002"],
            "source_link_id": [1, 2],
        },
        geometry=[LineString([(0, 0), (5, 0)]), LineString([(10, 0), (5, 0)])],
        crs=crs,
    ).to_file(core_dir / "core_segments.gpkg", layer="core_segments", driver="GPKG")
    for panel_id, link_id, geometry, upstream, downstream in (
        ("P001", 1, LineString([(0, 0), (10, 0)]), Point(0, 0), Point(10, 0)),
        ("P002", 2, LineString([(10, 0), (0, 0)]), Point(10, 0), Point(0, 0)),
    ):
        panel_dir = tmp_path / "panels" / panel_id / "rivgraph"
        panel_dir.mkdir(parents=True)
        gpd.GeoDataFrame(
            {"id_link": [link_id], "id_us_node": [1], "id_ds_node": [2]},
            geometry=[geometry],
            crs=crs,
        ).to_file(panel_dir / f"x_{panel_id}_links.gpkg", driver="GPKG")
        gpd.GeoDataFrame(
            {"id_node": [1, 2]}, geometry=[upstream, downstream], crs=crs,
        ).to_file(panel_dir / f"x_{panel_id}_nodes.gpkg", driver="GPKG")
    parent_path = tmp_path / "parent_directed_links.gpkg"
    gpd.GeoDataFrame(
        {"id_link": [99], "id_us_node": [100], "id_ds_node": [101]},
        geometry=[merged_geometry],
        crs=crs,
    ).to_file(parent_path, driver="GPKG")

    summary = assign_core_network_direction(
        merged_links_path=core_dir / "merged_links.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        panel_root=tmp_path / "panels",
        run_name="x",
        output_dir=core_dir,
        fallback_directed_links_path=parent_path,
    )

    assert summary["ambiguous_link_count"] == 0
    assert summary["parent_inherited_link_count"] == 1
    assert summary["parent_inherited_global_link_ids"] == [7]
    directed = gpd.read_file(core_dir / "merged_links_directed.gpkg", layer="merged_links")
    assert directed.loc[0, "id_us_node"] == 20
    assert directed.loc[0, "id_ds_node"] == 21
    assert directed.loc[0, "direction_status"] == "inherited_parent"


def test_assign_core_network_direction_completes_unique_global_single_path(
    tmp_path: Path,
) -> None:
    core_dir = tmp_path / "core"
    core_dir.mkdir(parents=True)
    crs = "EPSG:32611"
    merged_geometry = LineString([(0, 0), (5, 0), (10, 0)])
    gpd.GeoDataFrame(
        {
            "global_link_id": [7],
            "source_segments": ["P001:1:0;P002:2:0"],
            "id_node_a": [20],
            "id_node_b": [21],
        },
        geometry=[merged_geometry],
        crs=crs,
    ).to_file(core_dir / "merged_links.gpkg", layer="merged_links", driver="GPKG")
    gpd.GeoDataFrame(
        {
            "segment_id": ["P001:1:0", "P002:2:0"],
            "panel_id": ["P001", "P002"],
            "source_link_id": [1, 2],
        },
        geometry=[LineString([(0, 0), (5, 0)]), LineString([(10, 0), (5, 0)])],
        crs=crs,
    ).to_file(core_dir / "core_segments.gpkg", layer="core_segments", driver="GPKG")
    for panel_id, link_id, geometry, upstream, downstream in (
        ("P001", 1, LineString([(0, 0), (10, 0)]), Point(0, 0), Point(10, 0)),
        ("P002", 2, LineString([(10, 0), (0, 0)]), Point(10, 0), Point(0, 0)),
    ):
        panel_dir = tmp_path / "panels" / panel_id / "rivgraph"
        panel_dir.mkdir(parents=True)
        gpd.GeoDataFrame(
            {"id_link": [link_id], "id_us_node": [1], "id_ds_node": [2]},
            geometry=[geometry],
            crs=crs,
        ).to_file(panel_dir / f"x_{panel_id}_links.gpkg", driver="GPKG")
        gpd.GeoDataFrame(
            {"id_node": [1, 2]}, geometry=[upstream, downstream], crs=crs,
        ).to_file(panel_dir / f"x_{panel_id}_nodes.gpkg", driver="GPKG")
    parent_path = tmp_path / "parent_directed_links.gpkg"
    gpd.GeoDataFrame(
        {
            "id_link": [90, 91],
            "id_us_node": [100, 101],
            "id_ds_node": [101, 102],
        },
        geometry=[LineString([(0, 0), (4, 0)]), LineString([(4, 0), (10, 0)])],
        crs=crs,
    ).to_file(parent_path, driver="GPKG")

    summary = assign_core_network_direction(
        merged_links_path=core_dir / "merged_links.gpkg",
        core_segments_path=core_dir / "core_segments.gpkg",
        panel_root=tmp_path / "panels",
        run_name="x",
        output_dir=core_dir,
        fallback_directed_links_path=parent_path,
        require_single_source_sink_path=True,
    )

    assert summary["ambiguous_link_count"] == 0
    assert summary["global_single_path_assigned_global_link_ids"] == [7]
    directed = gpd.read_file(core_dir / "merged_links_directed.gpkg", layer="merged_links")
    assert directed.loc[0, "id_us_node"] == 20
    assert directed.loc[0, "id_ds_node"] == 21
    assert directed.loc[0, "direction_status"] == "global_single_path"

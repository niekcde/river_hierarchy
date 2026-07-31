from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from hierarchy_level_definition.manual_review.apply_link_reversals import (
    DEFAULT_LINKS_NAME,
    DEFAULT_LINK_CSV_NAME,
    DEFAULT_NODES_NAME,
    build_manual_direction_frame,
    filter_requests_to_example,
    load_requests,
    normalize_example_id,
    process_example,
)


def _write_review_network(
    base_dir: Path,
    *,
    example_id: str,
    links_rows: list[dict[str, object]],
    node_rows: list[dict[str, object]],
    write_manual_csv: bool = True,
) -> tuple[Path, Path, Path]:
    review_dir = base_dir / example_id
    review_dir.mkdir(parents=True, exist_ok=True)

    links = gpd.GeoDataFrame(links_rows, geometry="geometry", crs="EPSG:4326")
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")
    links_path = review_dir / DEFAULT_LINKS_NAME
    nodes_path = review_dir / DEFAULT_NODES_NAME
    manual_link_csv_path = review_dir / DEFAULT_LINK_CSV_NAME

    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")
    if write_manual_csv:
        build_manual_direction_frame(links, nodes).to_csv(manual_link_csv_path, index=False)

    return links_path, nodes_path, manual_link_csv_path


def test_process_example_reverses_links_and_runs_validation(tmp_path: Path) -> None:
    review_root = tmp_path / "manual_review_outputs"
    validation_root = tmp_path / "validation_outputs"
    example_id = "demo_valid"

    links_path, _, manual_csv_path = _write_review_network(
        review_root,
        example_id=example_id,
        links_rows=[
            {
                "id_link": 10,
                "id_nodes": "1, 2",
                "id_us_node": 1,
                "id_ds_node": 2,
                "manual_direction_reviewed": True,
                "manual_us_node": 1,
                "manual_ds_node": 2,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(0, 0), (1, 0)]),
            },
            {
                "id_link": 11,
                "id_nodes": "2, 3",
                "id_us_node": 2,
                "id_ds_node": 3,
                "manual_direction_reviewed": True,
                "manual_us_node": 2,
                "manual_ds_node": 3,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(1, 0), (2, 0)]),
            },
            {
                "id_link": 12,
                "id_nodes": "3, 2",
                "id_us_node": 3,
                "id_ds_node": 2,
                "manual_direction_reviewed": True,
                "manual_us_node": 3,
                "manual_ds_node": 2,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(2, 0), (1, 0)]),
            },
            {
                "id_link": 13,
                "id_nodes": "3, 4",
                "id_us_node": 3,
                "id_ds_node": 4,
                "manual_direction_reviewed": True,
                "manual_us_node": 3,
                "manual_ds_node": 4,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(2, 0), (3, 0)]),
            },
        ],
        node_rows=[
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(1, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(2, 0)},
            {"id_node": 4, "is_inlet": False, "is_outlet": True, "geometry": Point(3, 0)},
        ],
    )

    result = process_example(
        example_id,
        [12],
        review_root=review_root,
        validation_root=validation_root,
        old_backup_mode="error",
    )

    assert result["validation_is_valid"] is True
    assert Path(result["links_backup_path"]).exists()
    assert Path(result["manual_link_csv_backup_path"]).exists()

    edited_links = gpd.read_file(links_path).sort_values("id_link", kind="mergesort").reset_index(drop=True)
    edited_link = edited_links.loc[edited_links["id_link"] == 12].iloc[0]
    assert int(edited_link["id_us_node"]) == 2
    assert int(edited_link["id_ds_node"]) == 3
    assert edited_link["id_nodes"] == "2, 3"
    assert list(edited_link.geometry.coords) == [(1.0, 0.0), (2.0, 0.0)]

    manual_frame = pytest.importorskip("pandas").read_csv(manual_csv_path)
    manual_row = manual_frame.loc[manual_frame["link_id"] == 12].iloc[0]
    assert int(manual_row["usnode"]) == 2
    assert int(manual_row["dsnode"]) == 3
    assert bool(manual_row["geometry_reversed"]) is True

    report_path = validation_root / example_id / "validation_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["is_valid"] is True


def test_process_example_use_mode_rebuilds_from_original_backup(tmp_path: Path) -> None:
    review_root = tmp_path / "manual_review_outputs"
    validation_root = tmp_path / "validation_outputs"
    links_path, _, manual_csv_path = _write_review_network(
        review_root,
        example_id="baseline",
        links_rows=[
            {
                "id_link": 1,
                "id_nodes": "1, 2",
                "id_us_node": 1,
                "id_ds_node": 2,
                "manual_direction_reviewed": True,
                "manual_us_node": 1,
                "manual_ds_node": 2,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(0, 0), (1, 0)]),
            }
        ],
        node_rows=[
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": True, "geometry": Point(1, 0)},
        ],
    )

    process_example(
        "baseline", [1], review_root=review_root, validation_root=validation_root, old_backup_mode="error"
    )
    result = process_example(
        "baseline", [1], review_root=review_root, validation_root=validation_root, old_backup_mode="use"
    )

    edited = gpd.read_file(links_path).iloc[0]
    original = gpd.read_file(links_path.with_name("directed_links_old.gpkg")).iloc[0]
    assert (int(edited["id_us_node"]), int(edited["id_ds_node"])) == (2, 1)
    assert (int(original["id_us_node"]), int(original["id_ds_node"])) == (1, 2)
    assert result["old_backup_mode"] == "use"
    assert result["links_source_path"].endswith("directed_links_old.gpkg")
    manual = pd.read_csv(manual_csv_path).iloc[0]
    assert (int(manual["usnode"]), int(manual["dsnode"])) == (2, 1)


def test_process_example_use_mode_creates_missing_original_backups(tmp_path: Path) -> None:
    review_root = tmp_path / "manual_review_outputs"
    validation_root = tmp_path / "validation_outputs"
    links_path, _, manual_csv_path = _write_review_network(
        review_root,
        example_id="first_use",
        links_rows=[
            {
                "id_link": 1,
                "id_nodes": "1, 2",
                "id_us_node": 1,
                "id_ds_node": 2,
                "manual_direction_reviewed": True,
                "manual_us_node": 1,
                "manual_ds_node": 2,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(0, 0), (1, 0)]),
            }
        ],
        node_rows=[
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": True, "geometry": Point(1, 0)},
        ],
    )

    result = process_example(
        "first_use",
        [1],
        review_root=review_root,
        validation_root=validation_root,
        old_backup_mode="use",
    )

    links_backup = links_path.with_name("directed_links_old.gpkg")
    manual_backup = manual_csv_path.with_name("manual_link_directions_old.csv")
    assert links_backup.exists()
    assert manual_backup.exists()
    assert result["links_source_path"] == str(links_backup)
    assert result["manual_link_csv_backup_path"] == str(manual_backup)
    edited = gpd.read_file(links_path).iloc[0]
    original = gpd.read_file(links_backup).iloc[0]
    assert (int(edited["id_us_node"]), int(edited["id_ds_node"])) == (2, 1)
    assert (int(original["id_us_node"]), int(original["id_ds_node"])) == (1, 2)


def test_process_example_writes_invalid_validation_report(tmp_path: Path) -> None:
    review_root = tmp_path / "manual_review_outputs"
    validation_root = tmp_path / "validation_outputs"
    example_id = "demo_invalid"

    links_path, _, _ = _write_review_network(
        review_root,
        example_id=example_id,
        links_rows=[
            {
                "id_link": 20,
                "id_nodes": "1, 2",
                "id_us_node": 1,
                "id_ds_node": 2,
                "manual_direction_reviewed": True,
                "manual_us_node": 1,
                "manual_ds_node": 2,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(0, 0), (1, 0)]),
            },
            {
                "id_link": 21,
                "id_nodes": "2, 3",
                "id_us_node": 2,
                "id_ds_node": 3,
                "manual_direction_reviewed": True,
                "manual_us_node": 2,
                "manual_ds_node": 3,
                "geometry_reversed_manual": False,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(1, 0), (2, 0)]),
            },
        ],
        node_rows=[
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(1, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(2, 0)},
        ],
        write_manual_csv=False,
    )

    result = process_example(
        example_id,
        [20],
        review_root=review_root,
        validation_root=validation_root,
        old_backup_mode="error",
    )

    assert result["validation_is_valid"] is False
    assert result["validation_issues"]
    assert Path(result["links_backup_path"]).exists()

    edited_links = gpd.read_file(links_path)
    edited_link = edited_links.loc[edited_links["id_link"] == 20].iloc[0]
    assert int(edited_link["id_us_node"]) == 2
    assert int(edited_link["id_ds_node"]) == 1

    report_path = validation_root / example_id / "validation_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["is_valid"] is False
    assert report["issues"]


def test_load_requests_autodetects_semicolon_and_normalizes_example_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "flow_direction_adjustments.csv"
    csv_path.write_text("example_id;link_id\n2;85\n10;21\n", encoding="utf-8")

    requests = load_requests(csv_path)

    assert requests["example_id"].tolist() == ["02", "10"]


def test_load_requests_drops_empty_trailing_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "flow_direction_adjustments.csv"
    csv_path.write_text("example_id;link_id\n2;85\n;\n  ;  \n10;21\n", encoding="utf-8")

    requests = load_requests(csv_path)

    assert requests["example_id"].tolist() == ["02", "10"]
    assert requests["link_id"].tolist() == [85, 21]


def test_normalize_example_id_handles_integral_float_strings() -> None:
    assert normalize_example_id("2.0") == "02"
    assert normalize_example_id("10.0") == "10"


def test_filter_requests_to_example_matches_leading_zero_variants() -> None:
    requests = pd.DataFrame(
        {
            "example_id": ["02", "03", "05"],
            "link_id": [85, 21, 14],
        }
    )

    filtered_int_style = filter_requests_to_example(requests, example_id="2")
    filtered_zero_style = filter_requests_to_example(requests, example_id="02")

    assert filtered_int_style["example_id"].tolist() == ["02"]
    assert filtered_int_style["link_id"].tolist() == [85]
    assert filtered_zero_style["example_id"].tolist() == ["02"]
    assert filtered_zero_style["link_id"].tolist() == [85]


def test_filter_requests_to_example_raises_for_missing_network() -> None:
    requests = pd.DataFrame(
        {
            "example_id": ["02", "03"],
            "link_id": [85, 21],
        }
    )

    with pytest.raises(ValueError, match="No reversal requests found for example_id"):
        filter_requests_to_example(requests, example_id="5")


def test_process_example_supports_rivgraph_run_layout(tmp_path: Path) -> None:
    review_root = tmp_path / "runs"
    validation_root = tmp_path / "validation_outputs"
    example_id = "02"
    rivgraph_dir = review_root / example_id / "rivgraph"
    rivgraph_dir.mkdir(parents=True, exist_ok=True)

    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 85,
                "id_nodes": "1, 2",
                "id_us_node": 1,
                "id_ds_node": 2,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(0, 0), (1, 0)]),
            },
            {
                "id_link": 86,
                "id_nodes": "2, 3",
                "id_us_node": 2,
                "id_ds_node": 3,
                "is_inlet": False,
                "is_outlet": False,
                "geometry": LineString([(1, 0), (2, 0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(1, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(2, 0)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    links_path = rivgraph_dir / "02_links.gpkg"
    nodes_path = rivgraph_dir / "02_nodes.gpkg"
    links.to_file(links_path, driver="GPKG")
    nodes.to_file(nodes_path, driver="GPKG")

    result = process_example(
        example_id,
        [85],
        review_root=review_root,
        validation_root=validation_root,
        old_backup_mode="error",
    )

    assert result["links_path"].endswith("/02/rivgraph/02_links.gpkg")
    assert Path(result["links_backup_path"]).exists()
    assert Path(result["manual_link_csv_path"]).exists()

    edited_links = gpd.read_file(links_path)
    edited_link = edited_links.loc[edited_links["id_link"] == 85].iloc[0]
    assert int(edited_link["id_us_node"]) == 2
    assert int(edited_link["id_ds_node"]) == 1

    report_path = validation_root / example_id / "validation_report.json"
    assert report_path.exists()

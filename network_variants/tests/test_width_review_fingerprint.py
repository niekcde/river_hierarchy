from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from network_variants.width_review_archive import archive_base_width_review
from network_variants.width_review_fingerprint import (
    CORRUPT_STATUS,
    CURRENT_STATUS,
    DIRECTION_ONLY_CURRENT_STATUS,
    INCOMPLETE_REVIEW_STATUS,
    STALE_LINK_IDS_STATUS,
    STALE_MASK_STATUS,
    STALE_TOPOLOGY_STATUS,
    STALE_WIDTH_METHOD_STATUS,
    upgrade_review_fingerprints,
    validate_direction_only_review_continuity,
    validate_review_fingerprints,
    write_direction_only_review_continuity,
)


def _write_gpkg(frame: gpd.GeoDataFrame, path: Path) -> None:
    path.unlink(missing_ok=True)
    frame.to_file(path, driver="GPKG")


def _review_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    run = tmp_path / "rivgraph_centerline" / "outputs" / "runs" / "02"
    links_path = run / "rivgraph" / "02_links.gpkg"
    nodes_path = run / "rivgraph" / "02_nodes.gpkg"
    mask_path = run / "masks_cleaned" / "02_cleaned.tif"
    links_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2],
            "id_us_node": [10, 11],
            "id_ds_node": [11, 12],
            "wid_adj": [5.0, 8.0],
            "width_reviewed_flag": [True, True],
            "width_review_decision": ["good", "bad"],
            "width_update_applied": [False, True],
            "width_update_target_context_width": [np.nan, 7.0],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:32611",
    )
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [10, 11, 12],
            "is_inlet": [True, False, False],
            "is_outlet": [False, False, True],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs=links.crs,
    )
    _write_gpkg(links, links_path)
    _write_gpkg(nodes, nodes_path)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        crs=links.crs,
        transform=from_origin(0, 3, 1, 1),
        nodata=0,
    ) as destination:
        destination.write(np.ones((3, 3), dtype=np.uint8), 1)

    variant = tmp_path / "network_variants" / "output" / "02" / "states" / "S000_base" / "variant"
    (variant / "summary").mkdir(parents=True)
    (variant / "widths").mkdir(parents=True)
    (variant / "width_review").mkdir(parents=True)
    families = variant / "widths" / "link_width_families.csv"
    pd.DataFrame({"example_id": ["02"], "id_link": [1]}).to_csv(families, index=False)
    (variant / "summary" / "variant_manifest.json").write_text(
        json.dumps({"output_paths": {"link_width_families": str(families)}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"id_link": [1, 2], "review_decision": ["good", "bad"]}
    ).to_csv(variant / "width_review" / "width_review_decisions.csv", index=False)
    (variant / "width_review" / "width_review_apply_summary.json").write_text("{}", encoding="utf-8")
    result = archive_base_width_review(variant, source_links_path=links_path)
    archive = Path(result["archive_dir"])
    upgrade_review_fingerprints(
        archive,
        mask_path=mask_path,
        links_path=links_path,
        nodes_path=nodes_path,
    )
    return archive, mask_path, links_path, nodes_path, variant


def test_current_review_ignores_mutable_width_changes(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    links = gpd.read_file(links_path)
    links["wid_adj"] = [50.0, 80.0]
    links["width_update_applied"] = [True, False]
    _write_gpkg(links, links_path)

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == CURRENT_STATUS


def test_mask_change_marks_review_stale(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    with rasterio.open(mask, "r+") as destination:
        values = destination.read(1)
        values[0, 0] = 0
        destination.write(values, 1)

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == STALE_MASK_STATUS


def test_link_id_only_change_has_distinct_status(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    links = gpd.read_file(links_path)
    links["id_link"] = [101, 102]
    _write_gpkg(links, links_path)

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == STALE_LINK_IDS_STATUS
    proposal = result["link_id_remap_proposal"]
    assert proposal["status"] == "complete"
    assert proposal["requires_confirmation"] is True
    assert proposal["changed_link_count"] == 2
    assert proposal["mapping"] == [
        {"reviewed_id_link": 1, "current_id_link": 101},
        {"reviewed_id_link": 2, "current_id_link": 102},
    ]


def test_swapped_link_ids_are_detected_even_when_the_id_set_is_unchanged(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    links = gpd.read_file(links_path)
    links["id_link"] = [2, 1]
    _write_gpkg(links, links_path)

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == STALE_LINK_IDS_STATUS
    assert result["link_id_remap_proposal"]["mapping"] == [
        {"reviewed_id_link": 1, "current_id_link": 2},
        {"reviewed_id_link": 2, "current_id_link": 1},
    ]


def test_direction_or_topology_change_marks_review_stale(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    links = gpd.read_file(links_path)
    links.loc[links["id_link"] == 2, ["id_us_node", "id_ds_node"]] = [12, 11]
    _write_gpkg(links, links_path)

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == STALE_TOPOLOGY_STATUS


def test_explicit_whole_network_direction_continuity_is_accepted(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    links = gpd.read_file(links_path)
    links[["id_us_node", "id_ds_node"]] = links[["id_ds_node", "id_us_node"]]
    # Reverse coordinate order as a direction-aware raw writer may do.  The
    # correction is valid because the geometric line remains equal.
    links.geometry = links.geometry.map(lambda geometry: LineString(list(geometry.coords)[::-1]))
    nodes = gpd.read_file(nodes_path)
    nodes["is_inlet"], nodes["is_outlet"] = nodes["is_outlet"], nodes["is_inlet"]
    corrected_links = tmp_path / "corrected_links.gpkg"
    corrected_nodes = tmp_path / "corrected_nodes.gpkg"
    _write_gpkg(links, corrected_links)
    _write_gpkg(nodes, corrected_nodes)

    record = write_direction_only_review_continuity(
        archive,
        mask_path=mask,
        links_path=corrected_links,
        nodes_path=corrected_nodes,
        prior_exit_sides="SW",
        corrected_exit_sides="WS",
        output_path=tmp_path / "continuity.json",
    )
    result = validate_direction_only_review_continuity(
        archive,
        continuity_path=record["continuity_path"],
        mask_path=mask,
        links_path=corrected_links,
        nodes_path=corrected_nodes,
    )

    assert result["status"] == DIRECTION_ONLY_CURRENT_STATUS
    assert result["direction_continuity_audit"]["reversed_link_ids"] == [1, 2]


def test_width_method_change_marks_review_stale(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)

    result = validate_review_fingerprints(
        archive,
        mask_path=mask,
        links_path=links_path,
        nodes_path=nodes_path,
        width_method_version="future-method-v2",
    )

    assert result["status"] == STALE_WIDTH_METHOD_STATUS


def test_archive_hash_change_is_reported_as_corrupt(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    decisions = archive / "base_width_review_decisions.csv"
    decisions.write_text(decisions.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert result["status"] == CORRUPT_STATUS


def test_incomplete_review_is_fresh_but_not_current(tmp_path: Path) -> None:
    archive, mask, links_path, nodes_path, _ = _review_fixture(tmp_path)
    manifest_path = archive / "base_width_review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_completion_status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    upgraded = upgrade_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )
    validated = validate_review_fingerprints(
        archive, mask_path=mask, links_path=links_path, nodes_path=nodes_path
    )

    assert upgraded["status"] == INCOMPLETE_REVIEW_STATUS
    assert validated["status"] == INCOMPLETE_REVIEW_STATUS

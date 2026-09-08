from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from network_variants.island_setup import create_reviewed_island_setup


def _write_experiment(tmp_path: Path, *, with_fill: bool = True) -> Path:
    experiment = tmp_path / "02_indep__anchored"
    base = experiment / "states" / "S000_base"
    mask_dir = base / "variant" / "mask"
    directed_dir = base / "variant" / "directed"
    hierarchy_dir = base / "hierarchy"
    boundary_dir = experiment / "boundary_components"
    mask_dir.mkdir(parents=True)
    directed_dir.mkdir(parents=True)
    hierarchy_dir.mkdir(parents=True)
    boundary_dir.mkdir(parents=True)

    water = np.ones((9, 9), dtype=np.uint8)
    # The diagonal pixel must join the 2x2 component under the 8-connected rule.
    water[3:5, 3:5] = 0
    water[5, 5] = 0
    transform = from_origin(0, 9, 1, 1)
    with rasterio.open(
        mask_dir / "02__S000_base_collapsed.tif",
        "w",
        driver="GTiff",
        height=9,
        width=9,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(water, 1)
    labels = np.ones((9, 9), dtype=np.int16)
    labels[water == 0] = 2
    with rasterio.open(
        boundary_dir / "02_full_extent_land_water_labels.tif",
        "w",
        driver="GTiff",
        height=9,
        width=9,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(labels, 1)
    gpd.GeoDataFrame(
        {"id_link": [7]}, geometry=[LineString([(0, 0), (9, 9)])], crs="EPSG:3857"
    ).to_file(directed_dir / "02__S000_base_directed_links.gpkg", driver="GPKG")
    pd.DataFrame(
        [{
            "unit_id": 1, "bifurcation": 10, "confluence": 11,
            "is_compound": False, "depth_from_root": 2, "max_path_length": 100.0,
        }]
    ).to_csv(hierarchy_dir / "unit_summary.csv", index=False)
    if with_fill:
        edits_dir = experiment / "states" / "S001_unit_1" / "variant" / "mask"
        edits_dir.mkdir(parents=True)
        gpd.GeoDataFrame(
            {"action": ["add"]},
            geometry=[Polygon([(3, 4), (3, 6), (5, 6), (5, 4)])],
            crs="EPSG:3857",
        ).to_file(edits_dir / "collapse_edit_geometries.gpkg", driver="GPKG")
    return experiment


def test_create_reviewed_island_setup_writes_8_connected_audit(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    result = create_reviewed_island_setup(experiment, tiny_island_pixels=0)

    assert result.setup_dir == experiment / "island_only_setup_8connected"
    assert result.candidate_registry_path.is_file()
    assert result.candidate_geometries_path.is_file()
    assert result.manifest_path.is_file()
    assert result.equivalence_registry_path.is_file()
    registry = pd.read_csv(result.candidate_registry_path)
    assert len(registry) == 1
    assert registry.loc[0, "island_id"] == "island_02_001"
    assert registry.loc[0, "n_pixels"] == 5
    assert str(registry.loc[0, "current_unit_ids"]) == "1"
    assert registry.loc[0, "proposed_anchor_unit_id"] == 1
    assert registry.loc[0, "selection_status"] == "approved_pilot"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["candidate_definition"].startswith("enclosed dry 8-connected")
    assert manifest["approved_island_ids"] == ["island_02_001"]
    assert manifest["equivalence_check"] == "completed"


def test_create_reviewed_island_setup_marks_unanchored_candidates_for_review(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path, with_fill=False)
    result = create_reviewed_island_setup(experiment, tiny_island_pixels=0, dry_run=True)

    assert not result.setup_dir.exists()
    assert result.equivalence_registry_path is None
    assert result.candidate_registry.loc[0, "selection_status"] == "requires_anchor_review"
    assert result.summary["n_requires_anchor_review"] == 1
    assert result.summary["approved_island_ids"] == []


def test_create_reviewed_island_setup_completes_existing_candidate_only_setup(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    initial = create_reviewed_island_setup(experiment, tiny_island_pixels=0)
    initial.equivalence_registry_path.unlink()
    manifest = json.loads(initial.manifest_path.read_text())
    for key in [
        "equivalence_check", "equivalence_registry", "n_alias_existing_unit_state",
        "n_requires_island_state", "setup_operation",
    ]:
        manifest.pop(key, None)
    initial.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = create_reviewed_island_setup(experiment, tiny_island_pixels=0)

    assert completed.equivalence_registry_path.is_file()
    assert completed.summary["setup_operation"] == "completed_existing_candidate_setup"
    assert completed.summary["equivalence_check"] == "completed"

    reused = create_reviewed_island_setup(experiment, tiny_island_pixels=0)
    assert reused.equivalence_registry_path == completed.equivalence_registry_path
    assert reused.summary["setup_operation"] == "reused_existing_setup_and_equivalence"

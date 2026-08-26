from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import LineString

import network_variants.local_patch_execution as local_patch_execution

from network_variants.local_patch_execution import (
    _copy_raster_with_local_edit,
    _is_outer_context_retryable,
    _selected_units_from_hierarchy_products,
    _write_virtual_rasters_with_local_edit,
)


def test_outer_context_retry_is_limited_to_context_failures() -> None:
    assert _is_outer_context_retryable(
        ValueError("No safe local crop was found for the selected unit.")
    )
    assert _is_outer_context_retryable(
        ValueError("No local link or node lies within 80 m of boundary node 4.")
    )
    assert _is_outer_context_retryable(
        ValueError(
            "Collar 12 has identical reference-centerline projections and no "
            "safe outward continuation."
        )
    )
    assert not _is_outer_context_retryable(
        ValueError("A reversed strict parent connector exceeds the bounded correction distance.")
    )
    assert not _is_outer_context_retryable(
        ValueError("Dissolving computational cut interfaces produced an invalid realized graph.")
    )


def test_collar_context_retry_is_bounded_and_recorded(tmp_path, monkeypatch) -> None:
    calls: list[float | None] = []

    def fake_materialize_once(**kwargs):
        distance = kwargs.get("minimum_collar_context_m_override")
        calls.append(distance)
        if distance is None:
            raise ValueError(
                "No local link or node lies within 35.806 m of the downstream "
                "anchored boundary node 113."
            )
        manifest_path = tmp_path / "summary" / "variant_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            manifest={},
            output_dir=tmp_path,
        )

    monkeypatch.setattr(
        local_patch_execution,
        "_materialize_local_patch_network_variant_once",
        fake_materialize_once,
    )
    result = local_patch_execution._materialize_with_collar_context_retries(
        output_dir=tmp_path,
    )

    assert calls == [None, 1500.0]
    retry = result.manifest["local_patch"]["collar_context_retry"]
    assert retry["selected_minimum_collar_context_m"] == 1500.0
    assert [row["status"] for row in retry["attempts"]] == ["failed", "selected"]
    saved = json.loads((Path(result.output_dir) / "summary" / "variant_manifest.json").read_text())
    assert saved["local_patch"]["collar_context_retry"] == retry


def test_collar_context_retry_does_not_mask_topology_failure(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_materialize_once(**kwargs):
        nonlocal calls
        calls += 1
        raise ValueError(
            "Final shared-anchored local graph is inconsistent with the raster edit."
        )

    monkeypatch.setattr(
        local_patch_execution,
        "_materialize_local_patch_network_variant_once",
        fake_materialize_once,
    )
    try:
        local_patch_execution._materialize_with_collar_context_retries(
            output_dir=tmp_path,
        )
    except ValueError as error:
        assert "inconsistent with the raster edit" in str(error)
    else:
        raise AssertionError("A topology failure must not enter the collar retry ladder.")
    assert calls == 1


def _write(path: Path, values: np.ndarray, transform) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=str(values.dtype),
        crs="EPSG:32631",
        transform=transform,
        nodata=0,
    ) as destination:
        destination.write(values, 1)
    return path


def test_full_mask_materialization_applies_collapse_but_excludes_crop_cut(tmp_path: Path) -> None:
    # Non-integer pixels and large projected coordinates reproduce the floating
    # boundary case where DatasetReader.index previously floored an aligned
    # offset to the preceding column.
    parent_transform = from_origin(298319.67, 5753293.79, 7.318, 7.318)
    parent = np.ones((4, 4), dtype=np.uint8)
    parent_path = _write(tmp_path / "parent.tif", parent, parent_transform)
    local_window = Window(1, 1, 2, 2)
    local_transform = window_transform(local_window, parent_transform)
    local_base = np.ones((2, 2), dtype=np.uint8)
    local_collapsed = local_base.copy()
    local_collapsed[0, 0] = 0  # scientific collapse edit
    local_collapsed[0, 1] = 0  # computational cut only
    local_origin = np.ones((2, 2), dtype=np.int32)
    local_origin[0, 0] = 107
    local_origin[0, 1] = 0
    barrier = np.zeros((2, 2), dtype=np.uint8)
    barrier[0, 1] = 1

    collapsed_path, origin_path, changed_pixels = _copy_raster_with_local_edit(
        parent_mask_path=parent_path,
        local_base_mask_path=_write(tmp_path / "local_base.tif", local_base, local_transform),
        local_collapsed_mask_path=_write(
            tmp_path / "local_collapsed.tif", local_collapsed, local_transform
        ),
        local_origin_mask_path=_write(tmp_path / "local_origin.tif", local_origin, local_transform),
        cut_barrier_path=_write(tmp_path / "barrier.tif", barrier, local_transform),
        collapsed_output_path=tmp_path / "full_collapsed.tif",
        origin_output_path=tmp_path / "full_origin.tif",
    )

    with rasterio.open(collapsed_path) as source:
        full_collapsed = source.read(1)
    with rasterio.open(origin_path) as source:
        full_origin = source.read(1)
    assert changed_pixels == 1
    assert full_collapsed[1, 1] == 0
    assert full_origin[1, 1] == 107
    assert full_collapsed[1, 2] == 1
    assert full_origin[1, 2] == 1
    assert int(full_collapsed.sum()) == 15


def test_virtual_mask_publication_matches_materialized_result(tmp_path: Path) -> None:
    parent_transform = from_origin(298319.67, 5753293.79, 7.318, 7.318)
    parent = np.ones((4, 4), dtype=np.uint8)
    parent_path = _write(tmp_path / "parent.tif", parent, parent_transform)
    local_window = Window(1, 1, 2, 2)
    local_transform = window_transform(local_window, parent_transform)
    local_base = np.ones((2, 2), dtype=np.uint8)
    local_collapsed = local_base.copy()
    local_collapsed[0, 0] = 0
    local_collapsed[0, 1] = 0
    local_origin = np.ones((2, 2), dtype=np.int32)
    local_origin[0, 0] = 107
    local_origin[0, 1] = 0
    barrier = np.zeros((2, 2), dtype=np.uint8)
    barrier[0, 1] = 1
    paths = {
        "base": _write(tmp_path / "local_base.tif", local_base, local_transform),
        "collapsed": _write(tmp_path / "local_collapsed.tif", local_collapsed, local_transform),
        "origin": _write(tmp_path / "local_origin.tif", local_origin, local_transform),
        "barrier": _write(tmp_path / "barrier.tif", barrier, local_transform),
    }
    materialized_collapsed, materialized_origin, _ = _copy_raster_with_local_edit(
        parent_mask_path=parent_path,
        local_base_mask_path=paths["base"],
        local_collapsed_mask_path=paths["collapsed"],
        local_origin_mask_path=paths["origin"],
        cut_barrier_path=paths["barrier"],
        collapsed_output_path=tmp_path / "materialized_collapsed.tif",
        origin_output_path=tmp_path / "materialized_origin.tif",
    )
    virtual_collapsed, virtual_origin, changed_pixels, manifest = (
        _write_virtual_rasters_with_local_edit(
            parent_mask_path=parent_path,
            local_base_mask_path=paths["base"],
            local_collapsed_mask_path=paths["collapsed"],
            local_origin_mask_path=paths["origin"],
            cut_barrier_path=paths["barrier"],
            output_dir=tmp_path / "virtual",
            name="case",
        )
    )
    with rasterio.open(materialized_collapsed) as expected, rasterio.open(
        virtual_collapsed
    ) as actual:
        assert actual.shape == expected.shape
        assert actual.transform == expected.transform
        assert np.array_equal(actual.read(1), expected.read(1))
    with rasterio.open(materialized_origin) as expected, rasterio.open(
        virtual_origin
    ) as actual:
        assert actual.dtypes[0] == "int32"
        assert np.array_equal(actual.read(1), expected.read(1))
    assert changed_pixels == 1
    assert manifest.exists()
    assert not (tmp_path / "virtual" / "case_collapsed.tif").exists()


def test_selected_units_are_rebuilt_from_saved_base_hierarchy_without_redetection() -> None:
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (1, 1)]),
            },
            {
                "id_link": 11,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (1, -1)]),
            },
        ],
        crs="EPSG:3857",
    )
    summary = pd.DataFrame(
        [
            {
                "unit_id": 7,
                "bifurcation": 1,
                "confluence": 2,
                "n_paths": 2,
                "class": "simple",
                "min_path_length": 1.0,
                "max_path_length": 1.5,
            }
        ]
    )
    membership = pd.DataFrame(
        {"unit_id": [7, 7], "id_link": [10, 11]}
    )

    units = _selected_units_from_hierarchy_products(
        selected_unit_ids=(7,),
        unit_summary=summary,
        unit_link_membership=membership,
        parent_links=links,
    )

    assert set(units) == {7}
    assert units[7].bifurcation == 1
    assert units[7].confluence == 2
    assert units[7].node_set == {1, 2}
    assert units[7].edge_set == {(1, 2, 10), (1, 2, 11)}

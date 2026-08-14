from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform

from network_variants.local_patch_execution import _copy_raster_with_local_edit


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

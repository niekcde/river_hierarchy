"""Utilities for binary raster masks used by the RivGraph workflow."""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window, transform as window_transform


def crop_binary_mask(
    mask: np.ndarray,
    transform: Affine,
    *,
    padding: int = 1,
) -> tuple[np.ndarray, Affine, dict[str, int] | None]:
    """Crop a binary mask to the occupied pixel bounding box.

    Returns the cropped array, updated transform, and crop metadata. If the mask
    has no nonzero pixels, the original array and transform are returned and the
    crop metadata is ``None``.
    """

    if mask.ndim != 2:
        raise ValueError("crop_binary_mask expects a 2D array.")

    occupied_rows, occupied_cols = np.nonzero(mask)
    if occupied_rows.size == 0:
        return mask, transform, None

    row_min = max(int(occupied_rows.min()) - padding, 0)
    row_max = min(int(occupied_rows.max()) + padding + 1, mask.shape[0])
    col_min = max(int(occupied_cols.min()) - padding, 0)
    col_max = min(int(occupied_cols.max()) + padding + 1, mask.shape[1])

    if row_min == 0 and row_max == mask.shape[0] and col_min == 0 and col_max == mask.shape[1]:
        return mask, transform, {
            "row_min": 0,
            "row_max": mask.shape[0],
            "col_min": 0,
            "col_max": mask.shape[1],
        }

    window = Window(col_off=col_min, row_off=row_min, width=col_max - col_min, height=row_max - row_min)
    cropped = mask[row_min:row_max, col_min:col_max]
    cropped_transform = window_transform(window, transform)
    return cropped, cropped_transform, {
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
    }

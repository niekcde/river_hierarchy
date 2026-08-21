"""Small raster overlays published as full-extent GDAL VRT datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform


_GDAL_DATA_TYPES = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


@dataclass(frozen=True)
class VirtualRasterOverlay:
    vrt_path: Path
    base_path: Path
    overlay_path: Path
    window: Window
    base_value_overrides: tuple[tuple[int, int], ...] = ()

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": "virtual-raster-overlay-v1",
            "vrt_path": str(self.vrt_path.resolve()),
            "base_path": str(self.base_path.resolve()),
            "overlay_path": str(self.overlay_path.resolve()),
            "window": {
                "col_off": int(self.window.col_off),
                "row_off": int(self.window.row_off),
                "width": int(self.window.width),
                "height": int(self.window.height),
            },
            "base_value_overrides": {
                str(source): int(target)
                for source, target in self.base_value_overrides
            },
        }


def _aligned_overlay_window(
    base: rasterio.io.DatasetReader,
    overlay: rasterio.io.DatasetReader,
) -> Window:
    if base.crs != overlay.crs:
        raise ValueError("Virtual-raster base and overlay do not share a CRS.")
    if not np.isclose(base.transform.a, overlay.transform.a) or not np.isclose(
        base.transform.e, overlay.transform.e
    ):
        raise ValueError("Virtual-raster base and overlay do not share a pixel size.")
    col_float, row_float = (~base.transform) * (
        overlay.transform.c,
        overlay.transform.f,
    )
    col_off = int(round(float(col_float)))
    row_off = int(round(float(row_float)))
    if abs(float(col_float) - col_off) > 1e-6 or abs(float(row_float) - row_off) > 1e-6:
        raise ValueError("Virtual-raster overlay is not aligned to the base pixel grid.")
    window = Window(col_off, row_off, int(overlay.width), int(overlay.height))
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > base.width
        or window.row_off + window.height > base.height
    ):
        raise ValueError("Virtual-raster overlay lies outside the base raster.")
    if not np.allclose(
        tuple(window_transform(window, base.transform)),
        tuple(overlay.transform),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("Virtual-raster overlay transform does not match its base window.")
    return window


def _source_element(
    band: ET.Element,
    *,
    path: Path,
    width: int,
    height: int,
    dtype: str,
    destination_window: Window,
    value_overrides: dict[int, int] | None = None,
) -> None:
    source = ET.SubElement(
        band,
        "ComplexSource" if value_overrides else "SimpleSource",
    )
    filename = ET.SubElement(source, "SourceFilename", relativeToVRT="0")
    filename.text = str(path.resolve())
    ET.SubElement(source, "SourceBand").text = "1"
    ET.SubElement(
        source,
        "SourceProperties",
        RasterXSize=str(int(width)),
        RasterYSize=str(int(height)),
        DataType=_GDAL_DATA_TYPES[str(dtype)],
    )
    ET.SubElement(
        source,
        "SrcRect",
        xOff="0",
        yOff="0",
        xSize=str(int(width)),
        ySize=str(int(height)),
    )
    ET.SubElement(
        source,
        "DstRect",
        xOff=str(int(destination_window.col_off)),
        yOff=str(int(destination_window.row_off)),
        xSize=str(int(destination_window.width)),
        ySize=str(int(destination_window.height)),
    )
    if value_overrides:
        dtype_info = np.iinfo(np.dtype(dtype))
        control_points: dict[float, float] = {
            float(dtype_info.min): float(dtype_info.min),
            float(dtype_info.max): float(dtype_info.max),
        }
        for raw_source, raw_target in value_overrides.items():
            source_value = int(raw_source)
            target_value = int(raw_target)
            if source_value < dtype_info.min or source_value > dtype_info.max:
                raise ValueError(
                    f"VRT remap source {source_value} lies outside {dtype}."
                )
            # GDAL interpolates between LUT control points. Integer rasters are
            # kept on the identity line immediately beside each overridden
            # integer so every other integer value remains unchanged.
            if source_value > dtype_info.min:
                control_points[source_value - 0.25] = source_value - 0.25
            control_points[float(source_value)] = float(target_value)
            if source_value < dtype_info.max:
                control_points[source_value + 0.25] = source_value + 0.25
        ET.SubElement(source, "LUT").text = ",".join(
            f"{format(key, '.17g')}:{format(value, '.17g')}"
            for key, value in sorted(control_points.items())
        )


def write_overlay_vrt(
    *,
    base_path: str | Path,
    overlay_path: str | Path,
    output_path: str | Path,
    output_dtype: str | None = None,
    base_value_overrides: dict[int, int] | None = None,
) -> VirtualRasterOverlay:
    """Overlay a small aligned raster on a full base without copying the base."""
    base_path = Path(base_path)
    overlay_path = Path(overlay_path)
    output_path = Path(output_path)
    with rasterio.open(base_path) as base, rasterio.open(overlay_path) as overlay:
        if base.count != 1 or overlay.count != 1:
            raise ValueError("Virtual-raster overlays currently require one-band rasters.")
        window = _aligned_overlay_window(base, overlay)
        dtype = str(output_dtype or base.dtypes[0])
        if dtype not in _GDAL_DATA_TYPES:
            raise ValueError(f"Unsupported VRT output dtype {dtype!r}.")
        root = ET.Element(
            "VRTDataset",
            rasterXSize=str(int(base.width)),
            rasterYSize=str(int(base.height)),
        )
        ET.SubElement(root, "SRS").text = base.crs.to_wkt() if base.crs else ""
        ET.SubElement(root, "GeoTransform").text = ", ".join(
            format(float(value), ".17g") for value in base.transform.to_gdal()
        )
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=_GDAL_DATA_TYPES[dtype],
            band="1",
        )
        if base.nodata is not None:
            ET.SubElement(band, "NoDataValue").text = format(float(base.nodata), ".17g")
        _source_element(
            band,
            path=base_path,
            width=base.width,
            height=base.height,
            dtype=base.dtypes[0],
            destination_window=Window(0, 0, base.width, base.height),
            value_overrides=base_value_overrides,
        )
        _source_element(
            band,
            path=overlay_path,
            width=overlay.width,
            height=overlay.height,
            dtype=overlay.dtypes[0],
            destination_window=window,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return VirtualRasterOverlay(
        vrt_path=output_path,
        base_path=base_path,
        overlay_path=overlay_path,
        window=window,
        base_value_overrides=tuple(
            sorted((int(source), int(target)) for source, target in (base_value_overrides or {}).items())
        ),
    )


__all__ = ["VirtualRasterOverlay", "write_overlay_vrt"]

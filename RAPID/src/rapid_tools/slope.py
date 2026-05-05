from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SlopeConfig:
    min_slope: float = 1e-6
    preferred_length_field: str = "len"


REQUIRED_NODE_COLUMNS = {
    "id_node",
    "sword_wse",
    "sword_wse_field",
    "sword_wse_fill_method",
}


def _select_variant_lengths(
    links: gpd.GeoDataFrame,
    *,
    preferred_length_field: str,
) -> tuple[pd.Series, pd.Series]:
    geometry_length = pd.Series(np.nan, index=links.index, dtype=float)
    if links.geometry.name in links.columns and links.geometry.notna().any():
        if links.crs is not None and not links.crs.is_geographic:
            geometry_length = links.geometry.length.astype(float)

    field_length = pd.Series(np.nan, index=links.index, dtype=float)
    if preferred_length_field in links.columns:
        field_length = pd.to_numeric(links[preferred_length_field], errors="coerce")

    use_geometry = geometry_length.gt(0)
    lengths = geometry_length.where(use_geometry, field_length)
    length_source = pd.Series(
        np.where(use_geometry, "geometry", preferred_length_field),
        index=links.index,
        dtype="object",
    )

    if lengths.isna().any() or (lengths <= 0).any():
        bad = links.loc[lengths.isna() | lengths.le(0), "id_link"].tolist() if "id_link" in links.columns else []
        raise ValueError(
            "Could not resolve a positive variant length for all links."
            f" Problem id_link values: {bad[:10]}"
        )
    return lengths.astype(float), length_source


def compute_link_slopes(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig | None = None,
) -> pd.DataFrame:
    config = config or SlopeConfig()
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(nodes.columns))
    if missing:
        raise ValueError(
            "Directed nodes are missing required SWORD WSE columns: "
            + ", ".join(missing)
        )
    for column in ("id_link", "id_us_node", "id_ds_node"):
        if column not in links.columns:
            raise ValueError(f"Directed links are missing required column '{column}'.")

    lengths, length_source = _select_variant_lengths(
        links,
        preferred_length_field=config.preferred_length_field,
    )

    node_attrs = (
        nodes[
            [
                "id_node",
                "sword_wse",
                "sword_wse_field",
                "sword_wse_fill_method",
                "sword_wse_fallback_used",
                "sword_node_id",
            ]
        ]
        .copy()
        .rename(
            columns={
                "sword_wse": "node_wse",
                "sword_wse_field": "node_wse_field",
                "sword_wse_fill_method": "node_wse_fill_method",
                "sword_wse_fallback_used": "node_wse_fallback_used",
                "sword_node_id": "node_sword_node_id",
            }
        )
    )
    node_attrs["id_node"] = pd.to_numeric(node_attrs["id_node"], errors="coerce").astype("Int64")

    frame = links[["id_link", "id_us_node", "id_ds_node"]].copy()
    frame["link_length_m"] = lengths
    frame["link_length_source"] = length_source
    frame["id_us_node"] = pd.to_numeric(frame["id_us_node"], errors="coerce").astype("Int64")
    frame["id_ds_node"] = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype("Int64")

    us = node_attrs.rename(
        columns={
            "id_node": "id_us_node",
            "node_wse": "wse_us",
            "node_wse_field": "wse_us_field",
            "node_wse_fill_method": "wse_us_fill_method",
            "node_wse_fallback_used": "wse_us_fallback_used",
            "node_sword_node_id": "sword_us_node_id",
        }
    )
    ds = node_attrs.rename(
        columns={
            "id_node": "id_ds_node",
            "node_wse": "wse_ds",
            "node_wse_field": "wse_ds_field",
            "node_wse_fill_method": "wse_ds_fill_method",
            "node_wse_fallback_used": "wse_ds_fallback_used",
            "node_sword_node_id": "sword_ds_node_id",
        }
    )

    frame = frame.merge(us, on="id_us_node", how="left").merge(ds, on="id_ds_node", how="left")
    frame["raw_slope"] = (pd.to_numeric(frame["wse_us"], errors="coerce") - pd.to_numeric(frame["wse_ds"], errors="coerce")) / frame["link_length_m"]
    missing_wse = frame["wse_us"].isna() | frame["wse_ds"].isna()
    non_positive = frame["raw_slope"].le(0) | frame["raw_slope"].isna()
    frame["slope_used"] = frame["raw_slope"].where(~missing_wse & ~non_positive, config.min_slope).astype(float)
    frame["slope_adjusted"] = missing_wse | non_positive
    frame["slope_reason"] = np.select(
        [missing_wse, non_positive],
        ["missing_wse", "non_positive_raw"],
        default="ok",
    )
    frame["slope_minimum_applied"] = frame["slope_used"].eq(config.min_slope) & frame["slope_adjusted"]
    return frame

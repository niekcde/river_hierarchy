from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .utils import DEFAULT_CRS, read_table

REQUIRED_COLUMNS = ["station_id", "station_name", "lat", "lon", "country", "agency", "drainage_area", "river_name"]
OPTIONAL_RENAMES = {
    "station_id": ["site_no", "site", "site_number", "station_number"],
    "station_name": ["site_name", "name", "station_nm"],
    "lat": ["latitude", "dec_lat_va", "y"],
    "lon": ["longitude", "dec_long_va", "x"],
    "country": ["country_code"],
    "agency": ["provider", "network", "source_agency"],
    "drainage_area": ["drainagearea", "catchment_area", "area", "facc"],
    "river_name": ["stream_name", "river", "watercourse_name"],
}


def load_gauges(path: str) -> pd.DataFrame:
    return read_table(path)


def clean_gauges(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    rename_map: dict[str, str] = {}
    lower_cols = {column.lower(): column for column in working.columns}
    for target, candidates in OPTIONAL_RENAMES.items():
        if target in working.columns:
            continue
        for candidate in candidates:
            source = lower_cols.get(candidate.lower())
            if source:
                rename_map[source] = target
                break
    if rename_map:
        working = working.rename(columns=rename_map)

    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA

    working["station_id"] = working["station_id"].astype("string").str.strip()
    working["station_name"] = working["station_name"].astype("string").fillna(pd.NA)
    working["country"] = working["country"].astype("string").str.upper().fillna(pd.NA)
    working["agency"] = working["agency"].astype("string").fillna(pd.NA)
    working["river_name"] = working["river_name"].astype("string").fillna(pd.NA)
    working["lat"] = pd.to_numeric(working["lat"], errors="coerce")
    working["lon"] = pd.to_numeric(working["lon"], errors="coerce")
    working["drainage_area"] = pd.to_numeric(working["drainage_area"], errors="coerce")

    working = working.dropna(subset=["station_id", "lat", "lon"]).copy()
    working = working[working["lat"].between(-90, 90) & working["lon"].between(-180, 180)].copy()
    working = working.drop_duplicates(subset=["country", "station_id"], keep="first").reset_index(drop=True)
    working["station_key"] = working.apply(
        lambda row: f"{row['country']}:{row['station_id']}" if pd.notna(row["country"]) else str(row["station_id"]),
        axis=1,
    )

    ordered = ["station_key", *REQUIRED_COLUMNS]
    extras = [column for column in working.columns if column not in ordered]
    return working[ordered + extras]


def gauges_to_geodataframe(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs=DEFAULT_CRS)
    return gpd.GeoDataFrame(frame.copy(), geometry=geometry, crs=DEFAULT_CRS)


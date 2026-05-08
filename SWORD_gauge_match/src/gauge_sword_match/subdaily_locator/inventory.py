from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..utils import read_table

INVENTORY_REQUIRED_COLUMNS = ["station_id", "country", "lat", "lon"]


def autodetect_inventory_path(input_path: str | Path) -> Path | None:
    input_path = Path(input_path)
    candidates = [
        input_path.parent / "gauges_cleaned.parquet",
        input_path.parent / "crosswalk_best.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_gauge_inventory(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    missing = [column for column in INVENTORY_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Gauge inventory at {path} is missing required columns: {', '.join(missing)}")

    working = frame.copy()
    station_key_from_file = None
    if "station_key" in working.columns:
        station_key_from_file = working["station_key"].astype("string").str.strip()
        station_key_parts = station_key_from_file.str.split(":", n=1, expand=True)
        if station_key_parts.shape[1] == 2:
            country_from_key = station_key_parts[0].astype("string").str.upper()
            station_id_from_key = station_key_parts[1].astype("string").str.strip()
            working["country"] = working["country"].astype("string").str.upper().fillna(pd.NA)
            working["country"] = working["country"].where(
                working["country"].notna() & working["country"].str.len().gt(0),
                country_from_key,
            )
            working["station_id"] = station_id_from_key.where(
                station_id_from_key.notna() & station_id_from_key.str.len().gt(0),
                working["station_id"].astype("string").str.strip(),
            )
        else:
            working["station_id"] = working["station_id"].astype("string").str.strip()
            working["country"] = working["country"].astype("string").str.upper().fillna(pd.NA)
    else:
        working["station_id"] = working["station_id"].astype("string").str.strip()
        working["country"] = working["country"].astype("string").str.upper().fillna(pd.NA)
    working["lat"] = pd.to_numeric(working["lat"], errors="coerce")
    working["lon"] = pd.to_numeric(working["lon"], errors="coerce")
    if "station_name" not in working.columns:
        working["station_name"] = pd.NA
    working["station_name"] = working["station_name"].astype("string").fillna(pd.NA)
    working["station_key"] = working["country"] + ":" + working["station_id"]
    working = working.dropna(subset=["station_id", "country", "lat", "lon"]).copy()
    working = working.drop_duplicates(subset=["station_key"], keep="first").reset_index(drop=True)
    return working


def enrich_seeds_with_inventory_matches(
    seeds: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    max_snap_distance_m: float,
    station_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    if seeds.empty:
        return seeds.copy()

    prepared = seeds.copy()
    prepared["inventory_station_id"] = pd.NA
    prepared["inventory_station_key"] = pd.NA
    prepared["inventory_station_name"] = pd.NA
    prepared["inventory_resolution_method"] = pd.NA
    prepared["inventory_distance_m"] = pd.NA

    inventory_by_country = {
        country: group.reset_index(drop=True).copy()
        for country, group in inventory.groupby("country", dropna=False)
    }
    inventory_by_key = {str(row["station_key"]): row for _, row in inventory.iterrows()}

    for idx, row in prepared.iterrows():
        station_key = str(row["station_key"])
        country = str(row["country"]).upper()
        country_inventory = inventory_by_country.get(country)
        if country_inventory is None or country_inventory.empty:
            continue

        override_station_key = (station_overrides or {}).get(station_key)
        if override_station_key is not None:
            match = inventory_by_key.get(str(override_station_key))
            if match is not None:
                prepared.at[idx, "inventory_station_id"] = match["station_id"]
                prepared.at[idx, "inventory_station_key"] = match["station_key"]
                prepared.at[idx, "inventory_station_name"] = match["station_name"]
                prepared.at[idx, "inventory_resolution_method"] = "inventory_curated_override"
                prepared.at[idx, "inventory_distance_m"] = _distance_for_row(row, match)
                continue

        exact = country_inventory[country_inventory["station_key"] == station_key]
        if not exact.empty:
            match = exact.iloc[0]
            prepared.at[idx, "inventory_station_id"] = match["station_id"]
            prepared.at[idx, "inventory_station_key"] = match["station_key"]
            prepared.at[idx, "inventory_station_name"] = match["station_name"]
            prepared.at[idx, "inventory_resolution_method"] = "inventory_exact_station_key"
            prepared.at[idx, "inventory_distance_m"] = _distance_for_row(row, match)
            continue

        seed_lat = _to_float(row.get("lat"))
        seed_lon = _to_float(row.get("lon"))
        if seed_lat is None or seed_lon is None:
            continue

        distances = country_inventory.apply(lambda inv_row: _distance_for_row(row, inv_row), axis=1)
        if distances.empty:
            continue
        best_idx = distances.idxmin()
        best_distance = distances.loc[best_idx]
        if pd.isna(best_distance) or best_distance > max_snap_distance_m:
            continue
        match = country_inventory.loc[best_idx]
        prepared.at[idx, "inventory_station_id"] = match["station_id"]
        prepared.at[idx, "inventory_station_key"] = match["station_key"]
        prepared.at[idx, "inventory_station_name"] = match["station_name"]
        prepared.at[idx, "inventory_resolution_method"] = "inventory_nearest_gauge"
        prepared.at[idx, "inventory_distance_m"] = float(best_distance)

    return prepared


def _distance_for_row(seed_row: pd.Series, inv_row: pd.Series) -> float | None:
    seed_lat = _to_float(seed_row.get("lat"))
    seed_lon = _to_float(seed_row.get("lon"))
    inv_lat = _to_float(inv_row.get("lat"))
    inv_lon = _to_float(inv_row.get("lon"))
    if seed_lat is None or seed_lon is None or inv_lat is None or inv_lon is None:
        return None
    return _haversine_distance_m(seed_lon, seed_lat, inv_lon, inv_lat)


def _haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

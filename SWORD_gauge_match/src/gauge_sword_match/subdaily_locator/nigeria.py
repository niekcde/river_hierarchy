from __future__ import annotations

from typing import Any

import pandas as pd

NIGERIA_NIHSA_DASHBOARD_URL = "https://nihsa.gov.ng/flood-forecast-dashboard/"
NIGERIA_NIHSA_PUBLIC_API_URL = "https://nihsa.gov.ng/flood-forecast-dashboard/api/data"
NIGERIA_NIHSA_DATA_REQUEST_URL = "https://nihsa.gov.ng/data-request/"
NIGERIA_GRDC_STATION_NAME_MAP = {
    "1837253": "FOGGO",
    "1837255": "BUNGA",
}
NIGERIA_GRDC_RIVER_NAME_MAP = {
    "1837253": "RIVER JAMAARE",
    "1837255": "RIVER JAMAARE",
}


class NigeriaNihsaClient:
    def __init__(self) -> None:
        self.dashboard_url = NIGERIA_NIHSA_DASHBOARD_URL
        self.public_api_url = NIGERIA_NIHSA_PUBLIC_API_URL
        self.data_request_url = NIGERIA_NIHSA_DATA_REQUEST_URL


def locate_nigeria_subdaily_station(
    seed_row: pd.Series,
    *,
    client: NigeriaNihsaClient,
) -> dict[str, Any]:
    source_station_id = str(seed_row["source_station_id"]).strip()
    station_name = NIGERIA_GRDC_STATION_NAME_MAP.get(source_station_id)
    river_name = NIGERIA_GRDC_RIVER_NAME_MAP.get(source_station_id)
    candidate_site_numbers = [source_station_id]
    if station_name:
        candidate_site_numbers.append(station_name)

    notes = (
        "No public station-level Nigeria discharge API was identified for this seed. "
        f"The official NIHSA public dashboard ({client.dashboard_url}) exposes a flood-risk layer via "
        f"{client.public_api_url}, but not a station-resolved discharge timeseries endpoint that can be "
        "reconciled to the hierarchy seed gauges. NIHSA instead exposes a manual hydrological data-request "
        f"workflow at {client.data_request_url}."
    )
    if station_name:
        notes += f" The GRDC seed corresponds to station `{station_name}`"
        if river_name:
            notes += f" on `{river_name}`"
        notes += "."

    return {
        "station_key": str(seed_row["station_key"]),
        "country": str(seed_row["country"]).upper(),
        "source_station_id": source_station_id,
        "lat": _to_float(seed_row.get("lat")),
        "lon": _to_float(seed_row.get("lon")),
        "occurrence_count": int(seed_row.get("occurrence_count", 1) or 1),
        "example_ids": seed_row.get("example_ids", ""),
        "down_values": seed_row.get("down_values", ""),
        "inventory_station_id": _nullable_str(seed_row.get("inventory_station_id")),
        "inventory_station_key": _nullable_str(seed_row.get("inventory_station_key")),
        "inventory_station_name": _nullable_str(seed_row.get("inventory_station_name")),
        "inventory_resolution_method": _nullable_str(seed_row.get("inventory_resolution_method")),
        "inventory_distance_m": _to_float(seed_row.get("inventory_distance_m")),
        "provider": "nigeria_nihsa",
        "status": "unresolved",
        "resolution_method": "provider_station_api_not_public",
        "candidate_site_numbers": ",".join(candidate_site_numbers),
        "resolved_monitoring_location_id": None,
        "resolved_site_number": None,
        "resolved_station_name": None,
        "resolution_distance_m": None,
        "monitoring_location_found": False,
        "discharge_series_found": False,
        "subdaily_discharge_found": False,
        "discharge_series_count": 0,
        "instantaneous_series_count": 0,
        "primary_instantaneous_series_count": 0,
        "daily_series_count": 0,
        "instantaneous_begin": None,
        "instantaneous_end": None,
        "daily_begin": None,
        "daily_end": None,
        "daily_coverage_type": "none",
        "notes": notes,
    }


def _nullable_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

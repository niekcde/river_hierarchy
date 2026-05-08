from __future__ import annotations

from typing import Any

import pandas as pd

RUSSIA_GMVO_URL = "https://gmvo.skniivh.ru/"
RUSSIA_GIS_CP_VODA_CLOSED_URL = "https://sslgis.favr.ru/"
RUSSIA_GIS_CP_VODA_ACCESS_INSTRUCTION_URL = (
    "https://rwec.ru/dl/gis/"
    "%D0%98%D0%BD%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%86%D0%B8%D1%8F%20%D0%B4%D0%BB%D1%8F%20"
    "%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B4%D0%BE%D1%81%D1%82%D1%83%D0%BF%D0%B0%20"
    "%D0%BA%20%D0%A1%D0%B5%D0%B3%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D0%BC%20%D0%93%D0%98%D0%A1%20%D0%A6%D0%9F%20"
    "%D0%92%D0%BE%D0%B4%D0%B0.pdf"
)
RUSSIA_GRDC_STATION_NAME_MAP = {
    "2909150": "IGARKA",
    "2909152": "POD. TUNGUSKA",
}
RUSSIA_GRDC_RIVER_NAME_MAP = {
    "2909150": "YENISEY",
    "2909152": "YENISEY",
}


class RussiaGmvoClient:
    def __init__(self) -> None:
        self.legacy_portal_url = RUSSIA_GMVO_URL
        self.closed_contour_url = RUSSIA_GIS_CP_VODA_CLOSED_URL
        self.access_instruction_url = RUSSIA_GIS_CP_VODA_ACCESS_INSTRUCTION_URL


def locate_russia_subdaily_station(
    seed_row: pd.Series,
    *,
    client: RussiaGmvoClient,
) -> dict[str, Any]:
    source_station_id = str(seed_row["source_station_id"]).strip()
    station_name = RUSSIA_GRDC_STATION_NAME_MAP.get(source_station_id)
    river_name = RUSSIA_GRDC_RIVER_NAME_MAP.get(source_station_id)
    candidate_site_numbers = [source_station_id]
    if station_name:
        candidate_site_numbers.append(station_name)

    notes = (
        f"The legacy Russian AIS GMVO portal ({client.legacy_portal_url}) is publicly reachable but currently "
        "states that the resource has been decommissioned. Its replacement is the GIS CP Water closed contour "
        f"({client.closed_contour_url}), which requires controlled access and dedicated onboarding instructions "
        f"({client.access_instruction_url}). No public station-level discharge API or public station-resolved "
        "timeseries endpoint was identified for this hierarchy seed."
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
        "provider": "russia_gmvo",
        "status": "unresolved",
        "resolution_method": "provider_portal_closed_access",
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

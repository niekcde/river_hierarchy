from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

MRC_TIMESERIES_BASE_URL = "https://timeseries.api.mrcmekong.org/api/v1"
CAMBODIA_MRC_CURATED_STATION_OVERRIDES = {
    "2569002": "019801",  # Chroy Chang Var
    "2569003": "019802",  # Kompong Cham
    "2569004": "014901",  # Kratie
    "2569005": "014501",  # Stung Treng
}
LAOS_MRC_CURATED_STATION_OVERRIDES = {
    "2469260": "013901",  # Pakse
}
THAILAND_MRC_CURATED_STATION_OVERRIDES = {
    "2969090": "012001",  # Nong Khai
}


class MekongMrcClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = "gauge-sword-match/0.1.0",
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self._inventory_cache: list[dict[str, Any]] | None = None
        self._corrected_data_cache: dict[str, dict[str, Any]] = {}

    def fetch_time_series_inventory(self) -> list[dict[str, Any]]:
        if self._inventory_cache is not None:
            return self._inventory_cache

        payload = self._get_json(f"{MRC_TIMESERIES_BASE_URL}/ts/inventory/timeSeriesList")
        if not isinstance(payload, list):
            raise RuntimeError("MRC time-series inventory returned an unexpected payload.")

        rows = [dict(row) for row in payload if isinstance(row, dict)]
        self._inventory_cache = rows
        return rows

    def fetch_corrected_time_series_data(self, unique_id: str) -> dict[str, Any]:
        normalized_unique_id = str(unique_id).strip()
        cached = self._corrected_data_cache.get(normalized_unique_id)
        if cached is not None:
            return cached

        payload = self._get_json(f"{MRC_TIMESERIES_BASE_URL}/ts/data/timeSeriesCorrectedData/{normalized_unique_id}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"MRC corrected time-series endpoint returned an unexpected payload for {normalized_unique_id}.")

        self._corrected_data_cache[normalized_unique_id] = payload
        return payload

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        full_url = url
        if params:
            encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
            full_url = f"{url}?{encoded}"

        request = Request(
            full_url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MRC time-series HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"MRC time-series request failed for {request.full_url}: {exc.reason}") from exc


def locate_cambodia_subdaily_station(
    seed_row: pd.Series,
    *,
    client: MekongMrcClient,
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    return _locate_mrc_country_subdaily_station(
        seed_row,
        client=client,
        country_code="KH",
        country_name="Cambodia",
        station_overrides=CAMBODIA_MRC_CURATED_STATION_OVERRIDES,
        max_resolution_distance_m=max_resolution_distance_m,
        now_utc=now_utc,
    )


def locate_laos_subdaily_station(
    seed_row: pd.Series,
    *,
    client: MekongMrcClient,
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    return _locate_mrc_country_subdaily_station(
        seed_row,
        client=client,
        country_code="LA",
        country_name="Lao PDR",
        station_overrides=LAOS_MRC_CURATED_STATION_OVERRIDES,
        max_resolution_distance_m=max_resolution_distance_m,
        now_utc=now_utc,
    )


def locate_thailand_subdaily_station(
    seed_row: pd.Series,
    *,
    client: MekongMrcClient,
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    return _locate_mrc_country_subdaily_station(
        seed_row,
        client=client,
        country_code="TH",
        country_name="Thailand",
        station_overrides=THAILAND_MRC_CURATED_STATION_OVERRIDES,
        max_resolution_distance_m=max_resolution_distance_m,
        now_utc=now_utc,
    )


def _locate_mrc_country_subdaily_station(
    seed_row: pd.Series,
    *,
    client: MekongMrcClient,
    country_code: str,
    country_name: str,
    station_overrides: dict[str, str],
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    inventory_rows = client.fetch_time_series_inventory()
    country_rows = [row for row in inventory_rows if _nullable_str(row.get("countryCode")) == country_code]
    stations = _group_inventory_by_station(country_rows)

    source_station_id = str(seed_row["source_station_id"]).strip()
    candidate_site_numbers = [source_station_id]
    station_code_override = station_overrides.get(source_station_id)
    if station_code_override:
        candidate_site_numbers.append(station_code_override)

    selected_station = None
    resolution_method = "provider_referential_nearest_station"
    if station_code_override:
        selected_station = stations.get(station_code_override)
        resolution_method = "provider_curated_station_code"

    if selected_station is None:
        selected_station, resolution_distance_m = _select_nearest_station(
            list(stations.values()),
            lat=_to_float(seed_row.get("lat")),
            lon=_to_float(seed_row.get("lon")),
        )
    else:
        resolution_distance_m = _haversine_distance_m(
            _to_float(seed_row.get("lon")) or 0.0,
            _to_float(seed_row.get("lat")) or 0.0,
            selected_station["lon"],
            selected_station["lat"],
        )

    if selected_station is None or resolution_distance_m is None:
        return _build_result(
            seed_row,
            provider="mrc_timeseries",
            status="unresolved",
            resolution_method="provider_referential_empty",
            candidate_site_numbers=candidate_site_numbers,
            notes=f"MRC time-series inventory did not return any {country_name} stations with usable coordinates.",
        )

    candidate_site_numbers.append(selected_station["station_code"])
    if resolution_distance_m > max_resolution_distance_m:
        return _build_result(
            seed_row,
            provider="mrc_timeseries",
            status="unresolved",
            resolution_method="provider_referential_no_plausible_match",
            candidate_site_numbers=candidate_site_numbers,
            notes=(
                f"Nearest MRC {country_name} station exceeded the maximum resolution distance "
                f"({resolution_distance_m:.1f} m > {float(max_resolution_distance_m):.1f} m)."
            ),
        )

    discharge_rows = [
        row
        for row in selected_station["time_series"]
        if _normalize_text(row.get("parameter")) == "discharge"
    ]
    if not discharge_rows:
        return _build_result(
            seed_row,
            provider="mrc_timeseries",
            status="resolved_no_discharge",
            resolution_method=resolution_method,
            candidate_site_numbers=candidate_site_numbers,
            inventory_station_id=selected_station["station_code"],
            inventory_station_key=f"{country_code}:{selected_station['station_code']}",
            inventory_station_name=selected_station["station_name"],
            inventory_resolution_method=resolution_method,
            inventory_distance_m=resolution_distance_m,
            monitoring_location_id=selected_station["station_code"],
            resolved_site_number=selected_station["station_code"],
            resolved_station_name=selected_station["station_name"],
            monitoring_location_found=True,
            resolution_distance_m=resolution_distance_m,
            notes=(
                "MRC time-series inventory resolved this station, but it currently exposes no discharge series here; "
                "only other parameters such as water level or rainfall were found."
            ),
        )

    now_resolved = now_utc or datetime.now(timezone.utc)
    recent_daily_cutoff = _subtract_months(now_resolved, 60).date()
    daily_dates = _collect_inventory_timestamps(discharge_rows)
    daily_begin = _min_timestamp_iso(daily_dates)
    daily_end = _max_timestamp_iso(daily_dates)
    daily_end_date = max(daily_dates).date() if daily_dates else None
    daily_coverage_type = "none"
    if daily_end_date is not None:
        daily_coverage_type = "recent_window" if daily_end_date >= recent_daily_cutoff else "historical_only"

    recent_subdaily_dates: list[pd.Timestamp] = []
    subdaily_series_count = 0
    for row in discharge_rows:
        unique_id = _nullable_str(row.get("uniqueId"))
        if unique_id is None:
            continue
        payload = client.fetch_corrected_time_series_data(unique_id)
        points = _extract_corrected_points(payload)
        if not points:
            continue
        if _series_has_subdaily_spacing(points):
            subdaily_series_count += 1
            recent_subdaily_dates.extend([point["timestamp"] for point in points])

    discharge_series_count = len(discharge_rows)
    daily_series_count = len(discharge_rows)
    instantaneous_series_count = subdaily_series_count
    discharge_series_found = bool(discharge_series_count)

    if instantaneous_series_count:
        status = "subdaily_found"
    elif daily_series_count and daily_coverage_type == "recent_window":
        status = "resolved_no_subdaily"
    elif daily_series_count:
        status = "resolved_historical_daily_only"
    else:
        status = "resolved_no_discharge"

    notes = "Resolved through the official MRC time-series inventory and corrected-data endpoints."
    if instantaneous_series_count:
        notes += " The matched discharge series is labeled as daily in the inventory, but recent corrected timestamps include spacing below 24 hours."

    return _build_result(
        seed_row,
        provider="mrc_timeseries",
        status=status,
        resolution_method=resolution_method,
        candidate_site_numbers=candidate_site_numbers,
        inventory_station_id=selected_station["station_code"],
        inventory_station_key=f"{country_code}:{selected_station['station_code']}",
        inventory_station_name=selected_station["station_name"],
        inventory_resolution_method=resolution_method,
        inventory_distance_m=resolution_distance_m,
        monitoring_location_id=selected_station["station_code"],
        resolved_site_number=selected_station["station_code"],
        resolved_station_name=selected_station["station_name"],
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        resolution_distance_m=resolution_distance_m,
        discharge_series_count=discharge_series_count,
        instantaneous_series_count=instantaneous_series_count,
        daily_series_count=daily_series_count,
        instantaneous_begin=_min_timestamp_iso(recent_subdaily_dates),
        instantaneous_end=_max_timestamp_iso(recent_subdaily_dates),
        daily_begin=daily_begin,
        daily_end=daily_end,
        daily_coverage_type=daily_coverage_type,
        notes=notes,
    )


def _group_inventory_by_station(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        station_code = _nullable_str(row.get("stationCode"))
        station_name = _nullable_str(row.get("stationName"))
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if station_code is None or station_name is None or lat is None or lon is None:
            continue

        station = grouped.get(station_code)
        if station is None:
            station = {
                "station_code": station_code,
                "station_name": station_name,
                "country": _nullable_str(row.get("country")),
                "country_code": _nullable_str(row.get("countryCode")),
                "lat": lat,
                "lon": lon,
                "time_series": [],
            }
            grouped[station_code] = station
        station["time_series"].append(row)
    return grouped


def _select_nearest_station(
    stations: list[dict[str, Any]],
    *,
    lat: float | None,
    lon: float | None,
) -> tuple[dict[str, Any] | None, float | None]:
    if lat is None or lon is None:
        return None, None

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for station in stations:
        station_code = _nullable_str(station.get("station_code"))
        station_lat = _to_float(station.get("lat"))
        station_lon = _to_float(station.get("lon"))
        if station_code is None or station_lat is None or station_lon is None:
            continue
        distance_m = _haversine_distance_m(lon, lat, station_lon, station_lat)
        candidates.append((distance_m, station_code, station))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance, _, best_station = candidates[0]
    return best_station, float(best_distance)


def _extract_corrected_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_points = payload.get("Points")
    if not isinstance(raw_points, list):
        return []

    points: list[dict[str, Any]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        timestamp = _parse_timestamp(item.get("Timestamp"))
        value_container = item.get("Value")
        numeric_value = None
        if isinstance(value_container, dict):
            numeric_value = _to_float(value_container.get("Numeric"))
        if timestamp is None or numeric_value is None:
            continue
        points.append({"timestamp": timestamp, "value": numeric_value})

    points.sort(key=lambda item: item["timestamp"])
    return points


def _collect_inventory_timestamps(rows: list[dict[str, Any]]) -> list[pd.Timestamp]:
    values: list[pd.Timestamp] = []
    for row in rows:
        start = _parse_timestamp(row.get("correctedStartTime"))
        end = _parse_timestamp(row.get("correctedEndTime"))
        if start is not None:
            values.append(start)
        if end is not None:
            values.append(end)
    return values


def _series_has_subdaily_spacing(points: list[dict[str, Any]]) -> bool:
    if len(points) < 2:
        return False

    timestamps = [point["timestamp"] for point in points]
    for left, right in zip(timestamps, timestamps[1:]):
        delta_hours = (right - left).total_seconds() / 3600.0
        if delta_hours > 0 and delta_hours < 24.0:
            return True
    return False


def _build_result(
    seed_row: pd.Series,
    *,
    provider: str,
    status: str,
    resolution_method: str,
    candidate_site_numbers: list[str] | None = None,
    inventory_station_id: str | None = None,
    inventory_station_key: str | None = None,
    inventory_station_name: str | None = None,
    inventory_resolution_method: str | None = None,
    inventory_distance_m: float | None = None,
    monitoring_location_id: str | None = None,
    resolved_site_number: str | None = None,
    resolved_station_name: str | None = None,
    monitoring_location_found: bool = False,
    discharge_series_found: bool = False,
    resolution_distance_m: float | None = None,
    discharge_series_count: int = 0,
    instantaneous_series_count: int = 0,
    daily_series_count: int = 0,
    instantaneous_begin: str | None = None,
    instantaneous_end: str | None = None,
    daily_begin: str | None = None,
    daily_end: str | None = None,
    daily_coverage_type: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "station_key": str(seed_row["station_key"]),
        "country": str(seed_row["country"]).upper(),
        "source_station_id": str(seed_row["source_station_id"]).strip(),
        "lat": _to_float(seed_row.get("lat")),
        "lon": _to_float(seed_row.get("lon")),
        "occurrence_count": int(seed_row.get("occurrence_count", 1) or 1),
        "example_ids": seed_row.get("example_ids", ""),
        "down_values": seed_row.get("down_values", ""),
        "inventory_station_id": inventory_station_id,
        "inventory_station_key": inventory_station_key,
        "inventory_station_name": inventory_station_name,
        "inventory_resolution_method": inventory_resolution_method,
        "inventory_distance_m": inventory_distance_m,
        "provider": provider,
        "status": status,
        "resolution_method": resolution_method,
        "candidate_site_numbers": ",".join([value for value in candidate_site_numbers or [] if value]),
        "resolved_monitoring_location_id": monitoring_location_id,
        "resolved_site_number": resolved_site_number,
        "resolved_station_name": resolved_station_name,
        "resolution_distance_m": resolution_distance_m,
        "monitoring_location_found": bool(monitoring_location_found),
        "discharge_series_found": bool(discharge_series_found),
        "subdaily_discharge_found": bool(instantaneous_series_count),
        "discharge_series_count": int(discharge_series_count),
        "instantaneous_series_count": int(instantaneous_series_count),
        "primary_instantaneous_series_count": int(instantaneous_series_count),
        "daily_series_count": int(daily_series_count),
        "instantaneous_begin": instantaneous_begin,
        "instantaneous_end": instantaneous_end,
        "daily_begin": daily_begin,
        "daily_end": daily_end,
        "daily_coverage_type": daily_coverage_type,
        "notes": notes,
    }


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _min_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    if not values:
        return None
    return min(values).isoformat()


def _max_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    if not values:
        return None
    return max(values).isoformat()


def _subtract_months(value: datetime, months: int) -> datetime:
    year = value.year
    month = value.month - months
    while month <= 0:
        year -= 1
        month += 12
    day = min(value.day, _days_in_month(year, month))
    return value.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    if month in {4, 6, 9, 11}:
        return 30
    return 31


def _nullable_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: object) -> str:
    text = _nullable_str(value)
    return text.lower() if text else ""


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

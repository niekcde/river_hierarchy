from __future__ import annotations

import json
import math
import ssl
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ABN_NIGER_BASIN_WORKSPACE_ID = "70320af0-832c-43ed-865d-092d117f5662"
ABN_MAINSERVICE_BASE_URL = "https://mainservice.abn.ne"
ABN_GEOSERVER_WFS_URL = "https://geoserver.abn.ne/geoserver/dynbas/ows"


class NigerBasinAbnClient:
    def __init__(
        self,
        *,
        workspace_id: str = ABN_NIGER_BASIN_WORKSPACE_ID,
        timeout_seconds: float = 30.0,
        user_agent: str = "gauge-sword-match/0.1.0",
        verify_ssl: bool = False,
    ) -> None:
        self.workspace_id = str(workspace_id).strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self._station_cache: list[dict[str, Any]] | None = None
        if verify_ssl:
            self._ssl_context = ssl.create_default_context()
        else:
            # The public ABN hosts currently require relaxed certificate checks in strict clients.
            self._ssl_context = ssl._create_unverified_context()

    def fetch_discharge_station_features(self) -> list[dict[str, Any]]:
        if self._station_cache is not None:
            return self._station_cache

        payload = self._get_json(
            ABN_GEOSERVER_WFS_URL,
            query={
                "service": "WFS",
                "version": "1.0.0",
                "request": "GetFeature",
                "typeName": "dynbas:discharge",
                "outputFormat": "application/json",
            },
        )
        features = payload.get("features")
        if not isinstance(features, list):
            raise RuntimeError("ABN GeoServer discharge layer returned an unexpected payload.")

        rows: list[dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            rows.append(feature)

        self._station_cache = rows
        return rows

    def fetch_place_tabs(self, place_id: str, *, layer_id: str = "discharge") -> list[dict[str, Any]]:
        payload = self._get_json(
            f"{ABN_MAINSERVICE_BASE_URL}/data-analytics/{self.workspace_id}/tabs-by-place",
            query={
                "placeId": str(place_id).strip(),
                "layerId": str(layer_id).strip(),
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("ABN tabs-by-place endpoint returned an unexpected payload.")
        return [item for item in payload if isinstance(item, dict)]

    def fetch_place_timeseries(
        self,
        place_id: str,
        *,
        tab_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, str] = {
            "placeId": str(place_id).strip(),
            "tabId": str(tab_id).strip(),
        }
        if start:
            query["start"] = start
        if end:
            query["end"] = end

        payload = self._get_json(
            f"{ABN_MAINSERVICE_BASE_URL}/data-analytics/{self.workspace_id}/ts-by-placeId",
            query=query,
        )
        if not isinstance(payload, list):
            raise RuntimeError("ABN ts-by-placeId endpoint returned an unexpected payload.")
        return [item for item in payload if isinstance(item, dict)]

    def _get_json(self, url: str, *, query: dict[str, str] | None = None) -> Any:
        full_url = url
        if query:
            full_url = f"{url}?{urlencode(query)}"
        request = Request(
            full_url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ABN HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"ABN request failed for {request.full_url}: {exc.reason}") from exc


def locate_mali_subdaily_station(
    seed_row: pd.Series,
    *,
    client: NigerBasinAbnClient,
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    stations = client.fetch_discharge_station_features()
    best_station, best_distance = _select_best_station_match(
        stations,
        lat=_to_float(seed_row.get("lat")),
        lon=_to_float(seed_row.get("lon")),
    )

    source_station_id = str(seed_row["source_station_id"]).strip()
    if best_station is None or best_distance is None:
        return _build_result(
            seed_row,
            provider="niger_basin_abn",
            status="unresolved",
            resolution_method="provider_referential_empty",
            candidate_site_numbers=[source_station_id],
            notes="ABN did not return any discharge station features with usable coordinates.",
        )

    properties = best_station.get("properties") if isinstance(best_station, dict) else None
    geometry = best_station.get("geometry") if isinstance(best_station, dict) else None
    query_id = _nullable_str(properties.get("query_id") if isinstance(properties, dict) else None)
    station_name = _station_name_from_query_id(query_id)
    candidate_site_numbers = [source_station_id]
    if query_id:
        candidate_site_numbers.append(query_id)

    if query_id is None or best_distance > max_resolution_distance_m:
        return _build_result(
            seed_row,
            provider="niger_basin_abn",
            status="unresolved",
            resolution_method="provider_referential_no_plausible_match",
            candidate_site_numbers=candidate_site_numbers,
            notes=(
                "Nearest ABN discharge station exceeded the maximum resolution distance "
                f"({best_distance:.1f} m > {float(max_resolution_distance_m):.1f} m)."
            ),
        )

    tabs = client.fetch_place_tabs(query_id, layer_id="discharge")
    discharge_tab = _select_discharge_tab(tabs)
    if discharge_tab is None:
        return _build_result(
            seed_row,
            provider="niger_basin_abn",
            status="resolved_no_discharge",
            resolution_method="provider_referential_nearest_station",
            candidate_site_numbers=candidate_site_numbers,
            monitoring_location_id=query_id,
            resolved_site_number=station_name,
            resolved_station_name=station_name,
            monitoring_location_found=True,
            resolution_distance_m=best_distance,
            notes="ABN resolved the station, but no discharge tab was exposed for this place.",
        )

    tab_id = _nullable_str(discharge_tab.get("id"))
    start = _nullable_str(discharge_tab.get("xAxis", {}).get("start") if isinstance(discharge_tab.get("xAxis"), dict) else None)
    end = _nullable_str(discharge_tab.get("xAxis", {}).get("end") if isinstance(discharge_tab.get("xAxis"), dict) else None)
    daily_begin = _normalize_timestamp_text(start)
    daily_end = _normalize_timestamp_text(end)

    if tab_id is None:
        return _build_result(
            seed_row,
            provider="niger_basin_abn",
            status="resolved_no_discharge",
            resolution_method="provider_referential_nearest_station",
            candidate_site_numbers=candidate_site_numbers,
            monitoring_location_id=query_id,
            resolved_site_number=station_name,
            resolved_station_name=station_name,
            monitoring_location_found=True,
            resolution_distance_m=best_distance,
            notes="ABN exposed a discharge tab, but it had no tab identifier.",
        )

    series_payload = client.fetch_place_timeseries(query_id, tab_id=tab_id, start=start, end=end)
    points, series_identifier, value_type = _extract_discharge_points(series_payload)
    daily_series_count = int(bool(points))
    discharge_series_found = bool(daily_series_count)
    subdaily_found = _series_has_subdaily_spacing(points)
    instantaneous_series_count = int(subdaily_found)

    resolved_now = now_utc or datetime.now(timezone.utc)
    recent_daily_cutoff = _subtract_months(resolved_now, 18).date()
    timestamps = [point["timestamp"] for point in points]
    daily_coverage_type = "none"
    if timestamps:
        daily_coverage_type = "recent_window" if max(timestamps).date() >= recent_daily_cutoff else "historical_only"
        daily_begin = _format_timestamp(min(timestamps))
        daily_end = _format_timestamp(max(timestamps))

    if instantaneous_series_count:
        status = "subdaily_found"
    elif daily_series_count and daily_coverage_type == "recent_window":
        status = "resolved_no_subdaily"
    elif daily_series_count:
        status = "resolved_historical_daily_only"
    else:
        status = "resolved_no_discharge"

    notes = "Resolved through the official ABN GeoServer discharge layer and mainservice data-analytics endpoints."
    if value_type:
        notes += f" Provider valueType was `{value_type}`."
    if daily_series_count and not subdaily_found:
        notes += " The returned discharge timestamps were daily at midnight, so this was not counted as subdaily."
    if series_identifier:
        notes += f" Series identifier: {series_identifier}."
    if isinstance(properties, dict) and properties.get("year") is not None:
        notes += f" GeoServer station layer reported year {properties.get('year')}."
    if geometry and isinstance(geometry, dict):
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            notes += f" Resolved by exact/nearest ABN station feature at ({coordinates[1]}, {coordinates[0]})."

    return _build_result(
        seed_row,
        provider="niger_basin_abn",
        status=status,
        resolution_method="provider_referential_nearest_station",
        candidate_site_numbers=candidate_site_numbers,
        monitoring_location_id=query_id,
        resolved_site_number=station_name,
        resolved_station_name=station_name,
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        resolution_distance_m=best_distance,
        discharge_series_count=daily_series_count,
        instantaneous_series_count=instantaneous_series_count,
        instantaneous_begin=_format_timestamp(min(timestamps)) if subdaily_found and timestamps else None,
        instantaneous_end=_format_timestamp(max(timestamps)) if subdaily_found and timestamps else None,
        daily_series_count=daily_series_count,
        daily_begin=daily_begin,
        daily_end=daily_end,
        daily_coverage_type=daily_coverage_type,
        notes=notes,
    )


def _select_best_station_match(
    stations: list[dict[str, Any]],
    *,
    lat: float | None,
    lon: float | None,
) -> tuple[dict[str, Any] | None, float | None]:
    if lat is None or lon is None:
        return None, None

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for station in stations:
        query_id = _nullable_str(
            station.get("properties", {}).get("query_id") if isinstance(station.get("properties"), dict) else None
        )
        coordinates = station.get("geometry", {}).get("coordinates") if isinstance(station.get("geometry"), dict) else None
        if query_id is None or not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        station_lon = _to_float(coordinates[0])
        station_lat = _to_float(coordinates[1])
        if station_lon is None or station_lat is None:
            continue
        distance_m = _haversine_distance_m(lon, lat, station_lon, station_lat)
        candidates.append((distance_m, query_id, station))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance, _, best_station = candidates[0]
    return best_station, float(best_distance)


def _select_discharge_tab(tabs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tab in tabs:
        display_name = _nullable_str(tab.get("displayName"))
        if display_name and display_name.strip().lower() == "discharge":
            return tab
    for tab in tabs:
        tab_id = _nullable_str(tab.get("id"))
        if tab_id and tab_id.strip().lower() == "rwd2":
            return tab
    return None


def _extract_discharge_points(series_payload: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None, str | None]:
    points: list[dict[str, Any]] = []
    series_identifier = None
    value_type = None

    for item in series_payload:
        if _nullable_str(item.get("displayName")) not in {"Discharge", "discharge"}:
            continue
        charts = item.get("charts")
        if not isinstance(charts, list):
            continue
        for chart in charts:
            if not isinstance(chart, dict):
                continue
            series_identifier = series_identifier or _nullable_str(chart.get("id"))
            value_type = value_type or _nullable_str(chart.get("valueType"))
            raw_points = chart.get("data")
            if not isinstance(raw_points, list):
                continue
            for raw_point in raw_points:
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                    continue
                timestamp = _parse_abn_timestamp(raw_point[0])
                value = _to_float(raw_point[1])
                if timestamp is None or value is None:
                    continue
                points.append({"timestamp": timestamp, "value": value})

    points.sort(key=lambda item: item["timestamp"])
    return points, series_identifier, value_type


def _series_has_subdaily_spacing(points: list[dict[str, Any]]) -> bool:
    if len(points) < 2:
        return False

    timestamps = [point["timestamp"] for point in points]
    for left, right in zip(timestamps, timestamps[1:]):
        delta_hours = (right - left).total_seconds() / 3600.0
        if 0 < delta_hours < 24.0:
            return True
    return False


def _build_result(
    seed_row: pd.Series,
    *,
    provider: str,
    status: str,
    resolution_method: str,
    candidate_site_numbers: list[str] | None = None,
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
        "inventory_station_id": _nullable_str(seed_row.get("inventory_station_id")),
        "inventory_station_key": _nullable_str(seed_row.get("inventory_station_key")),
        "inventory_station_name": _nullable_str(seed_row.get("inventory_station_name")),
        "inventory_resolution_method": _nullable_str(seed_row.get("inventory_resolution_method")),
        "inventory_distance_m": _to_float(seed_row.get("inventory_distance_m")),
        "provider": provider,
        "status": status,
        "resolution_method": resolution_method,
        "candidate_site_numbers": ",".join(candidate_site_numbers or []),
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


def _station_name_from_query_id(query_id: str | None) -> str | None:
    if query_id is None:
        return None
    text = query_id.strip().rstrip("/")
    if not text:
        return None
    return text.rsplit("/", 1)[-1]


def _parse_abn_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _normalize_timestamp_text(value: str | None) -> str | None:
    parsed = _parse_abn_timestamp(value)
    return _format_timestamp(parsed) if parsed is not None else value


def _format_timestamp(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
    return normalized.isoformat(timespec="seconds")


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

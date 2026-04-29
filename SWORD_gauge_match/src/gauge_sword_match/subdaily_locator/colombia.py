from __future__ import annotations

import csv
import io
import json
import math
import ssl
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

COLOMBIA_FEWS_BASE_URL = "https://fews.ideam.gov.co/colombia"
COLOMBIA_FEWS_STATION_CSV_URL = f"{COLOMBIA_FEWS_BASE_URL}/data/ReporteTablaEstaciones.csv"


class ColombiaIdeamFewsClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = "gauge-sword-match/0.1.0",
        verify_ssl: bool = False,
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self._station_cache: list[dict[str, Any]] | None = None
        if verify_ssl:
            self._ssl_context = ssl.create_default_context()
        else:
            # The official FEWS legacy host is valid, but its certificate chain often fails strict clients.
            self._ssl_context = ssl._create_unverified_context()

    def fetch_station_inventory(self) -> list[dict[str, Any]]:
        if self._station_cache is not None:
            return self._station_cache

        text = self._get_text(COLOMBIA_FEWS_STATION_CSV_URL)
        reader = csv.DictReader(io.StringIO(text))
        rows: list[dict[str, Any]] = []
        for row in reader:
            if isinstance(row, dict):
                rows.append(dict(row))

        self._station_cache = rows
        return rows

    def fetch_discharge_payload(self, station_id: str) -> dict[str, Any]:
        normalized_station_id = str(station_id).strip()
        return self._get_json(f"{COLOMBIA_FEWS_BASE_URL}/jsonQ/{normalized_station_id}Qobs.json")

    def _get_text(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "Accept": "text/plain, text/csv, application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                return response.read().decode("utf-8-sig", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Colombia IDEAM FEWS HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Colombia IDEAM FEWS request failed for {request.full_url}: {exc.reason}") from exc

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Colombia IDEAM FEWS HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Colombia IDEAM FEWS request failed for {request.full_url}: {exc.reason}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Colombia IDEAM FEWS returned an unexpected payload for {request.full_url}")
        return payload


def locate_colombia_subdaily_station(
    seed_row: pd.Series,
    *,
    client: ColombiaIdeamFewsClient,
    max_resolution_distance_m: float = 5_000.0,
) -> dict[str, Any]:
    stations = client.fetch_station_inventory()
    best_station, best_distance = _select_best_station_match(
        stations,
        lat=_to_float(seed_row.get("lat")),
        lon=_to_float(seed_row.get("lon")),
    )

    source_station_id = str(seed_row["source_station_id"]).strip()
    if best_station is None or best_distance is None:
        return _build_result(
            seed_row,
            provider="colombia_ideam_fews",
            status="unresolved",
            resolution_method="provider_referential_empty",
            candidate_site_numbers=[source_station_id],
            notes="IDEAM FEWS did not return any station referential rows with usable coordinates.",
        )

    station_id = _nullable_str(best_station.get("id"))
    station_name = _nullable_str(best_station.get("nombre"))
    resolution_method = "provider_referential_nearest_station"
    candidate_site_numbers = [source_station_id]
    if station_id:
        candidate_site_numbers.append(station_id)

    if station_id is None or best_distance > max_resolution_distance_m:
        return _build_result(
            seed_row,
            provider="colombia_ideam_fews",
            status="unresolved",
            resolution_method="provider_referential_no_plausible_match",
            candidate_site_numbers=candidate_site_numbers,
            notes=(
                "Nearest IDEAM FEWS hydrology station exceeded the maximum resolution distance "
                f"({best_distance:.1f} m > {float(max_resolution_distance_m):.1f} m)."
            ),
        )

    payload = client.fetch_discharge_payload(station_id)
    observed_points = _extract_series_points(payload, "obs")
    sensor_points = _extract_series_points(payload, "sen")
    series_points = {
        "obs": observed_points,
        "sen": sensor_points,
    }
    populated_series = {name: points for name, points in series_points.items() if points}
    subdaily_series = {name: points for name, points in populated_series.items() if _series_has_subdaily_spacing(points)}

    all_discharge_points = [point for points in populated_series.values() for point in points]
    instantaneous_begin = _format_timestamp(min(point["timestamp"] for point in all_discharge_points)) if all_discharge_points else None
    instantaneous_end = _format_timestamp(max(point["timestamp"] for point in all_discharge_points)) if all_discharge_points else None
    discharge_series_count = len(populated_series)
    instantaneous_series_count = len(subdaily_series)
    discharge_series_found = bool(discharge_series_count)

    if instantaneous_series_count:
        status = "subdaily_found"
    elif discharge_series_found:
        status = "resolved_no_subdaily"
    else:
        status = "resolved_no_discharge"

    series_names = []
    if observed_points:
        series_names.append("observed discharge")
    if sensor_points:
        series_names.append("sensor discharge")
    notes = (
        "Resolved through the official IDEAM FEWS station referential and `jsonQ` discharge series. "
        "This locator does not yet query a separate daily archive."
    )
    if series_names:
        notes += f" Provider discharge values were present in: {', '.join(series_names)}."

    return _build_result(
        seed_row,
        provider="colombia_ideam_fews",
        status=status,
        resolution_method=resolution_method,
        candidate_site_numbers=candidate_site_numbers,
        inventory_station_id=station_id,
        inventory_station_key=f"CO:{station_id}",
        inventory_station_name=station_name,
        inventory_resolution_method=resolution_method,
        inventory_distance_m=best_distance,
        monitoring_location_id=station_id,
        resolved_site_number=station_id,
        resolved_station_name=station_name,
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        resolution_distance_m=best_distance,
        discharge_series_count=discharge_series_count,
        instantaneous_series_count=instantaneous_series_count,
        instantaneous_begin=instantaneous_begin,
        instantaneous_end=instantaneous_end,
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
        station_id = _nullable_str(station.get("id"))
        station_lat = _to_float(station.get("lat"))
        station_lon = _to_float(station.get("lng"))
        if station_id is None or station_lat is None or station_lon is None:
            continue
        distance_m = _haversine_distance_m(lon, lat, station_lon, station_lat)
        candidates.append((distance_m, station_id, station))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance, _, best_station = candidates[0]
    return best_station, float(best_distance)


def _extract_series_points(payload: dict[str, Any], series_name: str) -> list[dict[str, Any]]:
    series = payload.get(series_name)
    if not isinstance(series, dict):
        return []

    raw_points = series.get("data")
    if not isinstance(raw_points, list):
        return []

    points: list[dict[str, Any]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        timestamp = _parse_fews_timestamp(item[0])
        value = _to_float(item[1])
        if timestamp is None or value is None:
            continue
        points.append({"timestamp": timestamp, "value": value})

    points.sort(key=lambda item: item["timestamp"])
    return points


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


def _parse_fews_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


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

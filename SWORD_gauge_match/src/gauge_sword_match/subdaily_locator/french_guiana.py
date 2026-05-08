from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FRANCE_HUBEAU_BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
FRENCH_GUIANA_DEPARTMENT_CODE = "973"


class FranceHubeauClient:
    def __init__(self, *, timeout_seconds: float = 30.0, user_agent: str = "gauge-sword-match/0.1.0") -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self._department_station_cache: dict[str, list[dict[str, Any]]] = {}

    def fetch_department_stations(self, code_departement: str = FRENCH_GUIANA_DEPARTMENT_CODE) -> list[dict[str, Any]]:
        normalized_code = str(code_departement).strip()
        cached = self._department_station_cache.get(normalized_code)
        if cached is not None:
            return cached

        records: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._get_json(
                f"{FRANCE_HUBEAU_BASE_URL}/referentiel/stations",
                {
                    "code_departement": normalized_code,
                    "size": 200,
                    "page": page,
                    "format": "json",
                },
            )
            data = payload.get("data") or []
            for record in data:
                if isinstance(record, dict):
                    records.append(record)
            next_url = _nullable_str(payload.get("next"))
            if not next_url or not data:
                break
            page += 1

        self._department_station_cache[normalized_code] = records
        return records

    def fetch_realtime_discharge_bounds(self, station_id: str) -> tuple[str | None, str | None, int]:
        asc_payload = self._get_json(
            f"{FRANCE_HUBEAU_BASE_URL}/observations_tr",
            {
                "code_entite": station_id,
                "grandeur_hydro": "Q",
                "size": 1,
                "sort": "asc",
                "format": "json",
            },
        )
        count = _to_int(asc_payload.get("count")) or 0
        if count <= 0:
            return None, None, 0

        desc_payload = self._get_json(
            f"{FRANCE_HUBEAU_BASE_URL}/observations_tr",
            {
                "code_entite": station_id,
                "grandeur_hydro": "Q",
                "size": 1,
                "sort": "desc",
                "format": "json",
            },
        )

        earliest = _extract_first_date_field(asc_payload.get("data"), "date_obs")
        latest = _extract_first_date_field(desc_payload.get("data"), "date_obs")
        return earliest, latest, count

    def fetch_daily_discharge_bounds(self, station_id: str, *, page_size: int = 10_000) -> tuple[str | None, str | None, int]:
        payload = self._get_json(
            f"{FRANCE_HUBEAU_BASE_URL}/obs_elab",
            {
                "code_entite": station_id,
                "grandeur_hydro_elab": "QmnJ",
                "size": int(page_size),
                "sort": "asc",
                "format": "json",
            },
        )
        count = _to_int(payload.get("count")) or 0
        if count <= 0:
            return None, None, 0

        first_date = _extract_first_date_field(payload.get("data"), "date_obs_elab")
        last_date = _extract_last_date_field(payload.get("data"), "date_obs_elab")
        next_url = _nullable_str(payload.get("next"))

        while next_url:
            payload = self._get_json_url(next_url)
            last_date = _extract_last_date_field(payload.get("data"), "date_obs_elab") or last_date
            next_url = _nullable_str(payload.get("next"))

        return first_date, last_date, count

    def fetch_realtime_discharge_values(self, station_id: str, *, page_size: int = 20_000) -> pd.DataFrame:
        payload = self._get_json(
            f"{FRANCE_HUBEAU_BASE_URL}/observations_tr",
            {
                "code_entite": station_id,
                "grandeur_hydro": "Q",
                "size": int(page_size),
                "sort": "asc",
                "format": "json",
            },
        )
        records = _extract_realtime_records(payload)
        next_url = _nullable_str(payload.get("next"))
        while next_url:
            payload = self._get_json_url(next_url)
            records.extend(_extract_realtime_records(payload))
            next_url = _nullable_str(payload.get("next"))

        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
        return self._get_json_url(f"{url}?{encoded}")

    def _get_json_url(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"France Hubeau HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"France Hubeau request failed for {request.full_url}: {exc.reason}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"France Hubeau returned an unexpected payload for {request.full_url}")
        return payload


def locate_french_guiana_subdaily_station(
    seed_row: pd.Series,
    *,
    client: FranceHubeauClient,
    max_resolution_distance_m: float = 5_000.0,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    stations = client.fetch_department_stations(FRENCH_GUIANA_DEPARTMENT_CODE)
    best_station, best_distance = _select_best_station_match(
        stations,
        lat=_to_float(seed_row.get("lat")),
        lon=_to_float(seed_row.get("lon")),
    )

    source_station_id = str(seed_row["source_station_id"]).strip()
    if best_station is None or best_distance is None:
        return _build_result(
            seed_row,
            provider="france_hubeau",
            status="unresolved",
            resolution_method="provider_referential_empty",
            candidate_site_numbers=[source_station_id],
            notes="France Hubeau did not return any Guyane hydrometry stations with usable coordinates.",
        )

    station_id = _nullable_str(best_station.get("code_station"))
    station_name = _nullable_str(best_station.get("libelle_station"))
    resolution_method = "provider_referential_nearest_station"
    candidate_site_numbers = [source_station_id]
    if station_id:
        candidate_site_numbers.append(station_id)

    if station_id is None or best_distance > max_resolution_distance_m:
        return _build_result(
            seed_row,
            provider="france_hubeau",
            status="unresolved",
            resolution_method="provider_referential_no_plausible_match",
            candidate_site_numbers=candidate_site_numbers,
            notes=(
                "Nearest France Hubeau Guyane station exceeded the maximum resolution distance "
                f"({best_distance:.1f} m > {float(max_resolution_distance_m):.1f} m)."
            ),
        )

    resolved_now = now_utc or datetime.now(timezone.utc)
    recent_daily_cutoff = _subtract_months(resolved_now, 60).date()
    instantaneous_begin, instantaneous_end, realtime_count = client.fetch_realtime_discharge_bounds(station_id)
    daily_begin, daily_end, daily_count = client.fetch_daily_discharge_bounds(station_id)

    instantaneous_series_count = int(realtime_count > 0 and instantaneous_begin is not None and instantaneous_end is not None)
    daily_series_count = int(daily_count > 0 and daily_begin is not None and daily_end is not None)
    discharge_series_found = bool(instantaneous_series_count or daily_series_count)
    daily_end_date = _parse_date(daily_end)
    daily_coverage_type = "none"
    if daily_end_date is not None:
        daily_coverage_type = "recent_window" if daily_end_date >= recent_daily_cutoff else "historical_only"

    if instantaneous_series_count:
        status = "subdaily_found"
    elif daily_series_count and daily_coverage_type == "recent_window":
        status = "resolved_no_subdaily"
    elif daily_series_count:
        status = "resolved_historical_daily_only"
    else:
        status = "resolved_no_discharge"

    return _build_result(
        seed_row,
        provider="france_hubeau",
        status=status,
        resolution_method=resolution_method,
        candidate_site_numbers=candidate_site_numbers,
        inventory_station_id=station_id,
        inventory_station_key=f"GF:{station_id}",
        inventory_station_name=station_name,
        inventory_resolution_method=resolution_method,
        inventory_distance_m=best_distance,
        monitoring_location_id=station_id,
        resolved_site_number=station_id,
        resolved_station_name=station_name,
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        resolution_distance_m=best_distance,
        discharge_series_count=instantaneous_series_count + daily_series_count,
        instantaneous_series_count=instantaneous_series_count,
        daily_series_count=daily_series_count,
        instantaneous_begin=instantaneous_begin,
        instantaneous_end=instantaneous_end,
        daily_begin=daily_begin,
        daily_end=daily_end,
        daily_coverage_type=daily_coverage_type,
        notes="Resolved through the official France Hubeau Guyane hydrometry referential.",
    )


def _select_best_station_match(
    stations: list[dict[str, Any]],
    *,
    lat: float | None,
    lon: float | None,
) -> tuple[dict[str, Any] | None, float | None]:
    if lat is None or lon is None:
        return None, None

    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for station in stations:
        station_lat = _to_float(station.get("latitude_station"))
        station_lon = _to_float(station.get("longitude_station"))
        station_id = _nullable_str(station.get("code_station"))
        if station_lat is None or station_lon is None or station_id is None:
            continue
        distance_m = _haversine_distance_m(lon, lat, station_lon, station_lat)
        in_service_rank = 0 if bool(station.get("en_service")) else 1
        candidates.append((distance_m, in_service_rank, station_id, station))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_distance, _, _, best_station = candidates[0]
    return best_station, float(best_distance)


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
        "inventory_station_id": inventory_station_id or _nullable_str(seed_row.get("inventory_station_id")),
        "inventory_station_key": inventory_station_key or _nullable_str(seed_row.get("inventory_station_key")),
        "inventory_station_name": inventory_station_name or _nullable_str(seed_row.get("inventory_station_name")),
        "inventory_resolution_method": inventory_resolution_method or _nullable_str(seed_row.get("inventory_resolution_method")),
        "inventory_distance_m": inventory_distance_m if inventory_distance_m is not None else _to_float(seed_row.get("inventory_distance_m")),
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


def _extract_first_date_field(records: Any, field_name: str) -> str | None:
    if not isinstance(records, list) or not records:
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        value = _nullable_str(record.get(field_name))
        if value is not None:
            return value
    return None


def _extract_last_date_field(records: Any, field_name: str) -> str | None:
    if not isinstance(records, list) or not records:
        return None
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        value = _nullable_str(record.get(field_name))
        if value is not None:
            return value
    return None


def _extract_realtime_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp_text = _nullable_str(row.get("date_obs"))
        raw_value = _to_float(row.get("resultat_obs"))
        if timestamp_text is None or raw_value is None:
            continue
        records.append(
            {
                "time": timestamp_text,
                # Hubeau realtime discharge observations are exposed in l/s.
                "value_m3s": raw_value / 1000.0,
                "raw_value": raw_value,
                "raw_unit_of_measure": "l/s",
                "unit_of_measure": "m3/s",
                "continuite_obs_hydro": row.get("continuite_obs_hydro"),
            }
        )
    return records


def _subtract_months(moment: datetime, months: int) -> datetime:
    year = moment.year
    month = moment.month - months
    while month <= 0:
        year -= 1
        month += 12
    day = min(moment.day, _days_in_month(year, month))
    return moment.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in {4, 6, 9, 11} else 31


def _parse_date(value: str | None) -> datetime.date | None:
    text = _nullable_str(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
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


def _to_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

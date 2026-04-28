from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CANADA_WATEROFFICE_BASE_URL = "https://wateroffice.ec.gc.ca/services"
CANADA_CURATED_INVENTORY_OVERRIDES = {
    # GRDC 4208871 is "EMBARRAS AIRPORT" on the Athabasca River and aligns with
    # WSC 07DD001 "ATHABASCA RIVER AT EMBARRAS AIRPORT" from the official
    # Wateroffice station reference index.
    "CA:4208871": "CA:07DD001",
}


class CanadaWaterofficeClient:
    def __init__(self, *, timeout_seconds: float = 30.0, user_agent: str = "gauge-sword-match/0.1.0") -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent

    def fetch_discharge_unit_values(
        self,
        station_id: str,
        *,
        start_datetime_utc: datetime,
        end_datetime_utc: datetime,
    ) -> pd.DataFrame:
        return self._get_csv_frame(
            "real_time_data/csv/inline",
            {
                "stations[]": station_id,
                "parameters[]": "47",
                "start_date": start_datetime_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end_datetime_utc.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def fetch_discharge_daily_values(
        self,
        station_id: str,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        return self._get_csv_frame(
            "daily_data/csv/inline",
            {
                "stations[]": station_id,
                "parameters[]": "flow",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    def _get_csv_frame(self, path: str, params: dict[str, Any]) -> pd.DataFrame:
        encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
        request = Request(
            f"{CANADA_WATEROFFICE_BASE_URL}/{path}?{encoded}",
            headers={
                "Accept": "text/csv",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8-sig", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Canada Wateroffice HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Canada Wateroffice request failed for {request.full_url}: {exc.reason}") from exc

        payload = payload.strip()
        if not payload:
            return pd.DataFrame()

        try:
            return pd.read_csv(StringIO(payload))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def locate_canada_subdaily_station(
    seed_row: pd.Series,
    *,
    client: CanadaWaterofficeClient,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    station_key = str(seed_row["station_key"])
    inventory_station_id = _nullable_str(seed_row.get("inventory_station_id"))
    inventory_resolution_method = _nullable_str(seed_row.get("inventory_resolution_method"))

    if inventory_station_id is None or inventory_resolution_method not in {
        "inventory_exact_station_key",
        "inventory_nearest_gauge",
        "inventory_curated_override",
    }:
        return _build_result(
            seed_row,
            provider="canada_wateroffice",
            status="unresolved",
            resolution_method="inventory_not_found",
            candidate_site_numbers=[str(seed_row["source_station_id"]).strip()],
            notes="Canada locator currently supports only direct Canada inventory station IDs; this seed is not a direct match.",
        )

    resolved_now = now_utc or datetime.now(timezone.utc)
    subdaily_start = _subtract_months(resolved_now, 18)
    recent_daily_cutoff = _subtract_months(resolved_now, 60).date()
    historical_daily_start = date(1900, 1, 1)
    station_id = inventory_station_id

    unit_values = client.fetch_discharge_unit_values(
        station_id,
        start_datetime_utc=subdaily_start,
        end_datetime_utc=resolved_now,
    )
    daily_values = client.fetch_discharge_daily_values(
        station_id,
        start_date=historical_daily_start,
        end_date=resolved_now.date(),
    )

    instantaneous_series_count = int(not unit_values.empty)
    daily_series_count = int(not daily_values.empty)
    discharge_series_found = bool(instantaneous_series_count or daily_series_count)
    daily_dates = _extract_time_values(daily_values)
    daily_begin = _min_timestamp_iso(daily_dates)
    daily_end = _max_timestamp_iso(daily_dates)
    daily_coverage_type = "none"
    if daily_dates:
        daily_coverage_type = (
            "recent_window" if max(daily_dates).date() >= recent_daily_cutoff else "historical_only"
        )

    if instantaneous_series_count:
        status = "subdaily_found"
    elif daily_series_count and daily_coverage_type == "recent_window":
        status = "resolved_no_subdaily"
    elif daily_series_count:
        status = "resolved_historical_daily_only"
    else:
        status = "resolved_no_discharge"

    inst_times = _extract_time_values(unit_values)
    return _build_result(
        seed_row,
        provider="canada_wateroffice",
        status=status,
        resolution_method=inventory_resolution_method,
        candidate_site_numbers=[station_id],
        monitoring_location_id=station_id,
        resolved_site_number=station_id,
        resolved_station_name=_nullable_str(seed_row.get("inventory_station_name")),
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        resolution_distance_m=_to_float(seed_row.get("inventory_distance_m")),
        discharge_series_count=instantaneous_series_count + daily_series_count,
        instantaneous_series_count=instantaneous_series_count,
        daily_series_count=daily_series_count,
        instantaneous_begin=_min_timestamp_iso(inst_times),
        instantaneous_end=_max_timestamp_iso(inst_times),
        daily_begin=daily_begin,
        daily_end=daily_end,
        daily_coverage_type=daily_coverage_type,
    )


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


def _extract_time_values(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if frame.empty:
        return []
    candidate_columns = [column for column in frame.columns if "date" in str(column).lower() or "time" in str(column).lower()]
    for column in candidate_columns:
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        valid = [value for value in parsed if pd.notna(value)]
        if valid:
            return valid
    return []


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


def _min_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    if not values:
        return None
    return min(values).isoformat()


def _max_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    if not values:
        return None
    return max(values).isoformat()


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

from __future__ import annotations

from datetime import date, datetime, timezone
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pandas as pd

BRAZIL_ANA_SERVICE_BASE_URL = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx"
BRAZIL_CURATED_INVENTORY_OVERRIDES = {
    # GRDC 3652455 is "ITAPEBI" on the Jequitinhonha River. ANA inventory
    # station 54950000 is the matching hydrometric station by name, river, and
    # municipality, but the coordinate offset is larger than the generic snap
    # threshold used for automatic reconciliation.
    "BR:3652455": "BR:54950000",
    # Manual review accepted ANA 16661000 as a usable downstream substitute for
    # GRDC 3636201 in the hierarchy example set.
    "BR:3636201": "BR:16661000",
    # Manual review accepted ANA 17710000 as the single plausible hydrometric
    # station representing both GRDC 3637150 and 3637152 in the hierarchy
    # example set.
    "BR:3637150": "BR:17710000",
    "BR:3637152": "BR:17710000",
}


class BrazilAnaHydroClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        user_agent: str = "gauge-sword-match/0.1.0",
        max_retries: int = 2,
        retry_pause_seconds: float = 1.0,
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self.max_retries = max(0, int(max_retries))
        self.retry_pause_seconds = max(0.0, float(retry_pause_seconds))

    def fetch_subdaily_discharge_values(
        self,
        station_id: str,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        root = self._get_xml_root(
            "DadosHidrometeorologicos",
            {
                "codEstacao": station_id,
                "dataInicio": start_date.strftime("%d/%m/%Y"),
                "dataFim": end_date.strftime("%d/%m/%Y"),
            },
        )
        records: list[dict[str, Any]] = []
        for row in _find_row_elements(root, "DadosHidrometereologicos"):
            timestamp_text = _xml_child_text(row, "DataHora")
            discharge_value = _parse_decimal(_xml_child_text(row, "Vazao"))
            if not timestamp_text or discharge_value is None:
                continue
            records.append(
                {
                    "DateTime": timestamp_text,
                    "Vazao": discharge_value,
                    "has_discharge_value": discharge_value is not None,
                }
            )
        return pd.DataFrame.from_records(records)

    def fetch_daily_discharge_values(
        self,
        station_id: str,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        frames = [
            self._fetch_daily_discharge_values_for_consistency_level(
                station_id,
                start_date=start_date,
                end_date=end_date,
                consistency_level=consistency_level,
            )
            for consistency_level in (1, 2)
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["Date"] = combined["Date"].astype("string")
        combined = combined.sort_values(["Date", "NivelConsistencia"]).drop_duplicates(subset=["Date"], keep="first")
        combined = combined.reset_index(drop=True)
        return combined

    def _fetch_daily_discharge_values_for_consistency_level(
        self,
        station_id: str,
        *,
        start_date: date,
        end_date: date,
        consistency_level: int,
    ) -> pd.DataFrame:
        root = self._get_xml_root(
            "HidroSerieHistorica",
            {
                "codEstacao": station_id,
                "dataInicio": start_date.strftime("%d/%m/%Y"),
                "dataFim": end_date.strftime("%d/%m/%Y"),
                "tipoDados": "3",
                "nivelConsistencia": str(consistency_level),
            },
        )
        records: list[dict[str, Any]] = []
        for row in _find_row_elements(root, "SerieHistorica"):
            month_anchor = _parse_timestamp(_xml_child_text(row, "DataHora"))
            if month_anchor is None:
                continue
            for day in range(1, 32):
                discharge_value = _parse_decimal(_xml_child_text(row, f"Vazao{day:02d}"))
                if discharge_value is None or day > _days_in_month(month_anchor.year, month_anchor.month):
                    continue
                records.append(
                    {
                        "Date": f"{month_anchor.year:04d}-{month_anchor.month:02d}-{day:02d}",
                        "Vazao": discharge_value,
                        "NivelConsistencia": consistency_level,
                    }
                )
        return pd.DataFrame.from_records(records)

    def _get_xml_root(self, operation: str, params: dict[str, Any]) -> ET.Element:
        encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
        request = Request(
            f"{BRAZIL_ANA_SERVICE_BASE_URL}/{operation}?{encoded}",
            headers={
                "Accept": "application/xml, text/xml",
                "User-Agent": self.user_agent,
            },
        )
        payload = b""
        attempt = 0
        while True:
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read()
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    attempt += 1
                    if self.retry_pause_seconds > 0:
                        time.sleep(self.retry_pause_seconds)
                    continue
                raise RuntimeError(f"Brazil ANA HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
            except (TimeoutError, URLError) as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    if self.retry_pause_seconds > 0:
                        time.sleep(self.retry_pause_seconds)
                    continue
                reason = exc.reason if isinstance(exc, URLError) else str(exc)
                raise RuntimeError(f"Brazil ANA request failed for {request.full_url}: {reason}") from exc

        try:
            return ET.fromstring(payload)
        except ET.ParseError as exc:
            raise RuntimeError(f"Brazil ANA returned malformed XML for {request.full_url}") from exc


def locate_brazil_subdaily_station(
    seed_row: pd.Series,
    *,
    client: BrazilAnaHydroClient,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    inventory_station_id = _nullable_str(seed_row.get("inventory_station_id"))
    inventory_resolution_method = _nullable_str(seed_row.get("inventory_resolution_method"))

    if inventory_station_id is None or inventory_resolution_method not in {
        "inventory_exact_station_key",
        "inventory_nearest_gauge",
        "inventory_curated_override",
    }:
        return _build_result(
            seed_row,
            provider="brazil_ana",
            status="unresolved",
            resolution_method="inventory_not_found",
            candidate_site_numbers=[str(seed_row["source_station_id"]).strip()],
            notes="Brazil locator currently supports only ANA station IDs resolved through the local Brazil inventory.",
        )

    resolved_now = now_utc or datetime.now(timezone.utc)
    subdaily_start = _subtract_months(resolved_now, 18).date()
    recent_daily_cutoff = _subtract_months(resolved_now, 60).date()
    historical_daily_start = date(1900, 1, 1)
    station_id = inventory_station_id

    subdaily_values = client.fetch_subdaily_discharge_values(
        station_id,
        start_date=subdaily_start,
        end_date=resolved_now.date(),
    )
    daily_values = client.fetch_daily_discharge_values(
        station_id,
        start_date=historical_daily_start,
        end_date=resolved_now.date(),
    )

    has_telemetric_rows = bool(not subdaily_values.empty)
    telemetric_discharge_mask = (
        subdaily_values["has_discharge_value"]
        if "has_discharge_value" in subdaily_values.columns
        else subdaily_values.get("Vazao", pd.Series(dtype=float)).notna()
    )
    has_subdaily_discharge_values = bool(
        has_telemetric_rows and pd.Series(telemetric_discharge_mask).fillna(False).astype(bool).any()
    )
    instantaneous_series_count = int(has_subdaily_discharge_values)
    daily_series_count = int(not daily_values.empty)
    discharge_series_found = bool(instantaneous_series_count or daily_series_count)
    daily_dates = _extract_time_values(daily_values)
    daily_begin = _min_timestamp_iso(daily_dates)
    daily_end = _max_timestamp_iso(daily_dates)
    daily_coverage_type = "none"
    if daily_dates:
        daily_coverage_type = "recent_window" if max(daily_dates).date() >= recent_daily_cutoff else "historical_only"

    if instantaneous_series_count:
        status = "subdaily_found"
    elif daily_series_count and daily_coverage_type == "recent_window":
        status = "resolved_no_subdaily"
    elif daily_series_count:
        status = "resolved_historical_daily_only"
    else:
        status = "resolved_no_discharge"

    instantaneous_dates = _extract_time_values(subdaily_values)
    notes = None
    if has_telemetric_rows and not has_subdaily_discharge_values and not daily_series_count:
        notes = "ANA telemetric records were returned for this station, but all discharge values were blank."
    return _build_result(
        seed_row,
        provider="brazil_ana",
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
        instantaneous_begin=_min_timestamp_iso(instantaneous_dates),
        instantaneous_end=_max_timestamp_iso(instantaneous_dates),
        daily_begin=daily_begin,
        daily_end=daily_end,
        daily_coverage_type=daily_coverage_type,
        notes=notes,
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


def _find_row_elements(root: ET.Element, row_name: str) -> list[ET.Element]:
    return list(root.findall(f".//{{*}}{row_name}"))


def _xml_child_text(element: ET.Element, name: str) -> str | None:
    child = element.find(f"./{{*}}{name}")
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _parse_decimal(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(".", "").replace(",", ".") if text.count(",") == 1 and text.count(".") > 1 else text
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


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


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


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
    return 30 if month in {4, 6, 9, 11} else 31


def _min_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    valid = [value for value in values if value is not None and not pd.isna(value)]
    if not valid:
        return None
    return min(valid).isoformat()


def _max_timestamp_iso(values: list[pd.Timestamp]) -> str | None:
    valid = [value for value in values if value is not None and not pd.isna(value)]
    if not valid:
        return None
    return max(valid).isoformat()


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

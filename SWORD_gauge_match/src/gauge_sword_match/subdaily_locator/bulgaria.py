from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

BULGARIA_APPD_HIDROLOGY_URL = "https://www.appd-bg.org/hidrology-en"
BULGARIA_DANUBE_GRDC_STATION_MAP = {
    "6842700": "Svishtov",
    "6842800": "Ruse",
    "6842900": "Silistra",
}
BULGARIA_DANUBE_NAME_ALIASES = {
    "Svishtov": {"Svishtov"},
    "Ruse": {"Ruse"},
    "Silistra": {"Silistra", "Силистра"},
}


@dataclass
class BulgariaDailyStationRecord:
    station_name: str
    kilometre: float | None
    water_level_cm: float | None
    discharge_m3s: float | None


@dataclass
class BulgariaHydrologySnapshot:
    report_date: date | None
    daily_records: dict[str, BulgariaDailyStationRecord]
    automated_station_names: set[str]
    graph_station_names: set[str]


class BulgariaAppdClient:
    def __init__(self, *, timeout_seconds: float = 30.0, user_agent: str = "gauge-sword-match/0.1.0") -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self.hydrology_url = "https://www.appd-bg.org/hidrology-en"
        self._snapshot_cache: BulgariaHydrologySnapshot | None = None

    def fetch_hydrology_snapshot(self) -> BulgariaHydrologySnapshot:
        if self._snapshot_cache is not None:
            return self._snapshot_cache

        text = self._get_text(self.hydrology_url)
        snapshot = _parse_hydrology_snapshot(text)
        self._snapshot_cache = snapshot
        return snapshot

    def _get_text(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Bulgaria APPD HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Bulgaria APPD request failed for {request.full_url}: {exc.reason}") from exc


def locate_bulgaria_subdaily_station(
    seed_row: pd.Series,
    *,
    client: BulgariaAppdClient,
) -> dict[str, Any]:
    source_station_id = str(seed_row["source_station_id"]).strip()
    target_station_name = _bg_station_map().get(source_station_id)
    if target_station_name is None:
        return _build_result(
            seed_row,
            provider="bulgaria_appd",
            status="unresolved",
            resolution_method="provider_station_mapping_not_configured",
            candidate_site_numbers=[source_station_id],
            notes="No Bulgaria APPD Danube station mapping is configured for this GRDC station seed.",
        )

    snapshot = client.fetch_hydrology_snapshot()
    daily_record = _lookup_daily_station_record(snapshot.daily_records, target_station_name)
    candidate_site_numbers = [source_station_id, target_station_name]

    if daily_record is None:
        return _build_result(
            seed_row,
            provider="bulgaria_appd",
            status="unresolved",
            resolution_method="provider_daily_snapshot_missing",
            candidate_site_numbers=candidate_site_numbers,
            notes=(
                "The official APPD Danube hydrology page was fetched, but the mapped station was not present "
                "in the public daily hydrometeorological discharge table."
            ),
        )

    resolved_station_name = daily_record.station_name
    resolved_station_id = _format_provider_station_id(daily_record.station_name, daily_record.kilometre)
    normalized_station_name = _normalize_text(target_station_name)
    automated_station_present = normalized_station_name in snapshot.automated_station_names
    graph_station_present = normalized_station_name in snapshot.graph_station_names
    if daily_record.kilometre is not None:
        candidate_site_numbers.append(f"{daily_record.kilometre:.2f} km")

    daily_date_iso = snapshot.report_date.isoformat() if snapshot.report_date is not None else None
    discharge_series_found = daily_record.discharge_m3s is not None
    daily_series_count = int(discharge_series_found)
    if discharge_series_found:
        status = "resolved_no_subdaily"
        daily_coverage_type = "recent_window"
    else:
        status = "resolved_no_discharge"
        daily_coverage_type = "none"

    if graph_station_present:
        graph_note = "APPD also publishes a public last-24-hours automated water-level graph for this station, but not a discharge graph or discharge API."
    elif automated_station_present:
        graph_note = "APPD lists this station in its automated water-level section, but does not expose subdaily discharge there."
    else:
        graph_note = "APPD currently exposes this station only in the daily Danube hydrometeorological discharge table."

    notes = (
        "Resolved through the official Bulgaria APPD Danube hydrology page. "
        "Public discharge was found in the daily hydrometeorological station table, but no public subdaily discharge series was identified. "
        f"{graph_note}"
    )

    return _build_result(
        seed_row,
        provider="bulgaria_appd",
        status=status,
        resolution_method="provider_curated_station_name",
        candidate_site_numbers=candidate_site_numbers,
        monitoring_location_id=resolved_station_id,
        resolved_site_number=resolved_station_id,
        resolved_station_name=resolved_station_name,
        monitoring_location_found=True,
        discharge_series_found=discharge_series_found,
        discharge_series_count=daily_series_count,
        daily_series_count=daily_series_count,
        daily_begin=daily_date_iso,
        daily_end=daily_date_iso,
        daily_coverage_type=daily_coverage_type,
        notes=notes,
    )


def _parse_hydrology_snapshot(text: str) -> BulgariaHydrologySnapshot:
    report_date = _parse_report_date(text)
    daily_records = _parse_daily_records(text)
    automated_station_names = _parse_automated_station_names(text)
    graph_station_names = _parse_graph_station_names(text)
    return BulgariaHydrologySnapshot(
        report_date=report_date,
        daily_records=daily_records,
        automated_station_names=automated_station_names,
        graph_station_names=graph_station_names,
    )


def _parse_report_date(text: str) -> date | None:
    match = re.search(r"Water levels on the bulgarian section of the Danube river\s+(\d{2}\.\d{2}\.\d{4})", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%d.%m.%Y", utc=False).date()
    except (TypeError, ValueError):
        return None


def _parse_daily_records(text: str) -> dict[str, BulgariaDailyStationRecord]:
    section = _extract_section(text, "Hydrometeorological stations</h3>", "<h3>Automated gauging stations</h3>")
    rows = _extract_first_table_rows(section)
    records: dict[str, BulgariaDailyStationRecord] = {}
    for row in rows[1:]:
        if len(row) < 4:
            continue
        station_name = _strip_html(row[0])
        if not station_name:
            continue
        record = BulgariaDailyStationRecord(
            station_name=station_name,
            kilometre=_parse_float(row[1]),
            water_level_cm=_parse_float(row[2]),
            discharge_m3s=_parse_float(row[3]),
        )
        records[_canonical_bg_name(station_name)] = record
    return records


def _parse_automated_station_names(text: str) -> set[str]:
    section = _extract_section(text, "<h3>Automated gauging stations</h3>", "<h3>Water level graphs for the last 24 hours</h3>")
    rows = _extract_first_table_rows(section)
    names: set[str] = set()
    for row in rows[1:]:
        if not row:
            continue
        station_name = _strip_html(row[0])
        if station_name:
            names.add(_canonical_bg_name(station_name))
    return names


def _parse_graph_station_names(text: str) -> set[str]:
    section = _extract_section(text, "<h3>Water level graphs for the last 24 hours</h3>", "</section>")
    names: set[str] = set()
    for match in re.finditer(r"<h4>(.*?)\(\s*[-0-9.]+\s*km\s*\)</h4>", section, re.IGNORECASE | re.DOTALL):
        station_name = _strip_html(match.group(1))
        if station_name:
            names.add(_canonical_bg_name(station_name))
    return names


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start_index = text.find(start_marker)
    if start_index < 0:
        return ""
    end_index = text.find(end_marker, start_index)
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]


def _extract_first_table_rows(text: str) -> list[list[str]]:
    table_match = re.search(r"<table\b.*?>.*?</table>", text, re.IGNORECASE | re.DOTALL)
    if not table_match:
        return []

    rows: list[list[str]] = []
    for row_match in re.finditer(r"<tr\b.*?>(.*?)</tr>", table_match.group(0), re.IGNORECASE | re.DOTALL):
        cells = [
            _strip_html(cell_match.group(1))
            for cell_match in re.finditer(r"<t[dh]\b.*?>(.*?)</t[dh]>", row_match.group(1), re.IGNORECASE | re.DOTALL)
        ]
        if cells:
            rows.append(cells)
    return rows


def _lookup_daily_station_record(
    daily_records: dict[str, BulgariaDailyStationRecord],
    target_station_name: str,
) -> BulgariaDailyStationRecord | None:
    for alias in _bg_name_aliases().get(target_station_name, {target_station_name}):
        record = daily_records.get(_normalize_text(alias))
        if record is not None:
            return record
    return None


def _bg_station_map() -> dict[str, str]:
    return {
        "6842700": "Svishtov",
        "6842800": "Ruse",
        "6842900": "Silistra",
    }


def _bg_name_aliases() -> dict[str, set[str]]:
    return {
        "Svishtov": {"Svishtov"},
        "Ruse": {"Ruse"},
        "Silistra": {"Silistra", "Силистра"},
    }


def _format_provider_station_id(station_name: str, kilometre: float | None) -> str:
    if kilometre is None:
        return station_name
    return f"{station_name} ({kilometre:.2f} km)"


def _canonical_bg_name(value: object) -> str:
    normalized = _normalize_text(value)
    for canonical_name, aliases in _bg_name_aliases().items():
        alias_keys = {_normalize_text(alias) for alias in aliases}
        if normalized in alias_keys:
            return _normalize_text(canonical_name)
    return normalized


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


def _strip_html(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = html.unescape(text).replace("\xa0", " ")
    return " ".join(text.split())


def _normalize_text(value: object) -> str:
    text = _strip_html(value).casefold()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _nullable_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_float(value: object) -> float | None:
    text = _strip_html(value)
    if not text:
        return None
    normalized = text.replace(" ", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CHILE_DGA_INVENTORY_URL = "https://rest-sit.mop.gob.cl/arcgis/rest/services/DGA/Red_Hidrometrica/MapServer/0/query"
CHILE_DGA_ALERTS_URL = "https://rest-sit.mop.gob.cl/arcgis/rest/services/DGA/ALERTAS/MapServer/0/query"
CHILE_SUPPORTED_INVENTORY_RESOLUTION_METHODS = {
    "inventory_exact_station_key",
    "inventory_nearest_gauge",
    "inventory_curated_override",
}
CHILE_ALERT_CODE_FIELD = "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.CODBNA"
CHILE_ALERT_NAME_FIELD = "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.NOMBRERED"
CHILE_ALERT_FLUVIOMETRICA_FIELD = "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.FLUVIOMETRICA"
CHILE_ALERT_TRANSMISSION_FIELD = "SITMOP_PROD.SITMOP_DESA.TG_RED_HIDROMETEO.TIPOTRASMISION"
CHILE_ALERT_MOD_CODE_FIELD = "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_codest"
CHILE_ALERT_MOD_TIME_FIELD = "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_fechra"
CHILE_ALERT_MOD_VALUE_FIELD = "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_valor"
CHILE_ALERT_MOD_ALERT_FIELD = "SITMOP_PROD.SDE.V_DGA_GIS_ALERTAS.mod_alerta"


class ChileDgaClient:
    def __init__(self, *, timeout_seconds: float = 30.0, user_agent: str = "gauge-sword-match/0.1.0") -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent

    def fetch_inventory_station_records(self, station_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(
            CHILE_DGA_INVENTORY_URL,
            {
                "where": f"COD_BNA = '{_escape_sql_literal(station_id)}'",
                "outFields": "COD_BNA,NOM_ESTACION,TIPO_ESTACION,VIGENCIA,INSTITUCION,LATITUD,LONGITUD,AREA_DRENAJE_KM2",
                "returnGeometry": "false",
                "f": "pjson",
            },
        )
        return _get_attribute_records(payload)

    def fetch_alert_records_for_station_prefix(self, station_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(
            CHILE_DGA_ALERTS_URL,
            {
                "where": f"{CHILE_ALERT_CODE_FIELD} LIKE '{_escape_sql_literal(station_id)}%'",
                "outFields": ",".join(
                    [
                        CHILE_ALERT_CODE_FIELD,
                        CHILE_ALERT_NAME_FIELD,
                        CHILE_ALERT_FLUVIOMETRICA_FIELD,
                        CHILE_ALERT_TRANSMISSION_FIELD,
                        CHILE_ALERT_MOD_CODE_FIELD,
                        CHILE_ALERT_MOD_TIME_FIELD,
                        CHILE_ALERT_MOD_VALUE_FIELD,
                        CHILE_ALERT_MOD_ALERT_FIELD,
                    ]
                ),
                "returnGeometry": "false",
                "f": "pjson",
            },
        )
        return _get_attribute_records(payload)

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
        request = Request(
            f"{url}?{encoded}",
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
            raise RuntimeError(f"Chile DGA HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Chile DGA request failed for {request.full_url}: {exc.reason}") from exc

        if "error" in payload:
            message = payload["error"].get("message") or "Unknown ArcGIS error"
            details = payload["error"].get("details") or []
            detail_text = "; ".join(str(detail).strip() for detail in details if str(detail).strip())
            raise RuntimeError(f"Chile DGA query failed for {request.full_url}: {message}{': ' + detail_text if detail_text else ''}")
        return payload


def locate_chile_subdaily_station(
    seed_row: pd.Series,
    *,
    client: ChileDgaClient,
) -> dict[str, Any]:
    inventory_station_id = _nullable_str(seed_row.get("inventory_station_id"))
    inventory_resolution_method = _nullable_str(seed_row.get("inventory_resolution_method"))

    if inventory_station_id is None or inventory_resolution_method not in CHILE_SUPPORTED_INVENTORY_RESOLUTION_METHODS:
        return _build_result(
            seed_row,
            provider="chile_dga",
            status="unresolved",
            resolution_method="inventory_not_found",
            candidate_site_numbers=[str(seed_row["source_station_id"]).strip()],
            notes="Chile locator currently supports only DGA station IDs resolved through the local Chile inventory.",
        )

    inventory_records = client.fetch_inventory_station_records(inventory_station_id)
    official_station = _select_official_inventory_station(inventory_records, station_id=inventory_station_id)
    if official_station is None:
        return _build_result(
            seed_row,
            provider="chile_dga",
            status="unresolved",
            resolution_method="inventory_not_confirmed",
            candidate_site_numbers=[inventory_station_id],
            notes="Resolved local Chile inventory station ID could not be confirmed in the official DGA fluviometric inventory.",
        )

    alert_records = client.fetch_alert_records_for_station_prefix(inventory_station_id)
    alert_record = _select_best_alert_record(alert_records, station_id=inventory_station_id)
    resolved_station_name = _nullable_str(official_station.get("NOM_ESTACION")) or _nullable_str(seed_row.get("inventory_station_name"))

    if alert_record is not None:
        alert_code = _nullable_str(alert_record.get(CHILE_ALERT_MOD_CODE_FIELD)) or _nullable_str(alert_record.get(CHILE_ALERT_CODE_FIELD))
        alert_time = _timestamp_iso_from_epoch_millis(alert_record.get(CHILE_ALERT_MOD_TIME_FIELD))
        return _build_result(
            seed_row,
            provider="chile_dga",
            status="subdaily_found",
            resolution_method=inventory_resolution_method,
            candidate_site_numbers=[inventory_station_id, alert_code] if alert_code else [inventory_station_id],
            monitoring_location_id=alert_code or inventory_station_id,
            resolved_site_number=inventory_station_id,
            resolved_station_name=resolved_station_name,
            monitoring_location_found=True,
            discharge_series_found=True,
            resolution_distance_m=_to_float(seed_row.get("inventory_distance_m")),
            discharge_series_count=1,
            instantaneous_series_count=1,
            instantaneous_begin=alert_time,
            instantaneous_end=alert_time,
            notes=(
                "Official DGA ALERTAS exposes a current timestamped record for this station. "
                "This locator does not yet query a historical daily or subdaily archive for Chile."
            ),
        )

    return _build_result(
        seed_row,
        provider="chile_dga",
        status="resolved_no_subdaily",
        resolution_method=inventory_resolution_method,
        candidate_site_numbers=[inventory_station_id],
        monitoring_location_id=inventory_station_id,
        resolved_site_number=inventory_station_id,
        resolved_station_name=resolved_station_name,
        monitoring_location_found=True,
        discharge_series_found=False,
        resolution_distance_m=_to_float(seed_row.get("inventory_distance_m")),
        notes=(
            "Official DGA fluviometric inventory station confirmed, but no live ALERTAS record was returned. "
            "This locator does not yet query a historical daily or subdaily archive for Chile."
        ),
    )


def _get_attribute_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features = payload.get("features") or []
    records: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes")
        if isinstance(attributes, dict):
            records.append(attributes)
    return records


def _select_official_inventory_station(records: list[dict[str, Any]], *, station_id: str) -> dict[str, Any] | None:
    station_id = str(station_id).strip()
    exact_records = [record for record in records if _nullable_str(record.get("COD_BNA")) == station_id]
    if not exact_records:
        return None
    fluviometric = [record for record in exact_records if _normalize_text(record.get("TIPO_ESTACION")) == "fluviometricas"]
    if fluviometric:
        return fluviometric[0]
    return exact_records[0]


def _select_best_alert_record(records: list[dict[str, Any]], *, station_id: str) -> dict[str, Any] | None:
    station_id = str(station_id).strip()
    matching = []
    for record in records:
        code = _nullable_str(record.get(CHILE_ALERT_CODE_FIELD))
        if code is None or not code.startswith(station_id):
            continue
        if _normalize_text(record.get(CHILE_ALERT_FLUVIOMETRICA_FIELD)) != "vig":
            continue
        if _timestamp_iso_from_epoch_millis(record.get(CHILE_ALERT_MOD_TIME_FIELD)) is None:
            continue
        matching.append(record)
    if not matching:
        return None
    matching.sort(
        key=lambda item: (
            _to_float(item.get(CHILE_ALERT_MOD_TIME_FIELD)) or float("-inf"),
            _nullable_str(item.get(CHILE_ALERT_MOD_CODE_FIELD)) or "",
        ),
        reverse=True,
    )
    return matching[0]


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


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


def _timestamp_iso_from_epoch_millis(value: object) -> str | None:
    millis = _to_float(value)
    if millis is None:
        return None
    try:
        parsed = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


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

from __future__ import annotations

import json
import math
import os
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..utils import expand_point_bbox

USGS_WATER_API_BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"
USGS_DISCHARGE_PARAMETER_CODE = "00060"


@dataclass(slots=True)
class MonitoringLocation:
    monitoring_location_id: str
    monitoring_location_number: str | None
    monitoring_location_name: str | None
    lat: float | None
    lon: float | None


class USGSWaterDataClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        user_agent: str = "gauge-sword-match/0.1.0",
        api_key: str | None = None,
        request_pause_seconds: float = 0.0,
        max_retries: int = 4,
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.user_agent = user_agent
        self.api_key = _nullable_str(api_key) or _nullable_str(os.environ.get("USGS_WATER_API_KEY")) or _nullable_str(os.environ.get("USGS_API_KEY"))
        self.request_pause_seconds = max(0.0, float(request_pause_seconds))
        self.max_retries = max(0, int(max_retries))

    def fetch_discharge_metadata(self, monitoring_location_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(
            "time-series-metadata",
            {
                "f": "json",
                "monitoring_location_id": monitoring_location_id,
                "parameter_code": USGS_DISCHARGE_PARAMETER_CODE,
                "limit": 200,
            },
        )
        return _get_feature_list(payload)

    def fetch_monitoring_locations_by_number(self, monitoring_location_number: str) -> list[MonitoringLocation]:
        payload = self._get_json(
            "monitoring-locations",
            {
                "f": "json",
                "monitoring_location_number": monitoring_location_number,
                "limit": 20,
            },
        )
        return [_parse_monitoring_location(feature) for feature in _get_feature_list(payload)]

    def fetch_monitoring_locations_nearby(
        self,
        *,
        lon: float,
        lat: float,
        radius_m: float,
        limit: int = 25,
    ) -> list[MonitoringLocation]:
        bbox = expand_point_bbox(lon, lat, radius_m)
        payload = self._get_json(
            "monitoring-locations",
            {
                "f": "json",
                "bbox": ",".join(f"{value:.8f}" for value in bbox),
                "site_type_code": "ST",
                "limit": max(1, int(limit)),
            },
        )
        return [_parse_monitoring_location(feature) for feature in _get_feature_list(payload)]

    def fetch_continuous_values(
        self,
        *,
        time_series_id: str,
        start_datetime_utc: str,
        end_datetime_utc: str,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        features: list[dict[str, Any]] = []
        offset = 0
        normalized_limit = max(1, int(limit))
        while True:
            payload = self._get_json(
                "continuous",
                {
                    "f": "json",
                    "time_series_id": time_series_id,
                    "datetime": f"{start_datetime_utc}/{end_datetime_utc}",
                    "limit": normalized_limit,
                    "offset": offset,
                },
            )
            page_features = _get_feature_list(payload)
            if not page_features:
                break
            features.extend(page_features)
            if len(page_features) < normalized_limit:
                break
            offset += normalized_limit

        records: list[dict[str, Any]] = []
        for feature in features:
            properties = feature.get("properties") or {}
            timestamp_text = _nullable_str(properties.get("time"))
            value_text = _nullable_str(properties.get("value"))
            value = _to_float(value_text)
            if timestamp_text is None or value is None:
                continue
            records.append(
                {
                    "time": timestamp_text,
                    "value": value,
                    "unit_of_measure": _nullable_str(properties.get("unit_of_measure")),
                    "qualifier": _nullable_str(properties.get("qualifier")),
                    "approval_status": _nullable_str(properties.get("approval_status")),
                    "time_series_id": _nullable_str(properties.get("time_series_id")) or time_series_id,
                    "monitoring_location_id": _nullable_str(properties.get("monitoring_location_id")),
                }
            )
        return pd.DataFrame.from_records(records)

    def _get_json(self, collection: str, params: dict[str, Any]) -> dict[str, Any]:
        query_params = {key: value for key, value in params.items() if value is not None}
        if self.api_key is not None and "api_key" not in query_params:
            query_params["api_key"] = self.api_key
        encoded_params = urlencode(query_params, doseq=True)
        request = Request(
            f"{USGS_WATER_API_BASE_URL}/{collection}/items?{encoded_params}",
            headers={
                "Accept": "application/geo+json, application/json",
                "User-Agent": self.user_agent,
                **({"X-Api-Key": self.api_key} if self.api_key is not None else {}),
            },
        )
        if self.request_pause_seconds > 0:
            time.sleep(self.request_pause_seconds)

        attempt = 0
        while True:
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if attempt < self.max_retries and exc.code in {429, 500, 502, 503, 504}:
                    retry_after = _to_float(exc.headers.get("Retry-After"))
                    sleep_seconds = retry_after if retry_after is not None else min(60.0, 2.0 * (attempt + 1))
                    time.sleep(max(0.0, sleep_seconds))
                    attempt += 1
                    continue
                raise RuntimeError(f"USGS API HTTP {exc.code} for {request.full_url}: {detail or exc.reason}") from exc
            except (TimeoutError, socket.timeout, URLError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(10.0, 1.0 * (attempt + 1)))
                    attempt += 1
                    continue
                reason = exc.reason if isinstance(exc, URLError) else str(exc)
                raise RuntimeError(f"USGS API request failed for {request.full_url}: {reason}") from exc


def locate_usgs_subdaily_station(
    seed_row: pd.Series,
    *,
    client: USGSWaterDataClient,
    search_radius_m: float = 5_000.0,
    nearby_limit: int = 25,
    max_resolution_distance_m: float = 5_000.0,
) -> dict[str, Any]:
    station_key = str(seed_row["station_key"])
    country = str(seed_row["country"]).upper()
    raw_station_id = str(seed_row["source_station_id"]).strip()
    lat = _to_float(seed_row.get("lat"))
    lon = _to_float(seed_row.get("lon"))
    candidate_specs = build_usgs_site_number_candidates(
        raw_station_id,
        inventory_station_id=_nullable_str(seed_row.get("inventory_station_id")),
        inventory_resolution_method=_nullable_str(seed_row.get("inventory_resolution_method")),
    )
    candidate_site_numbers = [candidate["site_number"] for candidate in candidate_specs]

    for idx, candidate in enumerate(candidate_specs):
        site_number = candidate["site_number"]
        candidate_method = candidate["resolution_method"]
        monitoring_location_id = build_monitoring_location_id(site_number)
        metadata_features = client.fetch_discharge_metadata(monitoring_location_id)
        if metadata_features:
            metadata_distance = _metadata_distance_m(metadata_features, seed_lat=lat, seed_lon=lon)
            if not _is_spatially_plausible(metadata_distance, max_resolution_distance_m=max_resolution_distance_m):
                continue
            summary = summarize_discharge_metadata(metadata_features)
            return _build_result(
                seed_row,
                provider="usgs",
                status="subdaily_found" if summary["subdaily_discharge_found"] else "resolved_no_subdaily",
                resolution_method=candidate_method,
                monitoring_location_id=monitoring_location_id,
                resolved_site_number=site_number,
                monitoring_location_found=True,
                discharge_series_found=True,
                resolution_distance_m=metadata_distance,
                metadata_summary=summary,
                candidate_site_numbers=candidate_site_numbers,
            )

        monitoring_locations = client.fetch_monitoring_locations_by_number(site_number)
        exact_locations = [
            location
            for location in monitoring_locations
            if str(location.monitoring_location_number or "").strip() == site_number
        ]
        if exact_locations:
            location = exact_locations[0]
            location_distance = _location_distance_m(location, seed_lat=lat, seed_lon=lon)
            if not _is_spatially_plausible(location_distance, max_resolution_distance_m=max_resolution_distance_m):
                continue
            return _build_result(
                seed_row,
                provider="usgs",
                status="resolved_no_subdaily",
                resolution_method=candidate_method,
                monitoring_location_id=location.monitoring_location_id,
                resolved_site_number=site_number,
                resolved_station_name=location.monitoring_location_name,
                monitoring_location_found=True,
                discharge_series_found=False,
                resolution_distance_m=location_distance,
                metadata_summary=summarize_discharge_metadata([]),
                candidate_site_numbers=candidate_site_numbers,
            )

    if lat is None or lon is None:
        return _build_result(
            seed_row,
            provider="usgs",
            status="unresolved",
            resolution_method="unresolved",
            candidate_site_numbers=candidate_site_numbers,
            notes="No direct USGS station ID match and no coordinates available for nearby search",
        )

    nearby_locations = sorted(
        client.fetch_monitoring_locations_nearby(lon=lon, lat=lat, radius_m=search_radius_m, limit=nearby_limit),
        key=lambda item: (_location_distance_m(item, seed_lat=lat, seed_lon=lon) or math.inf),
    )
    if not nearby_locations:
        return _build_result(
            seed_row,
            provider="usgs",
            status="unresolved",
            resolution_method="unresolved",
            candidate_site_numbers=candidate_site_numbers,
            notes="No direct USGS station ID match and no nearby USGS monitoring location found",
        )

    best_location_without_discharge: MonitoringLocation | None = None
    for location in nearby_locations:
        location_distance = _location_distance_m(location, seed_lat=lat, seed_lon=lon)
        if not _is_spatially_plausible(location_distance, max_resolution_distance_m=max_resolution_distance_m):
            continue
        metadata_features = client.fetch_discharge_metadata(location.monitoring_location_id)
        if metadata_features:
            summary = summarize_discharge_metadata(metadata_features)
            return _build_result(
                seed_row,
                provider="usgs",
                status="subdaily_found" if summary["subdaily_discharge_found"] else "resolved_no_subdaily",
                resolution_method="spatial_nearest_discharge_station",
                monitoring_location_id=location.monitoring_location_id,
                resolved_site_number=location.monitoring_location_number,
                resolved_station_name=location.monitoring_location_name,
                monitoring_location_found=True,
                discharge_series_found=True,
                resolution_distance_m=location_distance,
                metadata_summary=summary,
                candidate_site_numbers=candidate_site_numbers,
            )
        if best_location_without_discharge is None:
            best_location_without_discharge = location

    if best_location_without_discharge is not None:
        return _build_result(
            seed_row,
            provider="usgs",
            status="resolved_no_subdaily",
            resolution_method="spatial_nearest_monitoring_location",
            monitoring_location_id=best_location_without_discharge.monitoring_location_id,
            resolved_site_number=best_location_without_discharge.monitoring_location_number,
            resolved_station_name=best_location_without_discharge.monitoring_location_name,
            monitoring_location_found=True,
            discharge_series_found=False,
            resolution_distance_m=_location_distance_m(best_location_without_discharge, seed_lat=lat, seed_lon=lon),
            metadata_summary=summarize_discharge_metadata([]),
            candidate_site_numbers=candidate_site_numbers,
            notes="Resolved by nearby USGS monitoring location but no discharge time series metadata were found",
        )

    return _build_result(
        seed_row,
        provider="usgs",
        status="unresolved",
        resolution_method="unresolved",
        candidate_site_numbers=candidate_site_numbers,
        notes="USGS nearby search completed without a usable match",
    )


def build_usgs_site_number_candidates(
    raw_station_id: str,
    *,
    inventory_station_id: str | None = None,
    inventory_resolution_method: str | None = None,
) -> list[dict[str, str]]:
    station_id = str(raw_station_id).strip()
    candidates: list[dict[str, str]] = []

    def add_candidate(site_number: str | None, resolution_method: str) -> None:
        if site_number is None:
            return
        normalized = str(site_number).strip()
        if not normalized:
            return
        if any(existing["site_number"] == normalized for existing in candidates):
            return
        candidates.append({"site_number": normalized, "resolution_method": resolution_method})

    if inventory_station_id is not None:
        add_candidate(inventory_station_id, inventory_resolution_method or "inventory_station_id")

    add_candidate(station_id, "direct_site_number")
    if station_id.isdigit() and len(station_id) < 8:
        add_candidate(station_id.zfill(8), "zero_padded_site_number")
    return candidates


def build_monitoring_location_id(site_number: str) -> str:
    return f"USGS-{site_number}"


def summarize_discharge_metadata(features: list[dict[str, Any]]) -> dict[str, Any]:
    instantaneous_features: list[dict[str, Any]] = []
    daily_features: list[dict[str, Any]] = []

    for feature in features:
        properties = feature.get("properties") or {}
        computation_identifier = str(properties.get("computation_identifier") or "").strip().lower()
        computation_period_identifier = str(properties.get("computation_period_identifier") or "").strip().lower()
        if computation_identifier == "instantaneous" or computation_period_identifier == "points":
            instantaneous_features.append(feature)
        if computation_period_identifier == "daily":
            daily_features.append(feature)

    instantaneous_begins = [_coerce_timestamp((feature.get("properties") or {}).get("begin")) for feature in instantaneous_features]
    instantaneous_ends = [_coerce_timestamp((feature.get("properties") or {}).get("end")) for feature in instantaneous_features]
    primary_instantaneous_series_count = sum(
        1
        for feature in instantaneous_features
        if str((feature.get("properties") or {}).get("primary") or "").strip().lower() == "primary"
    )

    return {
        "discharge_series_found": bool(features),
        "subdaily_discharge_found": bool(instantaneous_features),
        "discharge_series_count": len(features),
        "instantaneous_series_count": len(instantaneous_features),
        "primary_instantaneous_series_count": primary_instantaneous_series_count,
        "daily_series_count": len(daily_features),
        "instantaneous_begin": _min_timestamp_iso(instantaneous_begins),
        "instantaneous_end": _max_timestamp_iso(instantaneous_ends),
    }


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
    metadata_summary: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    summary = metadata_summary or summarize_discharge_metadata([])
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
        "subdaily_discharge_found": bool(summary["subdaily_discharge_found"]),
        "discharge_series_count": int(summary["discharge_series_count"]),
        "instantaneous_series_count": int(summary["instantaneous_series_count"]),
        "primary_instantaneous_series_count": int(summary["primary_instantaneous_series_count"]),
        "daily_series_count": int(summary["daily_series_count"]),
        "instantaneous_begin": summary["instantaneous_begin"],
        "instantaneous_end": summary["instantaneous_end"],
        "notes": notes,
    }


def _get_feature_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features = payload.get("features")
    if isinstance(features, list):
        return [feature for feature in features if isinstance(feature, dict)]
    return []


def _parse_monitoring_location(feature: dict[str, Any]) -> MonitoringLocation:
    properties = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    lon = _to_float(coordinates[0] if len(coordinates) > 0 else None)
    lat = _to_float(coordinates[1] if len(coordinates) > 1 else None)
    return MonitoringLocation(
        monitoring_location_id=str(feature.get("id") or properties.get("id") or "").strip(),
        monitoring_location_number=_nullable_str(properties.get("monitoring_location_number")),
        monitoring_location_name=_nullable_str(properties.get("monitoring_location_name")),
        lat=lat,
        lon=lon,
    )


def _metadata_distance_m(features: list[dict[str, Any]], *, seed_lat: float | None, seed_lon: float | None) -> float | None:
    if seed_lat is None or seed_lon is None:
        return None
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        lon = _to_float(coordinates[0] if len(coordinates) > 0 else None)
        lat = _to_float(coordinates[1] if len(coordinates) > 1 else None)
        if lat is None or lon is None:
            continue
        return _haversine_distance_m(seed_lon, seed_lat, lon, lat)
    return None


def _location_distance_m(location: MonitoringLocation, *, seed_lat: float | None, seed_lon: float | None) -> float | None:
    if seed_lat is None or seed_lon is None or location.lat is None or location.lon is None:
        return None
    return _haversine_distance_m(seed_lon, seed_lat, location.lon, location.lat)


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


def _is_spatially_plausible(distance_m: float | None, *, max_resolution_distance_m: float) -> bool:
    if distance_m is None:
        return True
    return distance_m <= max(0.0, float(max_resolution_distance_m))


def _coerce_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _min_timestamp_iso(values: list[pd.Timestamp | pd.NaT]) -> str | None:
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return None
    return min(valid).isoformat()


def _max_timestamp_iso(values: list[pd.Timestamp | pd.NaT]) -> str | None:
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return None
    return max(valid).isoformat()


def _nullable_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

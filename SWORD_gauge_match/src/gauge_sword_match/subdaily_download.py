from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import pandas as pd

from .canada_manual_download import load_canada_manual_archive
from .chile_manual_excel import load_chile_manual_archive
from .mrc_rating_curve import load_mrc_manual_waterlevel_archive
from .subdaily_locator.brazil import BrazilAnaHydroClient
from .subdaily_locator.canada import CanadaWaterofficeClient
from .subdaily_locator.chile import (
    CHILE_ALERT_CODE_FIELD,
    CHILE_ALERT_FLUVIOMETRICA_FIELD,
    CHILE_ALERT_MOD_CODE_FIELD,
    CHILE_ALERT_MOD_TIME_FIELD,
    CHILE_ALERT_MOD_VALUE_FIELD,
    ChileDgaClient,
)
from .subdaily_locator.colombia import ColombiaIdeamFewsClient
from .subdaily_locator.french_guiana import FranceHubeauClient
from .subdaily_locator.mekong_mrc import MekongMrcClient
from .subdaily_locator.usgs import USGSWaterDataClient
from .us_manual_download import load_us_manual_archive
from .utils import ensure_directory, get_logger, read_table, write_table

LOGGER = get_logger("subdaily_download")

DEFAULT_TARGET_START_DATE = date(2010, 1, 1)
DEFAULT_MINIMUM_COMPLETENESS = 0.70
DEFAULT_MAX_GAP_DAYS = 183.0
DEFAULT_FALLBACK_YEARS = (10, 5, 2, 1)
DEFAULT_EXTENDED_TARGET_START_DATE = date(2000, 1, 1)
USGS_CONTINUOUS_CHUNK_DAYS = 365
USGS_CONTINUOUS_MIN_CHUNK_DAYS = 30
BRAZIL_TELEMETRIC_CHUNK_DAYS = 180
BRAZIL_TELEMETRIC_MIN_CHUNK_DAYS = 30
DEFAULT_COUNTRY_TIMESERIES_FILENAME = "subdaily_timeseries.parquet"
DEFAULT_COUNTRY_MANIFEST_FILENAME = "subdaily_download_manifest.csv"
DEFAULT_COUNTRY_SUMMARY_FILENAME = "subdaily_country_download_summary.csv"
CANADA_DUPLICATE_STATION_KEYS_TO_SKIP = {
    "CA:4208871",
    "CA:4214590",
}

INTERNAL_SERIES_COLUMNS = [
    "time",
    "discharge",
    "raw_discharge",
    "unit_of_measure",
    "raw_unit_of_measure",
    "unit_normalized",
    "provider_series_name",
    "provider_series_id",
]
OUTPUT_TIMESERIES_COLUMNS = [
    "station_key",
    "station_id",
    "country",
    "time",
    "discharge",
    "variable",
    "source_function",
    "provider",
    "provider_station_id",
    "provider_series_id",
    "provider_series_name",
    "unit_of_measure",
    "raw_discharge",
    "raw_unit_of_measure",
    "unit_normalized",
    "window_strategy",
]
SUPPORTED_DOWNLOAD_PROVIDERS = {
    "usgs",
    "canada_wateroffice",
    "brazil_ana",
    "france_hubeau",
    "colombia_ideam_fews",
    "mrc_timeseries",
    "chile_dga",
}


@dataclass(slots=True)
class ProviderSeriesResult:
    frame: pd.DataFrame
    notes: str | None = None
    manifest_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class WindowAssessment:
    label: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    row_count: int
    median_timestep_hours: float | None
    completeness_ratio: float | None
    max_gap_days: float | None
    acceptable: bool


def download_subdaily_from_audit(
    audit_path: str | Path,
    *,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    countries: Sequence[str] | None = None,
    target_start_date: date = DEFAULT_TARGET_START_DATE,
    minimum_completeness: float = DEFAULT_MINIMUM_COMPLETENESS,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    fallback_years: Sequence[int] = DEFAULT_FALLBACK_YEARS,
    clients: Mapping[str, Any] | None = None,
    provider_contexts: Mapping[str, Any] | None = None,
    now_utc: datetime | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows = _load_download_station_rows(audit_path, countries=countries)
    runtime_now = now_utc or datetime.now(timezone.utc)
    provider_clients = _build_default_clients(clients)
    provider_cache: dict[tuple[str, str], ProviderSeriesResult] = {}

    timeseries_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    station_records = station_rows.to_dict(orient="records")

    progress_context = (
        click.progressbar(
            station_records,
            label="Downloading subdaily stations",
            item_show_func=lambda item: None if item is None else f"{item.get('country', '')}:{item.get('station_key', '')}",
        )
        if show_progress
        else nullcontext(station_records)
    )

    with progress_context as iterator:
        for row in iterator:
            provider = str(row["provider"]).strip()
            provider_station_id = _nullable_str(row.get("resolved_site_number"))
            manifest_base = _build_manifest_base(
                row,
                runtime_now=runtime_now,
                target_start_date=target_start_date,
            )

            if provider_station_id is None:
                manifest_rows.append(
                    {
                        **manifest_base,
                        "download_status": "missing_provider_station_id",
                        "notes": "Audit row is missing a resolved provider station identifier.",
                    }
                )
                continue

            try:
                cache_key = (provider, provider_station_id)
                if cache_key not in provider_cache:
                    provider_cache[cache_key] = _fetch_provider_series(
                        row,
                        provider_clients=provider_clients,
                        provider_context=provider_contexts.get(provider) if provider_contexts else None,
                        runtime_now=runtime_now,
                        target_start_date=target_start_date,
                    )

                provider_result = provider_cache[cache_key]
                raw_frame = provider_result.frame.copy()
                prepared_raw = _prepare_internal_series(raw_frame)
                raw_assessment = _assess_window(prepared_raw, label="raw_available")
                selected_frame, selected_assessment, window_strategy = _select_download_window(
                    prepared_raw,
                    provider=provider,
                    target_start_date=target_start_date,
                    minimum_completeness=minimum_completeness,
                    max_gap_days=max_gap_days,
                    fallback_years=fallback_years,
                )

                if selected_frame.empty:
                    manifest_rows.append(
                        {
                            **manifest_base,
                            "download_status": "no_data_returned",
                            "raw_row_count": raw_assessment.row_count,
                            "notes": _join_notes(provider_result.notes, "Provider returned no usable subdaily observations."),
                        }
                    )
                    continue

                station_series = _materialize_station_series(
                    selected_frame,
                    row=row,
                    provider=provider,
                    window_strategy=window_strategy,
                )
                timeseries_frames.append(station_series)

                manifest_rows.append(
                    {
                        **manifest_base,
                        "download_status": "ok",
                        "window_strategy": window_strategy,
                        "raw_returned_start": _timestamp_iso(raw_assessment.start),
                        "raw_returned_end": _timestamp_iso(raw_assessment.end),
                        "raw_row_count": raw_assessment.row_count,
                        "selected_start": _timestamp_iso(selected_assessment.start),
                        "selected_end": _timestamp_iso(selected_assessment.end),
                        "selected_row_count": selected_assessment.row_count,
                        "median_timestep_hours": selected_assessment.median_timestep_hours,
                        "completeness_ratio": selected_assessment.completeness_ratio,
                        "max_gap_days": selected_assessment.max_gap_days,
                        "unit_of_measure": _first_non_null(selected_frame.get("unit_of_measure")),
                        "raw_unit_of_measure": _first_non_null(selected_frame.get("raw_unit_of_measure")),
                        "unit_normalized": bool(_first_non_null(selected_frame.get("unit_normalized"), default=False)),
                        "provider_series_name": _first_non_null(selected_frame.get("provider_series_name")),
                        "provider_series_id": _first_non_null(selected_frame.get("provider_series_id")),
                        **(provider_result.manifest_metadata or {}),
                        "notes": provider_result.notes,
                    }
                )
            except Exception as exc:
                manifest_rows.append(
                    {
                        **manifest_base,
                        "download_status": "error",
                        "notes": str(exc),
                    }
                )

    timeseries = (
        pd.concat(timeseries_frames, ignore_index=True)
        if timeseries_frames
        else pd.DataFrame(columns=OUTPUT_TIMESERIES_COLUMNS)
    )
    manifest = pd.DataFrame.from_records(manifest_rows)

    if not timeseries.empty:
        timeseries = timeseries.sort_values(["station_key", "time"]).reset_index(drop=True)
    if not manifest.empty:
        manifest = manifest.sort_values(["country", "station_key"]).reset_index(drop=True)

    if output_path is not None:
        write_table(timeseries, output_path)
        LOGGER.info("Subdaily timeseries wrote %s rows to %s", len(timeseries), output_path)
    if manifest_path is not None:
        write_table(manifest, manifest_path)
        LOGGER.info("Subdaily download manifest wrote %s station rows to %s", len(manifest), manifest_path)

    return timeseries, manifest


def download_subdaily_to_country_outputs(
    audit_path: str | Path,
    *,
    output_dir: str | Path,
    countries: Sequence[str] | None = None,
    target_start_date: date = DEFAULT_TARGET_START_DATE,
    minimum_completeness: float = DEFAULT_MINIMUM_COMPLETENESS,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    fallback_years: Sequence[int] = DEFAULT_FALLBACK_YEARS,
    clients: Mapping[str, Any] | None = None,
    now_utc: datetime | None = None,
    show_progress: bool = False,
    timeseries_filename: str = DEFAULT_COUNTRY_TIMESERIES_FILENAME,
    manifest_filename: str = DEFAULT_COUNTRY_MANIFEST_FILENAME,
    summary_filename: str = DEFAULT_COUNTRY_SUMMARY_FILENAME,
) -> pd.DataFrame:
    station_rows = _load_download_station_rows(audit_path, countries=countries)
    selected_countries = sorted(
        {
            str(value).strip().upper()
            for value in station_rows.get("country", pd.Series(dtype="string")).dropna().tolist()
            if str(value).strip()
        }
    )
    if not selected_countries:
        raise ValueError("No subdaily_found stations matched the requested country filter.")

    runtime_now = now_utc or datetime.now(timezone.utc)
    output_dir = ensure_directory(output_dir)
    summary_rows: list[dict[str, Any]] = []

    for country in selected_countries:
        country_dir = ensure_directory(output_dir / country)
        country_output_path = country_dir / timeseries_filename
        country_manifest_path = country_dir / manifest_filename
        timeseries, manifest = download_subdaily_from_audit(
            audit_path,
            output_path=country_output_path,
            manifest_path=country_manifest_path,
            countries=[country],
            target_start_date=target_start_date,
            minimum_completeness=minimum_completeness,
            max_gap_days=max_gap_days,
            fallback_years=fallback_years,
            clients=clients,
            provider_contexts=_build_country_provider_contexts(country_dir),
            now_utc=runtime_now,
            show_progress=show_progress,
        )

        status_counts = (
            manifest["download_status"].value_counts(dropna=False).to_dict()
            if not manifest.empty and "download_status" in manifest.columns
            else {}
        )
        summary_rows.append(
            {
                "country": country,
                "requested_start": f"{target_start_date.isoformat()}T00:00:00Z",
                "requested_end": runtime_now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "station_count": int(len(manifest)),
                "ok_count": int(status_counts.get("ok", 0)),
                "error_count": int(status_counts.get("error", 0)),
                "no_data_count": int(status_counts.get("no_data_returned", 0)),
                "missing_provider_station_id_count": int(status_counts.get("missing_provider_station_id", 0)),
                "timeseries_row_count": int(len(timeseries)),
                "timeseries_path": str(country_output_path),
                "manifest_path": str(country_manifest_path),
            }
        )

    summary = pd.DataFrame.from_records(summary_rows)
    summary_path = output_dir / summary_filename
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        if "country" in existing_summary.columns:
            existing_summary["country"] = existing_summary["country"].astype("string").str.upper()
            existing_summary = existing_summary[~existing_summary["country"].isin(selected_countries)].copy()
            summary = pd.concat([existing_summary, summary], ignore_index=True, sort=False)
    summary["country"] = summary["country"].astype("string").str.upper()
    summary = summary.sort_values("country").reset_index(drop=True)
    write_table(summary, summary_path)
    LOGGER.info("Per-country subdaily download summary wrote %s rows to %s", len(summary), summary_path)
    return summary


def _load_download_station_rows(
    audit_path: str | Path,
    *,
    countries: Sequence[str] | None = None,
) -> pd.DataFrame:
    audit_path = Path(audit_path)
    if audit_path.suffix.lower() == ".csv":
        audit = pd.read_csv(audit_path, dtype="string")
    else:
        audit = read_table(audit_path)
    required_columns = {"station_key", "country", "provider", "status", "resolved_site_number"}
    missing = required_columns - set(audit.columns)
    if missing:
        raise ValueError(
            "Subdaily download audit is missing required columns: " + ", ".join(sorted(missing))
        )

    working = audit.copy()
    working["status"] = working["status"].astype("string").fillna(pd.NA)
    working = working[working["status"].str.lower() == "subdaily_found"].copy()

    if countries:
        allowed = {str(value).strip().upper() for value in countries if str(value).strip()}
        working["country"] = working["country"].astype("string").str.upper()
        working = working[working["country"].isin(sorted(allowed))].copy()
    else:
        working["country"] = working["country"].astype("string").str.upper()

    working["provider"] = working["provider"].astype("string").fillna(pd.NA)
    canada_duplicate_mask = (
        working["provider"].astype("string").eq("canada_wateroffice")
        & working["station_key"].astype("string").isin(CANADA_DUPLICATE_STATION_KEYS_TO_SKIP)
    )
    if canada_duplicate_mask.any():
        working = working[~canada_duplicate_mask].copy()
    working["resolved_site_number"] = working.apply(
        lambda row: _normalize_provider_station_id(
            row.get("resolved_site_number"),
            provider=row.get("provider"),
            inventory_station_key=row.get("inventory_station_key"),
        ),
        axis=1,
    )
    unsupported = sorted(set(working["provider"].dropna()) - SUPPORTED_DOWNLOAD_PROVIDERS)
    if unsupported:
        raise ValueError(
            "Subdaily downloader does not support these providers yet: " + ", ".join(unsupported)
        )

    working = working.drop_duplicates(subset=["station_key"], keep="first").reset_index(drop=True)
    return working.sort_values(["country", "station_key"]).reset_index(drop=True)


def _build_default_clients(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    clients = {
        "usgs": USGSWaterDataClient(),
        "canada_wateroffice": CanadaWaterofficeClient(),
        "brazil_ana": BrazilAnaHydroClient(),
        "france_hubeau": FranceHubeauClient(),
        "colombia_ideam_fews": ColombiaIdeamFewsClient(),
        "mrc_timeseries": MekongMrcClient(),
        "chile_dga": ChileDgaClient(),
    }
    if overrides:
        clients.update(dict(overrides))
    return clients


def _build_country_provider_contexts(country_dir: Path) -> dict[str, Any]:
    chile_excel_dir = country_dir / "excel_download"
    manual_download_dir = country_dir / "manual_download"
    mrc_waterlevel_dir = country_dir / "Waterlevel"
    contexts: dict[str, Any] = {}
    if chile_excel_dir.exists() and chile_excel_dir.is_dir():
        contexts["chile_dga"] = {"manual_excel_dir": chile_excel_dir}
    if manual_download_dir.exists() and manual_download_dir.is_dir():
        contexts["canada_wateroffice"] = {"manual_download_dir": manual_download_dir}
        contexts["usgs"] = {"manual_download_dir": manual_download_dir}
    if mrc_waterlevel_dir.exists() and mrc_waterlevel_dir.is_dir():
        contexts["mrc_timeseries"] = {"manual_waterlevel_dir": mrc_waterlevel_dir}
    return contexts


def _fetch_provider_series(
    row: pd.Series,
    *,
    provider_clients: Mapping[str, Any],
    provider_context: Mapping[str, Any] | None,
    runtime_now: datetime,
    target_start_date: date,
) -> ProviderSeriesResult:
    provider = str(row["provider"]).strip()
    if provider == "usgs":
        return _fetch_usgs_subdaily_series(
            row,
            client=provider_clients["usgs"],
            provider_context=provider_context,
            runtime_now=runtime_now,
            target_start_date=target_start_date,
        )
    if provider == "canada_wateroffice":
        return _fetch_canada_subdaily_series(
            row,
            client=provider_clients["canada_wateroffice"],
            provider_context=provider_context,
            runtime_now=runtime_now,
            target_start_date=target_start_date,
        )
    if provider == "brazil_ana":
        return _fetch_brazil_subdaily_series(
            row,
            client=provider_clients["brazil_ana"],
            runtime_now=runtime_now,
            target_start_date=target_start_date,
        )
    if provider == "france_hubeau":
        return _fetch_french_guiana_subdaily_series(
            row,
            client=provider_clients["france_hubeau"],
        )
    if provider == "colombia_ideam_fews":
        return _fetch_colombia_subdaily_series(
            row,
            client=provider_clients["colombia_ideam_fews"],
        )
    if provider == "mrc_timeseries":
        return _fetch_mrc_subdaily_series(
            row,
            client=provider_clients["mrc_timeseries"],
            provider_context=provider_context,
        )
    if provider == "chile_dga":
        return _fetch_chile_subdaily_series(
            row,
            client=provider_clients["chile_dga"],
            provider_context=provider_context,
        )
    raise NotImplementedError(f"Unsupported provider for subdaily download: {provider}")


def _fetch_usgs_subdaily_series(
    row: pd.Series,
    *,
    client: USGSWaterDataClient,
    provider_context: Mapping[str, Any] | None,
    runtime_now: datetime,
    target_start_date: date,
) -> ProviderSeriesResult:
    provider_station_id = str(row["resolved_site_number"]).strip()
    manual_download_dir = None if provider_context is None else provider_context.get("manual_download_dir")
    if manual_download_dir is not None:
        manual_frame, manual_notes = load_us_manual_archive(provider_station_id, manual_download_dir)
        if not manual_frame.empty:
            return ProviderSeriesResult(manual_frame, notes=manual_notes)

    monitoring_location_id = f"USGS-{provider_station_id}"
    metadata = client.fetch_discharge_metadata(monitoring_location_id)
    time_series_id = _select_usgs_instantaneous_time_series_id(metadata)
    if time_series_id is None:
        return ProviderSeriesResult(_empty_internal_series(), notes="No USGS instantaneous discharge time series metadata were returned.")

    frame = _fetch_usgs_continuous_values_chunked(
        client,
        time_series_id=time_series_id,
        start_datetime_utc=pd.Timestamp(f"{target_start_date.isoformat()}T00:00:00Z"),
        end_datetime_utc=pd.Timestamp(runtime_now.astimezone(timezone.utc)),
    )
    if frame.empty:
        return ProviderSeriesResult(_empty_internal_series(), notes="USGS continuous endpoint returned no rows for the selected discharge time series.")

    unit_value = _normalize_unit_scalar(_first_non_null(frame.get("unit_of_measure")))
    raw_values = pd.to_numeric(frame.get("value"), errors="coerce")
    converted_values, unit_of_measure, unit_normalized, raw_unit_of_measure = _convert_known_units(
        raw_values,
        default_unit=unit_value,
    )
    normalized = pd.DataFrame(
        {
            "time": _parse_time_series(frame.get("time")),
            "discharge": converted_values,
            "raw_discharge": raw_values,
            "unit_of_measure": unit_of_measure,
            "raw_unit_of_measure": raw_unit_of_measure,
            "unit_normalized": unit_normalized,
            "provider_series_name": "continuous_00060",
            "provider_series_id": frame.get("time_series_id").astype("string") if "time_series_id" in frame.columns else time_series_id,
        }
    )
    notes = None
    if unit_value is not None:
        notes = f"USGS continuous series downloaded from time_series_id `{time_series_id}`."
    return ProviderSeriesResult(normalized, notes=notes)


def _fetch_canada_subdaily_series(
    row: pd.Series,
    *,
    client: CanadaWaterofficeClient,
    provider_context: Mapping[str, Any] | None,
    runtime_now: datetime,
    target_start_date: date,
) -> ProviderSeriesResult:
    provider_station_id = str(row["resolved_site_number"]).strip()
    manual_download_dir = None if provider_context is None else provider_context.get("manual_download_dir")
    if manual_download_dir is not None:
        manual_frame, manual_notes = load_canada_manual_archive(provider_station_id, manual_download_dir)
        if not manual_frame.empty:
            return ProviderSeriesResult(manual_frame, notes=manual_notes)

    frame = client.fetch_discharge_unit_values(
        provider_station_id,
        start_datetime_utc=datetime.combine(target_start_date, datetime.min.time(), tzinfo=timezone.utc),
        end_datetime_utc=runtime_now.astimezone(timezone.utc),
    )
    if frame.empty:
        return ProviderSeriesResult(_empty_internal_series(), notes="Canada Wateroffice unit-values endpoint returned no rows.")

    time_column = _guess_time_column(frame)
    value_column = _guess_value_column(frame)
    normalized = pd.DataFrame(
        {
            "time": _parse_time_series(frame.get(time_column)),
            "discharge": pd.to_numeric(frame.get(value_column), errors="coerce"),
            "raw_discharge": pd.to_numeric(frame.get(value_column), errors="coerce"),
            "unit_of_measure": "m3/s",
            "raw_unit_of_measure": "m3/s",
            "unit_normalized": True,
            "provider_series_name": "unit_values_47",
            "provider_series_id": provider_station_id,
        }
    )
    return ProviderSeriesResult(normalized, notes=f"Canada Wateroffice discharge unit-values downloaded from station `{provider_station_id}`.")


def _fetch_brazil_subdaily_series(
    row: pd.Series,
    *,
    client: BrazilAnaHydroClient,
    runtime_now: datetime,
    target_start_date: date,
) -> ProviderSeriesResult:
    provider_station_id = str(row["resolved_site_number"]).strip()
    frame = _fetch_brazil_subdaily_values_chunked(
        client,
        station_id=provider_station_id,
        start_date=target_start_date,
        end_date=runtime_now.date(),
    )
    if frame.empty:
        return ProviderSeriesResult(_empty_internal_series(), notes="Brazil ANA telemetric discharge endpoint returned no rows.")

    raw_values = pd.to_numeric(frame.get("Vazao"), errors="coerce")
    normalized = pd.DataFrame(
        {
            "time": _parse_time_series(frame.get("DateTime")),
            "discharge": raw_values,
            "raw_discharge": raw_values,
            "unit_of_measure": "m3/s",
            "raw_unit_of_measure": "m3/s",
            "unit_normalized": True,
            "provider_series_name": "telemetric_vazao",
            "provider_series_id": provider_station_id,
        }
    )
    return ProviderSeriesResult(normalized, notes=f"Brazil ANA telemetric discharge downloaded from station `{provider_station_id}`.")


def _fetch_french_guiana_subdaily_series(
    row: pd.Series,
    *,
    client: FranceHubeauClient,
) -> ProviderSeriesResult:
    provider_station_id = str(row["resolved_site_number"]).strip()
    frame = client.fetch_realtime_discharge_values(provider_station_id)
    if frame.empty:
        return ProviderSeriesResult(_empty_internal_series(), notes="France Hubeau realtime discharge endpoint returned no rows.")

    normalized = pd.DataFrame(
        {
            "time": _parse_time_series(frame.get("time")),
            "discharge": pd.to_numeric(frame.get("value_m3s"), errors="coerce"),
            "raw_discharge": pd.to_numeric(frame.get("raw_value"), errors="coerce"),
            "unit_of_measure": frame.get("unit_of_measure"),
            "raw_unit_of_measure": frame.get("raw_unit_of_measure"),
            "unit_normalized": True,
            "provider_series_name": "observations_tr_Q",
            "provider_series_id": provider_station_id,
        }
    )
    return ProviderSeriesResult(
        normalized,
        notes=(
            "France Hubeau realtime discharge downloaded from `observations_tr`. "
            "The public API retains about one month of realtime history, so historical fallback is not available here."
        ),
    )


def _fetch_colombia_subdaily_series(
    row: pd.Series,
    *,
    client: ColombiaIdeamFewsClient,
) -> ProviderSeriesResult:
    provider_station_id = _normalize_provider_station_id(
        row.get("resolved_site_number"),
        provider="colombia_ideam_fews",
        inventory_station_key=row.get("inventory_station_key"),
    )
    if provider_station_id is None:
        return ProviderSeriesResult(_empty_internal_series(), notes="Colombia downloader could not derive a FEWS station identifier from the audit row.")
    payload = client.fetch_discharge_payload(provider_station_id)
    observed = _extract_colombia_points(payload, "obs")
    sensor = _extract_colombia_points(payload, "sen")

    if observed:
        chosen = observed
        chosen_name = "jsonQ_obs"
        notes = "IDEAM FEWS observed discharge series downloaded from `jsonQ/<station_id>Qobs.json`."
        if sensor:
            notes += " Sensor discharge rows were also available, but the observed series was preferred."
    elif sensor:
        chosen = sensor
        chosen_name = "jsonQ_sen"
        notes = "IDEAM FEWS sensor discharge series downloaded from `jsonQ/<station_id>Qobs.json`."
    else:
        return ProviderSeriesResult(_empty_internal_series(), notes="IDEAM FEWS discharge payload contained no usable time-value pairs.")

    normalized = pd.DataFrame.from_records(
        [
            {
                "time": point["time"],
                "discharge": point["value"],
                "raw_discharge": point["value"],
                "unit_of_measure": pd.NA,
                "raw_unit_of_measure": pd.NA,
                "unit_normalized": False,
                "provider_series_name": chosen_name,
                "provider_series_id": provider_station_id,
            }
            for point in chosen
        ]
    )
    return ProviderSeriesResult(normalized, notes=notes)


def _fetch_mrc_subdaily_series(
    row: pd.Series,
    *,
    client: MekongMrcClient,
    provider_context: Mapping[str, Any] | None = None,
) -> ProviderSeriesResult:
    provider_station_id = _normalize_provider_station_id(
        row.get("resolved_site_number"),
        provider="mrc_timeseries",
        inventory_station_key=row.get("inventory_station_key"),
    )
    if provider_station_id is None:
        return ProviderSeriesResult(_empty_internal_series(), notes="MRC downloader could not derive a station code from the audit row.")
    manual_waterlevel_dir = None if provider_context is None else provider_context.get("manual_waterlevel_dir")
    if manual_waterlevel_dir is not None:
        manual_result = load_mrc_manual_waterlevel_archive(provider_station_id, manual_waterlevel_dir)
        if manual_result is not None:
            manual_frame, manifest_metadata, manual_notes = manual_result
            return ProviderSeriesResult(
                manual_frame,
                notes=manual_notes,
                manifest_metadata=manifest_metadata,
            )
    inventory_rows = client.fetch_time_series_inventory()
    discharge_rows = [
        item
        for item in inventory_rows
        if _nullable_str(item.get("stationCode")) == provider_station_id
        and _normalize_text(item.get("parameter")) == "discharge"
    ]
    if not discharge_rows:
        return ProviderSeriesResult(_empty_internal_series(), notes="MRC station inventory contained no discharge series for the resolved station code.")

    records: list[dict[str, Any]] = []
    unique_series_ids: list[str] = []
    for item in discharge_rows:
        unique_id = _nullable_str(item.get("uniqueId"))
        if unique_id is None:
            continue
        unique_series_ids.append(unique_id)
        payload = client.fetch_corrected_time_series_data(unique_id)
        for point in _extract_mrc_points(payload):
            records.append(
                {
                    "time": point["time"],
                    "discharge": point["value"],
                    "raw_discharge": point["value"],
                    "unit_of_measure": pd.NA,
                    "raw_unit_of_measure": pd.NA,
                    "unit_normalized": False,
                    "provider_series_name": "timeSeriesCorrectedData",
                    "provider_series_id": unique_id,
                }
            )

    if not records:
        return ProviderSeriesResult(_empty_internal_series(), notes="MRC corrected-series endpoint returned no usable points for the resolved discharge station.")

    normalized = pd.DataFrame.from_records(records)
    notes = f"MRC corrected discharge downloaded from {len(set(unique_series_ids))} series for station `{provider_station_id}`."
    return ProviderSeriesResult(normalized, notes=notes)


def _fetch_chile_subdaily_series(
    row: pd.Series,
    *,
    client: ChileDgaClient,
    provider_context: Mapping[str, Any] | None = None,
) -> ProviderSeriesResult:
    provider_station_id = str(row["resolved_site_number"]).strip()
    manual_excel_dir = None if provider_context is None else provider_context.get("manual_excel_dir")
    if manual_excel_dir is not None:
        manual_frame, manual_notes = load_chile_manual_archive(provider_station_id, manual_excel_dir)
        if not manual_frame.empty:
            return ProviderSeriesResult(manual_frame, notes=manual_notes)

    records = client.fetch_alert_records_for_station_prefix(provider_station_id)
    rows: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        code = _nullable_str(item.get(CHILE_ALERT_CODE_FIELD))
        if code is None or not code.startswith(provider_station_id):
            continue
        if _normalize_text(item.get(CHILE_ALERT_FLUVIOMETRICA_FIELD)) != "vig":
            continue
        timestamp = _parse_epoch_millis(item.get(CHILE_ALERT_MOD_TIME_FIELD))
        value = _to_float(item.get(CHILE_ALERT_MOD_VALUE_FIELD))
        series_id = _nullable_str(item.get(CHILE_ALERT_MOD_CODE_FIELD)) or code
        if timestamp is None or value is None:
            continue
        rows.append(
            {
                "time": timestamp,
                "discharge": value,
                "raw_discharge": value,
                "unit_of_measure": pd.NA,
                "raw_unit_of_measure": pd.NA,
                "unit_normalized": False,
                "provider_series_name": "alertas_live",
                "provider_series_id": series_id,
            }
        )

    if not rows:
        return ProviderSeriesResult(_empty_internal_series(), notes="Chile DGA live ALERTAS query returned no usable timestamped discharge rows.")

    normalized = pd.DataFrame.from_records(rows)
    return ProviderSeriesResult(
        normalized,
        notes=(
            "Chile DGA download currently captures only the live ALERTAS snapshot rows exposed by the public ArcGIS service. "
            "A historical subdaily archive is not queried here yet."
        ),
    )


def _select_download_window(
    frame: pd.DataFrame,
    *,
    provider: str,
    target_start_date: date,
    minimum_completeness: float,
    max_gap_days: float,
    fallback_years: Sequence[int],
) -> tuple[pd.DataFrame, WindowAssessment, str]:
    if frame.empty:
        return frame.copy(), _assess_window(frame, label="empty"), "no_data"

    provider_series_names = set(frame.get("provider_series_name", pd.Series(dtype="string")).dropna().astype(str).tolist())
    if provider == "chile_dga" and provider_series_names <= {"alertas_live"}:
        assessment = _assess_window(frame, label="live_snapshot")
        return frame.copy(), assessment, "live_snapshot_only"

    full_candidate = frame.copy()
    full_assessment = _assess_window(full_candidate, label="full_available")
    target_window_years = max(1, int(fallback_years[0])) if fallback_years else 10
    raw_span_years = _window_span_years(full_assessment)

    if raw_span_years is None or raw_span_years < float(target_window_years):
        strategy = "full_available_short_record"
        if not _is_acceptable(
            full_assessment,
            minimum_completeness=minimum_completeness,
            max_gap_days=max_gap_days,
        ):
            strategy = f"{strategy}_gap_fallback"
        return full_candidate, full_assessment, strategy

    latest = frame["time"].max()
    earliest = frame["time"].min()
    latest_target_start = latest - pd.DateOffset(years=target_window_years)
    candidates: list[tuple[pd.DataFrame, WindowAssessment, str]] = []
    seen_ranges: set[tuple[pd.Timestamp | None, pd.Timestamp | None]] = set()

    def add_candidate(
        candidate_frame: pd.DataFrame,
        label: str,
        *,
        expected_start: pd.Timestamp | None = None,
        expected_end: pd.Timestamp | None = None,
    ) -> None:
        if candidate_frame.empty:
            return
        assessment = _assess_window(
            candidate_frame,
            label=label,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        range_key = (expected_start or assessment.start, expected_end or assessment.end)
        if range_key in seen_ranges:
            return
        seen_ranges.add(range_key)
        candidates.append((candidate_frame, assessment, label))

    add_candidate(
        frame[frame["time"] >= latest_target_start].copy(),
        f"latest_{target_window_years}y",
        expected_start=max(latest_target_start, earliest),
        expected_end=latest,
    )

    for anchor, label in (
        (target_start_date, f"since_{target_start_date.year}"),
        (DEFAULT_EXTENDED_TARGET_START_DATE, f"since_{DEFAULT_EXTENDED_TARGET_START_DATE.year}"),
    ):
        anchor_ts = pd.Timestamp(anchor.isoformat(), tz="UTC")
        add_candidate(
            frame[frame["time"] >= anchor_ts].copy(),
            label,
            expected_start=max(anchor_ts, earliest),
            expected_end=latest,
        )

    for candidate, assessment, strategy in candidates:
        if _is_acceptable(assessment, minimum_completeness=minimum_completeness, max_gap_days=max_gap_days):
            return candidate, assessment, strategy

    widest_candidate = max(
        candidates,
        key=lambda item: (
            _window_span_years(item[1]) or -1.0,
            item[1].row_count,
        ),
    )
    return widest_candidate[0], widest_candidate[1], f"{widest_candidate[2]}_gap_fallback"


def _is_acceptable(
    assessment: WindowAssessment,
    *,
    minimum_completeness: float,
    max_gap_days: float,
) -> bool:
    if assessment.row_count <= 0:
        return False
    if assessment.max_gap_days is None:
        return assessment.row_count > 0
    return float(assessment.max_gap_days) <= float(max_gap_days)


def _assess_window(
    frame: pd.DataFrame,
    *,
    label: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> WindowAssessment:
    if frame.empty:
        return WindowAssessment(
            label=label,
            start=None,
            end=None,
            row_count=0,
            median_timestep_hours=None,
            completeness_ratio=None,
            max_gap_days=None,
            acceptable=False,
        )

    working = frame.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    times = working["time"]
    start = times.iloc[0]
    end = times.iloc[-1]
    row_count = int(len(working))
    effective_start = expected_start if expected_start is not None else start
    effective_end = expected_end if expected_end is not None else end
    boundary_times = list(times)
    if expected_start is not None and expected_start < start:
        boundary_times = [expected_start, *boundary_times]
    if expected_end is not None and expected_end > end:
        boundary_times = [*boundary_times, expected_end]

    if row_count == 1 and expected_start is None and expected_end is None:
        return WindowAssessment(
            label=label,
            start=start,
            end=end,
            row_count=row_count,
            median_timestep_hours=None,
            completeness_ratio=1.0,
            max_gap_days=0.0,
            acceptable=True,
        )

    diffs = pd.Series(boundary_times, dtype="datetime64[ns, UTC]").diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty and expected_start is None and expected_end is None:
        return WindowAssessment(
            label=label,
            start=start,
            end=end,
            row_count=row_count,
            median_timestep_hours=None,
            completeness_ratio=1.0,
            max_gap_days=0.0,
            acceptable=True,
        )

    if diffs.empty:
        median_step_seconds = float((effective_end - effective_start).total_seconds()) if effective_end > effective_start else 0.0
    else:
        median_step_seconds = float(diffs.median())
    if median_step_seconds <= 0:
        expected_count = row_count
    else:
        expected_count = max(row_count, int(((effective_end - effective_start).total_seconds() / median_step_seconds)) + 1)
    completeness_ratio = row_count / expected_count if expected_count > 0 else None
    max_gap_value = float(diffs.max() / 86_400.0) if not diffs.empty else 0.0
    return WindowAssessment(
        label=label,
        start=start,
        end=end,
        row_count=row_count,
        median_timestep_hours=median_step_seconds / 3600.0,
        completeness_ratio=completeness_ratio,
        max_gap_days=max_gap_value,
        acceptable=False,
    )


def _window_span_years(assessment: WindowAssessment) -> float | None:
    if assessment.start is None or assessment.end is None:
        return None
    return (assessment.end - assessment.start).total_seconds() / 86_400.0 / 365.25


def _materialize_station_series(
    frame: pd.DataFrame,
    *,
    row: pd.Series,
    provider: str,
    window_strategy: str,
) -> pd.DataFrame:
    station_key = str(row["station_key"])
    country = str(row["country"]).strip().upper()
    station_id = station_key.split(":", 1)[1] if ":" in station_key else str(row.get("source_station_id") or station_key)
    provider_station_id = str(row["resolved_site_number"]).strip()

    working = frame.copy()
    working["station_key"] = station_key
    working["station_id"] = station_id
    working["country"] = country
    working["variable"] = "discharge"
    working["source_function"] = provider
    working["provider"] = provider
    working["provider_station_id"] = provider_station_id
    working["window_strategy"] = window_strategy

    grouped = (
        working.groupby(
            ["station_key", "station_id", "country", "time", "variable", "source_function", "provider", "provider_station_id", "window_strategy"],
            as_index=False,
        )
        .agg(
            discharge=("discharge", "mean"),
            provider_series_id=("provider_series_id", "first"),
            provider_series_name=("provider_series_name", "first"),
            unit_of_measure=("unit_of_measure", "first"),
            raw_discharge=("raw_discharge", "mean"),
            raw_unit_of_measure=("raw_unit_of_measure", "first"),
            unit_normalized=("unit_normalized", "max"),
        )
        .sort_values(["station_key", "time"])
        .reset_index(drop=True)
    )
    return grouped[OUTPUT_TIMESERIES_COLUMNS].copy()


def _prepare_internal_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_internal_series()

    working = frame.copy()
    for column in INTERNAL_SERIES_COLUMNS:
        if column not in working.columns:
            if column == "unit_normalized":
                working[column] = False
            else:
                working[column] = pd.NA

    working["time"] = _parse_time_series(working["time"])
    working["discharge"] = pd.to_numeric(working["discharge"], errors="coerce")
    working["raw_discharge"] = pd.to_numeric(working["raw_discharge"], errors="coerce")
    working["provider_series_name"] = working["provider_series_name"].astype("string")
    working["provider_series_id"] = working["provider_series_id"].astype("string")
    working["unit_of_measure"] = working["unit_of_measure"].astype("string")
    working["raw_unit_of_measure"] = working["raw_unit_of_measure"].astype("string")
    working["unit_normalized"] = working["unit_normalized"].fillna(False).astype(bool)

    working = working.dropna(subset=["time", "discharge"]).copy()
    working = working.sort_values(["time", "provider_series_id"]).drop_duplicates(
        subset=["time", "provider_series_id", "discharge"],
        keep="first",
    )
    return working.reset_index(drop=True)[INTERNAL_SERIES_COLUMNS].copy()


def _empty_internal_series() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERNAL_SERIES_COLUMNS)


def _fetch_usgs_continuous_values_chunked(
    client: USGSWaterDataClient,
    *,
    time_series_id: str,
    start_datetime_utc: pd.Timestamp,
    end_datetime_utc: pd.Timestamp,
    chunk_days: int = USGS_CONTINUOUS_CHUNK_DAYS,
    min_chunk_days: int = USGS_CONTINUOUS_MIN_CHUNK_DAYS,
) -> pd.DataFrame:
    normalized_start = pd.Timestamp(start_datetime_utc).tz_convert("UTC")
    normalized_end = pd.Timestamp(end_datetime_utc).tz_convert("UTC")
    if normalized_end < normalized_start:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    current_start = normalized_start
    delta = pd.Timedelta(days=max(1, int(chunk_days))) - pd.Timedelta(seconds=1)
    while current_start <= normalized_end:
        current_end = min(current_start + delta, normalized_end)
        frame = _fetch_usgs_continuous_values_adaptive(
            client,
            time_series_id=time_series_id,
            start_datetime_utc=current_start,
            end_datetime_utc=current_end,
            min_chunk_days=min_chunk_days,
        )
        if not frame.empty:
            frames.append(frame)
        current_start = current_end + pd.Timedelta(seconds=1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_usgs_continuous_values_adaptive(
    client: USGSWaterDataClient,
    *,
    time_series_id: str,
    start_datetime_utc: pd.Timestamp,
    end_datetime_utc: pd.Timestamp,
    min_chunk_days: int,
) -> pd.DataFrame:
    normalized_start = pd.Timestamp(start_datetime_utc).tz_convert("UTC")
    normalized_end = pd.Timestamp(end_datetime_utc).tz_convert("UTC")
    try:
        return client.fetch_continuous_values(
            time_series_id=time_series_id,
            start_datetime_utc=normalized_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_datetime_utc=normalized_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    except Exception as exc:
        window_days = max(1.0, (normalized_end - normalized_start).total_seconds() / 86_400.0)
        if not _is_timeout_like_error(exc) or window_days <= max(1, int(min_chunk_days)):
            raise

        midpoint = normalized_start + (normalized_end - normalized_start) / 2
        left_end = midpoint.floor("s")
        right_start = left_end + pd.Timedelta(seconds=1)
        if right_start > normalized_end or left_end <= normalized_start:
            raise

        left = _fetch_usgs_continuous_values_adaptive(
            client,
            time_series_id=time_series_id,
            start_datetime_utc=normalized_start,
            end_datetime_utc=left_end,
            min_chunk_days=min_chunk_days,
        )
        right = _fetch_usgs_continuous_values_adaptive(
            client,
            time_series_id=time_series_id,
            start_datetime_utc=right_start,
            end_datetime_utc=normalized_end,
            min_chunk_days=min_chunk_days,
        )
        frames = [frame for frame in (left, right) if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _fetch_brazil_subdaily_values_chunked(
    client: BrazilAnaHydroClient,
    *,
    station_id: str,
    start_date: date,
    end_date: date,
    chunk_days: int = BRAZIL_TELEMETRIC_CHUNK_DAYS,
    min_chunk_days: int = BRAZIL_TELEMETRIC_MIN_CHUNK_DAYS,
) -> pd.DataFrame:
    if end_date < start_date:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    current_start = start_date
    chunk_span = max(1, int(chunk_days))
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_span - 1), end_date)
        chunk = _fetch_brazil_subdaily_values_adaptive(
            client,
            station_id=station_id,
            start_date=current_start,
            end_date=current_end,
            min_chunk_days=max(1, int(min_chunk_days)),
        )
        if not chunk.empty:
            frames.append(chunk)
        current_start = current_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "DateTime" in combined.columns:
        combined = combined.sort_values("DateTime").drop_duplicates(subset=["DateTime", "Vazao"], keep="first")
    return combined.reset_index(drop=True)


def _fetch_brazil_subdaily_values_adaptive(
    client: BrazilAnaHydroClient,
    *,
    station_id: str,
    start_date: date,
    end_date: date,
    min_chunk_days: int,
) -> pd.DataFrame:
    try:
        return client.fetch_subdaily_discharge_values(
            station_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        window_days = (end_date - start_date).days + 1
        if not _is_timeout_like_error(exc) or window_days <= max(1, int(min_chunk_days)):
            raise

        midpoint = start_date + timedelta(days=(window_days // 2) - 1)
        if midpoint < start_date:
            raise
        left = _fetch_brazil_subdaily_values_adaptive(
            client,
            station_id=station_id,
            start_date=start_date,
            end_date=midpoint,
            min_chunk_days=min_chunk_days,
        )
        right_start = midpoint + timedelta(days=1)
        right = _fetch_brazil_subdaily_values_adaptive(
            client,
            station_id=station_id,
            start_date=right_start,
            end_date=end_date,
            min_chunk_days=min_chunk_days,
        )
        frames = [frame for frame in (left, right) if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _build_manifest_base(
    row: pd.Series,
    *,
    runtime_now: datetime,
    target_start_date: date,
) -> dict[str, Any]:
    return {
        "station_key": str(row["station_key"]),
        "country": str(row["country"]).strip().upper(),
        "provider": str(row["provider"]).strip(),
        "provider_station_id": _nullable_str(row.get("resolved_site_number")),
        "resolved_station_name": _nullable_str(row.get("resolved_station_name")),
        "requested_start": f"{target_start_date.isoformat()}T00:00:00Z",
        "requested_end": runtime_now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _is_timeout_like_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    if not message:
        return False
    return any(token in message for token in ["timed out", "timeout", "read operation timed out"])


def _guess_time_column(frame: pd.DataFrame) -> str:
    lower_columns = {str(column).lower(): str(column) for column in frame.columns}
    candidates = [
        "date",
        "datetime",
        "date/time",
        "date_heure",
        "obs_time",
        "timestamp",
    ]
    for candidate in candidates:
        if candidate in lower_columns:
            return lower_columns[candidate]
    for column in frame.columns:
        name = str(column).lower()
        if "date" in name or "time" in name:
            return str(column)
    raise ValueError("Could not identify a timestamp column in the provider response.")


def _guess_value_column(frame: pd.DataFrame) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    candidates = [
        "value",
        "value/valeur",
        "valeur",
        "flow",
        "discharge",
    ]
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in frame.columns:
        name = str(column).lower()
        if any(token in name for token in ["value", "flow", "discharge", "valeur"]):
            return str(column)

    numeric_columns = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    excluded = {"id", "station", "station_id", "parameter"}
    for column in numeric_columns:
        if str(column).lower() not in excluded:
            return column
    raise ValueError("Could not identify a discharge column in the provider response.")


def _select_usgs_instantaneous_time_series_id(features: list[dict[str, Any]]) -> str | None:
    primary_candidate: str | None = None
    first_candidate: str | None = None
    for feature in features:
        properties = feature.get("properties") or {}
        computation_identifier = (_nullable_str(properties.get("computation_identifier")) or "").strip().lower()
        computation_period_identifier = (_nullable_str(properties.get("computation_period_identifier")) or "").strip().lower()
        is_instantaneous = computation_identifier == "instantaneous" or computation_period_identifier == "points"
        if not is_instantaneous:
            continue
        time_series_id = _nullable_str(properties.get("id"))
        if time_series_id is None:
            continue
        if first_candidate is None:
            first_candidate = time_series_id
        if (_nullable_str(properties.get("primary")) or "").strip().lower() == "primary":
            primary_candidate = time_series_id
            break
    return primary_candidate or first_candidate


def _extract_colombia_points(payload: dict[str, Any], series_name: str) -> list[dict[str, Any]]:
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
        timestamp = _parse_time_scalar(item[0])
        value = _to_float(item[1])
        if timestamp is None or value is None:
            continue
        points.append({"time": timestamp, "value": value})
    points.sort(key=lambda item: item["time"])
    return points


def _extract_mrc_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_points = payload.get("Points")
    if not isinstance(raw_points, list):
        return []

    points: list[dict[str, Any]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        timestamp = _parse_time_scalar(item.get("Timestamp"))
        value_container = item.get("Value")
        numeric_value = _to_float(value_container.get("Numeric")) if isinstance(value_container, dict) else None
        if timestamp is None or numeric_value is None:
            continue
        points.append({"time": timestamp, "value": numeric_value})
    points.sort(key=lambda item: item["time"])
    return points


def _convert_known_units(
    values: pd.Series,
    *,
    default_unit: str | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    normalized_default_unit = _normalize_unit_scalar(default_unit)
    raw_unit_text = (normalized_default_unit or "").strip().lower()
    if raw_unit_text in {"ft3/s", "ft^3/s", "cfs", "cubic feet per second"}:
        converted = values * 0.028316846592
        return (
            converted,
            pd.Series(["m3/s"] * len(values), index=values.index, dtype="string"),
            pd.Series([True] * len(values), index=values.index, dtype="boolean"),
            pd.Series([normalized_default_unit] * len(values), index=values.index, dtype="string"),
        )
    return (
        values,
        pd.Series([normalized_default_unit if normalized_default_unit else pd.NA] * len(values), index=values.index, dtype="string"),
        pd.Series([False] * len(values), index=values.index, dtype="boolean"),
        pd.Series([normalized_default_unit if normalized_default_unit else pd.NA] * len(values), index=values.index, dtype="string"),
    )


def _parse_time_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, (list, tuple)):
        series = pd.Series(list(values))
    else:
        series = pd.Series(values)
    return pd.to_datetime(series.apply(_parse_time_scalar), errors="coerce", utc=True)


def _parse_time_scalar(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _parse_epoch_millis(value: Any) -> pd.Timestamp | None:
    numeric_value = _to_float(value)
    if numeric_value is None:
        return None
    try:
        parsed = pd.to_datetime(int(round(numeric_value)), unit="ms", utc=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _timestamp_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if value.tzinfo else value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_non_null(values: Any, *, default: Any = None) -> Any:
    for value in _iter_scalar_candidates(values):
        if not _is_missing_value(value):
            return value
    return default


def _join_notes(*notes: str | None) -> str | None:
    parts = [str(note).strip() for note in notes if note is not None and str(note).strip()]
    return " ".join(parts) if parts else None


def _normalize_text(value: Any) -> str:
    text = _nullable_str(value)
    return text.strip().lower() if text is not None else ""


def _nullable_str(value: Any) -> str | None:
    scalar = _first_non_null(value)
    if _is_missing_value(scalar):
        return None
    text = str(scalar).strip()
    return text or None


def _normalize_station_code_like(value: Any) -> str | None:
    text = _nullable_str(value)
    if text is None:
        return None
    if text.endswith(".0"):
        candidate = text[:-2]
        if candidate.replace("-", "").isalnum():
            return candidate
    return text


def _normalize_provider_station_id(
    value: Any,
    *,
    provider: Any,
    inventory_station_key: Any = None,
) -> str | None:
    provider_name = _nullable_str(provider)
    inventory_key = _nullable_str(inventory_station_key)
    if inventory_key and ":" in inventory_key:
        _, inventory_station_id = inventory_key.split(":", 1)
        if provider_name in {"colombia_ideam_fews", "mrc_timeseries", "chile_dga"}:
            return inventory_station_id

    text = _normalize_station_code_like(value)
    if text is None:
        return None
    if provider_name == "colombia_ideam_fews" and text.isdigit():
        return text.zfill(10)
    if provider_name == "mrc_timeseries" and text.isdigit():
        return text.zfill(6)
    if provider_name == "chile_dga" and text.isdigit():
        return text.zfill(8)
    return text


def _normalize_unit_scalar(value: Any) -> str | None:
    return _nullable_str(value)


def _to_float(value: Any) -> float | None:
    scalar = _first_non_null(value)
    if _is_missing_value(scalar):
        return None
    try:
        numeric = float(scalar)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.Series):
        return value.empty or value.isna().all()
    if isinstance(value, (list, tuple, set, dict, pd.Index)):
        return False
    if not pd.api.types.is_scalar(value) and not isinstance(value, (str, bytes)):
        return False
    result = pd.isna(value)
    if isinstance(result, (list, tuple, pd.Series, pd.Index)):
        return False
    if not pd.api.types.is_scalar(result) and not isinstance(result, (str, bytes)):
        return False
    try:
        return bool(result)
    except Exception:
        return False


def _iter_scalar_candidates(value: Any):
    if value is None:
        return
    if isinstance(value, pd.Series):
        for item in value.tolist():
            yield from _iter_scalar_candidates(item)
        return
    if isinstance(value, pd.Index):
        for item in value.tolist():
            yield from _iter_scalar_candidates(item)
        return
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_scalar_candidates(item)
        return
    if not pd.api.types.is_scalar(value) and not isinstance(value, (str, bytes)):
        try:
            iterable = list(value)
        except TypeError:
            yield value
            return
        for item in iterable:
            yield from _iter_scalar_candidates(item)
        return
    yield value

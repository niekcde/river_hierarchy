from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file


@dataclass(frozen=True, slots=True)
class ForcingConfig:
    time_column: str = "time"
    discharge_column: str = "discharge"
    timezone: str = "UTC"
    station_key_column: str = "station_key"
    station_key: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    resample_minutes: float | None = None
    resample_seconds: int | None = None
    interpolation_method: str = "time"
    output_cache_dir: str | None = None
    spinup_hours: float | None = None
    auto_spinup_from_k: bool = False
    spinup_k_multiplier: float = 1.1
    pad_end_with_spinup: bool = True


def _read_raw_forcing_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read parquet forcing file {path}. Install the parquet "
                "dependencies for pandas in the active environment."
            ) from exc
    raise ValueError(f"Unsupported forcing file format: {path.suffix}")


def _sanitize_cache_token(value: str | None, *, fallback: str) -> str:
    text = fallback if value is None or not str(value).strip() else str(value).strip()
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)
    return sanitized.strip("_") or fallback


def _format_timestamp_token(value: str | None, *, fallback: str) -> str:
    if value is None or not str(value).strip():
        return fallback
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return fallback
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def _maybe_infer_dt_seconds(forcing: pd.DataFrame) -> int | None:
    if len(forcing) < 2:
        return None
    diffs = forcing["time"].diff().dropna().dt.total_seconds()
    if diffs.empty or (diffs <= 0).any():
        return None
    unique = sorted({int(round(value)) for value in diffs.tolist()})
    if len(unique) != 1:
        return None
    return int(unique[0])


def _resolve_resample_seconds(config: ForcingConfig) -> int | None:
    seconds = config.resample_seconds
    minutes = config.resample_minutes
    if seconds is not None and minutes is not None:
        raise ValueError("Provide only one of resample_seconds or resample_minutes.")
    if seconds is not None:
        resolved = int(seconds)
        if resolved <= 0:
            raise ValueError("resample_seconds must be a positive integer when provided.")
        return resolved
    if minutes is None:
        return None
    minutes_float = float(minutes)
    if minutes_float <= 0:
        raise ValueError("resample_minutes must be positive when provided.")
    resolved = int(round(minutes_float * 60.0))
    if resolved <= 0:
        raise ValueError("resample_minutes resolved to a non-positive timestep in seconds.")
    if not np.isclose(minutes_float * 60.0, resolved, atol=1e-9):
        raise ValueError("resample_minutes must resolve to a whole number of seconds.")
    return resolved


def build_forcing_cache_path(
    forcing_path: str | Path,
    *,
    config: ForcingConfig | None = None,
) -> tuple[Path, Path] | None:
    config = config or ForcingConfig()
    if config.output_cache_dir is None or not str(config.output_cache_dir).strip():
        return None

    source_path = Path(forcing_path).expanduser().resolve()
    cache_dir = Path(config.output_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    station_token = _sanitize_cache_token(config.station_key, fallback="single_series")
    start_token = _format_timestamp_token(config.start_time, fallback="start_open")
    end_token = _format_timestamp_token(config.end_time, fallback="end_open")
    resample_seconds = _resolve_resample_seconds(config)
    resample_token = (
        f"{int(resample_seconds)}s"
        if resample_seconds is not None
        else "native_dt"
    )
    spinup_token = (
        f"spinup_{int(round(float(config.spinup_hours) * 3600.0))}s"
        if config.spinup_hours is not None
        else "spinup_off"
    )
    tail_token = "tail_same_as_spinup" if config.pad_end_with_spinup else "tail_off"
    source_token = _sanitize_cache_token(source_path.stem, fallback="forcing")
    stem = "__".join((source_token, station_token, start_token, end_token, resample_token, spinup_token, tail_token))
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"


def _load_normalized_forcing_csv(path: Path) -> pd.DataFrame:
    forcing = pd.read_csv(path)
    required = {"time", "discharge_cms"}
    missing = sorted(required - set(forcing.columns))
    if missing:
        raise ValueError(f"Cached/normalized forcing file {path} is missing required columns: {', '.join(missing)}")
    forcing["time"] = pd.to_datetime(forcing["time"], utc=True, errors="coerce")
    forcing["discharge_cms"] = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
    forcing = forcing.dropna(subset=["time", "discharge_cms"]).sort_values("time").reset_index(drop=True)
    if forcing.empty:
        raise ValueError(f"Cached/normalized forcing file {path} did not yield any valid rows.")
    if forcing["time"].duplicated().any():
        raise ValueError(f"Cached/normalized forcing file {path} contains duplicate timestamps.")
    return forcing


def _parse_time_bound(value: str | None, *, field_name: str) -> pd.Timestamp | None:
    if value is None or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse {field_name}: {value!r}")
    return parsed


def _normalize_raw_forcing_frame(
    raw_frame: pd.DataFrame,
    *,
    forcing_path: Path,
    config: ForcingConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata: dict[str, object] = {
        "forcing_source_path": str(forcing_path),
        "station_key_column": config.station_key_column,
        "station_key_requested": config.station_key,
        "forcing_start_time": config.start_time,
        "forcing_end_time": config.end_time,
        "forcing_resample_minutes": float(config.resample_minutes) if config.resample_minutes is not None else None,
        "forcing_resample_seconds": _resolve_resample_seconds(config),
        "forcing_interpolation_method": config.interpolation_method,
        "forcing_spinup_hours": float(config.spinup_hours) if config.spinup_hours is not None else None,
        "raw_row_count": int(len(raw_frame)),
    }

    frame = raw_frame.copy()
    selected_station_key: str | None = None
    if config.station_key_column in frame.columns:
        station_values = frame[config.station_key_column].dropna().astype("string").str.strip()
        unique_station_keys = sorted(value for value in station_values.unique().tolist() if value)
        metadata["available_station_keys"] = unique_station_keys
        metadata["available_station_key_count"] = int(len(unique_station_keys))
        if config.station_key is not None:
            selected_station_key = str(config.station_key).strip()
            frame = frame.loc[
                frame[config.station_key_column].astype("string").str.strip().eq(selected_station_key)
            ].copy()
            if frame.empty:
                raise ValueError(
                    f"Forcing file {forcing_path} does not contain any rows for station_key={selected_station_key!r}."
                )
        elif len(unique_station_keys) > 1:
            raise ValueError(
                f"Forcing file {forcing_path} contains multiple station keys. "
                "Provide --forcing-station-key explicitly."
            )
        elif len(unique_station_keys) == 1:
            selected_station_key = unique_station_keys[0]
            frame = frame.loc[
                frame[config.station_key_column].astype("string").str.strip().eq(selected_station_key)
            ].copy()
    elif config.station_key is not None:
        raise ValueError(
            f"Forcing file {forcing_path} is missing the station key column "
            f"{config.station_key_column!r} required for station_key={config.station_key!r}."
        )

    metadata["selected_station_key"] = selected_station_key
    metadata["selected_row_count_before_parse"] = int(len(frame))

    missing = [column for column in (config.time_column, config.discharge_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Forcing file {forcing_path} is missing required columns: {', '.join(missing)}")

    forcing = frame[[config.time_column, config.discharge_column]].copy()
    forcing = forcing.rename(columns={config.time_column: "time", config.discharge_column: "discharge_cms"})
    forcing["time"] = pd.to_datetime(forcing["time"], utc=True, errors="coerce")
    forcing["discharge_cms"] = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
    forcing = forcing.dropna(subset=["time", "discharge_cms"]).sort_values("time").reset_index(drop=True)
    if forcing.empty:
        raise ValueError(f"Forcing file {forcing_path} did not yield any valid time/discharge rows.")
    if forcing["time"].duplicated().any():
        raise ValueError(f"Forcing file {forcing_path} contains duplicate timestamps after parsing.")

    metadata["parsed_row_count"] = int(len(forcing))
    metadata["raw_dt_seconds"] = _maybe_infer_dt_seconds(forcing)

    start_time = _parse_time_bound(config.start_time, field_name="forcing_start_time")
    end_time = _parse_time_bound(config.end_time, field_name="forcing_end_time")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("forcing_start_time must be earlier than forcing_end_time.")
    if start_time is not None:
        forcing = forcing.loc[forcing["time"] >= start_time].copy()
    if end_time is not None:
        forcing = forcing.loc[forcing["time"] <= end_time].copy()
    forcing = forcing.reset_index(drop=True)
    metadata["truncated_row_count"] = int(len(forcing))
    if len(forcing) < 2:
        raise ValueError("Forcing normalization requires at least two timesteps after truncation.")

    resample_seconds = _resolve_resample_seconds(config)
    if resample_seconds is not None:
        if not str(config.interpolation_method).strip():
            raise ValueError("interpolation_method must be a non-empty string.")
        forcing = (
            forcing.set_index("time")["discharge_cms"]
            .resample(f"{int(resample_seconds)}s")
            .mean()
            .interpolate(method=config.interpolation_method)
            .reset_index()
        )
        forcing["discharge_cms"] = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
        forcing = forcing.dropna(subset=["time", "discharge_cms"]).sort_values("time").reset_index(drop=True)
        if len(forcing) < 2:
            raise ValueError("Forcing normalization requires at least two timesteps after resampling.")

    forcing = forcing[["time", "discharge_cms"]].copy()
    metadata["normalized_row_count"] = int(len(forcing))
    metadata["normalized_dt_seconds"] = infer_forcing_dt_seconds(forcing)
    return forcing, metadata


def _apply_constant_spinup(
    forcing: pd.DataFrame,
    *,
    spinup_hours: float | None,
    pad_end_with_spinup: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if spinup_hours is None:
        return forcing, {
            "spinup_enabled": False,
            "spinup_hours": None,
            "spinup_duration_seconds": 0,
            "spinup_row_count": 0,
            "spinup_discharge_cms": None,
            "end_padding_enabled": False,
            "end_padding_row_count": 0,
            "end_padding_discharge_cms": None,
        }

    spinup_hours_float = float(spinup_hours)
    if spinup_hours_float <= 0:
        raise ValueError("spinup_hours must be positive when provided.")

    dt_seconds = infer_forcing_dt_seconds(forcing)
    spinup_duration_seconds = int(np.ceil(spinup_hours_float * 3600.0))
    spinup_row_count = int(np.ceil(spinup_duration_seconds / dt_seconds))
    if spinup_row_count <= 0:
        return forcing, {
            "spinup_enabled": False,
            "spinup_hours": spinup_hours_float,
            "spinup_duration_seconds": 0,
            "spinup_row_count": 0,
            "spinup_discharge_cms": None,
            "end_padding_enabled": False,
            "end_padding_row_count": 0,
            "end_padding_discharge_cms": None,
        }

    first_time = forcing["time"].iloc[0]
    first_discharge = float(forcing["discharge_cms"].iloc[0])
    spinup_offsets = np.arange(spinup_row_count, 0, -1, dtype=int)
    spinup_times = first_time - pd.to_timedelta(spinup_offsets * dt_seconds, unit="s")
    spinup = pd.DataFrame(
        {
            "time": spinup_times,
            "discharge_cms": np.full(spinup_row_count, first_discharge, dtype=float),
        }
    )
    expanded = pd.concat([spinup, forcing], ignore_index=True)
    end_padding_row_count = 0
    end_padding_discharge_cms: float | None = None
    if pad_end_with_spinup:
        last_time = forcing["time"].iloc[-1]
        last_discharge = float(forcing["discharge_cms"].iloc[-1])
        end_padding_offsets = np.arange(1, spinup_row_count + 1, dtype=int)
        end_padding_times = last_time + pd.to_timedelta(end_padding_offsets * dt_seconds, unit="s")
        end_padding = pd.DataFrame(
            {
                "time": end_padding_times,
                "discharge_cms": np.full(spinup_row_count, last_discharge, dtype=float),
            }
        )
        expanded = pd.concat([expanded, end_padding], ignore_index=True)
        end_padding_row_count = spinup_row_count
        end_padding_discharge_cms = last_discharge
    return expanded, {
        "spinup_enabled": True,
        "spinup_hours": spinup_hours_float,
        "spinup_duration_seconds": int(spinup_row_count * dt_seconds),
        "spinup_row_count": spinup_row_count,
        "spinup_discharge_cms": first_discharge,
        "end_padding_enabled": bool(pad_end_with_spinup),
        "end_padding_row_count": int(end_padding_row_count),
        "end_padding_discharge_cms": end_padding_discharge_cms,
    }


def prepare_forcing_table(
    forcing_path: str | Path,
    *,
    config: ForcingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = config or ForcingConfig()
    source_path = Path(forcing_path).expanduser().resolve()
    cache_paths = build_forcing_cache_path(source_path, config=config)
    if cache_paths is not None:
        cache_csv_path, cache_metadata_path = cache_paths
        if cache_csv_path.exists():
            forcing = _load_normalized_forcing_csv(cache_csv_path)
            metadata: dict[str, object] = {}
            if cache_metadata_path.exists():
                metadata = json.loads(cache_metadata_path.read_text())
            metadata.update(
                {
                    "forcing_source_path": str(source_path),
                    "forcing_cache_csv": str(cache_csv_path),
                    "forcing_cache_metadata_json": str(cache_metadata_path),
                    "forcing_loaded_from_cache": True,
                    "forcing_spinup_hours": float(config.spinup_hours) if config.spinup_hours is not None else None,
                    "forcing_pad_end_with_spinup": bool(config.pad_end_with_spinup),
                    "normalized_row_count": int(len(forcing)),
                    "normalized_dt_seconds": infer_forcing_dt_seconds(forcing),
                }
            )
            return forcing, metadata

    raw_frame = _read_raw_forcing_frame(source_path)
    forcing, metadata = _normalize_raw_forcing_frame(
        raw_frame,
        forcing_path=source_path,
        config=config,
    )
    forcing, spinup_metadata = _apply_constant_spinup(
        forcing,
        spinup_hours=config.spinup_hours,
        pad_end_with_spinup=bool(config.pad_end_with_spinup),
    )
    metadata.update(spinup_metadata)
    metadata["forcing_loaded_from_cache"] = False
    metadata["normalized_row_count"] = int(len(forcing))
    metadata["normalized_dt_seconds"] = infer_forcing_dt_seconds(forcing)
    if cache_paths is not None:
        cache_csv_path, cache_metadata_path = cache_paths
        forcing.to_csv(cache_csv_path, index=False)
        metadata["forcing_cache_csv"] = str(cache_csv_path)
        metadata["forcing_cache_metadata_json"] = str(cache_metadata_path)
        cache_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    else:
        metadata["forcing_cache_csv"] = None
        metadata["forcing_cache_metadata_json"] = None
    return forcing, metadata


def load_forcing_table(
    forcing_path: str | Path,
    *,
    config: ForcingConfig | None = None,
) -> pd.DataFrame:
    forcing, _metadata = prepare_forcing_table(forcing_path, config=config)
    return forcing


def infer_forcing_dt_seconds(forcing: pd.DataFrame) -> int:
    if len(forcing) < 2:
        raise ValueError("At least two forcing timesteps are required to infer dt.")
    diffs = forcing["time"].diff().dropna().dt.total_seconds().astype(int)
    if (diffs <= 0).any():
        raise ValueError("Forcing timestamps must be strictly increasing.")
    dt = int(diffs.iloc[0])
    if not (diffs == dt).all():
        sample = sorted(set(int(value) for value in diffs.tolist()))
        raise ValueError(f"Forcing timestamps are irregular; observed timestep set: {sample}")
    return dt


def compute_inlet_weights(prepared_links: pd.DataFrame) -> pd.DataFrame:
    if "is_inlet" not in prepared_links.columns:
        raise ValueError("Prepared link table is missing 'is_inlet'.")
    inlet = prepared_links.loc[prepared_links["is_inlet"].fillna(False)].copy()
    if inlet.empty:
        raise ValueError("No inlet links were identified for RAPID forcing preparation.")

    weight_basis = pd.to_numeric(inlet["rapid_width_m"], errors="coerce").fillna(0.0)
    if float(weight_basis.sum()) <= 0:
        inlet["inlet_weight"] = 1.0 / len(inlet)
    else:
        inlet["inlet_weight"] = weight_basis / float(weight_basis.sum())
    return inlet[["reach_id", "inlet_weight"]]


def write_inflow_netcdf(
    prepared_links: pd.DataFrame,
    forcing: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    forcing_dt_seconds = infer_forcing_dt_seconds(forcing)
    inlet_weights = compute_inlet_weights(prepared_links)
    weight_by_reach = dict(zip(inlet_weights["reach_id"], inlet_weights["inlet_weight"]))

    reach_ids = prepared_links["reach_id"].astype(int).to_numpy()
    lons = prepared_links["centroid_x"].astype(float).to_numpy()
    lats = prepared_links["centroid_y"].astype(float).to_numpy()
    discharges = forcing["discharge_cms"].astype(float).to_numpy()
    volumes = np.zeros((len(forcing), len(prepared_links)), dtype=np.float32)
    for idx, reach_id in enumerate(reach_ids):
        weight = float(weight_by_reach.get(int(reach_id), 0.0))
        if weight == 0.0:
            continue
        volumes[:, idx] = discharges * forcing_dt_seconds * weight

    times = forcing["time"]
    base_time = times.iloc[0]
    offsets = ((times - base_time).dt.total_seconds()).astype(np.int32).to_numpy()
    bounds = np.column_stack([offsets, offsets + forcing_dt_seconds]).astype(np.int32)

    with netcdf_file(path, "w") as ds:
        ds.createDimension("time", len(forcing))
        ds.createDimension("rivid", len(prepared_links))
        ds.createDimension("nv", 2)

        m3_var = ds.createVariable("m3_riv", "f4", ("time", "rivid"))
        rivid_var = ds.createVariable("rivid", "i4", ("rivid",))
        time_var = ds.createVariable("time", "i4", ("time",))
        time_bnds_var = ds.createVariable("time_bnds", "i4", ("time", "nv"))
        lon_var = ds.createVariable("lon", "f8", ("rivid",))
        lat_var = ds.createVariable("lat", "f8", ("rivid",))
        ds.createVariable("crs", "i4", ())

        rivid_var[:] = reach_ids.astype(np.int32)
        time_var[:] = offsets
        time_bnds_var[:, :] = bounds
        lon_var[:] = lons
        lat_var[:] = lats
        m3_var[:, :] = volumes

        time_units = base_time.strftime("seconds since %Y-%m-%d %H:%M:%S +00:00")
        time_var.units = time_units.encode("utf-8")
        time_var.calendar = b"gregorian"
        time_var.bounds = b"time_bnds"
        ds.Conventions = b"CF-1.6"
        ds.featureType = b"timeSeries"
        ds.history = f"Created by RAPID forcing prep from {base_time.isoformat()}".encode("utf-8")

    return path

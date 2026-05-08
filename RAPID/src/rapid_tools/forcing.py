from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file


@dataclass(frozen=True, slots=True)
class ForcingConfig:
    time_column: str = "time"
    discharge_column: str = "discharge"
    timezone: str = "UTC"


def load_forcing_table(
    forcing_path: str | Path,
    *,
    config: ForcingConfig | None = None,
) -> pd.DataFrame:
    config = config or ForcingConfig()
    path = Path(forcing_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read parquet forcing file {path}. Install the parquet "
                "dependencies for pandas in the active environment."
            ) from exc
    else:
        raise ValueError(f"Unsupported forcing file format: {path.suffix}")

    missing = [c for c in (config.time_column, config.discharge_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"Forcing file {path} is missing required columns: {', '.join(missing)}")

    forcing = frame[[config.time_column, config.discharge_column]].copy()
    forcing = forcing.rename(columns={config.time_column: "time", config.discharge_column: "discharge_cms"})
    forcing["time"] = pd.to_datetime(forcing["time"], utc=True, errors="coerce")
    forcing["discharge_cms"] = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
    forcing = forcing.dropna(subset=["time", "discharge_cms"]).sort_values("time").reset_index(drop=True)
    if forcing.empty:
        raise ValueError(f"Forcing file {path} did not yield any valid time/discharge rows.")
    if forcing["time"].duplicated().any():
        raise ValueError(f"Forcing file {path} contains duplicate timestamps after parsing.")
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

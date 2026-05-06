from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file


@dataclass(frozen=True, slots=True)
class HydrographMetricConfig:
    event_start_time: str | None = None
    event_start_window_hours: float | None = None
    event_start_buffer_hours: float | None = None
    event_end_time: str | None = None
    event_end_buffer_hours: float | None = None


def _load_qout(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as ds:
        river_ids = np.array(ds.variables["rivid"].data, dtype=np.int64).copy()
        time_seconds = np.array(ds.variables["time"].data, dtype=np.float64).copy()
        qout = np.array(ds.variables["Qout"].data, dtype=np.float64).copy()
    return river_ids, time_seconds, qout


def _load_prep_links(prep_dir: Path) -> pd.DataFrame:
    path = prep_dir / "rapid_link_attributes.csv"
    if not path.exists():
        raise FileNotFoundError(f"RAPID link attributes were not found: {path}")
    return pd.read_csv(path)


def _load_forcing(prep_dir: Path) -> pd.DataFrame:
    path = prep_dir / "forcing_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Normalized forcing table was not found: {path}")
    forcing = pd.read_csv(path)
    if "time" not in forcing.columns or "discharge_cms" not in forcing.columns:
        raise ValueError(f"Normalized forcing table {path} is missing required columns.")
    forcing["time"] = pd.to_datetime(forcing["time"], utc=True, errors="coerce")
    forcing["discharge_cms"] = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
    forcing = forcing.dropna(subset=["time", "discharge_cms"]).reset_index(drop=True)
    if forcing.empty:
        raise ValueError(f"Normalized forcing table {path} is empty after parsing.")
    forcing["time_seconds"] = (forcing["time"] - forcing["time"].iloc[0]).dt.total_seconds().astype(float)
    return forcing


def _resolve_outlet_reach_ids(prepared_links: pd.DataFrame) -> list[int]:
    if "is_outlet" not in prepared_links.columns:
        raise ValueError("RAPID link attributes are missing 'is_outlet'.")
    outlet = prepared_links.loc[prepared_links["is_outlet"].fillna(False)].copy()
    if outlet.empty:
        raise ValueError("No outlet reaches were found in RAPID link attributes.")
    reach_ids = pd.to_numeric(outlet["reach_id"], errors="coerce").dropna().astype(int).tolist()
    if not reach_ids:
        raise ValueError("Outlet reaches could not be converted to integer reach IDs.")
    return sorted(set(reach_ids))


def _first_index_at_or_after(times: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(times, value, side="left"))
    if index >= len(times):
        raise ValueError("Requested event start time lies beyond the hydrograph duration.")
    return index


def _resolve_reference_time_window_min(
    forcing: pd.DataFrame,
    reference_time: str,
    *,
    buffer_hours: float | None,
) -> tuple[int, str]:
    reference_timestamp = pd.to_datetime(reference_time, utc=True, errors="coerce")
    if pd.isna(reference_timestamp):
        raise ValueError(f"Could not parse hydrograph reference time: {reference_time!r}")

    if buffer_hours is None:
        event_index = _first_index_at_or_after(
            forcing["time_seconds"].to_numpy(dtype=float),
            float((reference_timestamp - forcing["time"].iloc[0]).total_seconds()),
        )
        return event_index, "manual_input_time"

    if buffer_hours <= 0:
        raise ValueError("buffer_hours must be positive when provided.")

    buffer_delta = pd.to_timedelta(float(buffer_hours), unit="h")
    window = forcing.loc[
        forcing["time"].between(reference_timestamp - buffer_delta, reference_timestamp + buffer_delta)
    ].copy()
    if window.empty:
        raise ValueError(
            "The hydrograph reference-time search window did not contain any forcing timesteps."
        )
    min_discharge = float(window["discharge_cms"].min())
    event_index = int(window.loc[window["discharge_cms"].eq(min_discharge)].index.min())
    return event_index, "manual_input_min_window"


def _resolve_event_start(
    forcing: pd.DataFrame,
    config: HydrographMetricConfig,
) -> tuple[int, str]:
    if config.event_start_window_hours is not None and config.event_start_window_hours <= 0:
        raise ValueError("event_start_window_hours must be positive when provided.")

    if config.event_start_time:
        return _resolve_reference_time_window_min(
            forcing,
            config.event_start_time,
            buffer_hours=config.event_start_buffer_hours,
        )

    inflow = forcing["discharge_cms"].to_numpy(dtype=float)
    inflow_peak_index = int(np.argmax(inflow))
    search = forcing.iloc[: inflow_peak_index + 1].copy()
    if config.event_start_window_hours is not None:
        cutoff = float(config.event_start_window_hours) * 3600.0
        search = search.loc[search["time_seconds"] <= cutoff]
        if search.empty:
            raise ValueError(
                "The automatic event-start window did not include any forcing timesteps."
            )
        source = "auto_input_min_window"
    else:
        source = "auto_input_min_prepeak"

    min_discharge = float(search["discharge_cms"].min())
    event_index = int(search.loc[search["discharge_cms"].eq(min_discharge)].index.min())
    return event_index, source


def _resolve_event_end(
    forcing: pd.DataFrame,
    config: HydrographMetricConfig,
) -> tuple[int | None, str]:
    if not config.event_end_time:
        return None, ""
    return _resolve_reference_time_window_min(
        forcing,
        config.event_end_time,
        buffer_hours=config.event_end_buffer_hours,
    )


def _interpolated_falling_crossing_time(
    time_seconds: np.ndarray,
    discharge: np.ndarray,
    *,
    peak_index: int,
    threshold: float,
) -> float:
    for idx in range(peak_index + 1, len(discharge)):
        q_prev = float(discharge[idx - 1])
        q_curr = float(discharge[idx])
        if q_curr > threshold:
            continue
        t_prev = float(time_seconds[idx - 1])
        t_curr = float(time_seconds[idx])
        if q_prev <= threshold or np.isclose(q_prev, q_curr):
            return t_curr
        fraction = (q_prev - threshold) / (q_prev - q_curr)
        return t_prev + fraction * (t_curr - t_prev)
    return float("nan")


def _maybe_time_iso(base_times: pd.Series, seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    timestamp = base_times.iloc[0] + pd.to_timedelta(seconds, unit="s")
    return timestamp.isoformat()


def summarize_outlet_hydrograph(
    prep_dir: str | Path,
    qout_path: str | Path,
    *,
    config: HydrographMetricConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = config or HydrographMetricConfig()
    prep_path = Path(prep_dir).expanduser().resolve()
    prepared_links = _load_prep_links(prep_path)
    forcing = _load_forcing(prep_path)
    river_ids, time_seconds, qout = _load_qout(qout_path)

    if qout.shape[0] != len(time_seconds):
        raise ValueError("Qout time dimension does not match the time vector length.")
    if len(forcing) != len(time_seconds):
        raise ValueError(
            "Normalized forcing length does not match the RAPID Qout time dimension."
        )
    if not np.allclose(
        forcing["time_seconds"].to_numpy(dtype=float),
        time_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Normalized forcing timestamps do not align with the RAPID Qout time vector."
        )

    outlet_reach_ids = _resolve_outlet_reach_ids(prepared_links)
    index_by_reach = {int(reach_id): idx for idx, reach_id in enumerate(river_ids)}
    missing = [reach_id for reach_id in outlet_reach_ids if reach_id not in index_by_reach]
    if missing:
        raise ValueError(
            "Could not find all outlet reach IDs in the RAPID Qout file. "
            f"Missing reach IDs: {missing}"
        )
    outlet_indices = [index_by_reach[reach_id] for reach_id in outlet_reach_ids]
    outlet_q = qout[:, outlet_indices].sum(axis=1).astype(float)

    outlet_hydrograph = pd.DataFrame(
        {
            "time_seconds": time_seconds.astype(float),
            "time_utc": forcing["time"],
            "q_outlet_cms": outlet_q,
            "q_inflow_cms": forcing["discharge_cms"].astype(float),
        }
    )

    event_start_index, event_start_source = _resolve_event_start(forcing, config)
    event_end_index, event_end_source = _resolve_event_end(forcing, config)
    baseline_outlet = float(outlet_q[event_start_index])
    baseline_input = float(forcing.loc[event_start_index, "discharge_cms"])
    recession_baseline_outlet = (
        float(outlet_q[event_end_index]) if event_end_index is not None else baseline_outlet
    )
    recession_baseline_input = (
        float(forcing.loc[event_end_index, "discharge_cms"])
        if event_end_index is not None
        else baseline_input
    )

    peak_index_rel = int(np.argmax(outlet_q[event_start_index:]))
    peak_index = event_start_index + peak_index_rel
    peak_time_seconds = float(time_seconds[peak_index])
    peak_discharge = float(outlet_q[peak_index])
    peak_excess = peak_discharge - baseline_outlet
    peak_excess_recession = peak_discharge - recession_baseline_outlet

    inflow = forcing["discharge_cms"].to_numpy(dtype=float)
    inflow_peak_index = int(np.argmax(inflow))
    inflow_peak_time_seconds = float(time_seconds[inflow_peak_index])
    inflow_peak_discharge = float(inflow[inflow_peak_index])

    thresholds = {
        "fall_time_seconds": recession_baseline_outlet,
        "fall_time_50_seconds": (
            recession_baseline_outlet + 0.5 * peak_excess_recession
            if peak_excess_recession > 0
            else float("nan")
        ),
        "fall_time_10_seconds": (
            recession_baseline_outlet + 0.1 * peak_excess_recession
            if peak_excess_recession > 0
            else float("nan")
        ),
        "e_folding_time_seconds": (
            recession_baseline_outlet + (peak_excess_recession / np.e)
            if peak_excess_recession > 0
            else float("nan")
        ),
    }
    falling_times = {
        name: _interpolated_falling_crossing_time(
            time_seconds,
            outlet_q,
            peak_index=peak_index,
            threshold=value,
        )
        for name, value in thresholds.items()
        if not pd.isna(value)
    }
    for threshold_name, threshold_value in thresholds.items():
        if threshold_name not in falling_times:
            falling_times[threshold_name] = float("nan")

    time_to_peak_seconds = peak_time_seconds - float(time_seconds[event_start_index])
    metrics = {
        "event_start_source": event_start_source,
        "event_start_time_seconds": float(time_seconds[event_start_index]),
        "event_start_time_utc": forcing.loc[event_start_index, "time"].isoformat(),
        "event_start_input_discharge_cms": baseline_input,
        "event_start_outlet_discharge_cms": baseline_outlet,
        "event_end_source": event_end_source,
        "event_end_time_seconds": (
            float(time_seconds[event_end_index]) if event_end_index is not None else float("nan")
        ),
        "event_end_time_utc": (
            forcing.loc[event_end_index, "time"].isoformat() if event_end_index is not None else ""
        ),
        "event_end_input_discharge_cms": recession_baseline_input,
        "event_end_outlet_discharge_cms": recession_baseline_outlet,
        "event_duration_seconds": (
            float(time_seconds[event_end_index] - time_seconds[event_start_index])
            if event_end_index is not None
            else float("nan")
        ),
        "peak_time_seconds": peak_time_seconds,
        "peak_time_utc": _maybe_time_iso(forcing["time"], peak_time_seconds),
        "peak_discharge_cms": peak_discharge,
        "peak_height_cms": peak_discharge,
        "peak_excess_cms": peak_excess,
        "peak_excess_to_end_baseline_cms": peak_excess_recession,
        "time_to_peak_seconds": time_to_peak_seconds,
        "inflow_peak_time_seconds": inflow_peak_time_seconds,
        "inflow_peak_time_utc": _maybe_time_iso(forcing["time"], inflow_peak_time_seconds),
        "inflow_peak_discharge_cms": inflow_peak_discharge,
        "lag_to_inflow_peak_seconds": peak_time_seconds - inflow_peak_time_seconds,
        "peak_attenuation_ratio": (
            peak_discharge / inflow_peak_discharge if inflow_peak_discharge > 0 else float("nan")
        ),
        "outlet_volume_m3": float(np.trapz(outlet_q, time_seconds)),
        "outlet_excess_volume_m3": float(
            np.trapz(np.maximum(outlet_q - baseline_outlet, 0.0), time_seconds)
        ),
        "outlet_reach_count": len(outlet_reach_ids),
        "outlet_reach_ids": json.dumps(outlet_reach_ids),
        "hydrograph_duration_seconds": float(time_seconds[-1] - time_seconds[0]),
        "metric_config": json.dumps(asdict(config)),
    }
    metrics.update(falling_times)
    metrics["fall_time_seconds"] = (
        metrics["fall_time_seconds"] - peak_time_seconds
        if not pd.isna(metrics["fall_time_seconds"])
        else float("nan")
    )
    metrics["fall_time_50_seconds"] = (
        metrics["fall_time_50_seconds"] - peak_time_seconds
        if not pd.isna(metrics["fall_time_50_seconds"])
        else float("nan")
    )
    metrics["fall_time_10_seconds"] = (
        metrics["fall_time_10_seconds"] - peak_time_seconds
        if not pd.isna(metrics["fall_time_10_seconds"])
        else float("nan")
    )
    metrics["e_folding_time_seconds"] = (
        metrics["e_folding_time_seconds"] - peak_time_seconds
        if not pd.isna(metrics["e_folding_time_seconds"])
        else float("nan")
    )
    metrics["rise_rate_cms_per_hour"] = (
        peak_excess / (time_to_peak_seconds / 3600.0)
        if time_to_peak_seconds > 0
        else float("nan")
    )

    return outlet_hydrograph, metrics


def write_hydrograph_outputs(
    prep_dir: str | Path,
    qout_path: str | Path,
    output_dir: str | Path,
    *,
    config: HydrographMetricConfig | None = None,
) -> dict[str, object]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    outlet_hydrograph, metrics = summarize_outlet_hydrograph(
        prep_dir,
        qout_path,
        config=config,
    )

    hydrograph_csv = output_path / "outlet_hydrograph.csv"
    outlet_hydrograph.to_csv(hydrograph_csv, index=False)

    metrics_frame = pd.DataFrame([metrics])
    metrics_csv = output_path / "hydrograph_metrics.csv"
    metrics_frame.to_csv(metrics_csv, index=False)

    metrics_json = output_path / "hydrograph_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2, default=str))

    result = {
        "hydrograph_status": "computed",
        "outlet_hydrograph_csv": str(hydrograph_csv),
        "hydrograph_metrics_csv": str(metrics_csv),
        "hydrograph_metrics_json": str(metrics_json),
    }
    result.update(metrics)
    return result

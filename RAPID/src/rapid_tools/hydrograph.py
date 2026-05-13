from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class HydrographMetricConfig:
    start_mode: str = "auto_local_min"
    end_mode: str = "auto_local_min"
    manual_start_time: str | None = None
    max_start_search_window_hours: float | None = None
    manual_start_buffer_hours: float | None = None
    manual_end_time: str | None = None
    manual_end_buffer_hours: float | None = None
    smoothing_window_steps: int = 1
    min_peak_prominence_cms: float = 0.0
    min_trough_prominence_cms: float = 0.0
    min_separation_steps: int = 1
    end_fallback_mode: str = "series_end"
    event_definition_version: str = "v2_local_min"
    # Legacy aliases kept for backwards compatibility with older callers.
    event_start_time: str | None = None
    event_start_window_hours: float | None = None
    event_start_buffer_hours: float | None = None
    event_end_time: str | None = None
    event_end_buffer_hours: float | None = None


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


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


def _normalize_metric_config(config: HydrographMetricConfig) -> dict[str, object]:
    manual_start_time = config.manual_start_time or config.event_start_time
    manual_end_time = config.manual_end_time or config.event_end_time
    manual_start_buffer_hours = (
        config.manual_start_buffer_hours
        if config.manual_start_buffer_hours is not None
        else config.event_start_buffer_hours
    )
    manual_end_buffer_hours = (
        config.manual_end_buffer_hours
        if config.manual_end_buffer_hours is not None
        else config.event_end_buffer_hours
    )
    max_start_search_window_hours = (
        config.max_start_search_window_hours
        if config.max_start_search_window_hours is not None
        else config.event_start_window_hours
    )

    start_mode = config.start_mode
    end_mode = config.end_mode
    if manual_start_time is not None:
        start_mode = "manual"
    if manual_end_time is not None:
        end_mode = "manual"

    allowed_start_modes = {"auto_local_min", "manual"}
    allowed_end_modes = {"auto_local_min", "manual", "series_end"}
    if start_mode not in allowed_start_modes:
        raise ValueError(f"Unsupported start_mode {start_mode!r}.")
    if end_mode not in allowed_end_modes:
        raise ValueError(f"Unsupported end_mode {end_mode!r}.")
    if config.end_fallback_mode not in {"series_end", "error"}:
        raise ValueError("end_fallback_mode must be one of: series_end, error")
    if max_start_search_window_hours is not None and float(max_start_search_window_hours) <= 0:
        raise ValueError("max_start_search_window_hours must be positive when provided.")
    if manual_start_buffer_hours is not None and float(manual_start_buffer_hours) <= 0:
        raise ValueError("manual_start_buffer_hours must be positive when provided.")
    if manual_end_buffer_hours is not None and float(manual_end_buffer_hours) <= 0:
        raise ValueError("manual_end_buffer_hours must be positive when provided.")
    if int(config.smoothing_window_steps) <= 0:
        raise ValueError("smoothing_window_steps must be a positive integer.")
    if int(config.min_separation_steps) <= 0:
        raise ValueError("min_separation_steps must be a positive integer.")
    if float(config.min_peak_prominence_cms) < 0:
        raise ValueError("min_peak_prominence_cms cannot be negative.")
    if float(config.min_trough_prominence_cms) < 0:
        raise ValueError("min_trough_prominence_cms cannot be negative.")

    return {
        "start_mode": start_mode,
        "end_mode": end_mode,
        "manual_start_time": manual_start_time,
        "max_start_search_window_hours": (
            float(max_start_search_window_hours) if max_start_search_window_hours is not None else None
        ),
        "manual_start_buffer_hours": (
            float(manual_start_buffer_hours) if manual_start_buffer_hours is not None else None
        ),
        "manual_end_time": manual_end_time,
        "manual_end_buffer_hours": (
            float(manual_end_buffer_hours) if manual_end_buffer_hours is not None else None
        ),
        "smoothing_window_steps": int(config.smoothing_window_steps),
        "min_peak_prominence_cms": float(config.min_peak_prominence_cms),
        "min_trough_prominence_cms": float(config.min_trough_prominence_cms),
        "min_separation_steps": int(config.min_separation_steps),
        "end_fallback_mode": config.end_fallback_mode,
        "event_definition_version": config.event_definition_version,
    }


def _first_index_at_or_after(times: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(times, value, side="left"))
    if index >= len(times):
        raise ValueError("Requested hydrograph reference time lies beyond the hydrograph duration.")
    return index


def _resolve_reference_time_window_min(
    frame: pd.DataFrame,
    reference_time: str,
    *,
    value_column: str,
    buffer_hours: float | None,
    exact_source: str,
    window_source: str,
    min_index: int | None = None,
) -> tuple[int, str]:
    reference_timestamp = pd.to_datetime(reference_time, utc=True, errors="coerce")
    if pd.isna(reference_timestamp):
        raise ValueError(f"Could not parse hydrograph reference time: {reference_time!r}")

    if buffer_hours is None:
        event_index = _first_index_at_or_after(
            frame["time_seconds"].to_numpy(dtype=float),
            float((reference_timestamp - frame["time"].iloc[0]).total_seconds()),
        )
        if min_index is not None and event_index < min_index:
            raise ValueError("The selected hydrograph reference time lies before the required search bound.")
        return event_index, exact_source

    buffer_delta = pd.to_timedelta(float(buffer_hours), unit="h")
    window = frame.loc[frame["time"].between(reference_timestamp - buffer_delta, reference_timestamp + buffer_delta)]
    if min_index is not None:
        window = window.loc[window.index >= min_index]
    if window.empty:
        raise ValueError("The hydrograph reference-time search window did not contain any valid timesteps.")
    min_discharge = float(window[value_column].min())
    event_index = int(window.loc[window[value_column].eq(min_discharge)].index.min())
    return event_index, window_source


def _smooth_series(values: np.ndarray, window_steps: int) -> np.ndarray:
    if window_steps <= 1:
        return values.astype(float, copy=True)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window_steps, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _find_dominant_peak_index(
    values: np.ndarray,
    *,
    start_index: int = 0,
    prominence: float,
    min_separation_steps: int,
) -> int:
    search = np.asarray(values[start_index:], dtype=float)
    if search.size == 0:
        raise ValueError("Cannot identify a peak in an empty hydrograph segment.")
    fallback_local_index = int(np.argmax(search))
    peaks, _ = find_peaks(
        search,
        prominence=prominence,
        distance=max(min_separation_steps, 1),
    )
    if peaks.size > 0:
        best_local_index = int(peaks[np.argmax(search[peaks])])
        if search[best_local_index] >= search[fallback_local_index]:
            return start_index + best_local_index
    return start_index + fallback_local_index


def _find_local_minima(
    values: np.ndarray,
    *,
    start_index: int,
    end_index: int,
    prominence: float,
    min_separation_steps: int,
) -> np.ndarray:
    if end_index - start_index < 2:
        return np.array([], dtype=int)
    search = np.asarray(values[start_index : end_index + 1], dtype=float)
    candidates: list[int] = []
    for local_index in range(1, len(search) - 1):
        prev_value = float(search[local_index - 1])
        curr_value = float(search[local_index])
        next_value = float(search[local_index + 1])
        if curr_value <= prev_value and curr_value <= next_value and (curr_value < prev_value or curr_value < next_value):
            if prominence > 0:
                left_peak = float(np.max(search[: local_index + 1]))
                right_peak = float(np.max(search[local_index:]))
                trough_prominence = min(left_peak - curr_value, right_peak - curr_value)
                if trough_prominence < prominence:
                    continue
            candidate_index = start_index + local_index
            if candidates and candidate_index - candidates[-1] < max(min_separation_steps, 1):
                if values[candidate_index] <= values[candidates[-1]]:
                    candidates[-1] = candidate_index
                continue
            candidates.append(candidate_index)
    return np.asarray(candidates, dtype=int)


def _resolve_event_start(
    forcing: pd.DataFrame,
    inflow_smoothed: np.ndarray,
    normalized_config: dict[str, object],
) -> tuple[int, str, int]:
    inflow_peak_index = _find_dominant_peak_index(
        inflow_smoothed,
        start_index=0,
        prominence=float(normalized_config["min_peak_prominence_cms"]),
        min_separation_steps=int(normalized_config["min_separation_steps"]),
    )
    if normalized_config["start_mode"] == "manual":
        event_index, source = _resolve_reference_time_window_min(
            forcing,
            str(normalized_config["manual_start_time"]),
            value_column="discharge_cms",
            buffer_hours=normalized_config["manual_start_buffer_hours"],
            exact_source="manual_input_time",
            window_source="manual_input_min_window",
        )
        if event_index > inflow_peak_index:
            raise ValueError("Manual event start must lie at or before the forcing peak.")
        return event_index, source, inflow_peak_index

    search_end_index = inflow_peak_index
    if normalized_config["max_start_search_window_hours"] is not None:
        cutoff_seconds = float(normalized_config["max_start_search_window_hours"]) * 3600.0
        allowed = np.flatnonzero(forcing["time_seconds"].to_numpy(dtype=float) <= cutoff_seconds)
        if allowed.size == 0:
            raise ValueError("The automatic event-start window did not include any forcing timesteps.")
        search_end_index = min(search_end_index, int(allowed[-1]))

    minima = _find_local_minima(
        inflow_smoothed,
        start_index=0,
        end_index=search_end_index,
        prominence=float(normalized_config["min_trough_prominence_cms"]),
        min_separation_steps=int(normalized_config["min_separation_steps"]),
    )
    if minima.size > 0:
        source = (
            "auto_input_local_min_window"
            if normalized_config["max_start_search_window_hours"] is not None
            else "auto_input_local_min_prepeak"
        )
        return int(minima[-1]), source, inflow_peak_index

    fallback_index = int(
        np.argmin(forcing["discharge_cms"].to_numpy(dtype=float)[: search_end_index + 1])
    )
    source = (
        "auto_input_global_min_window_fallback"
        if normalized_config["max_start_search_window_hours"] is not None
        else "auto_input_global_min_prepeak_fallback"
    )
    return fallback_index, source, inflow_peak_index


def _resolve_event_end(
    forcing: pd.DataFrame,
    outlet_hydrograph: pd.DataFrame,
    outlet_smoothed: np.ndarray,
    normalized_config: dict[str, object],
    *,
    outlet_peak_index: int,
) -> tuple[int, str, bool]:
    if normalized_config["end_mode"] == "manual":
        event_index, source = _resolve_reference_time_window_min(
            forcing,
            str(normalized_config["manual_end_time"]),
            value_column="discharge_cms",
            buffer_hours=normalized_config["manual_end_buffer_hours"],
            exact_source="manual_input_time",
            window_source="manual_input_min_window",
        )
        return event_index, source, False

    if normalized_config["end_mode"] == "series_end":
        return len(outlet_hydrograph) - 1, "series_end_requested", True

    minima = _find_local_minima(
        outlet_smoothed,
        start_index=outlet_peak_index,
        end_index=len(outlet_smoothed) - 1,
        prominence=float(normalized_config["min_trough_prominence_cms"]),
        min_separation_steps=int(normalized_config["min_separation_steps"]),
    )
    valid = minima[minima > outlet_peak_index]
    if valid.size > 0:
        return int(valid[0]), "auto_outlet_local_min_postpeak", False

    if normalized_config["end_fallback_mode"] == "series_end":
        return len(outlet_hydrograph) - 1, "auto_outlet_series_end_fallback", True

    raise ValueError("Could not identify a post-peak outlet minimum and end_fallback_mode='error'.")


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
    normalized_config = _normalize_metric_config(config)

    prep_path = Path(prep_dir).expanduser().resolve()
    prepared_links = _load_prep_links(prep_path)
    forcing = _load_forcing(prep_path)
    river_ids, time_seconds, qout = _load_qout(qout_path)

    if qout.shape[0] != len(time_seconds):
        raise ValueError("Qout time dimension does not match the time vector length.")
    if len(forcing) != len(time_seconds):
        raise ValueError("Normalized forcing length does not match the RAPID Qout time dimension.")
    if not np.allclose(
        forcing["time_seconds"].to_numpy(dtype=float),
        time_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Normalized forcing timestamps do not align with the RAPID Qout time vector.")

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

    inflow = forcing["discharge_cms"].to_numpy(dtype=float)
    inflow_smoothed = _smooth_series(inflow, int(normalized_config["smoothing_window_steps"]))
    outlet_smoothed = _smooth_series(outlet_q, int(normalized_config["smoothing_window_steps"]))

    event_start_index, event_start_source, inflow_peak_index = _resolve_event_start(
        forcing,
        inflow_smoothed,
        normalized_config,
    )
    outlet_peak_index = _find_dominant_peak_index(
        outlet_smoothed,
        start_index=event_start_index,
        prominence=float(normalized_config["min_peak_prominence_cms"]),
        min_separation_steps=int(normalized_config["min_separation_steps"]),
    )
    event_end_index, event_end_source, event_end_censored = _resolve_event_end(
        forcing,
        outlet_hydrograph,
        outlet_smoothed,
        normalized_config,
        outlet_peak_index=outlet_peak_index,
    )

    if event_end_index < event_start_index:
        raise ValueError("Event end was resolved before event start.")
    if normalized_config["end_mode"] != "manual" and event_end_index < outlet_peak_index:
        raise ValueError("Event end was resolved before the outlet peak.")

    baseline_outlet = float(outlet_q[event_start_index])
    baseline_input = float(inflow[event_start_index])
    recession_baseline_outlet = float(outlet_q[event_end_index])
    recession_baseline_input = float(inflow[event_end_index])

    peak_time_seconds = float(time_seconds[outlet_peak_index])
    peak_discharge = float(outlet_q[outlet_peak_index])
    peak_excess = peak_discharge - baseline_outlet
    peak_excess_recession = peak_discharge - recession_baseline_outlet

    inflow_peak_time_seconds = float(time_seconds[inflow_peak_index])
    inflow_peak_discharge = float(inflow[inflow_peak_index])

    thresholds = {
        "fall_time_seconds": (
            recession_baseline_outlet if peak_excess_recession > 0 else float("nan")
        ),
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
            peak_index=outlet_peak_index,
            threshold=value,
        )
        for name, value in thresholds.items()
        if not pd.isna(value)
    }
    for threshold_name in thresholds:
        if threshold_name not in falling_times:
            falling_times[threshold_name] = float("nan")

    time_to_peak_seconds = peak_time_seconds - float(time_seconds[event_start_index])
    metrics = {
        "event_definition_version": normalized_config["event_definition_version"],
        "event_start_source": event_start_source,
        "event_start_detection_series": "forcing",
        "event_start_reference_peak_time_seconds": inflow_peak_time_seconds,
        "event_start_reference_peak_time_utc": _maybe_time_iso(forcing["time"], inflow_peak_time_seconds),
        "event_start_time_seconds": float(time_seconds[event_start_index]),
        "event_start_time_utc": forcing.loc[event_start_index, "time"].isoformat(),
        "event_start_input_discharge_cms": baseline_input,
        "event_start_outlet_discharge_cms": baseline_outlet,
        "event_end_source": event_end_source,
        "event_end_detection_series": "outlet" if event_end_source.startswith("auto_outlet") or event_end_source == "series_end_requested" else "forcing_reference",
        "event_end_reference_peak_time_seconds": peak_time_seconds,
        "event_end_reference_peak_time_utc": _maybe_time_iso(forcing["time"], peak_time_seconds),
        "event_end_censored": bool(event_end_censored),
        "event_end_time_seconds": float(time_seconds[event_end_index]),
        "event_end_time_utc": forcing.loc[event_end_index, "time"].isoformat(),
        "event_end_input_discharge_cms": recession_baseline_input,
        "event_end_outlet_discharge_cms": recession_baseline_outlet,
        "event_duration_seconds": float(time_seconds[event_end_index] - time_seconds[event_start_index]),
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
        "outlet_volume_m3": _integrate_trapezoid(outlet_q, time_seconds),
        "outlet_excess_volume_m3": _integrate_trapezoid(
            np.maximum(outlet_q - baseline_outlet, 0.0),
            time_seconds,
        ),
        "outlet_reach_count": len(outlet_reach_ids),
        "outlet_reach_ids": json.dumps(outlet_reach_ids),
        "hydrograph_duration_seconds": float(time_seconds[-1] - time_seconds[0]),
        "metric_config": json.dumps(normalized_config),
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

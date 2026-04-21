from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import EventDetectionConfig


def detect_events(timeseries: pd.DataFrame, config: EventDetectionConfig) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame(columns=_event_columns())

    records: list[dict[str, object]] = []
    for station_key, group in timeseries.groupby("station_key", sort=False):
        station_events = _detect_station_events(station_key, group, config)
        records.extend(station_events)
    if not records:
        return pd.DataFrame(columns=_event_columns())

    events = pd.DataFrame.from_records(records)
    if events.empty:
        return pd.DataFrame(columns=_event_columns())
    events = score_event_quality(events, config)
    events["station_event_rank"] = (
        events.groupby("station_key")["event_quality_score"].rank(method="first", ascending=False).astype(int)
    )
    return events.sort_values(["station_key", "peak_time"]).reset_index(drop=True)


def score_event_quality(events: pd.DataFrame, config: EventDetectionConfig) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    scored = events.copy()
    prominence_norm = np.clip(
        scored["prominence_ratio"].fillna(0) / max(config.min_peak_prominence_ratio, 1e-9),
        0,
        3,
    ) / 3
    rise_norm = np.clip(scored["rise_points"].fillna(0) / max(config.min_rise_points, 1), 0, 3) / 3
    monotonic_norm = np.clip(scored["rise_monotonic_fraction"].fillna(0), 0, 1)
    scored["event_quality_score"] = 0.4 * prominence_norm + 0.3 * rise_norm + 0.3 * monotonic_norm
    scored["selected_event"] = (
        scored["event_valid"]
        & (scored["rise_points"] >= config.min_rise_points)
        & (scored["prominence_ratio"] >= config.min_peak_prominence_ratio)
        & (scored["rise_monotonic_fraction"] >= config.min_monotonic_rise_fraction)
        & (scored["event_duration_hours"] >= config.min_event_duration_hours)
        & scored["t0_rise_t10_t90_hours"].notna()
        & scored["t0_rise_start_to_peak_hours"].notna()
    )
    return scored


def summarize_events(events: pd.DataFrame) -> dict[str, int | float]:
    if events.empty:
        return {
            "stations_with_timeseries": 0,
            "candidate_events": 0,
            "selected_events": 0,
            "selected_station_count": 0,
            "median_prominence_ratio": 0.0,
            "median_t0_rise_t10_t90_hours": 0.0,
        }

    selected = events[events["selected_event"]].copy()
    median_prominence = events["prominence_ratio"].median(skipna=True)
    median_t0 = events["t0_rise_t10_t90_hours"].median(skipna=True)
    return {
        "stations_with_timeseries": int(events["station_key"].nunique()),
        "candidate_events": int(len(events)),
        "selected_events": int(len(selected)),
        "selected_station_count": int(selected["station_key"].nunique()),
        "median_prominence_ratio": 0.0 if pd.isna(median_prominence) else float(median_prominence),
        "median_t0_rise_t10_t90_hours": 0.0 if pd.isna(median_t0) else float(median_t0),
    }


def _detect_station_events(
    station_key: str,
    frame: pd.DataFrame,
    config: EventDetectionConfig,
) -> list[dict[str, object]]:
    working = frame.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    if len(working) < 3:
        return []

    times = pd.to_datetime(working["time"])
    q_raw = pd.to_numeric(working["discharge"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(q_raw).all():
        return []
    q_smooth = (
        pd.Series(q_raw)
        .rolling(window=max(config.smoothing_window, 1), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )

    candidate_peaks = _find_candidate_peaks(q_smooth)
    selected_peaks = _select_separated_peaks(times, q_smooth, candidate_peaks, config.min_event_separation_hours)
    if not selected_peaks:
        return []

    records: list[dict[str, object]] = []
    start_search_delta = pd.Timedelta(hours=config.start_search_hours)
    end_search_delta = pd.Timedelta(hours=config.end_search_hours)
    pre_event_delta = pd.Timedelta(hours=config.pre_event_window_hours)

    for peak_idx in selected_peaks:
        peak_time = times.iloc[peak_idx]
        start_idx = _find_window_min_index(times, q_smooth, peak_idx, peak_time - start_search_delta, peak_idx, last=True)
        end_idx = _find_window_min_index(
            times,
            q_smooth,
            peak_idx + 1,
            peak_time + end_search_delta,
            len(times),
            last=False,
        )
        if start_idx is None or start_idx >= peak_idx:
            continue

        start_time = times.iloc[start_idx]
        end_time = times.iloc[end_idx] if end_idx is not None else pd.NaT
        peak_q = float(q_raw[peak_idx])
        peak_q_smooth = float(q_smooth[peak_idx])
        start_q = float(q_raw[start_idx])

        pre_window_start = start_time - pre_event_delta
        pre_mask = (times >= pre_window_start) & (times < start_time)
        pre_event_values = q_raw[pre_mask.to_numpy()]
        q0_pre_event_median = float(np.nanmedian(pre_event_values)) if len(pre_event_values) else np.nan
        baseline_q = q0_pre_event_median if np.isfinite(q0_pre_event_median) else start_q
        amplitude = peak_q_smooth - baseline_q

        rise_values = q_smooth[start_idx : peak_idx + 1]
        rise_times = times.iloc[start_idx : peak_idx + 1]
        rise_points = int(len(rise_values))
        rise_monotonic_fraction = float(np.mean(np.diff(rise_values) > 0)) if rise_points > 1 else 0.0
        t0_start_to_peak_hours = _duration_hours(start_time, peak_time)
        t10_time = pd.NaT
        t90_time = pd.NaT
        t0_rise_t10_t90_hours = np.nan
        if np.isfinite(amplitude) and amplitude > 0:
            q10 = baseline_q + 0.1 * amplitude
            q90 = baseline_q + 0.9 * amplitude
            t10_time = _interpolate_crossing_time(rise_times, rise_values, q10)
            t90_time = _interpolate_crossing_time(rise_times, rise_values, q90)
            if pd.notna(t10_time) and pd.notna(t90_time):
                t0_rise_t10_t90_hours = _duration_hours(t10_time, t90_time)

        event_duration_hours = _duration_hours(start_time, end_time) if pd.notna(end_time) else t0_start_to_peak_hours
        prominence_ratio = amplitude / max(abs(baseline_q), 1e-6) if np.isfinite(amplitude) else np.nan
        event_valid = bool(
            np.isfinite(peak_q)
            and np.isfinite(start_q)
            and np.isfinite(amplitude)
            and amplitude > 0
            and rise_points > 1
            and t0_start_to_peak_hours > 0
        )
        event_id = f"{station_key}:{peak_time.isoformat()}"

        records.append(
            {
                "event_id": event_id,
                "station_key": station_key,
                "station_id": working["station_id"].iloc[0],
                "country": working["country"].iloc[0],
                "source_function": working["source_function"].iloc[0] if "source_function" in working.columns else pd.NA,
                "start_time": start_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "start_discharge": start_q,
                "peak_discharge": peak_q,
                "peak_discharge_smooth": peak_q_smooth,
                "q0_pre_event_median": q0_pre_event_median,
                "q0_event_start_discharge": start_q,
                "event_amplitude": amplitude,
                "prominence_ratio": prominence_ratio,
                "rise_points": rise_points,
                "rise_monotonic_fraction": rise_monotonic_fraction,
                "event_duration_hours": event_duration_hours,
                "t10_time": t10_time,
                "t90_time": t90_time,
                "t0_rise_t10_t90_hours": t0_rise_t10_t90_hours,
                "t0_rise_start_to_peak_hours": t0_start_to_peak_hours,
                "event_valid": event_valid,
                "event_detection_config": str(asdict(config)),
            }
        )
    return records


def _find_candidate_peaks(values: np.ndarray) -> list[int]:
    if len(values) < 3:
        return []
    peaks: list[int] = []
    for idx in range(1, len(values) - 1):
        if np.isnan(values[idx - 1]) or np.isnan(values[idx]) or np.isnan(values[idx + 1]):
            continue
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            peaks.append(idx)
    return peaks


def _select_separated_peaks(
    times: pd.Series,
    values: np.ndarray,
    peaks: list[int],
    min_event_separation_hours: float,
) -> list[int]:
    if not peaks:
        return []
    min_delta = pd.Timedelta(hours=min_event_separation_hours)
    keep: list[int] = []
    for idx in sorted(peaks, key=lambda item: values[item], reverse=True):
        peak_time = times.iloc[idx]
        if any(abs(peak_time - times.iloc[other]) < min_delta for other in keep):
            continue
        keep.append(idx)
    return sorted(keep)


def _find_window_min_index(
    times: pd.Series,
    values: np.ndarray,
    start_idx: int,
    end_time: pd.Timestamp,
    default_end_idx: int,
    *,
    last: bool,
) -> int | None:
    # Use the deepest minimum in the search window, not the nearest local dip.
    # For the pre-peak search, ties are broken by taking the latest occurrence so
    # the event start stays as close to the peak as possible. For the post-peak
    # search, ties are broken by taking the earliest occurrence.
    if start_idx >= len(times):
        return None
    start_time = times.iloc[start_idx]
    if pd.isna(start_time):
        return None
    if last:
        mask = (times >= end_time) & (times < start_time if start_idx < len(times) else True)
    else:
        base_time = times.iloc[start_idx - 1] if start_idx > 0 else times.iloc[0]
        mask = (times > base_time) & (times <= end_time)
    indices = np.flatnonzero(mask.to_numpy())
    if len(indices) == 0:
        return None if last else None
    if not last:
        indices = indices[indices >= start_idx]
    if len(indices) == 0:
        return None
    window_values = values[indices]
    if np.isnan(window_values).all():
        return None
    min_value = np.nanmin(window_values)
    matches = indices[np.isclose(window_values, min_value, equal_nan=False)]
    if len(matches) == 0:
        return None
    return int(matches[-1] if last else matches[0])


def _interpolate_crossing_time(times: pd.Series, values: np.ndarray, level: float) -> pd.Timestamp:
    if len(values) == 0 or np.isnan(level):
        return pd.NaT
    if values[0] >= level:
        return times.iloc[0]
    for idx in range(1, len(values)):
        left = values[idx - 1]
        right = values[idx]
        if np.isnan(left) or np.isnan(right):
            continue
        if left <= level <= right:
            left_time = times.iloc[idx - 1]
            right_time = times.iloc[idx]
            if right == left:
                return right_time
            fraction = (level - left) / (right - left)
            return left_time + (right_time - left_time) * float(fraction)
    return pd.NaT


def _duration_hours(start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    if pd.isna(start_time) or pd.isna(end_time):
        return np.nan
    delta = end_time - start_time
    return float(delta.total_seconds() / 3600.0)


def _event_columns() -> list[str]:
    return [
        "event_id",
        "station_key",
        "station_id",
        "country",
        "source_function",
        "start_time",
        "peak_time",
        "end_time",
        "start_discharge",
        "peak_discharge",
        "peak_discharge_smooth",
        "q0_pre_event_median",
        "q0_event_start_discharge",
        "event_amplitude",
        "prominence_ratio",
        "rise_points",
        "rise_monotonic_fraction",
        "event_duration_hours",
        "t10_time",
        "t90_time",
        "t0_rise_t10_t90_hours",
        "t0_rise_start_to_peak_hours",
        "event_valid",
        "event_quality_score",
        "selected_event",
        "station_event_rank",
        "event_detection_config",
    ]

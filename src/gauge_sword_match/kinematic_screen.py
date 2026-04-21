from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KinematicConfig
from .hydraulics import (
    classify_kinematic_candidate,
    compute_froude,
    compute_reference_area,
    compute_reference_depth,
    compute_reference_velocity,
    compute_tplus,
    get_q0_value,
    get_t0_hours,
)
from .sword_io import SwordFileCatalog, load_reaches


def build_screening_inputs(
    events: pd.DataFrame,
    best_matches: pd.DataFrame,
    catalog: SwordFileCatalog,
    config: KinematicConfig,
) -> pd.DataFrame:
    selected_events = events[events["selected_event"]].copy()
    matched = best_matches[
        best_matches["reach_id"].notna()
        & best_matches["confidence_class"].isin(config.allowed_confidence_classes)
    ].copy()
    if selected_events.empty or matched.empty:
        return pd.DataFrame()

    match_columns = [
        "station_key",
        "station_id",
        "country",
        "reach_id",
        "sword_region",
        "sword_node_id",
        "confidence_class",
        "distance_m",
        "total_score",
    ]
    merged = selected_events.merge(matched[match_columns], on="station_key", how="inner", suffixes=("", "_match"))
    if merged.empty:
        return pd.DataFrame()

    reach_attrs = _load_reach_attributes(catalog, merged, config)
    if reach_attrs.empty:
        return pd.DataFrame()

    inputs = merged.merge(reach_attrs, on=["sword_region", "reach_id"], how="left")
    inputs["width_value"] = pd.to_numeric(inputs.get(config.width_field), errors="coerce")
    inputs["slope_value"] = pd.to_numeric(inputs.get(config.slope_field), errors="coerce")
    inputs["is_multichannel_hint"] = (
        pd.to_numeric(inputs.get("n_chan_max"), errors="coerce").fillna(1).gt(1)
        | pd.to_numeric(inputs.get("n_chan_mod"), errors="coerce").fillna(1).gt(1)
    )
    return inputs


def run_kinematic_screen(inputs: pd.DataFrame, config: KinematicConfig) -> pd.DataFrame:
    if inputs.empty:
        return pd.DataFrame(columns=_result_columns())

    records: list[dict[str, object]] = []
    for row in inputs.to_dict(orient="records"):
        width_value = _to_float(row.get("width_value"))
        slope_value = _to_float(row.get("slope_value"))
        for kb in config.kb_values:
            depth_y0 = compute_reference_depth(width_value, kb)
            area_a0 = compute_reference_area(width_value, depth_y0)
            for q0_method in config.q0_methods:
                q0 = get_q0_value(pd.Series(row), q0_method)
                v0 = compute_reference_velocity(q0, area_a0)
                froude_f0 = compute_froude(v0, depth_y0, gravity_m_s2=config.screening.gravity_m_s2)
                for t0_method in config.t0_methods:
                    t0_hours = get_t0_hours(pd.Series(row), t0_method)
                    tplus = compute_tplus(t0_hours, v0, slope_value, depth_y0)
                    valid_input = bool(
                        np.isfinite(width_value)
                        and width_value > 0
                        and np.isfinite(slope_value)
                        and slope_value >= config.screening.min_valid_slope
                        and np.isfinite(q0)
                        and q0 >= 0
                        and np.isfinite(t0_hours)
                        and t0_hours > 0
                        and np.isfinite(depth_y0)
                        and depth_y0 > 0
                        and np.isfinite(v0)
                    )
                    is_candidate = (
                        classify_kinematic_candidate(
                            froude_f0,
                            tplus,
                            regime_tplus_min=config.screening.regime_tplus_min,
                            regime_froude_t0=config.screening.regime_froude_t0,
                            regime_tplus_end=config.screening.regime_tplus_end,
                            regime_froude_end=config.screening.regime_froude_end,
                        )
                        if valid_input
                        else None
                    )
                    records.append(
                        {
                            "station_key": row["station_key"],
                            "event_id": row["event_id"],
                            "station_id": row.get("station_id"),
                            "country": row.get("country"),
                            "reach_id": row["reach_id"],
                            "sword_region": row["sword_region"],
                            "sword_node_id": row.get("sword_node_id"),
                            "confidence_class": row.get("confidence_class"),
                            "distance_m": row.get("distance_m"),
                            "total_score": row.get("total_score"),
                            "peak_time": row.get("peak_time"),
                            "kb": float(kb),
                            "q0_method": q0_method,
                            "t0_method": t0_method,
                            "width_field": config.width_field,
                            "slope_field": config.slope_field,
                            "width_value": width_value,
                            "slope_value": slope_value,
                            "q0_cms": q0,
                            "t0_hours": t0_hours,
                            "depth_y0_m": depth_y0,
                            "area_a0_m2": area_a0,
                            "velocity_v0_ms": v0,
                            "froude_f0": froude_f0,
                            "tplus": tplus,
                            "is_kinematic_candidate": is_candidate,
                            "regime_classification": _classify_result_label(is_candidate, valid_input),
                            "valid_input": valid_input,
                            "is_multichannel_hint": bool(row.get("is_multichannel_hint", False)),
                            "river_name": row.get("river_name"),
                            "n_chan_max": row.get("n_chan_max"),
                            "n_chan_mod": row.get("n_chan_mod"),
                            "slope_obs_reliable": row.get("slope_obs_reliable"),
                            "review_reason": _result_review_reason(
                                width_value=width_value,
                                slope_value=slope_value,
                                q0=q0,
                                t0_hours=t0_hours,
                                valid_input=valid_input,
                                is_multichannel_hint=bool(row.get("is_multichannel_hint", False)),
                            ),
                        }
                    )

    return pd.DataFrame.from_records(records, columns=_result_columns())


def summarize_kinematic_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=_summary_columns())

    working = results.copy()
    working["is_candidate_numeric"] = working["is_kinematic_candidate"].map(
        lambda value: np.nan if value is None else int(bool(value))
    )
    summary = (
        working.groupby(
            ["station_key", "station_id", "country", "reach_id", "sword_region", "sword_node_id"],
            as_index=False,
        )
        .agg(
            event_count=("event_id", "nunique"),
            result_count=("event_id", "count"),
            valid_result_count=("valid_input", "sum"),
            kinematic_candidate_count=("is_candidate_numeric", "sum"),
            mean_froude_f0=("froude_f0", "mean"),
            mean_tplus=("tplus", "mean"),
            width_value=("width_value", "first"),
            slope_value=("slope_value", "first"),
            confidence_class=("confidence_class", "first"),
            distance_m=("distance_m", "first"),
            total_score=("total_score", "first"),
            is_multichannel_hint=("is_multichannel_hint", "max"),
            n_chan_max=("n_chan_max", "max"),
            n_chan_mod=("n_chan_mod", "max"),
            q0_methods=("q0_method", lambda values: ",".join(sorted(set(values.astype(str))))),
            t0_methods=("t0_method", lambda values: ",".join(sorted(set(values.astype(str))))),
            kb_values=("kb", lambda values: ",".join(str(int(value)) if float(value).is_integer() else str(value) for value in sorted(set(values)))),
        )
    )
    summary["kinematic_fraction"] = summary["kinematic_candidate_count"] / summary["valid_result_count"].replace(0, np.nan)
    summary["any_kinematic_candidate"] = summary["kinematic_candidate_count"] > 0
    summary["stable_kinematic_candidate"] = summary["valid_result_count"].gt(0) & summary["kinematic_fraction"].eq(1.0)
    summary["review_flag"] = (
        summary["valid_result_count"].eq(0)
        | summary["kinematic_fraction"].between(0, 1, inclusive="neither")
        | summary["is_multichannel_hint"]
    )
    summary["review_reason"] = summary.apply(_summary_review_reason, axis=1)
    return summary[_summary_columns()].sort_values(["stable_kinematic_candidate", "kinematic_fraction"], ascending=[False, False])


def _load_reach_attributes(
    catalog: SwordFileCatalog,
    merged: pd.DataFrame,
    config: KinematicConfig,
) -> pd.DataFrame:
    columns = [
        "reach_id",
        "river_name",
        "n_chan_max",
        "n_chan_mod",
        "slope_obs_reliable",
        config.width_field,
        config.slope_field,
    ]
    frames: list[pd.DataFrame] = []
    for region, group in merged.groupby("sword_region"):
        reach_ids = group["reach_id"].dropna().unique().tolist()
        reaches = load_reaches(catalog, regions=[region], columns=columns, reach_ids=reach_ids)
        if reaches.empty:
            continue
        keep_columns = [column for column in ["sword_region", *columns] if column in reaches.columns]
        frames.append(pd.DataFrame(reaches[keep_columns]).copy())
    if not frames:
        return pd.DataFrame(columns=["sword_region", "reach_id", *columns])
    return pd.concat(frames, ignore_index=True)


def _classify_result_label(is_candidate: bool | None, valid_input: bool) -> str:
    if not valid_input:
        return "invalid"
    return "kinematic_candidate" if bool(is_candidate) else "non_kinematic_candidate"


def _result_review_reason(
    *,
    width_value: float,
    slope_value: float,
    q0: float,
    t0_hours: float,
    valid_input: bool,
    is_multichannel_hint: bool,
) -> str:
    reasons: list[str] = []
    if not np.isfinite(width_value) or width_value <= 0:
        reasons.append("invalid_width")
    if not np.isfinite(slope_value) or slope_value <= 0:
        reasons.append("invalid_slope")
    if not np.isfinite(q0) or q0 < 0:
        reasons.append("invalid_q0")
    if not np.isfinite(t0_hours) or t0_hours <= 0:
        reasons.append("invalid_t0")
    if not valid_input:
        reasons.append("invalid_hydraulics")
    if is_multichannel_hint:
        reasons.append("multichannel_hint")
    return ",".join(dict.fromkeys(reasons)) if reasons else ""


def _summary_review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if int(row.get("valid_result_count", 0)) == 0:
        reasons.append("no_valid_results")
    fraction = row.get("kinematic_fraction")
    if pd.notna(fraction) and 0 < float(fraction) < 1:
        reasons.append("assumption_sensitive")
    if bool(row.get("is_multichannel_hint", False)):
        reasons.append("multichannel_hint")
    return ",".join(reasons) if reasons else ""


def _to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def _result_columns() -> list[str]:
    return [
        "station_key",
        "event_id",
        "station_id",
        "country",
        "reach_id",
        "sword_region",
        "sword_node_id",
        "confidence_class",
        "distance_m",
        "total_score",
        "peak_time",
        "kb",
        "q0_method",
        "t0_method",
        "width_field",
        "slope_field",
        "width_value",
        "slope_value",
        "q0_cms",
        "t0_hours",
        "depth_y0_m",
        "area_a0_m2",
        "velocity_v0_ms",
        "froude_f0",
        "tplus",
        "is_kinematic_candidate",
        "regime_classification",
        "valid_input",
        "is_multichannel_hint",
        "river_name",
        "n_chan_max",
        "n_chan_mod",
        "slope_obs_reliable",
        "review_reason",
    ]


def _summary_columns() -> list[str]:
    return [
        "station_key",
        "station_id",
        "country",
        "reach_id",
        "sword_region",
        "sword_node_id",
        "event_count",
        "result_count",
        "valid_result_count",
        "kinematic_candidate_count",
        "kinematic_fraction",
        "any_kinematic_candidate",
        "stable_kinematic_candidate",
        "mean_froude_f0",
        "mean_tplus",
        "width_value",
        "slope_value",
        "confidence_class",
        "distance_m",
        "total_score",
        "is_multichannel_hint",
        "n_chan_max",
        "n_chan_mod",
        "q0_methods",
        "t0_methods",
        "kb_values",
        "review_flag",
        "review_reason",
    ]

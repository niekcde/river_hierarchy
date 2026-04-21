from __future__ import annotations

import pandas as pd

from .utils import write_json, write_table


def compute_kinematic_metrics(results: pd.DataFrame, summary: pd.DataFrame) -> dict[str, int | float]:
    if results.empty and summary.empty:
        return {
            "result_rows": 0,
            "valid_result_rows": 0,
            "event_count": 0,
            "station_count": 0,
            "kinematic_candidate_rows": 0,
            "any_kinematic_station_count": 0,
            "stable_kinematic_station_count": 0,
            "multichannel_station_count": 0,
        }

    return {
        "result_rows": int(len(results)),
        "valid_result_rows": int(results["valid_input"].sum()) if "valid_input" in results.columns else 0,
        "event_count": int(results["event_id"].nunique()) if "event_id" in results.columns else 0,
        "station_count": int(summary["station_key"].nunique()) if "station_key" in summary.columns else 0,
        "kinematic_candidate_rows": int(results["is_kinematic_candidate"].fillna(False).astype(bool).sum())
        if "is_kinematic_candidate" in results.columns
        else 0,
        "any_kinematic_station_count": int(summary["any_kinematic_candidate"].sum())
        if "any_kinematic_candidate" in summary.columns
        else 0,
        "stable_kinematic_station_count": int(summary["stable_kinematic_candidate"].sum())
        if "stable_kinematic_candidate" in summary.columns
        else 0,
        "multichannel_station_count": int(summary["is_multichannel_hint"].sum())
        if "is_multichannel_hint" in summary.columns
        else 0,
    }


def export_kinematic_metrics(results: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    write_json(compute_kinematic_metrics(results, summary), path)


def build_kinematic_review_queue(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    return summary[summary["review_flag"]].copy().reset_index(drop=True)


def export_kinematic_review_queue(summary: pd.DataFrame, path) -> None:
    write_table(build_kinematic_review_queue(summary), path)

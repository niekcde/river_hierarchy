from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from .hydrograph import _load_qout


SECONDS_PER_HOUR = 3600.0
BASE_COMPARISON_COLUMNS = {
    "peak_time_seconds": "delta_peak_time_hours",
    "peak_discharge_cms": "delta_peak_discharge_cms",
    "time_to_peak_seconds": "delta_rise_time_hours",
    "fall_time_seconds": "delta_fall_time_hours",
    "peak_attenuation_ratio": "delta_peak_attenuation_ratio",
    "lag_to_inflow_peak_seconds": "delta_lag_to_inflow_peak_hours",
}
SECOND_COLUMNS_TO_HOURS = {
    "time_to_peak_seconds": "time_to_peak_hours",
    "lag_to_inflow_peak_seconds": "lag_to_inflow_peak_hours",
    "event_duration_seconds": "event_duration_hours",
    "fall_time_seconds": "fall_time_hours",
    "fall_time_50_seconds": "fall_time_50_hours",
    "fall_time_10_seconds": "fall_time_10_hours",
    "e_folding_time_seconds": "e_folding_time_hours",
    "hydrograph_duration_seconds": "hydrograph_duration_hours",
}


@dataclass(frozen=True, slots=True)
class ReachWaveStack:
    reaches: Any
    time_seconds: np.ndarray
    reach_ids: np.ndarray
    qout: np.ndarray


@dataclass(frozen=True, slots=True)
class _LoadedWaveState:
    experiment_path: Path
    run_registry: pd.DataFrame
    run_row: pd.Series
    wave_stack: ReachWaveStack
    outlet_hydro: pd.DataFrame


def _as_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} in {directory}")
    return matches[0]


def _parse_selected_unit_ids(value: object) -> tuple[int, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    ids: list[int] = []
    for part in parts:
        try:
            ids.append(int(float(part)))
        except (TypeError, ValueError):
            continue
    return tuple(ids)


def _parse_int_list(value: object) -> tuple[int, ...]:
    return _parse_selected_unit_ids(value)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) ** 2)))


def _align_for_lag(reference: np.ndarray, candidate: np.ndarray, lag_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_steps < 0:
        return reference[-lag_steps:], candidate[: candidate.size + lag_steps]
    if lag_steps > 0:
        return reference[: reference.size - lag_steps], candidate[lag_steps:]
    return reference, candidate


def compare_hydrographs(
    reference_hydrograph: pd.DataFrame,
    candidate_hydrograph: pd.DataFrame,
    *,
    q_column: str = "q_outlet_cms",
    max_lag_steps: int = 48,
) -> dict[str, float]:
    if q_column not in reference_hydrograph.columns or q_column not in candidate_hydrograph.columns:
        raise ValueError(f"Hydrograph comparison requires '{q_column}' in both inputs.")

    merged = reference_hydrograph[["time_utc", q_column]].merge(
        candidate_hydrograph[["time_utc", q_column]],
        on="time_utc",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    if merged.empty:
        raise ValueError("Hydrographs do not share any common timestamps.")

    x = merged[f"{q_column}_reference"].to_numpy(dtype=float)
    y = merged[f"{q_column}_candidate"].to_numpy(dtype=float)
    time_seconds = merged.get("time_seconds_reference")
    if time_seconds is None and "time_seconds" in reference_hydrograph.columns:
        ref_times = reference_hydrograph[["time_utc", "time_seconds"]]
        merged = merged.merge(ref_times, on="time_utc", how="left")
        time_seconds = merged["time_seconds"]
    dt_seconds = float(np.median(np.diff(time_seconds.to_numpy(dtype=float)))) if time_seconds is not None and len(merged) > 1 else float("nan")

    best_lag = 0
    best_corr = -np.inf
    max_lag = max(0, int(max_lag_steps))
    for lag_steps in range(-max_lag, max_lag + 1):
        ref_aligned, cand_aligned = _align_for_lag(x, y, lag_steps)
        corr = _safe_corr(ref_aligned, cand_aligned)
        corr_score = corr if np.isfinite(corr) else -np.inf
        if corr_score > best_corr:
            best_corr = corr_score
            best_lag = lag_steps

    ref_shifted, cand_shifted = _align_for_lag(x, y, best_lag)
    rmse_unshifted = _safe_rmse(x, y)
    rmse_shifted = _safe_rmse(ref_shifted, cand_shifted)
    peak_reference = float(np.max(x)) if x.size else float("nan")
    nrmse_shifted = rmse_shifted / peak_reference if np.isfinite(peak_reference) and peak_reference > 0 else float("nan")

    return {
        "hydrograph_lag_vs_reference_steps": float(best_lag),
        "hydrograph_lag_vs_reference_hours": float(best_lag * dt_seconds / SECONDS_PER_HOUR) if np.isfinite(dt_seconds) else float("nan"),
        "hydrograph_corr_vs_reference": _safe_corr(x, y),
        "hydrograph_corr_shifted_vs_reference": _safe_corr(ref_shifted, cand_shifted),
        "hydrograph_rmse_vs_reference": rmse_unshifted,
        "hydrograph_rmse_shifted_vs_reference": rmse_shifted,
        "hydrograph_nrmse_shifted_vs_reference": nrmse_shifted,
    }


def _infer_example_id(experiment_dir: Path, state_registry: pd.DataFrame | None = None) -> str:
    parts = experiment_dir.name.split("_")
    if parts and parts[0]:
        return parts[0]
    if state_registry is not None and "directed_links_path" in state_registry.columns:
        non_null = state_registry["directed_links_path"].dropna().astype(str)
        if not non_null.empty:
            stem = Path(non_null.iloc[0]).stem
            return stem.split("__")[0]
    return experiment_dir.name


def _build_bubble_summary(unit_summary: pd.DataFrame | None) -> pd.DataFrame | None:
    if unit_summary is None or unit_summary.empty:
        return None
    required = {"unit_id", "compound_bubble_id", "compound_bubble_role"}
    if not required.issubset(unit_summary.columns):
        return None

    working = unit_summary.copy()
    working["unit_id"] = pd.to_numeric(working["unit_id"], errors="coerce").astype("Int64")
    working["compound_bubble_id"] = pd.to_numeric(working["compound_bubble_id"], errors="coerce").astype("Int64")
    if "unit_node_count" in working.columns:
        working["unit_node_count"] = pd.to_numeric(working["unit_node_count"], errors="coerce").astype(float)

    working = working.loc[
        working["compound_bubble_id"].notna()
        & working["compound_bubble_role"].isin(["bubble_root", "bubble_member"])
        & working["unit_id"].notna()
    ].copy()
    if working.empty:
        return None

    rows: list[dict[str, Any]] = []
    for bubble_id, group in working.groupby("compound_bubble_id", sort=True):
        unit_ids = sorted(group["unit_id"].astype(int).tolist())
        root_rows = group.loc[group["compound_bubble_role"].eq("bubble_root"), "unit_id"]
        bubble_root_unit_id = int(root_rows.iloc[0]) if not root_rows.empty else int(min(unit_ids))
        total_unit_nodes = float(group["unit_node_count"].sum()) if "unit_node_count" in group.columns else float("nan")
        rows.append(
            {
                "compound_bubble_id": int(bubble_id),
                "bubble_unit_count": int(len(unit_ids)),
                "bubble_unit_ids": ",".join(str(unit_id) for unit_id in unit_ids),
                "bubble_root_unit_id": bubble_root_unit_id,
                "bubble_total_unit_nodes": total_unit_nodes,
            }
        )
    return pd.DataFrame(rows)


def _prepare_state_frame(state_registry: pd.DataFrame | None) -> pd.DataFrame | None:
    if state_registry is None or state_registry.empty:
        return None
    frame = state_registry.copy()
    frame["selected_unit_ids_tuple"] = frame["selected_unit_ids"].apply(_parse_selected_unit_ids) if "selected_unit_ids" in frame.columns else [()] * len(frame)
    frame["selected_unit_count"] = frame["selected_unit_ids_tuple"].apply(len)
    frame["selected_unit_id"] = frame["selected_unit_ids_tuple"].apply(lambda values: values[0] if len(values) == 1 else pd.NA)
    prefixed = frame.rename(columns={column: f"state_{column}" for column in frame.columns if column != "state_id"})
    return prefixed


def _prepare_prep_frame(prep_registry: pd.DataFrame | None) -> pd.DataFrame | None:
    if prep_registry is None or prep_registry.empty:
        return None
    return prep_registry.rename(columns={column: f"prep_{column}" for column in prep_registry.columns if column != "state_id"})


def _prepare_transition_frame(transition_registry: pd.DataFrame | None) -> pd.DataFrame | None:
    if transition_registry is None or transition_registry.empty:
        return None
    frame = transition_registry.copy()
    if "child_state_id" not in frame.columns:
        return None
    return frame.rename(
        columns={
            column: ("state_id" if column == "child_state_id" else f"transition_{column}")
            for column in frame.columns
        }
    )


def _add_hour_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for seconds_column, hours_column in SECOND_COLUMNS_TO_HOURS.items():
        if seconds_column in frame.columns:
            frame[hours_column] = pd.to_numeric(frame[seconds_column], errors="coerce") / SECONDS_PER_HOUR
    return frame


def _load_experiment_tables(experiment_dir: str | Path) -> dict[str, Any]:
    experiment_path = _as_path(experiment_dir)
    state_registry = _load_csv_if_exists(experiment_path / "state_registry.csv")
    tables = {
        "experiment_dir": experiment_path,
        "example_id": _infer_example_id(experiment_path, state_registry),
        "run_registry": pd.read_csv(experiment_path / "rapid_run_registry.csv"),
        "state_registry": state_registry,
        "prep_registry": _load_csv_if_exists(experiment_path / "rapid_prep_registry.csv"),
        "transition_registry": _load_csv_if_exists(experiment_path / "transition_registry.csv"),
        "base_collapse_ranking": _load_csv_if_exists(experiment_path / "states" / "S000_base" / "hierarchy" / "collapse_ranking.csv"),
        "base_unit_summary": _load_csv_if_exists(experiment_path / "states" / "S000_base" / "hierarchy" / "unit_summary.csv"),
    }
    return tables


def load_state_hydrograph(run_registry_or_experiment_dir: pd.DataFrame | str | Path, state_id: str) -> pd.DataFrame:
    if isinstance(run_registry_or_experiment_dir, pd.DataFrame):
        run_registry = run_registry_or_experiment_dir
    else:
        run_registry = pd.read_csv(_as_path(run_registry_or_experiment_dir) / "rapid_run_registry.csv")
    row = run_registry.loc[run_registry["state_id"].eq(state_id)]
    if row.empty:
        raise KeyError(f"State '{state_id}' was not found in the RAPID run registry.")
    hydro = pd.read_csv(row.iloc[0]["outlet_hydrograph_csv"])
    if "time_utc" in hydro.columns:
        hydro["time_utc"] = pd.to_datetime(hydro["time_utc"], utc=True, errors="coerce")
    return hydro


def build_experiment_analysis_frame(
    experiment_dir: str | Path,
    *,
    base_state_id: str = "S000_base",
    max_lag_hours: float = 24.0,
) -> pd.DataFrame:
    tables = _load_experiment_tables(experiment_dir)
    run_registry = tables["run_registry"].copy()
    run_registry["experiment_dir"] = str(tables["experiment_dir"])
    run_registry["example_id"] = tables["example_id"]
    run_registry["peak_time_utc"] = pd.to_datetime(run_registry.get("peak_time_utc"), utc=True, errors="coerce")
    run_registry["event_start_time_utc"] = pd.to_datetime(run_registry.get("event_start_time_utc"), utc=True, errors="coerce")
    run_registry["event_end_time_utc"] = pd.to_datetime(run_registry.get("event_end_time_utc"), utc=True, errors="coerce")
    run_registry = _add_hour_columns(run_registry)

    state_frame = _prepare_state_frame(tables["state_registry"])
    if state_frame is not None:
        run_registry = run_registry.merge(state_frame, on="state_id", how="left")
    prep_frame = _prepare_prep_frame(tables["prep_registry"])
    if prep_frame is not None:
        run_registry = run_registry.merge(prep_frame, on="state_id", how="left")
    transition_frame = _prepare_transition_frame(tables["transition_registry"])
    if transition_frame is not None:
        run_registry = run_registry.merge(transition_frame, on="state_id", how="left")

    run_registry["scenario_family"] = "unit"
    if "state_selection_type" in run_registry.columns:
        run_registry.loc[run_registry["state_selection_type"].eq("compound_bubble"), "scenario_family"] = "compound_bubble"
    if "state_is_compound_state" in run_registry.columns:
        compound_mask = run_registry["state_is_compound_state"].eq(True)
        run_registry.loc[compound_mask, "scenario_family"] = "compound_bubble"
    if "state_state_role" in run_registry.columns:
        run_registry.loc[run_registry["state_state_role"].eq("base"), "scenario_family"] = "base"
    else:
        run_registry.loc[run_registry["state_id"].eq(base_state_id), "scenario_family"] = "base"

    if "state_selected_unit_count" in run_registry.columns:
        run_registry["selected_unit_count"] = pd.to_numeric(run_registry["state_selected_unit_count"], errors="coerce").fillna(0).astype(int)
        run_registry["selected_unit_id"] = pd.to_numeric(run_registry["state_selected_unit_id"], errors="coerce").astype("Int64")
        run_registry["selected_unit_ids_tuple"] = run_registry["state_selected_unit_ids_tuple"]
    else:
        run_registry["selected_unit_count"] = 0
        run_registry["selected_unit_id"] = pd.Series([pd.NA] * len(run_registry), dtype="Int64")
        run_registry["selected_unit_ids_tuple"] = [()] * len(run_registry)

    ranking = tables["base_collapse_ranking"]
    if ranking is not None and not ranking.empty:
        ranking_frame = ranking.copy()
        ranking_frame["selected_unit_id"] = pd.to_numeric(ranking_frame["unit_id"], errors="coerce").astype("Int64")
        ranking_frame = ranking_frame.rename(
            columns={
                column: f"base_unit_{column}"
                for column in ranking_frame.columns
                if column not in {"unit_id", "selected_unit_id"}
            }
        )
        run_registry = run_registry.merge(ranking_frame, on="selected_unit_id", how="left")

    bubble_summary = _build_bubble_summary(tables["base_unit_summary"])
    run_registry["compound_bubble_id"] = pd.Series([pd.NA] * len(run_registry), dtype="Float64")
    if "state_compound_bubble_id" in run_registry.columns:
        run_registry["compound_bubble_id"] = pd.to_numeric(run_registry["state_compound_bubble_id"], errors="coerce").astype("Float64")
    if "base_unit_compound_bubble_id" in run_registry.columns:
        unit_bubble = pd.to_numeric(run_registry["base_unit_compound_bubble_id"], errors="coerce").astype("Float64")
        run_registry["compound_bubble_id"] = run_registry["compound_bubble_id"].fillna(unit_bubble)
    if bubble_summary is not None and not bubble_summary.empty:
        run_registry = run_registry.merge(bubble_summary, on="compound_bubble_id", how="left")

    base_row = run_registry.loc[run_registry["state_id"].eq(base_state_id)]
    if base_row.empty:
        raise ValueError(f"Base state '{base_state_id}' was not found in {tables['experiment_dir']}.")
    base_row = base_row.iloc[0]
    for metric_column, delta_column in BASE_COMPARISON_COLUMNS.items():
        if metric_column not in run_registry.columns:
            continue
        base_value = pd.to_numeric(base_row.get(metric_column), errors="coerce")
        values = pd.to_numeric(run_registry[metric_column], errors="coerce")
        delta = values - base_value
        if metric_column.endswith("_seconds"):
            delta = delta / SECONDS_PER_HOUR
        run_registry[delta_column] = delta

    computed_rows = run_registry.loc[run_registry["hydrograph_status"].eq("computed")].copy()
    if not computed_rows.empty:
        base_hydro = load_state_hydrograph(run_registry, base_state_id)
        if "time_utc" in base_hydro.columns:
            dt_seconds = float(np.median(np.diff(base_hydro["time_seconds"].to_numpy(dtype=float)))) if len(base_hydro) > 1 and "time_seconds" in base_hydro.columns else float("nan")
            max_lag_steps = int(math.ceil(max_lag_hours * SECONDS_PER_HOUR / dt_seconds)) if np.isfinite(dt_seconds) and dt_seconds > 0 else 48
        else:
            max_lag_steps = 48
        comparison_rows: list[dict[str, Any]] = []
        for state_id in computed_rows["state_id"].tolist():
            candidate_hydro = load_state_hydrograph(run_registry, state_id)
            metrics = compare_hydrographs(base_hydro, candidate_hydro, max_lag_steps=max_lag_steps)
            metrics["state_id"] = state_id
            comparison_rows.append(metrics)
        comparison_frame = pd.DataFrame(comparison_rows).rename(
            columns={
                "hydrograph_lag_vs_reference_steps": "hydrograph_lag_vs_base_steps",
                "hydrograph_lag_vs_reference_hours": "hydrograph_lag_vs_base_hours",
                "hydrograph_corr_vs_reference": "hydrograph_corr_vs_base",
                "hydrograph_corr_shifted_vs_reference": "hydrograph_corr_shifted_vs_base",
                "hydrograph_rmse_vs_reference": "hydrograph_rmse_vs_base",
                "hydrograph_rmse_shifted_vs_reference": "hydrograph_rmse_shifted_vs_base",
                "hydrograph_nrmse_shifted_vs_reference": "hydrograph_nrmse_shifted_vs_base",
            }
        )
        run_registry = run_registry.merge(comparison_frame, on="state_id", how="left")

    run_registry["analysis_reference_state_id"] = base_state_id
    return run_registry


def build_combined_analysis_frame(
    independent_experiment_dir: str | Path,
    compound_bubble_experiment_dir: str | Path | None = None,
    *,
    base_state_id: str = "S000_base",
    max_lag_hours: float = 24.0,
) -> pd.DataFrame:
    frames = [
        build_experiment_analysis_frame(
            independent_experiment_dir,
            base_state_id=base_state_id,
            max_lag_hours=max_lag_hours,
        )
    ]
    if compound_bubble_experiment_dir is not None:
        frames.append(
            build_experiment_analysis_frame(
                compound_bubble_experiment_dir,
                base_state_id=base_state_id,
                max_lag_hours=max_lag_hours,
            )
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)

    bubble_reference = combined.loc[combined["scenario_family"].eq("compound_bubble")].copy()
    if not bubble_reference.empty:
        bubble_reference = bubble_reference.rename(
            columns={
                "state_id": "matching_compound_state_id",
                "peak_time_utc": "matching_compound_peak_time_utc",
                "delta_peak_time_hours": "matching_compound_delta_peak_time_hours",
                "delta_peak_discharge_cms": "matching_compound_delta_peak_discharge_cms",
                "delta_rise_time_hours": "matching_compound_delta_rise_time_hours",
                "delta_fall_time_hours": "matching_compound_delta_fall_time_hours",
                "hydrograph_lag_vs_base_hours": "matching_compound_hydrograph_lag_vs_base_hours",
            }
        )
        merge_columns = [
            "example_id",
            "compound_bubble_id",
            "matching_compound_state_id",
            "matching_compound_peak_time_utc",
            "matching_compound_delta_peak_time_hours",
            "matching_compound_delta_peak_discharge_cms",
            "matching_compound_delta_rise_time_hours",
            "matching_compound_delta_fall_time_hours",
            "matching_compound_hydrograph_lag_vs_base_hours",
        ]
        bubble_reference = bubble_reference[merge_columns].drop_duplicates(subset=["example_id", "compound_bubble_id"])
        combined = combined.merge(bubble_reference, on=["example_id", "compound_bubble_id"], how="left")

        for numerator, denominator, output in [
            ("delta_peak_time_hours", "matching_compound_delta_peak_time_hours", "unit_to_bubble_peak_time_ratio"),
            ("delta_peak_discharge_cms", "matching_compound_delta_peak_discharge_cms", "unit_to_bubble_peak_discharge_ratio"),
            ("delta_rise_time_hours", "matching_compound_delta_rise_time_hours", "unit_to_bubble_rise_time_ratio"),
            ("delta_fall_time_hours", "matching_compound_delta_fall_time_hours", "unit_to_bubble_fall_time_ratio"),
        ]:
            if numerator not in combined.columns or denominator not in combined.columns:
                continue
            denom = pd.to_numeric(combined[denominator], errors="coerce")
            num = pd.to_numeric(combined[numerator], errors="coerce")
            combined[output] = np.where(denom.abs() > 1e-12, num / denom, np.nan)

    return combined


def plot_effect_signature(
    analysis_frame: pd.DataFrame,
    *,
    x: str = "delta_peak_time_hours",
    y: str = "delta_peak_discharge_cms",
    color: str = "scenario_family",
    hover_name: str = "state_id",
    size: str | None = None,
    title: str = "Hydrograph effect signature",
):
    import plotly.express as px

    frame = analysis_frame.copy()
    if size is None:
        if "selected_unit_count" in frame.columns:
            size = "selected_unit_count"
        elif "bubble_unit_count" in frame.columns:
            size = "bubble_unit_count"
    plot_kwargs = {
        "x": x,
        "y": y,
        "color": color,
        "hover_name": hover_name,
        "title": title,
    }
    if size is not None and size in frame.columns:
        plot_kwargs["size"] = size
    return px.scatter(frame, **plot_kwargs)


def plot_hydrograph_overlay(
    run_registry_or_experiment_dir: pd.DataFrame | str | Path,
    state_ids: list[str],
    *,
    include_inflow: bool = True,
    reference_state_id: str | None = "S000_base",
    title: str | None = None,
):
    import plotly.graph_objects as go

    if isinstance(run_registry_or_experiment_dir, pd.DataFrame):
        run_registry = run_registry_or_experiment_dir
    else:
        run_registry = pd.read_csv(_as_path(run_registry_or_experiment_dir) / "rapid_run_registry.csv")

    ordered_state_ids = list(state_ids)
    if reference_state_id is not None and reference_state_id not in ordered_state_ids:
        ordered_state_ids = [reference_state_id] + ordered_state_ids

    fig = go.Figure()
    inflow_added = False
    for state_id in ordered_state_ids:
        hydro = load_state_hydrograph(run_registry, state_id)
        if include_inflow and not inflow_added:
            fig.add_trace(
                go.Scatter(
                    x=hydro["time_utc"],
                    y=hydro["q_inflow_cms"],
                    mode="lines",
                    name="Inflow",
                    line={"color": "black", "dash": "dash"},
                )
            )
            inflow_added = True
        fig.add_trace(
            go.Scatter(
                x=hydro["time_utc"],
                y=hydro["q_outlet_cms"],
                mode="lines",
                name=state_id,
            )
        )
    fig.update_layout(
        title=title or "Outlet hydrograph comparison",
        xaxis_title="time",
        yaxis_title="discharge (cms)",
    )
    return fig


def _resolve_reach_attributes_gpkg(prep_dir: Path) -> Path:
    manifest = _load_json_if_exists(prep_dir / "rapid_prep_manifest.json")
    if manifest is not None:
        paths = manifest.get("paths", {})
        gpkg = paths.get("rapid_reach_attributes_gpkg")
        if gpkg:
            path = _as_path(gpkg)
            if path.exists():
                return path
    fallback = prep_dir / "rapid_reach_attributes.gpkg"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "RAPID reach geometry export was not found. Re-run RAPID prep with the updated code so "
        "'rapid_reach_attributes.gpkg' is written."
    )


def load_reach_wave_stack(prep_dir: str | Path, qout_nc: str | Path) -> ReachWaveStack:
    import geopandas as gpd

    prep_path = _as_path(prep_dir)
    qout_path = _as_path(qout_nc)
    reaches = gpd.read_file(_resolve_reach_attributes_gpkg(prep_path))
    reaches["reach_id"] = pd.to_numeric(reaches["reach_id"], errors="coerce").astype(int)
    river_ids, time_seconds, qout = _load_qout(qout_path)
    reach_index = reaches.set_index("reach_id", drop=False)
    missing = [int(river_id) for river_id in river_ids if int(river_id) not in reach_index.index]
    if missing:
        raise ValueError(f"Reach geometry is missing RAPID reach IDs present in Qout: {missing[:10]}")
    reaches_ordered = reach_index.loc[river_ids.astype(int)].reset_index(drop=True)
    return ReachWaveStack(
        reaches=reaches_ordered,
        time_seconds=np.asarray(time_seconds, dtype=float),
        reach_ids=np.asarray(river_ids, dtype=int),
        qout=np.asarray(qout, dtype=float),
    )


def compute_wave_intensity_matrix(
    qout: np.ndarray,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> np.ndarray:
    values = np.asarray(qout, dtype=float)
    if values.ndim != 2:
        raise ValueError("qout must be a 2D array of shape (time, reach).")
    if end_index is None:
        end_index = values.shape[0] - 1
    start_index = max(0, int(start_index))
    end_index = min(values.shape[0] - 1, int(end_index))
    if end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index.")

    baseline = values[start_index, :]
    window = values[start_index : end_index + 1, :]
    peaks = np.max(window, axis=0)
    denominator = peaks - baseline
    intensity = np.zeros_like(window, dtype=float)
    valid = denominator > 0
    if np.any(valid):
        intensity[:, valid] = (window[:, valid] - baseline[valid]) / denominator[valid]
    return np.clip(intensity, 0.0, 1.0)


def _geometry_segments(geometry) -> list[np.ndarray]:
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "LineString":
        return [np.asarray(geometry.coords)]
    if geom_type == "MultiLineString":
        return [np.asarray(part.coords) for part in geometry.geoms]
    return []


def _resolve_event_window_indices(run_row: pd.Series, time_seconds: np.ndarray) -> tuple[int, int]:
    start_seconds = pd.to_numeric(run_row.get("event_start_time_seconds"), errors="coerce")
    end_seconds = pd.to_numeric(run_row.get("event_end_time_seconds"), errors="coerce")
    if pd.isna(start_seconds):
        start_seconds = float(time_seconds[0])
    if pd.isna(end_seconds):
        end_seconds = float(time_seconds[-1])
    start_index = int(np.searchsorted(time_seconds, float(start_seconds), side="left"))
    end_index = int(np.searchsorted(time_seconds, float(end_seconds), side="right")) - 1
    start_index = max(0, min(start_index, len(time_seconds) - 1))
    end_index = max(start_index, min(end_index, len(time_seconds) - 1))
    return start_index, end_index


def _load_wave_state(experiment_dir: str | Path, state_id: str) -> _LoadedWaveState:
    experiment_path = _as_path(experiment_dir)
    run_registry = pd.read_csv(experiment_path / "rapid_run_registry.csv")
    row = run_registry.loc[run_registry["state_id"].eq(state_id)]
    if row.empty:
        raise KeyError(f"State '{state_id}' was not found in {experiment_path / 'rapid_run_registry.csv'}.")
    run_row = row.iloc[0]

    prep_dir = _as_path(run_row["rapid_prep_dir"])
    qout_path = _as_path(run_row["qout_nc"])
    wave_stack = load_reach_wave_stack(prep_dir, qout_path)
    outlet_hydro = load_state_hydrograph(run_registry, state_id)
    return _LoadedWaveState(
        experiment_path=experiment_path,
        run_registry=run_registry,
        run_row=run_row,
        wave_stack=wave_stack,
        outlet_hydro=outlet_hydro,
    )


def _compute_line_widths(reaches: Any, linewidth_field: str) -> np.ndarray:
    if linewidth_field in reaches.columns:
        width_values = pd.to_numeric(reaches[linewidth_field], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        width_values = np.ones(len(reaches), dtype=float)
    width_min = float(np.nanmin(width_values)) if len(width_values) else 1.0
    width_max = float(np.nanmax(width_values)) if len(width_values) else 1.0
    if width_max > width_min:
        return 0.75 + 3.25 * ((width_values - width_min) / (width_max - width_min))
    return np.full(len(width_values), 2.0, dtype=float)


@dataclass(frozen=True, slots=True)
class _WaveRenderState:
    wave_stack: ReachWaveStack
    outlet_hydro: pd.DataFrame
    run_row: pd.Series
    reaches: Any
    frame_indices: list[int]
    start_index: int
    intensity: np.ndarray
    line_widths: np.ndarray
    bounds: np.ndarray
    cmap_name: str


@dataclass(frozen=True, slots=True)
class _WaveComparisonRenderState:
    reference_state_id: str
    target_state_id: str
    reference_reaches: Any
    target_reaches: Any
    reference_intensity: np.ndarray
    target_intensity: np.ndarray
    difference_intensity: np.ndarray
    frame_positions: list[int]
    frame_times: np.ndarray
    reference_line_widths: np.ndarray
    target_line_widths: np.ndarray
    full_bounds: np.ndarray
    zoom_bounds: np.ndarray
    cmap_name: str
    difference_cmap_name: str
    peak_shift_hours: float


def _build_wave_render_state(
    experiment_dir: str | Path,
    state_id: str,
    *,
    frame_stride: int,
    linewidth_field: str,
    use_event_window: bool,
    cmap_name: str,
) -> _WaveRenderState:
    loaded = _load_wave_state(experiment_dir, state_id)
    run_row = loaded.run_row
    wave_stack = loaded.wave_stack
    outlet_hydro = loaded.outlet_hydro

    if use_event_window:
        start_index, end_index = _resolve_event_window_indices(run_row, wave_stack.time_seconds)
    else:
        start_index, end_index = 0, len(wave_stack.time_seconds) - 1

    intensity = compute_wave_intensity_matrix(
        wave_stack.qout,
        start_index=start_index,
        end_index=end_index,
    )
    frame_indices = list(range(start_index, end_index + 1, max(1, int(frame_stride))))
    if frame_indices[-1] != end_index:
        frame_indices.append(end_index)

    reaches = wave_stack.reaches.copy()
    line_widths = _compute_line_widths(reaches, linewidth_field)

    return _WaveRenderState(
        wave_stack=wave_stack,
        outlet_hydro=outlet_hydro,
        run_row=run_row,
        reaches=reaches,
        frame_indices=frame_indices,
        start_index=start_index,
        intensity=intensity,
        line_widths=line_widths,
        bounds=reaches.total_bounds,
        cmap_name=cmap_name,
    )


def _align_time_indices(
    reference_time_seconds: np.ndarray,
    target_time_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_times = np.asarray(reference_time_seconds, dtype=float)
    target_times = np.asarray(target_time_seconds, dtype=float)
    if reference_times.shape == target_times.shape and np.allclose(reference_times, target_times):
        indices = np.arange(reference_times.size, dtype=int)
        return reference_times, indices, indices.copy()
    common_time, reference_indices, target_indices = np.intersect1d(
        reference_times,
        target_times,
        assume_unique=False,
        return_indices=True,
    )
    if common_time.size == 0:
        raise ValueError("Reference and target states do not share any common RAPID timesteps.")
    return common_time.astype(float), reference_indices.astype(int), target_indices.astype(int)


def _aggregate_parent_series(reaches: Any, values: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if "parent_link_id" not in reaches.columns:
        return np.full(len(reaches), np.nan, dtype=float), {}
    parent_ids = pd.to_numeric(reaches["parent_link_id"], errors="coerce").to_numpy(dtype=float)
    lookup: dict[int, np.ndarray] = {}
    finite_parent_ids = np.unique(parent_ids[np.isfinite(parent_ids)]).astype(int)
    for parent_link_id in finite_parent_ids.tolist():
        columns = np.flatnonzero(parent_ids == parent_link_id)
        if columns.size == 0:
            continue
        lookup[int(parent_link_id)] = np.nanmean(values[:, columns], axis=1)
    return parent_ids, lookup


def _aggregate_stable_parent_series(
    reaches: Any,
    values: np.ndarray,
    *,
    stable_field: str = "dominant_parent_link_id",
    lineage_field: str = "lineage_type",
    required_lineage: str = "unchanged_1to1",
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if stable_field not in reaches.columns:
        return np.full(len(reaches), np.nan, dtype=float), {}
    stable_parent_ids = pd.to_numeric(reaches[stable_field], errors="coerce").to_numpy(dtype=float)
    if lineage_field in reaches.columns:
        lineage_mask = reaches[lineage_field].astype(str).eq(required_lineage).to_numpy(dtype=bool)
    else:
        lineage_mask = np.ones(len(reaches), dtype=bool)

    lookup: dict[int, np.ndarray] = {}
    finite_parent_ids = np.unique(stable_parent_ids[np.isfinite(stable_parent_ids)]).astype(int)
    for parent_link_id in finite_parent_ids.tolist():
        columns = np.flatnonzero((stable_parent_ids == parent_link_id) & lineage_mask)
        if columns.size == 0:
            continue
        lookup[int(parent_link_id)] = np.nanmean(values[:, columns], axis=1)
    stable_parent_ids = np.where(lineage_mask, stable_parent_ids, np.nan)
    return stable_parent_ids, lookup


def _summarize_parent_dist_out(
    reaches: Any,
    *,
    parent_field: str,
    lineage_field: str | None = None,
    required_lineage: str | None = None,
    prefix: str,
) -> pd.DataFrame:
    if parent_field not in reaches.columns:
        return pd.DataFrame(columns=["parent_link_id"])
    working = reaches.copy()
    working[parent_field] = pd.to_numeric(working[parent_field], errors="coerce")
    if lineage_field is not None and required_lineage is not None and lineage_field in working.columns:
        working = working.loc[working[lineage_field].astype(str).eq(required_lineage)].copy()
    working = working.loc[working[parent_field].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["parent_link_id"])

    group = working.groupby(parent_field, sort=True)
    summary = pd.DataFrame({"parent_link_id": [int(value) for value in group.size().index.tolist()]})
    for source_column, target_suffix in [
        ("sword_us_dist_out", "dist_out_us_m"),
        ("sword_ds_dist_out", "dist_out_ds_m"),
    ]:
        if source_column in working.columns:
            values = group[source_column].mean().reset_index(drop=True)
            summary[f"{prefix}_{target_suffix}"] = pd.to_numeric(values, errors="coerce")
    us_column = f"{prefix}_dist_out_us_m"
    ds_column = f"{prefix}_dist_out_ds_m"
    if us_column in summary.columns and ds_column in summary.columns:
        summary[f"{prefix}_dist_out_mid_m"] = (summary[us_column] + summary[ds_column]) / 2.0
        summary[f"{prefix}_dist_out_mid_km"] = summary[f"{prefix}_dist_out_mid_m"] / 1000.0
    return summary


def _compute_shared_scale_difference(
    reference_series: np.ndarray,
    target_series: np.ndarray,
    *,
    activation_fraction: float,
) -> np.ndarray:
    reference = np.asarray(reference_series, dtype=float)
    target = np.asarray(target_series, dtype=float)
    if reference.shape != target.shape:
        raise ValueError("Reference and target series must share the same shape for difference rendering.")
    baseline = float(min(reference[0], target[0]))
    peak = float(max(np.nanmax(reference), np.nanmax(target)))
    scale = peak - baseline
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(target, dtype=float)
    activation_level = activation_fraction * scale
    active = np.maximum(reference - baseline, target - baseline) >= activation_level
    difference = (target - reference) / scale
    difference = np.where(active, difference, 0.0)
    return np.clip(difference, -1.0, 1.0)


def _compute_first_significant_difference_time(
    reference_series: np.ndarray,
    target_series: np.ndarray,
    time_seconds: np.ndarray,
    *,
    difference_fraction: float,
) -> float:
    reference = np.asarray(reference_series, dtype=float)
    target = np.asarray(target_series, dtype=float)
    times = np.asarray(time_seconds, dtype=float)
    baseline = float(min(reference[0], target[0]))
    peak = float(max(np.nanmax(reference), np.nanmax(target)))
    scale = peak - baseline
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    threshold = difference_fraction * scale
    activation = np.maximum(reference - baseline, target - baseline) >= threshold
    significant = activation & (np.abs(target - reference) >= threshold)
    if not np.any(significant):
        return float("nan")
    return float(times[int(np.argmax(significant))])


def _build_common_wave_window(
    reference: _LoadedWaveState,
    target: _LoadedWaveState,
    *,
    use_event_window: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_time, reference_indices, target_indices = _align_time_indices(
        reference.wave_stack.time_seconds,
        target.wave_stack.time_seconds,
    )

    if use_event_window:
        reference_start_index, reference_end_index = _resolve_event_window_indices(reference.run_row, reference.wave_stack.time_seconds)
        target_start_index, target_end_index = _resolve_event_window_indices(target.run_row, target.wave_stack.time_seconds)
        start_seconds = min(
            float(reference.wave_stack.time_seconds[reference_start_index]),
            float(target.wave_stack.time_seconds[target_start_index]),
        )
        end_seconds = max(
            float(reference.wave_stack.time_seconds[reference_end_index]),
            float(target.wave_stack.time_seconds[target_end_index]),
        )
    else:
        start_seconds = float(common_time[0])
        end_seconds = float(common_time[-1])

    time_mask = (common_time >= start_seconds) & (common_time <= end_seconds)
    if not np.any(time_mask):
        raise ValueError("Reference and target states do not overlap on the requested event window.")

    window_time = common_time[time_mask]
    reference_qout_window = np.asarray(reference.wave_stack.qout[reference_indices, :], dtype=float)[time_mask, :]
    target_qout_window = np.asarray(target.wave_stack.qout[target_indices, :], dtype=float)[time_mask, :]
    return window_time, reference_qout_window, target_qout_window


def _load_state_directed_links(state_dir: Path):
    import geopandas as gpd

    directed_dir = state_dir / "variant" / "directed"
    return gpd.read_file(_resolve_single_file(directed_dir, "*.gpkg"))


def _load_collapse_parent_link_ids(state_dir: Path) -> tuple[int, ...]:
    collapse_components = _load_csv_if_exists(state_dir / "variant" / "summary" / "collapse_components.csv")
    if collapse_components is None or collapse_components.empty or "link_ids" not in collapse_components.columns:
        return ()
    link_ids: list[int] = []
    for value in collapse_components["link_ids"].tolist():
        link_ids.extend(_parse_int_list(value))
    return tuple(sorted(set(link_ids)))


def _select_matched_parent_links_around_change(
    base_edges: pd.DataFrame,
    *,
    changed_parent_link_ids: set[int],
    matched_parent_link_ids: set[int],
    n_upstream: int,
    n_downstream: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    working = base_edges.copy()
    working["id_link"] = pd.to_numeric(working["id_link"], errors="coerce").astype("Int64")
    working["id_us_node"] = pd.to_numeric(working["id_us_node"], errors="coerce").astype("Int64")
    working["id_ds_node"] = pd.to_numeric(working["id_ds_node"], errors="coerce").astype("Int64")
    working = working.loc[working["id_link"].notna() & working["id_us_node"].notna() & working["id_ds_node"].notna()].copy()

    edge_by_id = {
        int(row.id_link): (int(row.id_us_node), int(row.id_ds_node))
        for row in working.itertuples(index=False)
    }
    incoming_by_node: dict[int, list[int]] = {}
    outgoing_by_node: dict[int, list[int]] = {}
    for link_id, (us_node, ds_node) in edge_by_id.items():
        outgoing_by_node.setdefault(us_node, []).append(link_id)
        incoming_by_node.setdefault(ds_node, []).append(link_id)

    changed_present = {link_id for link_id in changed_parent_link_ids if link_id in edge_by_id}
    if not changed_present:
        return [], []

    changed_upstream_nodes = {edge_by_id[link_id][0] for link_id in changed_present}
    changed_downstream_nodes = {edge_by_id[link_id][1] for link_id in changed_present}
    upstream_frontier_nodes = changed_upstream_nodes - changed_downstream_nodes
    downstream_frontier_nodes = changed_downstream_nodes - changed_upstream_nodes
    if not upstream_frontier_nodes:
        upstream_frontier_nodes = changed_upstream_nodes
    if not downstream_frontier_nodes:
        downstream_frontier_nodes = changed_downstream_nodes

    def walk(start_nodes: set[int], *, reverse: bool, limit: int) -> list[dict[str, int]]:
        queue: deque[tuple[int, int]] = deque((node_id, 0) for node_id in sorted(start_nodes))
        visited_nodes = set(start_nodes)
        selected: list[dict[str, int]] = []
        seen_links: set[int] = set()
        while queue and len(selected) < max(0, int(limit)):
            node_id, distance = queue.popleft()
            candidate_links = incoming_by_node.get(node_id, []) if reverse else outgoing_by_node.get(node_id, [])
            for link_id in candidate_links:
                next_node = edge_by_id[link_id][0] if reverse else edge_by_id[link_id][1]
                next_distance = distance + 1
                if link_id not in changed_present and link_id in matched_parent_link_ids and link_id not in seen_links:
                    selected.append(
                        {
                            "parent_link_id": int(link_id),
                            "graph_distance": int(next_distance),
                        }
                    )
                    seen_links.add(link_id)
                    if len(selected) >= int(limit):
                        break
                if next_node not in visited_nodes:
                    visited_nodes.add(next_node)
                    queue.append((next_node, next_distance))
        return selected

    upstream = walk(upstream_frontier_nodes, reverse=True, limit=n_upstream)
    downstream = walk(downstream_frontier_nodes, reverse=False, limit=n_downstream)
    return upstream, downstream


def compute_reach_timing_metrics(
    qout: np.ndarray,
    time_seconds: np.ndarray,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    threshold_fractions: tuple[float, ...] = (0.1, 0.5),
) -> pd.DataFrame:
    values = np.asarray(qout, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("qout must be a 1D or 2D array with time on axis 0.")
    times = np.asarray(time_seconds, dtype=float)
    if times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ValueError("time_seconds must be 1D and aligned with qout along axis 0.")
    if end_index is None:
        end_index = values.shape[0] - 1
    start_index = max(0, int(start_index))
    end_index = min(values.shape[0] - 1, int(end_index))
    if end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index.")

    window = values[start_index : end_index + 1, :]
    window_time = times[start_index : end_index + 1]
    baseline = window[0, :]
    peaks = np.nanmax(window, axis=0)
    amplitudes = peaks - baseline
    t_peak = np.full(window.shape[1], np.nan, dtype=float)
    metric_columns = {
        f"t{int(round(fraction * 100)):02d}_seconds": np.full(window.shape[1], np.nan, dtype=float)
        for fraction in threshold_fractions
    }

    for column_index in range(window.shape[1]):
        column = window[:, column_index]
        if not np.any(np.isfinite(column)):
            continue
        peak_index = int(np.nanargmax(column))
        t_peak[column_index] = float(window_time[peak_index])
        amplitude = amplitudes[column_index]
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        for fraction, metric_name in zip(threshold_fractions, metric_columns.keys(), strict=False):
            threshold = baseline[column_index] + float(fraction) * amplitude
            hit_indices = np.flatnonzero(column >= threshold)
            if hit_indices.size:
                metric_columns[metric_name][column_index] = float(window_time[int(hit_indices[0])])

    data: dict[str, Any] = {
        "baseline_discharge_cms": baseline,
        "peak_discharge_cms": peaks,
        "amplitude_discharge_cms": amplitudes,
        "t_peak_seconds": t_peak,
    }
    data.update(metric_columns)
    return pd.DataFrame(data)


def _expand_bounds(
    bounds: np.ndarray,
    full_bounds: np.ndarray,
    *,
    margin_fraction: float = 0.08,
    min_fraction: float = 0.28,
) -> np.ndarray:
    full_bounds = np.asarray(full_bounds, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    full_width = max(float(full_bounds[2] - full_bounds[0]), 1.0)
    full_height = max(float(full_bounds[3] - full_bounds[1]), 1.0)
    width = max(float(bounds[2] - bounds[0]), 0.0)
    height = max(float(bounds[3] - bounds[1]), 0.0)
    target_width = min(full_width, max(width * (1.0 + 2.0 * margin_fraction), full_width * min_fraction))
    target_height = min(full_height, max(height * (1.0 + 2.0 * margin_fraction), full_height * min_fraction))
    center_x = float((bounds[0] + bounds[2]) / 2.0)
    center_y = float((bounds[1] + bounds[3]) / 2.0)

    if target_width >= full_width:
        x0, x1 = float(full_bounds[0]), float(full_bounds[2])
    else:
        x0 = max(float(full_bounds[0]), min(center_x - target_width / 2.0, float(full_bounds[2] - target_width)))
        x1 = x0 + target_width
    if target_height >= full_height:
        y0, y1 = float(full_bounds[1]), float(full_bounds[3])
    else:
        y0 = max(float(full_bounds[1]), min(center_y - target_height / 2.0, float(full_bounds[3] - target_height)))
        y1 = y0 + target_height
    return np.asarray([x0, y0, x1, y1], dtype=float)


def _derive_comparison_zoom_bounds(
    target_reaches: Any,
    difference_intensity: np.ndarray,
    full_bounds: np.ndarray,
    *,
    zoom_bounds: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if zoom_bounds is not None:
        return _expand_bounds(np.asarray(zoom_bounds, dtype=float), np.asarray(full_bounds, dtype=float))
    if len(target_reaches) == 0 or difference_intensity.size == 0:
        return np.asarray(full_bounds, dtype=float)

    selected_mask = np.zeros(len(target_reaches), dtype=bool)
    if "parent_link_id" in target_reaches.columns:
        parent_ids = pd.to_numeric(target_reaches["parent_link_id"], errors="coerce").to_numpy(dtype=float)
        finite_parent_ids = np.unique(parent_ids[np.isfinite(parent_ids)]).astype(int)
        parent_scores: list[tuple[float, int]] = []
        for parent_link_id in finite_parent_ids.tolist():
            columns = np.flatnonzero(parent_ids == parent_link_id)
            if columns.size == 0:
                continue
            score = float(np.nanmax(np.abs(difference_intensity[:, columns])))
            if np.isfinite(score):
                parent_scores.append((score, parent_link_id))
        parent_scores.sort(key=lambda item: item[0], reverse=True)
        for _score, parent_link_id in parent_scores[:5]:
            selected_mask |= parent_ids == parent_link_id

    if not np.any(selected_mask):
        reach_scores = np.nanmax(np.abs(difference_intensity), axis=0)
        if np.all(~np.isfinite(reach_scores)):
            return np.asarray(full_bounds, dtype=float)
        top_count = min(12, len(reach_scores))
        top_indices = np.argsort(np.nan_to_num(reach_scores, nan=-np.inf))[-top_count:]
        selected_mask[top_indices] = True

    selected_reaches = target_reaches.loc[selected_mask]
    if selected_reaches.empty:
        return np.asarray(full_bounds, dtype=float)
    return _expand_bounds(selected_reaches.total_bounds, np.asarray(full_bounds, dtype=float))


def _build_wave_comparison_render_state(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str,
    frame_stride: int,
    linewidth_field: str,
    use_event_window: bool,
    cmap_name: str,
    difference_cmap_name: str,
    zoom_bounds: tuple[float, float, float, float] | list[float] | np.ndarray | None,
    difference_activation_fraction: float,
) -> _WaveComparisonRenderState:
    reference = _load_wave_state(experiment_dir, reference_state_id)
    target = _load_wave_state(experiment_dir, state_id)
    window_time, reference_qout_window, target_qout_window = _build_common_wave_window(
        reference,
        target,
        use_event_window=use_event_window,
    )
    reference_intensity = compute_wave_intensity_matrix(reference_qout_window, start_index=0, end_index=len(window_time) - 1)
    target_intensity = compute_wave_intensity_matrix(target_qout_window, start_index=0, end_index=len(window_time) - 1)

    frame_positions = list(range(0, len(window_time), max(1, int(frame_stride))))
    if frame_positions[-1] != len(window_time) - 1:
        frame_positions.append(len(window_time) - 1)
    frame_times = window_time[np.asarray(frame_positions, dtype=int)]

    reference_reaches = reference.wave_stack.reaches.copy()
    target_reaches = target.wave_stack.reaches.copy()
    reference_line_widths = _compute_line_widths(reference_reaches, linewidth_field)
    target_line_widths = _compute_line_widths(target_reaches, linewidth_field)

    target_parent_ids, target_parent_lookup = _aggregate_parent_series(target_reaches, target_qout_window)
    _reference_parent_ids, reference_parent_lookup = _aggregate_parent_series(reference_reaches, reference_qout_window)
    difference_intensity = np.full_like(target_intensity, np.nan, dtype=float)
    zero_series = np.zeros(target_qout_window.shape[0], dtype=float)
    for reach_index, parent_link_id in enumerate(target_parent_ids.tolist()):
        if np.isfinite(parent_link_id):
            parent_key = int(parent_link_id)
            target_series = target_parent_lookup.get(parent_key, target_qout_window[:, reach_index])
            reference_series = reference_parent_lookup.get(parent_key)
            if reference_series is None:
                reference_series = np.full_like(target_series, float(target_series[0]))
        else:
            target_series = target_qout_window[:, reach_index]
            reference_series = np.full_like(target_series, float(target_series[0]))
        difference_intensity[:, reach_index] = _compute_shared_scale_difference(
            reference_series,
            target_series,
            activation_fraction=difference_activation_fraction,
        )

    reference_bounds = np.asarray(reference_reaches.total_bounds, dtype=float)
    target_bounds = np.asarray(target_reaches.total_bounds, dtype=float)
    full_bounds = np.asarray(
        [
            min(reference_bounds[0], target_bounds[0]),
            min(reference_bounds[1], target_bounds[1]),
            max(reference_bounds[2], target_bounds[2]),
            max(reference_bounds[3], target_bounds[3]),
        ],
        dtype=float,
    )
    zoom_bounds_array = _derive_comparison_zoom_bounds(
        target_reaches,
        difference_intensity,
        full_bounds,
        zoom_bounds=zoom_bounds,
    )
    peak_shift_hours = pd.to_numeric(target.run_row.get("peak_time_seconds"), errors="coerce") - pd.to_numeric(
        reference.run_row.get("peak_time_seconds"), errors="coerce"
    )
    peak_shift_hours = float(peak_shift_hours / SECONDS_PER_HOUR) if pd.notna(peak_shift_hours) else float("nan")

    return _WaveComparisonRenderState(
        reference_state_id=reference_state_id,
        target_state_id=state_id,
        reference_reaches=reference_reaches,
        target_reaches=target_reaches,
        reference_intensity=reference_intensity,
        target_intensity=target_intensity,
        difference_intensity=difference_intensity,
        frame_positions=frame_positions,
        frame_times=frame_times,
        reference_line_widths=reference_line_widths,
        target_line_widths=target_line_widths,
        full_bounds=full_bounds,
        zoom_bounds=zoom_bounds_array,
        cmap_name=cmap_name,
        difference_cmap_name=difference_cmap_name,
        peak_shift_hours=peak_shift_hours,
    )


def _plot_wave_panel(
    ax: Any,
    reaches: Any,
    values: np.ndarray,
    line_widths: np.ndarray,
    *,
    bounds: np.ndarray,
    cmap: Any,
    norm: Any,
    title: str,
) -> None:
    missing_color = "#d9d9d9"
    for idx, geometry in enumerate(reaches.geometry):
        value = float(values[idx]) if idx < len(values) else float("nan")
        color = cmap(norm(value)) if np.isfinite(value) else missing_color
        for segment in _geometry_segments(geometry):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=float(line_widths[idx]),
                solid_capstyle="round",
            )
    ax.set_aspect("equal")
    ax.set_xlim(float(bounds[0]), float(bounds[2]))
    ax.set_ylim(float(bounds[1]), float(bounds[3]))
    ax.set_axis_off()
    ax.set_title(title)


def _render_wave_comparison_frames(
    render_state: _WaveComparisonRenderState,
    frame_dir: Path,
    *,
    dpi: int = 120,
) -> list[Path]:
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    frame_dir.mkdir(parents=True, exist_ok=True)
    base_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    difference_norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    base_cmap = plt.get_cmap(render_state.cmap_name)
    difference_cmap = plt.get_cmap(render_state.difference_cmap_name)
    zoom_bounds = np.asarray(render_state.zoom_bounds, dtype=float)

    frame_paths: list[Path] = []
    for frame_position, local_index in enumerate(render_state.frame_positions):
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
        current_seconds = float(render_state.frame_times[frame_position])

        _plot_wave_panel(
            axes[0, 0],
            render_state.reference_reaches,
            render_state.reference_intensity[local_index, :],
            render_state.reference_line_widths,
            bounds=render_state.full_bounds,
            cmap=base_cmap,
            norm=base_norm,
            title=render_state.reference_state_id,
        )
        _plot_wave_panel(
            axes[0, 1],
            render_state.target_reaches,
            render_state.target_intensity[local_index, :],
            render_state.target_line_widths,
            bounds=render_state.full_bounds,
            cmap=base_cmap,
            norm=base_norm,
            title=render_state.target_state_id,
        )
        _plot_wave_panel(
            axes[1, 0],
            render_state.target_reaches,
            render_state.difference_intensity[local_index, :],
            render_state.target_line_widths,
            bounds=render_state.full_bounds,
            cmap=difference_cmap,
            norm=difference_norm,
            title="Difference (variant - base)",
        )
        _plot_wave_panel(
            axes[1, 1],
            render_state.target_reaches,
            render_state.difference_intensity[local_index, :],
            render_state.target_line_widths,
            bounds=zoom_bounds,
            cmap=difference_cmap,
            norm=difference_norm,
            title="Zoomed difference",
        )

        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.add_patch(
                mpatches.Rectangle(
                    (float(zoom_bounds[0]), float(zoom_bounds[1])),
                    float(zoom_bounds[2] - zoom_bounds[0]),
                    float(zoom_bounds[3] - zoom_bounds[1]),
                    fill=False,
                    linewidth=1.1,
                    linestyle="--",
                    edgecolor="black",
                )
            )

        if np.isfinite(render_state.peak_shift_hours):
            subtitle = f"t = {current_seconds / SECONDS_PER_HOUR:.2f} h | Δpeak vs base = {render_state.peak_shift_hours:+.2f} h"
        else:
            subtitle = f"t = {current_seconds / SECONDS_PER_HOUR:.2f} h"
        fig.suptitle(f"{render_state.target_state_id} vs {render_state.reference_state_id}\n{subtitle}")

        frame_path = frame_dir / f"frame_{frame_position:04d}.png"
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def build_reach_timing_comparison(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str = "S000_base",
    use_event_window: bool = True,
    threshold_fractions: tuple[float, ...] = (0.1, 0.5),
    difference_fraction: float = 0.05,
):
    import geopandas as gpd

    reference = _load_wave_state(experiment_dir, reference_state_id)
    target = _load_wave_state(experiment_dir, state_id)
    window_time, reference_qout_window, target_qout_window = _build_common_wave_window(
        reference,
        target,
        use_event_window=use_event_window,
    )

    _reference_parent_ids, reference_parent_lookup = _aggregate_parent_series(reference.wave_stack.reaches, reference_qout_window)
    target_parent_ids, target_parent_lookup = _aggregate_parent_series(target.wave_stack.reaches, target_qout_window)

    target_reaches = target.wave_stack.reaches.copy()
    target_reaches["parent_link_id"] = pd.to_numeric(target_reaches["parent_link_id"], errors="coerce")
    parent_geometry = (
        target_reaches.loc[target_reaches["parent_link_id"].notna(), ["parent_link_id", "geometry"]]
        .copy()
        .dissolve(by="parent_link_id", as_index=False)
    )
    parent_geometry["parent_link_id"] = pd.to_numeric(parent_geometry["parent_link_id"], errors="coerce").astype(int)

    metric_suffixes = [f"t{int(round(fraction * 100)):02d}" for fraction in threshold_fractions]
    rows: list[dict[str, Any]] = []
    for parent_link_id in parent_geometry["parent_link_id"].tolist():
        target_series = target_parent_lookup.get(int(parent_link_id))
        reference_series = reference_parent_lookup.get(int(parent_link_id))
        if target_series is None:
            continue
        target_metrics = compute_reach_timing_metrics(
            target_series,
            window_time,
            threshold_fractions=threshold_fractions,
        ).iloc[0]
        if reference_series is not None:
            reference_metrics = compute_reach_timing_metrics(
                reference_series,
                window_time,
                threshold_fractions=threshold_fractions,
            ).iloc[0]
            first_difference_time_seconds = _compute_first_significant_difference_time(
                reference_series,
                target_series,
                window_time,
                difference_fraction=difference_fraction,
            )
        else:
            reference_metrics = pd.Series(dtype=float)
            first_difference_time_seconds = float("nan")

        row: dict[str, Any] = {
            "parent_link_id": int(parent_link_id),
            "first_significant_difference_time_seconds": first_difference_time_seconds,
            "first_significant_difference_time_hours": (
                first_difference_time_seconds / SECONDS_PER_HOUR if np.isfinite(first_difference_time_seconds) else float("nan")
            ),
            "reference_peak_discharge_cms": pd.to_numeric(reference_metrics.get("peak_discharge_cms"), errors="coerce"),
            "variant_peak_discharge_cms": pd.to_numeric(target_metrics.get("peak_discharge_cms"), errors="coerce"),
            "delta_peak_discharge_cms": (
                pd.to_numeric(target_metrics.get("peak_discharge_cms"), errors="coerce")
                - pd.to_numeric(reference_metrics.get("peak_discharge_cms"), errors="coerce")
            ),
        }
        for suffix in metric_suffixes:
            reference_value = pd.to_numeric(reference_metrics.get(f"{suffix}_seconds"), errors="coerce")
            target_value = pd.to_numeric(target_metrics.get(f"{suffix}_seconds"), errors="coerce")
            row[f"reference_{suffix}_seconds"] = reference_value
            row[f"variant_{suffix}_seconds"] = target_value
            row[f"delta_{suffix}_hours"] = (
                (target_value - reference_value) / SECONDS_PER_HOUR if pd.notna(reference_value) and pd.notna(target_value) else float("nan")
            )
        reference_peak_time = pd.to_numeric(reference_metrics.get("t_peak_seconds"), errors="coerce")
        target_peak_time = pd.to_numeric(target_metrics.get("t_peak_seconds"), errors="coerce")
        row["reference_t_peak_seconds"] = reference_peak_time
        row["variant_t_peak_seconds"] = target_peak_time
        row["delta_t_peak_hours"] = (
            (target_peak_time - reference_peak_time) / SECONDS_PER_HOUR
            if pd.notna(reference_peak_time) and pd.notna(target_peak_time)
            else float("nan")
        )
        rows.append(row)

    metrics_frame = pd.DataFrame(rows)
    comparison = gpd.GeoDataFrame(parent_geometry.merge(metrics_frame, on="parent_link_id", how="left"), geometry="geometry", crs=parent_geometry.crs)
    comparison["reference_state_id"] = reference_state_id
    comparison["variant_state_id"] = state_id
    return comparison


def plot_reach_timing_map(
    timing_comparison,
    *,
    column: str = "delta_t10_hours",
    title: str | None = None,
    cmap: str = "RdBu_r",
    zoom_bounds: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
    linewidth: float = 2.0,
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    frame = timing_comparison.copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 7))
    frame.plot(ax=ax, color="#d9d9d9", linewidth=0.8)
    valid = frame.loc[values.notna()].copy()
    if not valid.empty:
        valid_values = pd.to_numeric(valid[column], errors="coerce")
        value_min = float(valid_values.min())
        value_max = float(valid_values.max())
        if value_min < 0 < value_max:
            norm = mcolors.TwoSlopeNorm(vmin=value_min, vcenter=0.0, vmax=value_max)
        else:
            norm = mcolors.Normalize(vmin=value_min, vmax=value_max)
        valid.plot(ax=ax, column=column, cmap=cmap, linewidth=linewidth, legend=True, norm=norm)
    if zoom_bounds is not None:
        bounds = np.asarray(zoom_bounds, dtype=float)
        ax.set_xlim(float(bounds[0]), float(bounds[2]))
        ax.set_ylim(float(bounds[1]), float(bounds[3]))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or column)
    fig.tight_layout()
    return fig, ax


def build_parent_link_difference_timeseries(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str = "S000_base",
    use_event_window: bool = True,
    difference_fraction: float = 0.05,
    parent_link_ids: list[int] | tuple[int, ...] | None = None,
    top_n: int = 12,
) -> pd.DataFrame:
    reference = _load_wave_state(experiment_dir, reference_state_id)
    target = _load_wave_state(experiment_dir, state_id)
    window_time, reference_qout_window, target_qout_window = _build_common_wave_window(
        reference,
        target,
        use_event_window=use_event_window,
    )
    _reference_parent_ids, reference_parent_lookup = _aggregate_parent_series(reference.wave_stack.reaches, reference_qout_window)
    _target_parent_ids, target_parent_lookup = _aggregate_parent_series(target.wave_stack.reaches, target_qout_window)

    shared_parent_ids = sorted(set(reference_parent_lookup).intersection(set(target_parent_lookup)))
    if parent_link_ids is not None:
        selected_parent_ids = [int(parent_link_id) for parent_link_id in parent_link_ids if int(parent_link_id) in shared_parent_ids]
    else:
        ranked: list[tuple[float, int]] = []
        for parent_link_id in shared_parent_ids:
            difference = _compute_shared_scale_difference(
                reference_parent_lookup[parent_link_id],
                target_parent_lookup[parent_link_id],
                activation_fraction=difference_fraction,
            )
            ranked.append((float(np.nanmax(np.abs(difference))), int(parent_link_id)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected_parent_ids = [parent_link_id for _score, parent_link_id in ranked[: max(1, int(top_n))]]

    rows: list[dict[str, Any]] = []
    for parent_link_id in selected_parent_ids:
        difference = _compute_shared_scale_difference(
            reference_parent_lookup[parent_link_id],
            target_parent_lookup[parent_link_id],
            activation_fraction=difference_fraction,
        )
        first_difference_time = _compute_first_significant_difference_time(
            reference_parent_lookup[parent_link_id],
            target_parent_lookup[parent_link_id],
            window_time,
            difference_fraction=difference_fraction,
        )
        for time_seconds, value in zip(window_time.tolist(), difference.tolist(), strict=False):
            rows.append(
                {
                    "parent_link_id": int(parent_link_id),
                    "time_seconds": float(time_seconds),
                    "time_hours": float(time_seconds / SECONDS_PER_HOUR),
                    "difference": float(value),
                    "first_significant_difference_time_seconds": first_difference_time,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ordering = (
        frame[["parent_link_id", "first_significant_difference_time_seconds"]]
        .drop_duplicates()
        .sort_values(["first_significant_difference_time_seconds", "parent_link_id"], na_position="last")
    )
    frame["parent_link_id"] = pd.Categorical(
        frame["parent_link_id"],
        categories=ordering["parent_link_id"].tolist(),
        ordered=True,
    )
    return frame.sort_values(["parent_link_id", "time_seconds"]).reset_index(drop=True)


def build_matched_corridor_hydrograph_frame(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str = "S000_base",
    n_upstream: int = 5,
    n_downstream: int = 5,
    use_event_window: bool = True,
) -> pd.DataFrame:
    experiment_path = _as_path(experiment_dir)
    if state_id == reference_state_id:
        raise ValueError("Corridor comparison requires a target state that differs from the reference state.")

    reference = _load_wave_state(experiment_path, reference_state_id)
    target = _load_wave_state(experiment_path, state_id)
    window_time, reference_qout_window, target_qout_window = _build_common_wave_window(
        reference,
        target,
        use_event_window=use_event_window,
    )

    _reference_parent_ids, reference_parent_lookup = _aggregate_parent_series(reference.wave_stack.reaches, reference_qout_window)
    _target_parent_ids, target_parent_lookup = _aggregate_stable_parent_series(
        target.wave_stack.reaches,
        target_qout_window,
    )
    _ = _reference_parent_ids, _target_parent_ids

    target_state_dir = experiment_path / "states" / state_id
    reference_state_dir = experiment_path / "states" / reference_state_id
    changed_parent_link_ids = set(_load_collapse_parent_link_ids(target_state_dir))
    target_directed = _load_state_directed_links(target_state_dir)
    base_directed = _load_state_directed_links(reference_state_dir)

    if "lineage_type" not in target_directed.columns or "dominant_parent_link_id" not in target_directed.columns:
        raise ValueError("Target directed links do not contain lineage fields needed for matched-corridor selection.")
    matched_parent_link_ids = {
        int(parent_link_id)
        for parent_link_id in pd.to_numeric(
            target_directed.loc[target_directed["lineage_type"].eq("unchanged_1to1"), "dominant_parent_link_id"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    }

    upstream_links, downstream_links = _select_matched_parent_links_around_change(
        base_directed[["id_link", "id_us_node", "id_ds_node"]],
        changed_parent_link_ids=changed_parent_link_ids,
        matched_parent_link_ids=matched_parent_link_ids,
        n_upstream=n_upstream,
        n_downstream=n_downstream,
    )

    rows: list[dict[str, Any]] = []
    for side, selection in [("upstream", upstream_links), ("downstream", downstream_links)]:
        for ordinal, item in enumerate(selection, start=1):
            parent_link_id = int(item["parent_link_id"])
            reference_series = reference_parent_lookup.get(parent_link_id)
            target_series = target_parent_lookup.get(parent_link_id)
            if reference_series is None or target_series is None:
                continue
            for time_seconds, reference_q, target_q in zip(
                window_time.tolist(),
                reference_series.tolist(),
                target_series.tolist(),
                strict=False,
            ):
                rows.append(
                    {
                        "state_id": state_id,
                        "reference_state_id": reference_state_id,
                        "parent_link_id": parent_link_id,
                        "corridor_side": side,
                        "corridor_order": ordinal,
                        "graph_distance": int(item["graph_distance"]),
                        "time_seconds": float(time_seconds),
                        "time_hours": float(time_seconds / SECONDS_PER_HOUR),
                        "reference_q_cms": float(reference_q),
                        "variant_q_cms": float(target_q),
                        "delta_q_cms": float(target_q - reference_q),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["panel_label"] = frame.apply(
        lambda row: f"{row['corridor_side'][0].upper()}{int(row['corridor_order'])} | parent {int(row['parent_link_id'])}",
        axis=1,
    )
    return frame.sort_values(["corridor_side", "corridor_order", "time_seconds"]).reset_index(drop=True)


def summarize_matched_corridor_peak_timing(hydrograph_frame: pd.DataFrame) -> pd.DataFrame:
    if hydrograph_frame.empty:
        raise ValueError("hydrograph_frame is empty.")

    group_columns = ["corridor_side", "corridor_order", "panel_label", "parent_link_id", "graph_distance"]
    rows: list[dict[str, Any]] = []
    for keys, group in hydrograph_frame.groupby(group_columns, sort=True):
        group = group.sort_values("time_seconds").reset_index(drop=True)
        reference_peak_index = int(group["reference_q_cms"].idxmax())
        variant_peak_index = int(group["variant_q_cms"].idxmax())
        reference_peak = hydrograph_frame.loc[reference_peak_index]
        variant_peak = hydrograph_frame.loc[variant_peak_index]
        rows.append(
            {
                "corridor_side": keys[0],
                "corridor_order": int(keys[1]),
                "panel_label": keys[2],
                "parent_link_id": int(keys[3]),
                "graph_distance": int(keys[4]),
                "reference_peak_time_seconds": float(reference_peak["time_seconds"]),
                "reference_peak_time_hours": float(reference_peak["time_hours"]),
                "reference_peak_q_cms": float(reference_peak["reference_q_cms"]),
                "variant_peak_time_seconds": float(variant_peak["time_seconds"]),
                "variant_peak_time_hours": float(variant_peak["time_hours"]),
                "variant_peak_q_cms": float(variant_peak["variant_q_cms"]),
                "delta_peak_time_hours": float(variant_peak["time_hours"] - reference_peak["time_hours"]),
                "delta_peak_q_cms": float(variant_peak["variant_q_cms"] - reference_peak["reference_q_cms"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["corridor_side", "corridor_order"]).reset_index(drop=True)


def build_all_matched_link_summary(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str = "S000_base",
    use_event_window: bool = True,
    threshold_fractions: tuple[float, ...] = (0.1, 0.5),
) -> pd.DataFrame:
    experiment_path = _as_path(experiment_dir)
    if state_id == reference_state_id:
        raise ValueError("Matched-link summary requires a target state that differs from the reference state.")

    reference = _load_wave_state(experiment_path, reference_state_id)
    target = _load_wave_state(experiment_path, state_id)
    window_time, reference_qout_window, target_qout_window = _build_common_wave_window(
        reference,
        target,
        use_event_window=use_event_window,
    )

    _reference_parent_ids, reference_parent_lookup = _aggregate_parent_series(reference.wave_stack.reaches, reference_qout_window)
    _target_parent_ids, target_parent_lookup = _aggregate_stable_parent_series(
        target.wave_stack.reaches,
        target_qout_window,
    )

    target_reaches = target.wave_stack.reaches.copy()
    if "dominant_parent_link_id" not in target_reaches.columns:
        raise ValueError("Target reach attributes do not contain dominant_parent_link_id.")
    target_reaches["dominant_parent_link_id"] = pd.to_numeric(target_reaches["dominant_parent_link_id"], errors="coerce")
    if "lineage_type" not in target_reaches.columns:
        raise ValueError("Target reach attributes do not contain lineage_type.")
    stable_reaches = target_reaches.loc[
        target_reaches["lineage_type"].astype(str).eq("unchanged_1to1")
        & target_reaches["dominant_parent_link_id"].notna()
    ].copy()
    reference_dist_out = _summarize_parent_dist_out(
        reference.wave_stack.reaches,
        parent_field="parent_link_id",
        prefix="reference",
    )
    target_dist_out = _summarize_parent_dist_out(
        stable_reaches,
        parent_field="dominant_parent_link_id",
        prefix="variant",
    )

    metadata_rows: list[dict[str, Any]] = []
    for dominant_parent_link_id, group in stable_reaches.groupby("dominant_parent_link_id", sort=True):
        local_parent_ids = sorted(
            {
                int(value)
                for value in pd.to_numeric(group.get("parent_link_id"), errors="coerce").dropna().astype(int).tolist()
            }
        )
        matched_paths = sorted(
            {
                str(value)
                for value in group.get("matched_parent_node_path", pd.Series(dtype=object)).dropna().astype(str).tolist()
                if str(value).strip()
            }
        )
        metadata_rows.append(
            {
                "parent_link_id": int(dominant_parent_link_id),
                "variant_local_parent_link_ids": ",".join(str(value) for value in local_parent_ids),
                "matched_parent_node_paths": " | ".join(matched_paths),
                "variant_matched_reach_count": int(len(group)),
            }
        )
    metadata = pd.DataFrame(metadata_rows)

    matched_parent_link_ids = sorted(set(reference_parent_lookup).intersection(set(target_parent_lookup)))
    metric_suffixes = [f"t{int(round(fraction * 100)):02d}" for fraction in threshold_fractions]
    rows: list[dict[str, Any]] = []
    for parent_link_id in matched_parent_link_ids:
        reference_series = reference_parent_lookup[parent_link_id]
        target_series = target_parent_lookup[parent_link_id]
        reference_metrics = compute_reach_timing_metrics(
            reference_series,
            window_time,
            threshold_fractions=threshold_fractions,
        ).iloc[0]
        target_metrics = compute_reach_timing_metrics(
            target_series,
            window_time,
            threshold_fractions=threshold_fractions,
        ).iloc[0]

        row: dict[str, Any] = {
            "parent_link_id": int(parent_link_id),
            "reference_peak_time_seconds": pd.to_numeric(reference_metrics.get("t_peak_seconds"), errors="coerce"),
            "reference_peak_time_hours": pd.to_numeric(reference_metrics.get("t_peak_seconds"), errors="coerce") / SECONDS_PER_HOUR,
            "reference_peak_q_cms": pd.to_numeric(reference_metrics.get("peak_discharge_cms"), errors="coerce"),
            "variant_peak_time_seconds": pd.to_numeric(target_metrics.get("t_peak_seconds"), errors="coerce"),
            "variant_peak_time_hours": pd.to_numeric(target_metrics.get("t_peak_seconds"), errors="coerce") / SECONDS_PER_HOUR,
            "variant_peak_q_cms": pd.to_numeric(target_metrics.get("peak_discharge_cms"), errors="coerce"),
        }
        row["delta_peak_time_hours"] = row["variant_peak_time_hours"] - row["reference_peak_time_hours"]
        row["delta_peak_q_cms"] = row["variant_peak_q_cms"] - row["reference_peak_q_cms"]
        for suffix in metric_suffixes:
            reference_value = pd.to_numeric(reference_metrics.get(f"{suffix}_seconds"), errors="coerce")
            target_value = pd.to_numeric(target_metrics.get(f"{suffix}_seconds"), errors="coerce")
            row[f"reference_{suffix}_hours"] = reference_value / SECONDS_PER_HOUR if pd.notna(reference_value) else float("nan")
            row[f"variant_{suffix}_hours"] = target_value / SECONDS_PER_HOUR if pd.notna(target_value) else float("nan")
            row[f"delta_{suffix}_hours"] = (
                (target_value - reference_value) / SECONDS_PER_HOUR if pd.notna(reference_value) and pd.notna(target_value) else float("nan")
            )
        rows.append(row)

    summary = pd.DataFrame(rows).merge(metadata, on="parent_link_id", how="left")
    if not reference_dist_out.empty:
        summary = summary.merge(reference_dist_out, on="parent_link_id", how="left")
    if not target_dist_out.empty:
        summary = summary.merge(target_dist_out, on="parent_link_id", how="left")
    if "reference_dist_out_mid_km" in summary.columns:
        summary["dist_out_mid_km"] = summary["reference_dist_out_mid_km"]
    if "reference_dist_out_mid_m" in summary.columns:
        summary["dist_out_mid_m"] = summary["reference_dist_out_mid_m"]
    summary["state_id"] = state_id
    summary["reference_state_id"] = reference_state_id
    return summary.sort_values("reference_peak_time_hours").reset_index(drop=True)


def plot_corridor_difference_heatmap(
    difference_frame: pd.DataFrame,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
):
    import matplotlib.pyplot as plt

    if difference_frame.empty:
        raise ValueError("difference_frame is empty.")
    pivot = difference_frame.pivot(index="parent_link_id", columns="time_hours", values="difference")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(pivot.index))))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index.tolist()])
    if pivot.shape[1] > 1:
        tick_positions = np.linspace(0, pivot.shape[1] - 1, num=min(8, pivot.shape[1]), dtype=int)
        tick_hours = pivot.columns.to_numpy(dtype=float)[tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{value:.1f}" for value in tick_hours])
    ax.set_xlabel("time since forcing start (hours)")
    ax.set_ylabel("parent_link_id")
    ax.set_title(title or "Parent-link difference heatmap")
    fig.colorbar(image, ax=ax, label="shared-scale difference (variant - base)")
    fig.tight_layout()
    return fig, ax


def plot_matched_corridor_hydrographs(
    hydrograph_frame: pd.DataFrame,
    *,
    title: str | None = None,
    show_peak_markers: bool = False,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if hydrograph_frame.empty:
        raise ValueError("hydrograph_frame is empty.")

    side_order = ["upstream", "downstream"]
    grouped = (
        hydrograph_frame[["corridor_side", "corridor_order", "panel_label", "parent_link_id", "graph_distance"]]
        .drop_duplicates()
        .sort_values(["corridor_side", "corridor_order"])
    )
    upstream_orders = grouped.loc[grouped["corridor_side"].eq("upstream"), "corridor_order"]
    downstream_orders = grouped.loc[grouped["corridor_side"].eq("downstream"), "corridor_order"]
    rows = max(
        int(upstream_orders.max()) if not upstream_orders.empty else 0,
        int(downstream_orders.max()) if not downstream_orders.empty else 0,
    )
    if rows == 0:
        raise ValueError("No matched corridor links were available for plotting.")

    subplot_titles: list[str] = []
    for row_index in range(1, rows + 1):
        for side in side_order:
            entry = grouped.loc[
                grouped["corridor_side"].eq(side) & grouped["corridor_order"].eq(row_index)
            ]
            if entry.empty:
                subplot_titles.append(f"{side.title()} {row_index}")
            else:
                first = entry.iloc[0]
                subplot_titles.append(
                    f"{side.title()} {row_index}: parent {int(first['parent_link_id'])} (d={int(first['graph_distance'])})"
                )

    fig = make_subplots(
        rows=rows,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    show_legend = {"reference": True, "variant": True}
    color_map = {"reference": "#111111", "variant": "#d95f02"}
    name_map = {"reference": "Base", "variant": "Variant"}
    peak_summary = summarize_matched_corridor_peak_timing(hydrograph_frame) if show_peak_markers else None
    for row_index in range(1, rows + 1):
        for col_index, side in enumerate(side_order, start=1):
            subset = hydrograph_frame.loc[
                hydrograph_frame["corridor_side"].eq(side)
                & hydrograph_frame["corridor_order"].eq(row_index)
            ].copy()
            if subset.empty:
                continue
            for series_key, value_column in [("reference", "reference_q_cms"), ("variant", "variant_q_cms")]:
                fig.add_trace(
                    go.Scatter(
                        x=subset["time_hours"],
                        y=subset[value_column],
                        mode="lines",
                        name=name_map[series_key],
                        legendgroup=series_key,
                        showlegend=show_legend[series_key],
                        line={"color": color_map[series_key]},
                        hovertemplate=(
                            "parent_link_id=%{customdata[0]}<br>"
                            "time=%{x:.2f} h<br>"
                            f"{name_map[series_key]}=%{{y:.2f}} cms<extra></extra>"
                        ),
                        customdata=np.column_stack([subset["parent_link_id"].to_numpy(dtype=int)]),
                    ),
                    row=row_index,
                    col=col_index,
                )
                show_legend[series_key] = False
            if peak_summary is not None:
                peak_row = peak_summary.loc[
                    peak_summary["corridor_side"].eq(side)
                    & peak_summary["corridor_order"].eq(row_index)
                ]
                if not peak_row.empty:
                    peak_row = peak_row.iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_row["reference_peak_time_hours"]],
                            y=[peak_row["reference_peak_q_cms"]],
                            mode="markers",
                            name="Base peak",
                            legendgroup="reference_peak",
                            showlegend=False,
                            marker={"color": color_map["reference"], "size": 8, "symbol": "circle-open"},
                            hovertemplate=(
                                f"Base peak<br>parent_link_id={int(peak_row['parent_link_id'])}<br>"
                                "time=%{x:.2f} h<br>Q=%{y:.2f} cms<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=col_index,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_row["variant_peak_time_hours"]],
                            y=[peak_row["variant_peak_q_cms"]],
                            mode="markers",
                            name="Variant peak",
                            legendgroup="variant_peak",
                            showlegend=False,
                            marker={"color": color_map["variant"], "size": 8, "symbol": "x"},
                            hovertemplate=(
                                f"Variant peak<br>parent_link_id={int(peak_row['parent_link_id'])}<br>"
                                "time=%{x:.2f} h<br>Q=%{y:.2f} cms<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=col_index,
                    )

    for row_index in range(1, rows + 1):
        fig.update_yaxes(title_text="discharge (cms)", row=row_index, col=1)
    fig.update_xaxes(title_text="time since forcing start (hours)", row=rows, col=1)
    fig.update_xaxes(title_text="time since forcing start (hours)", row=rows, col=2)
    fig.update_layout(
        title=title or "Matched upstream/downstream corridor hydrographs",
        height=max(320, 260 * rows),
        width=1150,
    )
    return fig


def plot_matched_link_difference_scatter(
    matched_link_summary: pd.DataFrame,
    *,
    x: str = "reference_peak_time_hours",
    y: str = "delta_peak_time_hours",
    color: str = "delta_peak_q_cms",
    size: str | None = None,
    title: str | None = None,
):
    import plotly.express as px

    if matched_link_summary.empty:
        raise ValueError("matched_link_summary is empty.")

    frame = matched_link_summary.copy()
    hover_data = [
        "parent_link_id",
        "variant_local_parent_link_ids",
        "matched_parent_node_paths",
        "reference_peak_time_hours",
        "variant_peak_time_hours",
        "delta_peak_time_hours",
        "reference_peak_q_cms",
        "variant_peak_q_cms",
        "delta_peak_q_cms",
    ]
    plot_kwargs = {
        "x": x,
        "y": y,
        "color": color,
        "hover_data": [column for column in hover_data if column in frame.columns],
        "title": title or "All 100% matched links: base vs variant",
    }
    if size is not None and size in frame.columns:
        plot_kwargs["size"] = size
    return px.scatter(frame, **plot_kwargs)


def _render_wave_frames(
    render_state: _WaveRenderState,
    frame_dir: Path,
    state_id: str,
    *,
    dpi: int = 120,
) -> list[Path]:
    import matplotlib.pyplot as plt

    frame_dir.mkdir(parents=True, exist_ok=True)
    hydro_start = float(
        render_state.run_row.get(
            "event_start_time_seconds",
            render_state.wave_stack.time_seconds[render_state.frame_indices[0]],
        )
    )
    hydro_end = float(
        render_state.run_row.get(
            "event_end_time_seconds",
            render_state.wave_stack.time_seconds[render_state.frame_indices[-1]],
        )
    )
    hydro_window = render_state.outlet_hydro.loc[
        render_state.outlet_hydro["time_seconds"].between(hydro_start, hydro_end, inclusive="both")
    ].copy()
    if hydro_window.empty:
        hydro_window = render_state.outlet_hydro.copy()

    cmap = plt.get_cmap(render_state.cmap_name)
    frame_paths: list[Path] = []
    for frame_position, time_index in enumerate(render_state.frame_indices):
        local_index = time_index - render_state.start_index
        fig = plt.figure(figsize=(10, 5.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_hydro = fig.add_subplot(gs[0, 1])

        intensity_frame = render_state.intensity[local_index, :]
        for idx, geometry in enumerate(render_state.reaches.geometry):
            for segment in _geometry_segments(geometry):
                ax_map.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=cmap(float(intensity_frame[idx])),
                    linewidth=float(render_state.line_widths[idx]),
                    solid_capstyle="round",
                )
        ax_map.set_aspect("equal")
        ax_map.set_xlim(render_state.bounds[0], render_state.bounds[2])
        ax_map.set_ylim(render_state.bounds[1], render_state.bounds[3])
        ax_map.set_axis_off()
        current_seconds = float(render_state.wave_stack.time_seconds[time_index])
        ax_map.set_title(f"{state_id}\nWave intensity at t = {current_seconds / SECONDS_PER_HOUR:.2f} h")

        ax_hydro.plot(hydro_window["time_seconds"] / SECONDS_PER_HOUR, hydro_window["q_inflow_cms"], color="black", linestyle="--", label="Inflow")
        ax_hydro.plot(hydro_window["time_seconds"] / SECONDS_PER_HOUR, hydro_window["q_outlet_cms"], color="tab:blue", label="Outlet")
        ax_hydro.axvline(current_seconds / SECONDS_PER_HOUR, color="tab:red", linestyle=":")
        ax_hydro.set_xlabel("time since forcing start (hours)")
        ax_hydro.set_ylabel("discharge (cms)")
        ax_hydro.legend(loc="upper left")

        frame_path = frame_dir / f"frame_{frame_position:04d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def write_wave_gif(
    experiment_dir: str | Path,
    state_id: str,
    *,
    output_path: str | Path | None = None,
    frame_stride: int = 1,
    duration_ms: int = 120,
    linewidth_field: str = "rapid_branch_width_m",
    use_event_window: bool = True,
    cmap_name: str = "coolwarm",
) -> Path:
    from PIL import Image

    experiment_path = _as_path(experiment_dir)
    output = _as_path(output_path) if output_path is not None else experiment_path / f"{state_id}_wave.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    render_state = _build_wave_render_state(
        experiment_path,
        state_id,
        frame_stride=frame_stride,
        linewidth_field=linewidth_field,
        use_event_window=use_event_window,
        cmap_name=cmap_name,
    )
    with TemporaryDirectory(prefix="rapid-wave-frames-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        frame_paths = _render_wave_frames(render_state, tmp_dir, state_id)
        images: list[Image.Image] = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame_image:
                images.append(frame_image.convert("RGB").copy())
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(duration_ms)),
            loop=0,
            disposal=2,
        )
    return output


def _resolve_ffmpeg_executable(ffmpeg_path: str | Path | None = None) -> str:
    if ffmpeg_path is not None:
        path = _as_path(ffmpeg_path)
        if not path.exists():
            raise FileNotFoundError(f"Requested ffmpeg executable was not found: {path}")
        return str(path)
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise RuntimeError(
        "MP4 export requires ffmpeg, but no ffmpeg executable was found on PATH. "
        "Install ffmpeg or pass ffmpeg_path=... to write_wave_mp4(...)."
    )


def write_wave_comparison_mp4(
    experiment_dir: str | Path,
    state_id: str,
    *,
    reference_state_id: str = "S000_base",
    output_path: str | Path | None = None,
    frame_stride: int = 1,
    fps: int = 6,
    linewidth_field: str = "rapid_branch_width_m",
    use_event_window: bool = True,
    cmap_name: str = "coolwarm",
    difference_cmap_name: str = "RdBu_r",
    ffmpeg_path: str | Path | None = None,
    crf: int = 20,
    zoom_bounds: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
    difference_activation_fraction: float = 0.05,
) -> Path:
    if state_id == reference_state_id:
        raise ValueError("Comparison MP4 requires a target state that differs from the reference state.")
    experiment_path = _as_path(experiment_dir)
    output = (
        _as_path(output_path)
        if output_path is not None
        else experiment_path / f"{state_id}_vs_{reference_state_id}_wave.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = _resolve_ffmpeg_executable(ffmpeg_path)
    render_state = _build_wave_comparison_render_state(
        experiment_path,
        state_id,
        reference_state_id=reference_state_id,
        frame_stride=frame_stride,
        linewidth_field=linewidth_field,
        use_event_window=use_event_window,
        cmap_name=cmap_name,
        difference_cmap_name=difference_cmap_name,
        zoom_bounds=zoom_bounds,
        difference_activation_fraction=difference_activation_fraction,
    )
    with TemporaryDirectory(prefix="rapid-wave-frames-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        _render_wave_comparison_frames(render_state, tmp_dir)
        command = [
            executable,
            "-y",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
            str(output),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed while writing MP4:\n{stderr}") from exc
    return output


def write_wave_mp4(
    experiment_dir: str | Path,
    state_id: str,
    *,
    output_path: str | Path | None = None,
    frame_stride: int = 1,
    fps: int = 6,
    linewidth_field: str = "rapid_branch_width_m",
    use_event_window: bool = True,
    cmap_name: str = "coolwarm",
    ffmpeg_path: str | Path | None = None,
    crf: int = 20,
    reference_state_id: str = "S000_base",
    compare_to_reference: bool = True,
    difference_cmap_name: str = "RdBu_r",
    zoom_bounds: tuple[float, float, float, float] | list[float] | np.ndarray | None = None,
    difference_activation_fraction: float = 0.05,
) -> Path:
    experiment_path = _as_path(experiment_dir)
    output = _as_path(output_path) if output_path is not None else experiment_path / f"{state_id}_wave.mp4"
    if compare_to_reference and state_id != reference_state_id:
        return write_wave_comparison_mp4(
            experiment_path,
            state_id,
            reference_state_id=reference_state_id,
            output_path=output,
            frame_stride=frame_stride,
            fps=fps,
            linewidth_field=linewidth_field,
            use_event_window=use_event_window,
            cmap_name=cmap_name,
            difference_cmap_name=difference_cmap_name,
            ffmpeg_path=ffmpeg_path,
            crf=crf,
            zoom_bounds=zoom_bounds,
            difference_activation_fraction=difference_activation_fraction,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    executable = _resolve_ffmpeg_executable(ffmpeg_path)
    render_state = _build_wave_render_state(
        experiment_path,
        state_id,
        frame_stride=frame_stride,
        linewidth_field=linewidth_field,
        use_event_window=use_event_window,
        cmap_name=cmap_name,
    )
    with TemporaryDirectory(prefix="rapid-wave-frames-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        _render_wave_frames(render_state, tmp_dir, state_id)
        command = [
            executable,
            "-y",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
            str(output),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed while writing MP4:\n{stderr}") from exc
    return output


__all__ = [
    "build_all_matched_link_summary",
    "ReachWaveStack",
    "build_combined_analysis_frame",
    "build_matched_corridor_hydrograph_frame",
    "build_parent_link_difference_timeseries",
    "build_experiment_analysis_frame",
    "build_reach_timing_comparison",
    "compare_hydrographs",
    "compute_wave_intensity_matrix",
    "compute_reach_timing_metrics",
    "load_reach_wave_stack",
    "load_state_hydrograph",
    "plot_corridor_difference_heatmap",
    "plot_effect_signature",
    "plot_hydrograph_overlay",
    "plot_matched_link_difference_scatter",
    "plot_matched_corridor_hydrographs",
    "plot_reach_timing_map",
    "summarize_matched_corridor_peak_timing",
    "write_wave_comparison_mp4",
    "write_wave_gif",
    "write_wave_mp4",
]

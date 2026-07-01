from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.express as px


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "network_variants" / "output"

_DT_ERROR_PATTERN = re.compile(r"dt_min=([0-9.]+)\s*>\s*dt_max=([0-9.]+)")
_DT_RANGE_PATTERN = re.compile(r"bounds \[([0-9.]+),\s*([0-9.]+)\]")


def _normalize_example_id(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Example IDs cannot be empty.")
    return text.zfill(2) if text.isdigit() and len(text) < 2 else text


def _normalize_state_ids(
    state_ids: Sequence[str] | Mapping[str, Sequence[str]] | None,
    example_ids: Sequence[str],
) -> dict[str, list[str]] | None:
    if state_ids is None:
        return None
    if isinstance(state_ids, Mapping):
        return {
            _normalize_example_id(example_id): [str(state_id) for state_id in values]
            for example_id, values in state_ids.items()
        }
    normalized = [str(state_id) for state_id in state_ids]
    return {_normalize_example_id(example_id): list(normalized) for example_id in example_ids}


def _experiment_dir(
    example_id: str | int,
    *,
    mode: str = "indep",
    anchored_suffix: str = "__anchored",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    normalized = _normalize_example_id(example_id)
    mode_name = "compound_bubbles" if str(mode).strip().lower() == "compound" else "indep"
    return Path(output_root) / f"{normalized}_{mode_name}{anchored_suffix}"


def list_stability_states(
    example_ids: Sequence[str | int],
    *,
    mode: str = "indep",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    anchored_suffix: str = "__anchored",
    state_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    include_base: bool = True,
) -> pd.DataFrame:
    normalized_examples = [_normalize_example_id(example_id) for example_id in example_ids]
    normalized_state_ids = _normalize_state_ids(state_ids, normalized_examples)

    rows: list[dict[str, Any]] = []
    for example_id in normalized_examples:
        exp_dir = _experiment_dir(
            example_id,
            mode=mode,
            anchored_suffix=anchored_suffix,
            output_root=output_root,
        )
        if not exp_dir.exists():
            continue

        registry_path = exp_dir / "rapid_prep_registry.csv"
        registry = (
            pd.read_csv(registry_path)
            if registry_path.exists() and registry_path.stat().st_size > 1
            else pd.DataFrame()
        )
        wanted_state_ids = None if normalized_state_ids is None else normalized_state_ids.get(example_id, [])
        if wanted_state_ids is not None:
            selection = registry.loc[registry["state_id"].astype(str).isin(wanted_state_ids)].copy()
        else:
            selection = registry.copy()

        if include_base and not selection.empty and not selection["state_id"].astype(str).eq("S000_base").any():
            base_rows = registry.loc[registry["state_id"].astype(str).eq("S000_base")].copy()
            if not base_rows.empty:
                selection = pd.concat([base_rows, selection], ignore_index=True)
        elif include_base and selection.empty:
            base_rows = registry.loc[registry["state_id"].astype(str).eq("S000_base")].copy()
            selection = base_rows if not base_rows.empty else pd.DataFrame([{"state_id": "S000_base"}])

        for _, row in selection.iterrows():
            state_id = str(row["state_id"])
            state_dir = exp_dir / "states" / state_id
            rows.append(
                {
                    "example_id": example_id,
                    "mode": mode,
                    "experiment_dir": str(exp_dir),
                    "state_id": state_id,
                    "state_dir": str(state_dir),
                    "state_role": row.get("state_role", "base" if state_id == "S000_base" else "derived"),
                    "prep_status": row.get("status"),
                    "prep_error": row.get("error"),
                    "prep_row": row.to_dict(),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.drop_duplicates(subset=["example_id", "mode", "state_id"]).reset_index(drop=True)
    result["state_label"] = result["example_id"].astype(str) + " · " + result["state_id"].astype(str)
    return result


def _resolve_state_dt_bounds(prep_row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    status = str(prep_row.get("status", ""))
    if status == "prepared":
        rapid_k_min = pd.to_numeric(pd.Series([prep_row.get("rapid_k_min")]), errors="coerce").iloc[0]
        rapid_k_max = pd.to_numeric(pd.Series([prep_row.get("rapid_k_max")]), errors="coerce").iloc[0]
        if pd.notna(rapid_k_min) and pd.notna(rapid_k_max):
            return float(2.0 * rapid_k_max * 0.1), float(2.0 * rapid_k_min * 0.9)
    error = str(prep_row.get("error", "") or "")
    match = _DT_ERROR_PATTERN.search(error)
    if match:
        return float(match.group(1).rstrip(".")), float(match.group(2).rstrip("."))
    match = _DT_RANGE_PATTERN.search(error)
    if match:
        return float(match.group(1).rstrip(".")), float(match.group(2).rstrip("."))
    return None, None


def build_timestep_tail_frame(
    example_ids: Sequence[str | int],
    *,
    mode: str = "indep",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    anchored_suffix: str = "__anchored",
    state_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    include_base: bool = True,
) -> pd.DataFrame:
    selection = list_stability_states(
        example_ids,
        mode=mode,
        output_root=output_root,
        anchored_suffix=anchored_suffix,
        state_ids=state_ids,
        include_base=include_base,
    )
    if selection.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in selection.iterrows():
        rapid_path = Path(row["state_dir"]) / "rapid" / "prep" / "rapid_link_attributes.csv"
        if not rapid_path.exists() or rapid_path.stat().st_size <= 1:
            continue

        rapid = pd.read_csv(rapid_path)
        if "rapid_k" not in rapid.columns:
            continue
        rapid["rapid_k"] = pd.to_numeric(rapid["rapid_k"], errors="coerce")
        rapid = rapid.dropna(subset=["rapid_k"]).copy()
        if rapid.empty:
            continue

        dt_min, dt_max = _resolve_state_dt_bounds(row["prep_row"])
        min_row = rapid.loc[rapid["rapid_k"].idxmin()]
        max_row = rapid.loc[rapid["rapid_k"].idxmax()]
        for tail, subreach in (("dt_max", min_row), ("dt_min", max_row)):
            state_dt_value = dt_max if tail == "dt_max" else dt_min
            rows.append(
                {
                    "example_id": row["example_id"],
                    "mode": row["mode"],
                    "state_id": row["state_id"],
                    "state_label": row["state_label"],
                    "prep_status": row["prep_status"],
                    "tail": tail,
                    "state_dt_value": state_dt_value,
                    "rapid_k": float(subreach["rapid_k"]),
                    "parent_link_id": int(subreach["parent_link_id"]) if pd.notna(subreach.get("parent_link_id")) else None,
                    "id_link": int(subreach["id_link"]) if pd.notna(subreach.get("id_link")) else None,
                    "subreach_index": int(subreach["subreach_index"]) if pd.notna(subreach.get("subreach_index")) else None,
                    "rapid_effective_length_m": float(subreach["rapid_effective_length_m"])
                    if pd.notna(subreach.get("rapid_effective_length_m"))
                    else None,
                    "rapid_celerity_mps": float(subreach["rapid_celerity_mps"])
                    if pd.notna(subreach.get("rapid_celerity_mps"))
                    else None,
                    "rapid_width_m": float(subreach["rapid_width_m"])
                    if pd.notna(subreach.get("rapid_width_m"))
                    else None,
                    "rapid_subreach_width_m": float(subreach["rapid_subreach_width_m"])
                    if pd.notna(subreach.get("rapid_subreach_width_m"))
                    else None,
                    "slope_used": float(subreach["slope_used"]) if pd.notna(subreach.get("slope_used")) else None,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["tail_label"] = result["tail"].map(
        {"dt_min": "Largest dt_min culprit", "dt_max": "Smallest dt_max culprit"}
    )
    return result


def plot_effective_length_vs_dt(
    tail_frame: pd.DataFrame,
    *,
    tail: str,
    color: str = "prep_status",
) -> Any:
    frame = tail_frame.loc[tail_frame["tail"].astype(str).eq(str(tail))].copy()
    if frame.empty:
        raise ValueError(f"No rows found for tail={tail!r}.")

    title_tail = "dt_min" if str(tail) == "dt_min" else "dt_max"
    fig = px.scatter(
        frame,
        x="rapid_effective_length_m",
        y="state_dt_value",
        color=color if color in frame.columns else None,
        hover_data=[
            c
            for c in [
                "example_id",
                "state_id",
                "parent_link_id",
                "id_link",
                "subreach_index",
                "rapid_k",
                "rapid_celerity_mps",
                "rapid_width_m",
                "rapid_subreach_width_m",
                "slope_used",
            ]
            if c in frame.columns
        ],
        labels={
            "rapid_effective_length_m": "Effective subreach length [m]",
            "state_dt_value": f"State {title_tail} [s]",
            "prep_status": "Prep status",
        },
        title=f"Effective length vs {title_tail} culprit across states",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))
    return fig


def plot_timestep_tail_suite(
    tail_frame: pd.DataFrame,
    *,
    color: str = "prep_status",
) -> Any:
    frame = tail_frame.copy()
    if frame.empty:
        raise ValueError("tail_frame is empty.")
    fig = px.scatter(
        frame,
        x="rapid_effective_length_m",
        y="state_dt_value",
        color=color if color in frame.columns else None,
        facet_col="tail_label",
        hover_data=[
            c
            for c in [
                "example_id",
                "state_id",
                "parent_link_id",
                "id_link",
                "subreach_index",
                "rapid_k",
                "rapid_celerity_mps",
                "rapid_width_m",
                "rapid_subreach_width_m",
                "slope_used",
            ]
            if c in frame.columns
        ],
        labels={
            "rapid_effective_length_m": "Effective subreach length [m]",
            "state_dt_value": "State tail timestep [s]",
            "tail_label": "Tail",
            "prep_status": "Prep status",
        },
        title="Effective length against the dt_min / dt_max tail culprits",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))
    return fig


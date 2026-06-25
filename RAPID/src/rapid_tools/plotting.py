from __future__ import annotations

from collections import defaultdict
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from pandas.errors import EmptyDataError


SECONDS_PER_HOUR = 3600.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "network_variants" / "output"

CATEGORY_COLOR_MAP = {
    "single unit": "#1f77b4",
    "compound unit": "#d62728",
    "nested unit": "#2ca02c",
    "part of compound unit": "#ff7f0e",
}

TIMING_Y_METRIC_LABELS = {
    "centroid50_shift_hours": "Centroid shift above 50% peak [hours]",
    "centroid75_shift_hours": "Centroid shift above 75% peak [hours]",
    "peak_time_shift_hours": "Peak-time shift [hours]",
    "rise95_time_shift_hours": "Rise-95 timing shift [hours]",
    "plateau995_center_shift_hours": "99.5% peak-plateau center shift [hours]",
    "peak_window_center95_shift_hours": "95% peak-window center shift [hours]",
    "peak_window_centroid95_shift_hours": "95% peak-window centroid shift [hours]",
    "xcorr_lag_hours": "Cross-correlation lag [hours]",
    "delta_peak_time_hours": "Peak-time delta vs base [hours]",
    "delta_lag_to_inflow_peak_hours": "Lag-to-inflow-peak delta vs base [hours]",
    "hydrograph_lag_vs_base_hours": "Hydrograph lag vs base [hours]",
    "lag_to_inflow_peak_relative_to_base": "Lag-to-inflow-peak relative change vs base",
    "lag_to_inflow_peak_pct_shift_vs_base": "Lag-to-inflow-peak percent shift vs base [%]",
    "unit_to_bubble_lag_to_inflow_peak_relative": "Lag-to-inflow-peak relative change vs matching compound",
    "unit_to_bubble_lag_to_inflow_peak_pct_shift": "Lag-to-inflow-peak percent shift vs matching compound [%]",
}

GEOMETRY_X_METRIC_LABELS = {
    "along_network_frac": "Along-network fractional distance",
    "matched_len_frac": "Matched parent length fraction of network",
    "length_fraction": "Matched parent length fraction of network",
    "child_parent_len_frac": "Child length / matched parent length",
    "child_length_fraction": "Child length / matched parent length",
    "width_fraction_of_total": "Upstream width-share routed into target object",
    "width_fraction": "Upstream width-share routed into target object",
    "Lambda_matched": "Lambda_matched",
    "lambda_matched": "Lambda_matched",
    "neg_log10_lambda": "-log10(Lambda_matched)",
    "log_lambda": "-log10(Lambda_matched)",
    "length_weighted_log_lambda": "matched_len_frac * (-log10(Lambda_matched))",
    "child_parent_weighted_log_lambda": "(child_len / matched_parent_len) * (-log10(Lambda_matched))",
    "child_parent_width_weighted_log_lambda": "(child_len / matched_parent_len) * (-log10(Lambda_matched)) * width_fraction",
    "k_ratio_to_matched_base": "K_variant_realized / K_eq_matched_base",
}

PLOT_DIMENSION_LABELS = {
    "target_id": "Unit / bubble id",
    "example_id": "Example ID",
    "mode": "Experiment type",
    "plot_category": "Category",
    **GEOMETRY_X_METRIC_LABELS,
    **TIMING_Y_METRIC_LABELS,
}


def supported_timing_metrics() -> dict[str, str]:
    return dict(TIMING_Y_METRIC_LABELS)


def supported_timing_y_metrics() -> dict[str, str]:
    return dict(TIMING_Y_METRIC_LABELS)


def supported_timing_x_metrics() -> dict[str, str]:
    return dict(GEOMETRY_X_METRIC_LABELS)


def _normalize_example_id(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Example IDs cannot be empty.")
    return text.zfill(2) if text.isdigit() and len(text) < 2 else text


def _parse_int_list(value: object) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def _parse_optional_int_list(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def mandatory_edges(
    G: nx.DiGraph,
    source: object | None = None,
    sink: object | None = None,
) -> list[tuple[object, object]]:
    """
    Return all directed edges that lie on every source-to-sink path.

    This avoids enumerating all simple paths. Each original edge is converted
    into an intermediate edge-node, then NetworkX immediate dominators are
    computed on the transformed graph. Any edge-node on the sink dominator
    chain corresponds to an original edge that every source-to-sink path must
    traverse.

    Parameters
    ----------
    G
        Input directed acyclic graph as a standard ``networkx.DiGraph``.
    source
        Optional source node. When omitted, the unique node with in-degree 0 is
        inferred.
    sink
        Optional sink node. When omitted, the unique node with out-degree 0 is
        inferred.

    Returns
    -------
    list[tuple]
        Original directed edges ordered from source to sink.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([
    ...     ("A", "B"),
    ...     ("B", "C"),
    ...     ("B", "D"),
    ...     ("C", "E"),
    ...     ("D", "E"),
    ...     ("E", "F"),
    ... ])
    >>> mandatory_edges(G)
    [('A', 'B'), ('E', 'F')]
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")

    if source is None:
        sources = [node for node in G.nodes if G.in_degree(node) == 0]
        if len(sources) != 1:
            raise ValueError(f"Expected exactly one source, found {len(sources)}: {sources}")
        source = sources[0]

    if sink is None:
        sinks = [node for node in G.nodes if G.out_degree(node) == 0]
        if len(sinks) != 1:
            raise ValueError(f"Expected exactly one sink, found {len(sinks)}: {sinks}")
        sink = sinks[0]

    if source not in G:
        raise ValueError(f"source {source!r} is not in G")
    if sink not in G:
        raise ValueError(f"sink {sink!r} is not in G")
    if not nx.has_path(G, source, sink):
        raise ValueError(f"No directed path from source {source!r} to sink {sink!r}")

    H = nx.DiGraph()

    def original_node(node: object) -> tuple[str, object]:
        return ("node", node)

    def edge_node(u: object, v: object) -> tuple[str, object, object]:
        return ("edge", u, v)

    for u, v in G.edges:
        H.add_edge(original_node(u), edge_node(u, v))
        H.add_edge(edge_node(u, v), original_node(v))

    h_source = original_node(source)
    h_sink = original_node(sink)
    idom = nx.immediate_dominators(H, h_source)

    result: list[tuple[object, object]] = []
    cur = h_sink
    while cur != h_source:
        cur = idom[cur]
        if cur[0] == "edge":
            _, u, v = cur
            result.append((u, v))
    return result[::-1]


def _interp_first_crossing_time(t: np.ndarray, q: np.ndarray, level: float) -> float:
    idx = np.where(q >= level)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    q0, q1 = float(q[i - 1]), float(q[i])
    if q1 == q0:
        return t1
    return t0 + (level - q0) * (t1 - t0) / (q1 - q0)


def _plateau_center_time(t: np.ndarray, q: np.ndarray, frac: float = 0.995) -> float:
    peak = float(np.max(q))
    mask = q >= peak * frac
    if not np.any(mask):
        return float("nan")
    return float(np.mean(t[mask]))


def _weighted_centroid_time(t: np.ndarray, q: np.ndarray, frac: float = 0.5) -> float:
    peak = float(np.max(q))
    threshold = peak * frac
    mask = q >= threshold
    if not np.any(mask):
        return float("nan")
    weights = q[mask] - threshold
    if np.allclose(weights.sum(), 0):
        return float(np.mean(t[mask]))
    return float(np.sum(t[mask] * weights) / np.sum(weights))


def _peak_window_center_time(t: np.ndarray, q: np.ndarray, frac: float = 0.95) -> float:
    peak = float(np.max(q))
    mask = q >= peak * frac
    if not np.any(mask):
        return float("nan")
    return float((t[mask][0] + t[mask][-1]) / 2.0)


def _peak_window_centroid_time(t: np.ndarray, q: np.ndarray, frac: float = 0.95) -> float:
    peak = float(np.max(q))
    threshold = peak * frac
    mask = q >= threshold
    if not np.any(mask):
        return float("nan")
    weights = q[mask] - threshold
    if np.allclose(weights.sum(), 0):
        return float(np.mean(t[mask]))
    return float(np.sum(t[mask] * weights) / np.sum(weights))


def _xcorr_lag_hours(q_base: np.ndarray, q_other: np.ndarray, dt_seconds: float) -> float:
    qb = np.asarray(q_base, dtype=float) - np.mean(q_base)
    qo = np.asarray(q_other, dtype=float) - np.mean(q_other)
    corr = np.correlate(qo, qb, mode="full")
    lags = np.arange(-len(qo) + 1, len(qo))
    best_lag_steps = int(lags[np.argmax(corr)])
    return float(best_lag_steps * dt_seconds / SECONDS_PER_HOUR)


def _load_outlet(states_dir: Path, state_id: str) -> pd.DataFrame:
    df = pd.read_csv(states_dir / state_id / "rapid" / "run" / "outlet_hydrograph.csv")
    df = df[["time_seconds", "time_utc", "q_outlet_cms"]].copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def _read_csv_nonempty(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 1:
        return None
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return None


def _build_length_lookup(exp_dir: Path) -> tuple[pd.Series, float, str]:
    manifest = json.loads((exp_dir / "equivalent_resistance" / "equivalent_resistance_manifest.json").read_text())
    length_candidates = manifest["equivalent_resistance_config"]["length_field_candidates"]
    base_links = gpd.read_file(exp_dir / "states" / "S000_base" / "variant" / "widths" / "links_with_width_families.gpkg")
    length_field = next(column for column in length_candidates if column in base_links.columns)
    lookup = (
        base_links[["id_link", length_field]]
        .rename(columns={length_field: "base_link_len"})
        .assign(id_link=lambda d: pd.to_numeric(d["id_link"], errors="coerce").astype("Int64"))
        .dropna(subset=["id_link"])
        .drop_duplicates(subset=["id_link"])
        .set_index("id_link")["base_link_len"]
    )
    return lookup, float(lookup.sum()), length_field


def _matched_length_from_ids(link_ids_text: object, base_len_lookup: pd.Series) -> float:
    ids = _parse_int_list(link_ids_text)
    if not ids:
        return float("nan")
    values = base_len_lookup.reindex(ids)
    return float(values.sum()) if values.notna().any() else float("nan")


def _load_unit_summary(exp_dir: Path) -> pd.DataFrame:
    unit_summary = pd.read_csv(exp_dir / "states" / "S000_base" / "hierarchy" / "unit_summary.csv")
    unit_summary = unit_summary.rename(columns={"class": "unit_class"}).copy()
    unit_summary["unit_id"] = pd.to_numeric(unit_summary["unit_id"], errors="coerce").astype("Int64")
    if "compound_unit_id" in unit_summary.columns:
        unit_summary["compound_unit_id"] = pd.to_numeric(unit_summary["compound_unit_id"], errors="coerce").astype("Int64")
    return unit_summary[
        [
            "unit_id",
            "unit_class",
            "is_compound",
            "compound_unit_id",
            "compound_bubble_id",
            "in_compound_bubble",
            "compound_bubble_role",
        ]
    ].copy()


def _classify_plot_category(row: pd.Series) -> str:
    if row.get("target_type") == "compound_bubble":
        return "compound unit"
    if bool(row.get("is_compound", False)):
        return "compound unit"
    if pd.notna(row.get("compound_unit_id")):
        return "part of compound unit"
    if str(row.get("unit_class", "")) == "compound_or_nested_complex":
        return "nested unit"
    return "single unit"


def _build_experiment_frame(example_id: str, mode: str, exp_dir: Path) -> pd.DataFrame:
    states_dir = exp_dir / "states"
    run = _read_csv_nonempty(exp_dir / "rapid_run_registry.csv")
    comp = _read_csv_nonempty(exp_dir / "equivalent_resistance" / "equivalent_resistance_comparison.csv")
    matched = _read_csv_nonempty(exp_dir / "equivalent_resistance" / "matched_base_corridors.csv")
    child = _read_csv_nonempty(exp_dir / "equivalent_resistance" / "child_realized_objects.csv")
    if run is None or comp is None or matched is None or child is None:
        warnings.warn(
            f"Skipping {exp_dir.name}: one or more required output tables are missing or empty.",
            stacklevel=2,
        )
        return pd.DataFrame()
    unit_summary = _load_unit_summary(exp_dir)

    state_ids = run.loc[run["state_id"].ne("S000_base"), "state_id"].tolist()

    base = _load_outlet(states_dir, "S000_base")
    t_base = base["time_seconds"].to_numpy()
    q_base = base["q_outlet_cms"].to_numpy()
    dt_seconds = float(np.median(np.diff(t_base)))

    base_metrics = {
        "peak_time_seconds": float(t_base[np.argmax(q_base)]),
        "rise95_time_seconds": _interp_first_crossing_time(t_base, q_base, 0.95 * np.max(q_base)),
        "plateau995_center_seconds": _plateau_center_time(t_base, q_base, frac=0.995),
        "centroid50_seconds": _weighted_centroid_time(t_base, q_base, frac=0.50),
        "centroid75_seconds": _weighted_centroid_time(t_base, q_base, frac=0.75),
        "peak_window_center95_seconds": _peak_window_center_time(t_base, q_base, frac=0.95),
        "peak_window_centroid95_seconds": _peak_window_centroid_time(t_base, q_base, frac=0.95),
    }

    rows: list[dict[str, Any]] = []
    for state_id in state_ids:
        df = _load_outlet(states_dir, state_id)
        t = df["time_seconds"].to_numpy()
        q = df["q_outlet_cms"].to_numpy()
        metrics = {
            "state_id": state_id,
            "peak_time_seconds": float(t[np.argmax(q)]),
            "rise95_time_seconds": _interp_first_crossing_time(t, q, 0.95 * np.max(q)),
            "plateau995_center_seconds": _plateau_center_time(t, q, frac=0.995),
            "centroid50_seconds": _weighted_centroid_time(t, q, frac=0.50),
            "centroid75_seconds": _weighted_centroid_time(t, q, frac=0.75),
            "peak_window_center95_seconds": _peak_window_center_time(t, q, frac=0.95),
            "peak_window_centroid95_seconds": _peak_window_centroid_time(t, q, frac=0.95),
            "xcorr_lag_hours": _xcorr_lag_hours(q_base, q, dt_seconds),
        }
        for key, base_value in base_metrics.items():
            metrics[key.replace("_seconds", "_shift_hours")] = (metrics[key] - base_value) / SECONDS_PER_HOUR
        rows.append(metrics)

    summary = pd.DataFrame(rows).merge(
        comp[
            [
                "state_id",
                "child_object_id",
                "target_type",
                "target_id",
                "Lambda_matched",
                "K_eq_matched_base",
                "K_variant_realized",
                "boundary_relation",
                "comparison_valid_for_matched",
            ]
        ],
        on="state_id",
        how="left",
    )
    summary = summary.loc[summary["comparison_valid_for_matched"].fillna(False)].copy()
    summary["target_id"] = pd.to_numeric(summary["target_id"], errors="coerce").astype("Int64")

    summary = (
        summary.merge(
            matched[
                [
                    "state_id",
                    "child_object_id",
                    "matched_base_link_ids",
                    "matched_base_n_links",
                    "matched_base_upstream_node",
                    "matched_base_downstream_node",
                    "Q_unit_matched_base",
                ]
            ],
            on=["state_id", "child_object_id"],
            how="left",
        )
        .merge(
            child[
                [
                    "state_id",
                    "child_object_id",
                    "child_link_ids",
                    "child_object_type",
                    "child_upstream_node",
                    "child_downstream_node",
                    "Q_unit_variant",
                ]
            ],
            on=["state_id", "child_object_id"],
            how="left",
        )
    )

    base_len_lookup, total_base_len, length_field = _build_length_lookup(exp_dir)
    summary["matched_base_len"] = summary["matched_base_link_ids"].map(
        lambda value: _matched_length_from_ids(value, base_len_lookup)
    )
    summary["matched_len_frac"] = summary["matched_base_len"] / total_base_len
    summary["neg_log10_lambda"] = -np.log10(summary["Lambda_matched"])
    summary["length_weighted_log_lambda"] = summary["matched_len_frac"] * summary["neg_log10_lambda"]

    if mode == "indep":
        summary = summary.merge(unit_summary, left_on="target_id", right_on="unit_id", how="left")
    else:
        summary["unit_id"] = pd.Series([pd.NA] * len(summary), dtype="Int64")
        summary["unit_class"] = pd.NA
        summary["is_compound"] = True
        summary["compound_unit_id"] = pd.Series([pd.NA] * len(summary), dtype="Int64")
        summary["compound_bubble_id"] = summary["target_id"]
        summary["in_compound_bubble"] = True
        summary["compound_bubble_role"] = "compound_bubble"

    summary["plot_category"] = summary.apply(_classify_plot_category, axis=1)
    summary["mode"] = mode
    summary["experiment_dir"] = str(exp_dir)
    summary["base_length_field"] = length_field
    summary["example_id"] = example_id
    summary["plot_unit_id"] = summary["target_id"]
    return summary


def _build_analysis_metric_frame(
    example_id: str,
    *,
    output_root: Path,
    anchored_suffix: str,
) -> pd.DataFrame:
    from .analysis import build_combined_analysis_frame

    indep_dir = output_root / f"{example_id}_indep{anchored_suffix}"
    compound_dir = output_root / f"{example_id}_compound_bubbles{anchored_suffix}"
    if not indep_dir.exists():
        return pd.DataFrame()

    analysis = build_combined_analysis_frame(
        indep_dir,
        compound_dir if compound_dir.exists() else None,
    )
    if analysis.empty:
        return analysis

    analysis = analysis.loc[analysis["state_id"].ne("S000_base")].copy()
    keep_columns = [
        "experiment_dir",
        "state_id",
        "delta_peak_time_hours",
        "delta_lag_to_inflow_peak_hours",
        "hydrograph_lag_vs_base_hours",
        "lag_to_inflow_peak_relative_to_base",
        "lag_to_inflow_peak_pct_shift_vs_base",
        "unit_to_bubble_lag_to_inflow_peak_relative",
        "unit_to_bubble_lag_to_inflow_peak_pct_shift",
    ]
    present_columns = [column for column in keep_columns if column in analysis.columns]
    if "experiment_dir" not in present_columns or "state_id" not in present_columns:
        return pd.DataFrame()
    analysis = analysis[present_columns].copy()
    analysis["experiment_dir"] = analysis["experiment_dir"].astype(str)
    return analysis.drop_duplicates(subset=["experiment_dir", "state_id"])


def build_multi_example_timing_frame(
    example_ids: Sequence[str | int],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    anchored_suffix: str = "__anchored",
) -> pd.DataFrame:
    output_root = Path(output_root)
    frames: list[pd.DataFrame] = []
    analysis_frames: list[pd.DataFrame] = []
    for raw_example_id in example_ids:
        example_id = _normalize_example_id(raw_example_id)
        found_any = False
        found_nonempty = False
        for mode, name in (
            ("indep", f"{example_id}_indep{anchored_suffix}"),
            ("compound", f"{example_id}_compound_bubbles{anchored_suffix}"),
        ):
            exp_dir = output_root / name
            if not exp_dir.exists():
                continue
            found_any = True
            frame = _build_experiment_frame(example_id, mode, exp_dir)
            if not frame.empty:
                found_nonempty = True
                frames.append(frame)
        if not found_any:
            raise FileNotFoundError(f"No anchored experiment directories were found for example {example_id!r}.")
        if not found_nonempty:
            warnings.warn(
                f"Example {example_id!r} was found but has no non-empty anchored outputs to plot.",
                stacklevel=2,
            )
        analysis_frame = _build_analysis_metric_frame(
            example_id,
            output_root=output_root,
            anchored_suffix=anchored_suffix,
        )
        if not analysis_frame.empty:
            analysis_frames.append(analysis_frame)
    if not frames:
        raise ValueError("No example frames were built.")
    frame = pd.concat(frames, ignore_index=True)
    if analysis_frames:
        analysis_frame = pd.concat(analysis_frames, ignore_index=True).drop_duplicates(
            subset=["experiment_dir", "state_id"]
        )
        frame = frame.merge(analysis_frame, on=["experiment_dir", "state_id"], how="left")
    return frame


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _build_base_context(exp_dir: Path) -> dict[str, Any]:
    manifest = json.loads((exp_dir / "equivalent_resistance" / "equivalent_resistance_manifest.json").read_text())
    eq_cfg = manifest["equivalent_resistance_config"]
    length_field = eq_cfg["length_field_candidates"][0]

    base_variant = gpd.read_file(exp_dir / "states" / "S000_base" / "variant" / "widths" / "links_with_width_families.gpkg")
    width_field = _first_present(base_variant.columns, eq_cfg["width_field_candidates"])
    if width_field is None:
        raise ValueError(f"Could not resolve a width field for {exp_dir}.")

    base_variant = base_variant.copy()
    base_variant["id_link"] = pd.to_numeric(base_variant["id_link"], errors="coerce").astype("Int64")
    base_variant["id_us_node"] = pd.to_numeric(base_variant["id_us_node"], errors="coerce").astype("Int64")
    base_variant["id_ds_node"] = pd.to_numeric(base_variant["id_ds_node"], errors="coerce").astype("Int64")
    base_variant[length_field] = pd.to_numeric(base_variant[length_field], errors="coerce")
    base_variant[width_field] = pd.to_numeric(base_variant[width_field], errors="coerce")
    base_variant = base_variant.dropna(subset=["id_link"]).drop_duplicates(subset=["id_link"])

    len_lookup = base_variant.set_index("id_link")[length_field]
    width_lookup = base_variant.set_index("id_link")[width_field]
    us_lookup = base_variant.set_index("id_link")["id_us_node"]
    ds_lookup = base_variant.set_index("id_link")["id_ds_node"]

    base_directed_dir = exp_dir / "states" / "S000_base" / "variant" / "directed"
    directed_paths = sorted(base_directed_dir.glob("*_directed_links.gpkg"))
    if not directed_paths:
        raise FileNotFoundError(f"Could not resolve base directed links for {exp_dir}.")
    base_directed = gpd.read_file(directed_paths[0])
    base_directed = base_directed.copy()
    base_directed["id_link"] = pd.to_numeric(base_directed["id_link"], errors="coerce").astype("Int64")
    base_directed["id_us_node"] = pd.to_numeric(base_directed["id_us_node"], errors="coerce").astype("Int64")
    base_directed["id_ds_node"] = pd.to_numeric(base_directed["id_ds_node"], errors="coerce").astype("Int64")
    base_directed = base_directed.dropna(subset=["id_link", "id_us_node", "id_ds_node"]).drop_duplicates(subset=["id_link"])
    if "unit_ids" not in base_directed.columns:
        base_directed["unit_ids"] = None
    base_directed["width_value"] = base_directed["id_link"].map(width_lookup)
    base_graph = nx.DiGraph()
    for row in base_directed.itertuples(index=False):
        base_graph.add_edge(int(row.id_us_node), int(row.id_ds_node))
    mandatory_edge_pairs = set(mandatory_edges(base_graph))
    base_in_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    base_out_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base_directed.itertuples(index=False):
        edge_record = {
            "id_link": int(row.id_link),
            "id_us_node": int(row.id_us_node),
            "id_ds_node": int(row.id_ds_node),
            "unit_ids": row.unit_ids,
            "width_value": float(row.width_value) if pd.notna(row.width_value) else float("nan"),
            "is_mandatory": (int(row.id_us_node), int(row.id_ds_node)) in mandatory_edge_pairs,
        }
        base_in_edges[int(row.id_ds_node)].append(edge_record)
        base_out_edges[int(row.id_us_node)].append(edge_record)

    base_rapid = pd.read_csv(exp_dir / "states" / "S000_base" / "rapid" / "prep" / "rapid_link_attributes.csv")
    dist_cols = [column for column in ["parent_link_id", "network_us_dist_out_m", "network_ds_dist_out_m"] if column in base_rapid.columns]
    dist_df = base_rapid[dist_cols].copy()
    dist_df["parent_link_id"] = pd.to_numeric(dist_df["parent_link_id"], errors="coerce").astype("Int64")
    dist_df["network_us_dist_out_m"] = pd.to_numeric(dist_df["network_us_dist_out_m"], errors="coerce")
    dist_df["network_ds_dist_out_m"] = pd.to_numeric(dist_df["network_ds_dist_out_m"], errors="coerce")
    dist_df = dist_df.dropna(subset=["parent_link_id"]).drop_duplicates(subset=["parent_link_id"])
    dist_df["parent_mid_dist_out_m"] = (dist_df["network_us_dist_out_m"] + dist_df["network_ds_dist_out_m"]) / 2.0
    max_mid = float(dist_df["parent_mid_dist_out_m"].max()) if len(dist_df) else float("nan")
    if pd.notna(max_mid) and max_mid > 0:
        dist_df["along_network_frac"] = dist_df["parent_mid_dist_out_m"] / max_mid
    else:
        dist_df["along_network_frac"] = float("nan")
    dist_lookup = dist_df.set_index("parent_link_id")[["parent_mid_dist_out_m", "along_network_frac"]]

    return {
        "len_lookup": len_lookup,
        "width_lookup": width_lookup,
        "us_lookup": us_lookup,
        "ds_lookup": ds_lookup,
        "dist_lookup": dist_lookup,
        "base_in_edges": base_in_edges,
        "base_out_edges": base_out_edges,
        "mandatory_edge_pairs": mandatory_edge_pairs,
    }


def _child_length(exp_dir: Path, state_id: str, child_link_ids_text: object) -> float:
    child_link_ids = _parse_int_list(child_link_ids_text)
    if not child_link_ids:
        return float("nan")
    manifest = json.loads((exp_dir / "equivalent_resistance" / "equivalent_resistance_manifest.json").read_text())
    length_field = manifest["equivalent_resistance_config"]["length_field_candidates"][0]
    child_variant = gpd.read_file(exp_dir / "states" / state_id / "variant" / "widths" / "links_with_width_families.gpkg")
    child_variant = child_variant.drop_duplicates(subset=["id_link"])
    child_variant["id_link"] = pd.to_numeric(child_variant["id_link"], errors="coerce").astype("Int64")
    values = pd.to_numeric(child_variant.loc[child_variant["id_link"].isin(child_link_ids), length_field], errors="coerce")
    return float(values.sum())


def _is_unit_owned_link(unit_ids_value: object) -> bool:
    return len(_parse_optional_int_list(unit_ids_value)) > 0


def _first_non_unit_upstream_edge(row: pd.Series, ctx: dict[str, Any]) -> dict[str, Any] | None:
    start_node = pd.to_numeric(row.get("matched_base_upstream_node"), errors="coerce")
    if pd.isna(start_node):
        return None

    base_in_edges: dict[int, list[dict[str, Any]]] = ctx["base_in_edges"]
    queue: list[tuple[int, int]] = [(int(start_node), 0)]
    visited_nodes: set[int] = set()
    discovered_by_depth: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)

    while queue:
        node, depth = queue.pop(0)
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        for edge in base_in_edges.get(node, []):
            if _is_unit_owned_link(edge.get("unit_ids")):
                queue.append((int(edge["id_us_node"]), depth + 1))
            elif bool(edge.get("is_mandatory", False)):
                discovered_by_depth[depth][int(edge["id_link"])] = edge

    if not discovered_by_depth:
        return None

    first_depth = min(discovered_by_depth)
    candidates = discovered_by_depth[first_depth]
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return None


def _compute_target_base_share(row: pd.Series, ctx: dict[str, Any]) -> tuple[float, int | None]:
    upstream_edge = _first_non_unit_upstream_edge(row, ctx)
    if upstream_edge is None:
        return float("nan"), None

    target_links = set(_parse_int_list(row.get("matched_base_link_ids")))
    if not target_links:
        return float("nan"), int(upstream_edge["id_link"])

    base_graph = nx.DiGraph()
    base_out_edges: dict[int, list[dict[str, Any]]] = ctx["base_out_edges"]
    for us_node, edges in base_out_edges.items():
        for edge in edges:
            base_graph.add_edge(int(edge["id_us_node"]), int(edge["id_ds_node"]))
    if not nx.is_directed_acyclic_graph(base_graph):
        return float("nan"), int(upstream_edge["id_link"])

    start_node = int(upstream_edge["id_ds_node"])
    share_by_node: dict[int, float] = defaultdict(float)
    share_by_node[start_node] = 1.0
    entry_share = 0.0

    topo = list(nx.topological_sort(base_graph))
    try:
        start_index = topo.index(start_node)
    except ValueError:
        return float("nan"), int(upstream_edge["id_link"])

    for node in topo[start_index:]:
        incoming_share = float(share_by_node.get(int(node), 0.0))
        if incoming_share <= 0:
            continue
        outgoing = base_out_edges.get(int(node), [])
        if not outgoing:
            continue
        widths = np.asarray([edge["width_value"] for edge in outgoing], dtype=float)
        valid = np.isfinite(widths) & (widths > 0)
        if not np.any(valid):
            continue
        width_sum = float(widths[valid].sum())
        if width_sum <= 0:
            continue
        for edge, edge_width, is_valid in zip(outgoing, widths, valid, strict=False):
            if not is_valid:
                continue
            branch_share = incoming_share * (float(edge_width) / width_sum)
            edge_id = int(edge["id_link"])
            if edge_id in target_links:
                entry_share += branch_share
            else:
                share_by_node[int(edge["id_ds_node"])] += branch_share

    return float(entry_share), int(upstream_edge["id_link"])


def _width_fraction_of_total(row: pd.Series, ctx: dict[str, Any]) -> float:
    share, _ = _compute_target_base_share(row, ctx)
    return float(share) if np.isfinite(share) else float("nan")


def build_geometry_context_frame(plot_df: pd.DataFrame) -> pd.DataFrame:
    geom_df = plot_df.copy()
    contexts: dict[str, dict[str, Any]] = {}
    for exp_dir_str in geom_df["experiment_dir"].dropna().unique():
        contexts[str(exp_dir_str)] = _build_base_context(Path(exp_dir_str))

    matched_lens: list[float] = []
    matched_dist_mid: list[float] = []
    matched_dist_frac: list[float] = []
    child_lens: list[float] = []
    width_fracs: list[float] = []
    upstream_reference_link_ids: list[float] = []

    for row in geom_df.itertuples(index=False):
        exp_dir = Path(row.experiment_dir)
        ctx = contexts[str(exp_dir)]
        len_lookup = ctx["len_lookup"]
        dist_lookup = ctx["dist_lookup"]
        parent_ids = _parse_int_list(row.matched_base_link_ids)
        parent_series = len_lookup.reindex(parent_ids)
        matched_base_len = float(parent_series.sum()) if len(parent_ids) else float("nan")
        matched_lens.append(matched_base_len)

        if len(parent_ids):
            dist_frame = dist_lookup.reindex(parent_ids).reset_index()
            dist_frame["base_link_len"] = parent_series.values
            dist_frame = dist_frame.dropna(subset=["base_link_len"])
            weight_sum = float(dist_frame["base_link_len"].sum()) if len(dist_frame) else float("nan")
            if len(dist_frame) and pd.notna(weight_sum) and weight_sum > 0:
                matched_dist_mid.append(float(np.average(dist_frame["parent_mid_dist_out_m"], weights=dist_frame["base_link_len"])))
                matched_dist_frac.append(float(np.average(dist_frame["along_network_frac"], weights=dist_frame["base_link_len"])))
            else:
                matched_dist_mid.append(float("nan"))
                matched_dist_frac.append(float("nan"))
        else:
            matched_dist_mid.append(float("nan"))
            matched_dist_frac.append(float("nan"))

        row_series = pd.Series(row._asdict())
        child_lens.append(_child_length(exp_dir, row.state_id, row.child_link_ids))
        share, upstream_link_id = _compute_target_base_share(row_series, ctx)
        width_fracs.append(share)
        upstream_reference_link_ids.append(float(upstream_link_id) if upstream_link_id is not None else float("nan"))

    geom_df["matched_base_len"] = matched_lens
    geom_df["along_network_mid_dist_out_m"] = matched_dist_mid
    geom_df["along_network_frac"] = matched_dist_frac
    geom_df["child_len"] = child_lens
    geom_df["child_parent_len_frac"] = geom_df["child_len"] / geom_df["matched_base_len"]
    geom_df["upstream_reference_link_id"] = upstream_reference_link_ids
    geom_df["width_fraction_of_total_raw"] = width_fracs
    geom_df["width_fraction_of_total"] = pd.to_numeric(geom_df["width_fraction_of_total_raw"], errors="coerce").clip(upper=1.0)
    geom_df["child_parent_weighted_log_lambda"] = geom_df["child_parent_len_frac"] * geom_df["neg_log10_lambda"]
    geom_df["child_parent_width_weighted_log_lambda"] = (
        geom_df["child_parent_len_frac"]
        * geom_df["neg_log10_lambda"]
        * geom_df["width_fraction_of_total"]
    )
    geom_df["lambda_matched"] = geom_df["Lambda_matched"]
    geom_df["length_fraction"] = geom_df["matched_len_frac"]
    geom_df["child_length_fraction"] = geom_df["child_parent_len_frac"]
    geom_df["width_fraction"] = geom_df["width_fraction_of_total"]
    geom_df["log_lambda"] = geom_df["neg_log10_lambda"]
    if {"K_variant_realized", "K_eq_matched_base"}.issubset(geom_df.columns):
        denom = pd.to_numeric(geom_df["K_eq_matched_base"], errors="coerce")
        num = pd.to_numeric(geom_df["K_variant_realized"], errors="coerce")
        geom_df["k_ratio_to_matched_base"] = np.where(denom.abs() > 1e-12, num / denom, np.nan)
    return geom_df


def make_timing_scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str = "centroid50_shift_hours",
    color: str = "plot_category",
    symbol: str = "mode",
    title: str | None = None,
    hover_fields: Sequence[str] | None = None,
) -> Any:
    if x not in frame.columns:
        raise KeyError(f"X metric {x!r} is not available.")
    if y not in frame.columns:
        raise KeyError(f"Timing metric {y!r} is not available.")
    if hover_fields is None:
        hover_fields = ["example_id", "target_id"]

    labels = dict(PLOT_DIMENSION_LABELS)
    labels[x] = labels.get(x, x)
    labels[y] = labels.get(y, y)

    kwargs: dict[str, Any] = {
        "data_frame": frame,
        "x": x,
        "y": y,
        "hover_data": list(hover_fields),
        "labels": labels,
        "title": title or f"{labels.get(y, y)} vs {labels.get(x, x)}",
    }
    if symbol:
        kwargs["symbol"] = symbol

    if color == "along_network_frac":
        kwargs["color"] = color
        kwargs["color_continuous_scale"] = "Viridis"
    else:
        kwargs["color"] = color
        kwargs["color_discrete_map"] = CATEGORY_COLOR_MAP

    fig = px.scatter(**kwargs)
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="black")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

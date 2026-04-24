"""Shared runner utilities extracted from the legacy sampled/sensitivity runners."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd


def _compute_k_value(
    length_m: float,
    width_m: float,
    kb: float = 20.0,
    slope: float = 1e-3,
    n_manning: float = 0.35,
) -> float:
    if length_m <= 0 or width_m <= 0 or slope <= 0:
        return np.nan
    k_value = (3 / 5) * n_manning * (
        (length_m / (slope**0.5)) * ((kb ** (2 / 3)) / (width_m ** (2 / 3)))
    )
    return round(float(k_value), 4)


def _pick_source_node(G: nx.MultiDiGraph):
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    if not sources:
        return None, "no source nodes (in_degree==0)"
    if len(sources) == 1:
        return sources[0], None
    sources_sorted = sorted(sources, key=lambda n: G.nodes[n].get("x", 0.0))
    return sources_sorted[0], f"multiple sources; picked min-x={sources_sorted[0]}"


def _find_first_bifurcation_node(G: nx.MultiDiGraph, source):
    out_edges = list(G.out_edges(source, keys=True, data=True))
    if len(out_edges) == 0:
        return None, "source has no outgoing edge"
    if len(out_edges) != 1:
        return None, "source has multiple outgoing edges; expected exactly one"
    _, v, _, _ = out_edges[0]
    return v, None


def _pick_outlet_node(G: nx.MultiDiGraph):
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        return None, "no outlet nodes (out_degree==0)"
    if len(sinks) == 1:
        return sinks[0], None
    sinks_sorted = sorted(sinks, key=lambda n: G.nodes[n].get("x", 0.0))
    return sinks_sorted[-1], f"multiple outlets; picked max-x={sinks_sorted[-1]}"


def _find_final_confluence_node(G: nx.MultiDiGraph, outlet):
    in_edges = list(G.in_edges(outlet, keys=True, data=True))
    if len(in_edges) == 0:
        return None, "outlet has no incoming edge"
    if len(in_edges) != 1:
        return None, "outlet has multiple incoming edges; expected exactly one"
    u, _, _, _ = in_edges[0]
    return u, None


def _build_reach_graph(G: nx.MultiDiGraph, kb: float = 20.0, S_global: float = 1e-3):
    reach_info = {}
    node_to_out = {}
    node_to_in = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        rid = int(data["reach_id"])
        length = float(data["length"])
        width = float(data["width"])
        S_local = float(data.get("slope_local", S_global))
        k_val = _compute_k_value(length, width, kb=kb, slope=S_local)
        reach_info[rid] = {"u": u, "v": v, "K": k_val}
        node_to_out.setdefault(u, []).append(rid)
        node_to_in.setdefault(v, []).append(rid)

    RG = nx.DiGraph()
    for rid, info in reach_info.items():
        RG.add_node(rid, K=info["K"])
    for rid, info in reach_info.items():
        downstream = node_to_out.get(info["v"], [])
        for rid_dn in downstream:
            RG.add_edge(rid, rid_dn)
    return RG, reach_info, node_to_out, node_to_in


def compute_q_weighted_metrics(
    G: nx.MultiDiGraph,
    qout: np.ndarray,
    riv_order: list[int],
    dt_seconds: float,
    kb: float = 20.0,
    S_global: float = 1e-3,
    max_paths: int = 100,
    network_id: int | None = None,
):
    """Compute Q-weighted K metrics between first bifurcation and final confluence."""
    RG, _reach_info, node_to_out, node_to_in = _build_reach_graph(G, kb=kb, S_global=S_global)
    if not nx.is_directed_acyclic_graph(RG):
        return None, {"error": "reach graph is not a DAG"}

    rid_to_idx = {rid: i for i, rid in enumerate(riv_order)}
    for rid in RG.nodes:
        idx = rid_to_idx.get(rid)
        if idx is None:
            RG.nodes[rid]["w"] = 0.0
            continue
        RG.nodes[rid]["w"] = float(np.sum(qout[:, idx])) * float(dt_seconds)

    source, source_msg = _pick_source_node(G)
    if source is None:
        return None, {"error": source_msg}
    B, b_msg = _find_first_bifurcation_node(G, source)
    if B is None:
        return None, {"error": b_msg}
    outlet, outlet_msg = _pick_outlet_node(G)
    if outlet is None:
        return None, {"error": outlet_msg}
    C, c_msg = _find_final_confluence_node(G, outlet)
    if C is None:
        return None, {"error": c_msg}

    B_reaches = node_to_out.get(B, [])
    C_reaches = node_to_in.get(C, [])
    if not B_reaches or not C_reaches:
        return None, {"error": "B->C reach set empty"}

    desc = set()
    for r in B_reaches:
        desc |= nx.descendants(RG, r)
        desc.add(r)
    anc = set()
    for r in C_reaches:
        anc |= nx.ancestors(RG, r)
        anc.add(r)
    sub_nodes = desc & anc
    if not sub_nodes:
        return None, {"error": "no B->C subgraph nodes"}
    SG = RG.subgraph(sub_nodes).copy()

    tau = {}
    topo = list(nx.topological_sort(SG))
    for r in topo:
        preds = list(SG.predecessors(r))
        K_r = float(SG.nodes[r].get("K", np.nan))
        if len(preds) == 0:
            tau[r] = K_r
        else:
            num = sum(float(SG.nodes[p].get("w", 0.0)) * tau[p] for p in preds)
            den = sum(float(SG.nodes[p].get("w", 0.0)) for p in preds)
            tau[r] = K_r if den <= 0 else K_r + (num / den)

    W = sum(float(SG.nodes[r].get("w", 0.0)) for r in SG.nodes)
    if W <= 0:
        K_Q = np.nan
        tau_mean = np.nan
        tau_var = np.nan
    else:
        K_Q = sum(float(SG.nodes[r].get("w", 0.0)) * float(SG.nodes[r].get("K", 0.0)) for r in SG.nodes) / W
        tau_mean = sum(float(SG.nodes[r].get("w", 0.0)) * float(tau[r]) for r in SG.nodes) / W
        tau_var = sum(
            float(SG.nodes[r].get("w", 0.0)) * (float(tau[r]) - tau_mean) ** 2 for r in SG.nodes
        ) / W

    best_cap = -np.inf
    K_dom = np.nan
    n_paths = 0
    paths_exceeded = False
    for s in B_reaches:
        for t in C_reaches:
            for path in nx.all_simple_paths(SG, source=s, target=t):
                n_paths += 1
                if n_paths > max_paths:
                    paths_exceeded = True
                    break
                cap = min(float(SG.nodes[r].get("w", 0.0)) for r in path)
                if cap > best_cap:
                    best_cap = cap
                    K_dom = sum(float(SG.nodes[r].get("K", 0.0)) for r in path)
            if paths_exceeded:
                break
        if paths_exceeded:
            break

    info = {
        "B_node": B,
        "C_node": C,
        "source_node": source,
        "outlet_node": outlet,
        "source_note": source_msg,
        "outlet_note": outlet_msg,
    }
    if b_msg:
        info["B_note"] = b_msg
    if c_msg:
        info["C_note"] = c_msg

    metrics = {
        "K_Q": K_Q,
        "tau_mean": tau_mean,
        "tau_var": tau_var,
        "K_dom": K_dom,
        "n_paths": n_paths,
        "paths_exceeded": bool(paths_exceeded),
    }
    if paths_exceeded:
        nid_str = f" network_id={network_id}" if network_id is not None else ""
        print(f"[WARN]{nid_str}: paths exceeded max_paths={max_paths}; K_dom set to NaN")

    return metrics, info


def _find_outlet_reach_id(G: nx.MultiDiGraph) -> int | None:
    candidates = []
    for _u, v, _k, data in G.edges(keys=True, data=True):
        if G.out_degree(v) == 0:
            candidates.append((float(G.nodes[v]["x"]), int(data["reach_id"])))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _find_source_reach_id(G: nx.MultiDiGraph) -> tuple[int | None, str | None]:
    source, source_msg = _pick_source_node(G)
    if source is None:
        return None, source_msg
    out_edges = list(G.out_edges(source, keys=True, data=True))
    if len(out_edges) == 0:
        return None, "source has no outgoing edges"
    if len(out_edges) > 1:
        return None, "source has multiple outgoing edges"
    rid = int(out_edges[0][3].get("reach_id"))
    return rid, None


def _load_summary_table(sampled_dir: str | Path) -> pd.DataFrame:
    sampled_dir = Path(sampled_dir)
    parquet = sampled_dir / "summary.parquet"
    print("print parquet: ", parquet)
    csv_path = sampled_dir / "summary.csv"
    if parquet.exists():
        import duckdb

        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{parquet}')").df()
        print(df.shape[0])
        return df
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("summary.parquet or summary.csv not found in sampled_dir")


def _load_run_meta(sampled_dir: str | Path) -> dict:
    sampled_dir = Path(sampled_dir)
    meta_path = sampled_dir / "run_meta_sampling.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta_sampling.json not found in {sampled_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_recipes_path(sampled_dir: str | Path) -> Path:
    sampled_dir = Path(sampled_dir)
    for name in ["networks.jsonl.gz", "networks.json.gz"]:
        p = sampled_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("networks.jsonl.gz or networks.json.gz not found in sampled_dir")


def maximin_sample_ids(
    summary_df: pd.DataFrame,
    n_total: int,
    cols: tuple = ("k_mean", "ebi_mean"),
    seed: int = 123,
    log_cols: tuple = ("k_mean",),
    start: str = "extreme",
) -> list:
    """Greedy maximin selection on the specified summary columns."""
    if n_total <= 0:
        return []
    df = summary_df.copy().dropna(subset=[*cols, "network_id"])
    if df.empty:
        return []

    X = df.loc[:, cols].to_numpy(dtype=float)
    X_t = X.copy()
    for j, c in enumerate(cols):
        if c in log_cols:
            X_t[:, j] = np.log(X_t[:, j] + 1e-12)
    mu = X_t.mean(axis=0)
    sd = X_t.std(axis=0, ddof=0) + 1e-12
    Z = (X_t - mu) / sd
    m = Z.shape[0]
    if n_total >= m:
        return df["network_id"].astype(int).tolist()

    rng = np.random.default_rng(seed)
    if start == "random":
        first = int(rng.integers(0, m))
    elif start == "extreme":
        centroid = Z.mean(axis=0)
        d2 = np.sum((Z - centroid) ** 2, axis=1)
        first = int(np.argmax(d2))
    else:
        raise ValueError("start must be 'random' or 'extreme'")

    selected = np.empty(n_total, dtype=int)
    selected_mask = np.zeros(m, dtype=bool)
    selected[0] = first
    selected_mask[first] = True
    diff = Z - Z[first]
    min_d2 = np.sum(diff * diff, axis=1)
    min_d2[selected_mask] = -np.inf
    for i in range(1, n_total):
        nxt = int(np.argmax(min_d2))
        selected[i] = nxt
        selected_mask[nxt] = True
        diff = Z - Z[nxt]
        d2_new = np.sum(diff * diff, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[selected_mask] = -np.inf
    return df.iloc[selected]["network_id"].astype(int).tolist()


def resample(curve, N):
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, N)
    return np.interp(x_new, x_old, curve)


def build_first_part(
    dt,
    rise_hours: float = 12.0,
    fall_hours: float | None = None,
    peak: float = 5e-10,
    baseflow: float = 1e-10,
):
    """Build a dt-invariant runoff curve and return sampled forcing plus bounds."""
    if fall_hours is None:
        fall_hours = rise_hours

    T_A = 24 * 1 * 60 * 60
    T_B = 24 * 1 * 60 * 60
    T_C = 24 * 3 * 60 * 60
    T_D = float(rise_hours) * 60 * 60
    T_E = float(fall_hours) * 60 * 60
    T_G = 24 * 3 * 60 * 60

    durations = [T_A, T_B, T_C, T_D, T_E, T_G]
    T_total = sum(durations)
    N_base = 3000
    t_base = np.linspace(0, T_total, N_base)

    def segment_mask(t, t0, t1):
        return (t >= t0) & (t < t1)

    r = np.zeros_like(t_base)
    t1 = T_A
    baseflow = float(baseflow)
    peak = float(peak)
    t2 = t1 + T_B
    mask = segment_mask(t_base, t1, t2)
    r[mask] = np.linspace(0, baseflow, mask.sum())
    t3 = t2 + T_C
    mask = segment_mask(t_base, t2, t3)
    r[mask] = baseflow
    t4 = t3 + T_D
    mask = segment_mask(t_base, t3, t4)
    r[mask] = np.linspace(baseflow, peak, mask.sum())
    t5 = t4 + T_E
    mask = segment_mask(t_base, t4, t5)
    r[mask] = np.linspace(peak, baseflow, mask.sum())
    t6 = t5 + T_G
    mask = segment_mask(t_base, t5, t6)
    r[mask] = baseflow

    t = np.arange(0, T_total, dt)
    runOffC = np.interp(t, t_base, r)
    segment_bounds = {
        "A": (0.0, T_A),
        "B": (T_A, T_A + T_B),
        "C": (T_A + T_B, T_A + T_B + T_C),
        "D": (T_A + T_B + T_C, T_A + T_B + T_C + T_D),
        "E": (T_A + T_B + T_C + T_D, T_A + T_B + T_C + T_D + T_E),
        "G": (T_A + T_B + T_C + T_D + T_E, T_total),
    }
    return runOffC.tolist(), durations, segment_bounds


def _load_rapid_tools():
    """Import the shared RAPID helpers lazily for runner modules."""
    rapid_src_dir = Path(__file__).resolve().parents[4] / "RAPID" / "src"
    if str(rapid_src_dir) not in sys.path:
        sys.path.insert(0, str(rapid_src_dir))

    from rapid_tools.adapters.synthetic import (
        build_single_edge_graph,
        rivernetwork_to_rapid_graph,
    )
    from rapid_tools.engine import run_rapid
    from rapid_tools.prep import (
        compute_area_csv,
        compute_dt_from_K,
        compute_reach_ratios,
        create_conn_file,
        create_riv_file,
        create_routing_parameters,
        create_runoff,
    )

    return {
        "build_single_edge_graph": build_single_edge_graph,
        "compute_area_csv": compute_area_csv,
        "compute_dt_from_K": compute_dt_from_K,
        "compute_reach_ratios": compute_reach_ratios,
        "create_conn_file": create_conn_file,
        "create_riv_file": create_riv_file,
        "create_routing_parameters": create_routing_parameters,
        "create_runoff": create_runoff,
        "rivernetwork_to_rapid_graph": rivernetwork_to_rapid_graph,
        "run_rapid": run_rapid,
    }


__all__ = [
    "_compute_k_value",
    "_find_first_bifurcation_node",
    "_find_final_confluence_node",
    "_find_outlet_reach_id",
    "_find_source_reach_id",
    "_load_rapid_tools",
    "_load_run_meta",
    "_load_summary_table",
    "_pick_outlet_node",
    "_pick_recipes_path",
    "_pick_source_node",
    "_build_reach_graph",
    "build_first_part",
    "compute_q_weighted_metrics",
    "maximin_sample_ids",
    "resample",
]

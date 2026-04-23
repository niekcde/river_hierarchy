import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

from synthetic_admissable_networkx_part_save import RiverNetworkNX
try:
    from tqdm import tqdm
except ImportError:  # Fallback: no progress bar if tqdm isn't available.
    class _TqdmFallback:
        def __init__(self, iterable=None, **_kwargs):
            self.iterable = iterable
        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)
        def update(self, _n=1):
            return None
        def close(self):
            return None
    def tqdm(iterable=None, **_kwargs):
        return _TqdmFallback(iterable, **_kwargs)


def _pick_recipes_path(sampled_dir: str | Path) -> Path:
    sampled_dir = Path(sampled_dir)
    for name in ["networks.jsonl.gz", "networks.json.gz"]:
        p = sampled_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("networks.jsonl.gz or networks.json.gz not found in sampled_dir")


def compute_k_value(length_m: float, width_m: float) -> float:
    if length_m <= 0 or width_m <= 0:
        return np.nan
    kb = 20
    S = 1e-3
    n = 0.35
    k_value = (3 / 5) * n * ((length_m / (S**0.5)) * ((kb ** (2 / 3)) / (width_m ** (2 / 3))))
    return round(float(k_value), 4)


def build_k_graph(net) -> nx.DiGraph:
    """
    Build a directed graph with edge attribute K based on the same
    length/width logic used in synthetic_runs.
    """
    G_src = net.G
    G = nx.MultiDiGraph()
    for n, data in G_src.nodes(data=True):
        G.add_node(n, x=float(data.get("x", 0.0)))

    for u, v, k, data in G_src.edges(keys=True, data=True):
        xu = float(G_src.nodes[u].get("x", 0.0))
        xv = float(G_src.nodes[v].get("x", 0.0))
        if xv <= xu:
            continue
        width = float(data.get("w", 0.0))
        length = float(xv - xu)
        k_val = compute_k_value(length, width)
        
        if G.has_edge(u, v, k):
            # Keep the smaller K if a parallel edge exists.
            existing = G[u][v].get("K", np.nan)
            if np.isnan(existing) or (not np.isnan(k_val) and k_val < existing):
                G[u][v]["K"] = k_val
                G[u][v]["length"] = length
                G[u][v]["width"] = width
        else:
            G.add_edge(u, v, K=k_val, length=length, width=width)
    return G


def _pick_source(G: nx.DiGraph):
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    if not sources:
        return None, "no source nodes (in_degree==0)"
    if len(sources) == 1:
        return sources[0], None
    sources_sorted = sorted(sources, key=lambda n: G.nodes[n].get("x", 0.0))
    return sources_sorted[0], f"multiple sources; picked min-x={sources_sorted[0]}"


def _find_first_bifurcation(G: nx.DiGraph, source):
    current = source
    visited = {current}
    while True:
        out_edges = list(G.out_edges(current))
        if len(out_edges) == 0:
            return None, "reached sink before any bifurcation"
        successors = list(G.successors(current))
        if len(out_edges) == 1 and len(successors) == 1:
            current = successors[0]
            if current in visited:
                return None, "cycle detected while searching for first bifurcation"
            visited.add(current)
            continue
        return current, None


def _pick_outlet(G: nx.DiGraph):
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        return None, "no outlet nodes (out_degree==0)"
    if len(sinks) == 1:
        return sinks[0], None
    sinks_sorted = sorted(sinks, key=lambda n: G.nodes[n].get("x", 0.0))
    return sinks_sorted[-1], f"multiple outlets; picked max-x={sinks_sorted[-1]}"


def _find_final_confluence(G: nx.DiGraph, outlet):
    current = outlet
    visited = {current}
    while True:
        preds = list(G.predecessors(current))
        if len(preds) == 0:
            return None, "reached source before any confluence"
        if len(preds) == 1:
            current = preds[0]
            if current in visited:
                return None, "cycle detected while searching for final confluence"
            visited.add(current)
            continue
        return current, None


def _path_length(G: nx.DiGraph, path) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(G[u][v]["K"])
    return total


def compute_k_metrics_for_network(
    net,
    max_paths: int = 100,
):
    G = build_k_graph(net)
    if not nx.is_directed_acyclic_graph(G):
        return None, {"error": "graph is not a DAG"}

    source, source_msg = _pick_source(G)
    if source is None:
        return None, {"error": source_msg}

    B, b_msg = _find_first_bifurcation(G, source)
    if B is None:
        return None, {"error": b_msg}

    outlet, outlet_msg = _pick_outlet(G)
    if outlet is None:
        return None, {"error": outlet_msg}

    C, c_msg = _find_final_confluence(G, outlet)
    if C is None:
        return None, {"error": c_msg}

    if not nx.has_path(G, B, C):
        return None, {"error": "no path from B to C"}

    # Subgraph of paths from B to C
    desc = nx.descendants(G, B) | {B}
    anc = nx.ancestors(G, C) | {C}
    sub_nodes = desc & anc
    SG = G.subgraph(sub_nodes).copy()

    # Check for NaN K values in relevant edges
    for u, v, data in SG.edges(data=True):
        if np.isnan(data.get("K", np.nan)):
            return None, {"error": "NaN K value encountered"}

    K_min = nx.shortest_path_length(SG, source=B, target=C, weight="K")

    # Longest path in DAG via DP
    topo = list(nx.topological_sort(SG))
    dist = {n: -np.inf for n in SG.nodes}
    dist[B] = 0.0
    for u in topo:
        if dist[u] == -np.inf:
            continue
        for v in SG.successors(u):
            w = float(SG[u][v]["K"])
            dist[v] = max(dist[v], dist[u] + w)
    K_max = dist[C]

    # Enumerate paths up to max_paths
    total = 0.0
    n_paths = 0
    exceeded = False
    for path in nx.all_simple_paths(SG, source=B, target=C):
        n_paths += 1
        if n_paths > max_paths:
            exceeded = True
            break
        total += _path_length(SG, path)

    K_avg = np.nan
    if n_paths > 0 and not exceeded:
        K_avg = total / n_paths

    info = {
        "B": B,
        "C": C,
        "source": source,
        "outlet": outlet,
        "source_note": source_msg,
        "outlet_note": outlet_msg,
    }
    if b_msg:
        info["B_note"] = b_msg
    if c_msg:
        info["C_note"] = c_msg

    metrics = {
        "K_min": K_min,
        "K_max": K_max,
        "delta_K": K_max - K_min,
        "K_avg_path": K_avg,
        "n_paths": n_paths,
        "paths_exceeded": bool(exceeded),
    }
    return metrics, info


def compute_metrics(
    recipes_path: Path,
    network_ids: list[int],
    max_paths: int,
):
    rows = []
    target_ids = set(int(n) for n in network_ids)
    found_ids = set()
    progress = tqdm(total=len(target_ids), desc="Computing K metrics")

    def handle_network(nid: int, recipe: dict):
        try:
            net = RiverNetworkNX.from_recipe(recipe)
            metrics, info = compute_k_metrics_for_network(net, max_paths=max_paths)
            if metrics is None:
                row = {"network_id": int(nid), "error": info.get("error", "unknown")}
                rows.append(row)
                print(f"[WARN] network_id={nid}: {row['error']}")
            else:
                if metrics.get("paths_exceeded"):
                    print(
                        f"[WARN] network_id={nid}: "
                        f"paths exceeded max_paths={max_paths}; "
                        f"skipping K_avg_path"
                    )
                row = {"network_id": int(nid)}
                row.update(metrics)
                row.update(info)
                rows.append(row)
        except Exception as exc:
            rows.append({"network_id": int(nid), "error": str(exc)})
            print(f"[ERROR] network_id={nid}: {exc}")

    recipes_name = recipes_path.name
    if recipes_name.endswith("jsonl.gz"):
        with gzip.open(recipes_path, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx not in target_ids:
                    continue
                recipe = json.loads(line)
                handle_network(idx, recipe)
                found_ids.add(idx)
                progress.update(1)
                if len(found_ids) == len(target_ids):
                    break
    elif recipes_name.endswith("json.gz"):
        with gzip.open(recipes_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("networks.json.gz must contain a list of recipes")
        for idx, recipe in enumerate(data):
            if idx not in target_ids:
                continue
            handle_network(idx, recipe)
            found_ids.add(idx)
            progress.update(1)
            if len(found_ids) == len(target_ids):
                break
    else:
        raise ValueError(f"Unsupported recipes file: {recipes_path}")

    progress.close()

    missing = target_ids.difference(found_ids)
    for nid in sorted(missing):
        rows.append({"network_id": int(nid), "error": "network_id not found in recipes file"})
        print(f"[WARN] network_id={nid}: not found in recipes file")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute K-path metrics for selected networks.")
    parser.add_argument("--sampled-dir", required=True, help="Directory containing networks.jsonl.gz")
    parser.add_argument("--q-outlet", required=True, help="Path to q_outlet.parquet")
    parser.add_argument("--out", default="k_metrics.csv", help="Output CSV path")
    parser.add_argument("--max-paths", type=int, default=100, help="Max paths before skipping K_avg_path")
    args = parser.parse_args()

    q_outlet_path = Path(args.q_outlet)
    if not q_outlet_path.exists():
        raise FileNotFoundError(f"q_outlet.parquet not found: {q_outlet_path}")

    recipes_path = _pick_recipes_path(args.sampled_dir)
    df_ids = pd.read_parquet(q_outlet_path, columns=["network_id"])
    network_ids = df_ids["network_id"].dropna().astype(int).unique().tolist()

    if not network_ids:
        print("No network_ids found in q_outlet.parquet")
        sys.exit(1)

    print(f"Found {len(network_ids)} unique network_ids from {q_outlet_path}")
    df_out = compute_metrics(recipes_path, network_ids, max_paths=args.max_paths)
    out_path = Path(args.out)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()

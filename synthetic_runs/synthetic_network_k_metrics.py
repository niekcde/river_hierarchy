import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from synthetic_runs.core import RiverNetworkNX
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


def build_k_graph(net) -> tuple[nx.DiGraph, dict, dict]:
    """
    Build a reach graph (edges-as-nodes) to correctly handle MultiDiGraph
    parallel edges. Returns (reach_graph, node_to_out, node_to_in).
    """
    G_src = net.G
    RG = nx.DiGraph()
    node_to_out = {}
    node_to_in = {}
    for u, v, k, data in G_src.edges(keys=True, data=True):
        xu = float(G_src.nodes[u].get("x", 0.0))
        xv = float(G_src.nodes[v].get("x", 0.0))
        if xv <= xu:
            continue
        width = float(data.get("w", 0.0))
        length = float(xv - xu)
        k_val = compute_k_value(length, width)
        edge_id = (u, v, k)
        RG.add_node(edge_id, K=k_val, u=u, v=v, length=length, width=width)
        node_to_out.setdefault(u, []).append(edge_id)
        node_to_in.setdefault(v, []).append(edge_id)
    for edge_id, attrs in RG.nodes(data=True):
        v = attrs["v"]
        for dn in node_to_out.get(v, []):
            RG.add_edge(edge_id, dn)
    return RG, node_to_out, node_to_in


def _pick_source(G: nx.DiGraph | nx.MultiDiGraph):
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    if not sources:
        return None, "no source nodes (in_degree==0)"
    if len(sources) == 1:
        return sources[0], None
    sources_sorted = sorted(sources, key=lambda n: G.nodes[n].get("x", 0.0))
    return sources_sorted[0], f"multiple sources; picked min-x={sources_sorted[0]}"


def _find_first_bifurcation(G: nx.DiGraph | nx.MultiDiGraph, source):
    out_edges = list(G.out_edges(source, keys=True, data=True))
    if len(out_edges) == 0:
        return None, "source has no outgoing edge"
    if len(out_edges) != 1:
        return None, "source has multiple outgoing edges; expected exactly one"
    _, v, _, _ = out_edges[0]
    return v, None


def _pick_outlet(G: nx.DiGraph | nx.MultiDiGraph):
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if not sinks:
        return None, "no outlet nodes (out_degree==0)"
    if len(sinks) == 1:
        return sinks[0], None
    sinks_sorted = sorted(sinks, key=lambda n: G.nodes[n].get("x", 0.0))
    return sinks_sorted[-1], f"multiple outlets; picked max-x={sinks_sorted[-1]}"


def _find_final_confluence(G: nx.DiGraph | nx.MultiDiGraph, outlet):
    in_edges = list(G.in_edges(outlet, keys=True, data=True))
    if len(in_edges) == 0:
        return None, "outlet has no incoming edge"
    if len(in_edges) != 1:
        return None, "outlet has multiple incoming edges; expected exactly one"
    u, _, _, _ = in_edges[0]
    return u, None


def _path_length(G: nx.DiGraph, path) -> float:
    return sum(float(G.nodes[r]["K"]) for r in path)


def compute_k_metrics_for_network(
    net,
    max_paths: int = 100,
):
    G_nodes = net.G
    source, source_msg = _pick_source(G_nodes)
    if source is None:
        return None, {"error": source_msg}

    B, b_msg = _find_first_bifurcation(G_nodes, source)
    if B is None:
        return None, {"error": b_msg}

    outlet, outlet_msg = _pick_outlet(G_nodes)
    if outlet is None:
        return None, {"error": outlet_msg}

    C, c_msg = _find_final_confluence(G_nodes, outlet)
    if C is None:
        return None, {"error": c_msg}

    RG, node_to_out, node_to_in = build_k_graph(net)
    if not nx.is_directed_acyclic_graph(RG):
        return None, {"error": "reach graph is not a DAG"}

    B_reaches = node_to_out.get(B, [])
    C_reaches = node_to_in.get(C, [])
    if not B_reaches or not C_reaches:
        return None, {"error": "B->C reach set empty"}

    # Subgraph of paths from B to C
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

    # Check for NaN K values in relevant edges
    for r, data in SG.nodes(data=True):
        if np.isnan(data.get("K", np.nan)):
            return None, {"error": "NaN K value encountered"}

    # Min/Max path sums over reach-nodes (node-weighted DAG DP)
    topo = list(nx.topological_sort(SG))
    dist_min = {n: np.inf for n in SG.nodes}
    dist_max = {n: -np.inf for n in SG.nodes}
    for r in topo:
        K_r = float(SG.nodes[r]["K"])
        preds = list(SG.predecessors(r))
        if r in B_reaches:
            dist_min[r] = K_r
            dist_max[r] = K_r
            continue
        if not preds:
            continue
        dist_min[r] = K_r + min(dist_min[p] for p in preds)
        dist_max[r] = K_r + max(dist_max[p] for p in preds)

    K_min_candidates = [dist_min[r] for r in C_reaches if dist_min[r] < np.inf]
    K_max_candidates = [dist_max[r] for r in C_reaches if dist_max[r] > -np.inf]
    if not K_min_candidates or not K_max_candidates:
        return None, {"error": "no valid B->C path distances"}
    K_min = min(K_min_candidates)
    K_max = max(K_max_candidates)

    # Enumerate paths up to max_paths
    total = 0.0
    n_paths = 0
    exceeded = False
    for s in B_reaches:
        for t in C_reaches:
            for path in nx.all_simple_paths(SG, source=s, target=t):
                n_paths += 1
                if n_paths > max_paths:
                    exceeded = True
                    break
                total += _path_length(SG, path)
            if exceeded:
                break
        if exceeded:
            break

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

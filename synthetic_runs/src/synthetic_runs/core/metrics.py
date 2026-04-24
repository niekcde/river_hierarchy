"""Shared network metrics extracted from the preserved synthetic legacy code."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def x_midpoints(G, x_attr="x"):
    xs = sorted({G.nodes[n][x_attr] for n in G.nodes})
    mids = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    return xs, mids


def edges_spanning_x(G, x0):
    spanning = []
    for u, v, _k, data in G.edges(keys=True, data=True):
        xu = u[0]
        xv = v[0]
        if min(xu, xv) < x0 < max(xu, xv):
            spanning.append((u, v, data))
    return spanning


def ebi(w):
    x = w / w.sum()
    entropy = -np.sum(x * np.log2(x))
    return 2**entropy


def metrics_by_midpoint(G, x_attr="x", width_attr="width"):
    _xs, mids = x_midpoints(G, x_attr=x_attr)
    values = []
    for x0 in mids[1:-1]:
        spanning = edges_spanning_x(G, x0)
        ws = np.array([float(data[width_attr]) for _, _, data in spanning])
        values.append(ebi(ws))
    return np.mean(values), np.max(values)


def compute_k(L, W, kb=20, S=1e-3, n=0.35):
    return (3 / 5) * n * (L / math.sqrt(S)) * (kb ** (2 / 3)) / (W ** (2 / 3))


def k_stats_from_graph(G: nx.MultiDiGraph, x_stability=0.3):
    ks = []
    for u, v, _k, data in G.edges(keys=True, data=True):
        w = float(data.get("w", 0.0))
        if w <= 0:
            continue
        xu = float(G.nodes[u]["x"])
        xv = float(G.nodes[v]["x"])
        L = xv - xu
        if L <= 0:
            continue
        ks.append(compute_k(L, w))

    if not ks:
        return None

    ks = np.array(ks, dtype=float)
    kmin, kmax = ks.min(), ks.max()
    ratio = (kmax / kmin) if kmin > 0 else np.inf

    ebi_mean, ebi_max = metrics_by_midpoint(G, width_attr="w")
    return {
        "k_sum": float(ks.sum()),
        "k_min": float(kmin),
        "k_max": float(kmax),
        "k_mean": float(ks.mean()),
        "k_ratio": float(ratio),
        "ebi_mean": ebi_mean,
        "ebi_max": ebi_max,
        "admissible": bool(ratio <= (1 - x_stability) / x_stability),
    }


def admissable(ratio, x_stability=0.1):
    return ratio <= (1 - x_stability) / x_stability


__all__ = [
    "admissable",
    "compute_k",
    "ebi",
    "edges_spanning_x",
    "k_stats_from_graph",
    "metrics_by_midpoint",
    "x_midpoints",
]

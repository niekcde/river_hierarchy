"""Plotly helpers for inspecting synthetic network recipes."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_runs.core import RiverNetworkNX


_EDGE_COLORS = {
    ("main", None): "#6b7280",
    ("connector", None): "#9ca3af",
    ("corridor", "A"): "#1d4ed8",
    ("corridor", "B"): "#ea580c",
    ("loop", "A"): "#059669",
    ("loop", "B"): "#7c3aed",
    ("cross", None): "#dc2626",
}


def iter_recipe_records(
    recipes_gz_path: str | Path,
    *,
    network_ids: Sequence[int] | None = None,
) -> Iterable[tuple[int, dict]]:
    """Yield ``(network_id, recipe)`` pairs from a ``.jsonl.gz`` recipe file."""
    selected = None if network_ids is None else {int(i) for i in network_ids}
    with gzip.open(recipes_gz_path, "rt", encoding="utf-8") as f:
        for network_id, line in enumerate(f):
            if selected is not None and network_id not in selected:
                continue
            yield network_id, json.loads(line)


def load_recipe_records(
    recipes_gz_path: str | Path,
    *,
    network_ids: Sequence[int] | None = None,
) -> list[tuple[int, dict]]:
    """Load recipe records while preserving the requested network-id order."""
    records = list(iter_recipe_records(recipes_gz_path, network_ids=network_ids))
    if network_ids is None:
        return records

    by_id = {network_id: recipe for network_id, recipe in records}
    ordered = []
    for network_id in network_ids:
        network_id = int(network_id)
        if network_id not in by_id:
            raise ValueError(f"Network {network_id} not found in {recipes_gz_path}")
        ordered.append((network_id, by_id[network_id]))
    return ordered


def recipe_break_summary(recipe: dict) -> str:
    """Return a compact human-readable description of the recipe breaks."""
    breaks = recipe.get("breaks", [])
    if not breaks:
        return "no-break"

    pieces = []
    for b in breaks:
        kind = b.get("kind", "unknown")
        if kind == "loop":
            pieces.append(
                f"loop {b['from_branch']} {b['xb']:.0f}-{b['xr']:.0f} "
                f"w=({b['w1']:.0f},{b['w2']:.0f})"
            )
        elif kind == "cross":
            pieces.append(
                f"cross {b['from_branch']}->{b['to_branch']} "
                f"{b['xb']:.0f}->{b['xr']:.0f} w={b['w_cross']:.0f}"
            )
        else:
            pieces.append(kind)
    return " | ".join(pieces)


def recipe_title(recipe: dict, network_id: int | None = None) -> str:
    """Return a compact title for single-network or subplot display."""
    prefix = f"id {network_id}" if network_id is not None else "recipe"
    wa = float(recipe.get("initial_split", {}).get("WA", np.nan))
    wb = float(recipe.get("initial_split", {}).get("WB", np.nan))
    return f"{prefix} | WA/WB={wa:.0f}/{wb:.0f} | {recipe_break_summary(recipe)}"


def recipe_summary_frame(
    recipes_gz_path: str | Path,
    *,
    network_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build a table summarizing recipe structure and widths."""
    rows = []
    for network_id, recipe in load_recipe_records(recipes_gz_path, network_ids=network_ids):
        rows.append(
            {
                "network_id": int(network_id),
                "geometry_id": recipe.get("geometry_id"),
                "sample_id": recipe.get("sample_id"),
                "sample_mode": recipe.get("sample_mode"),
                "WA": float(recipe.get("initial_split", {}).get("WA", np.nan)),
                "WB": float(recipe.get("initial_split", {}).get("WB", np.nan)),
                "n_breaks": int(len(recipe.get("breaks", []))),
                "break_summary": recipe_break_summary(recipe),
            }
        )
    return pd.DataFrame(rows)


def _edge_branch(data: dict) -> str | None:
    if data.get("kind") == "corridor":
        return data.get("branch")
    if data.get("kind") == "loop":
        return data.get("from_branch")
    return None


def _edge_color(data: dict) -> str:
    return _EDGE_COLORS.get((data.get("kind"), _edge_branch(data)), "#111827")


def _node_y(net: RiverNetworkNX, tag: str, x: float, *, N: int) -> float:
    xg, yA, yB = net._precompute_corridor_centerlines(N=N)
    if tag == "A":
        return float(np.interp(x, xg, yA))
    if tag == "B":
        return float(np.interp(x, xg, yB))
    return 0.0


def _node_tag(node, data: dict) -> str:
    tag = data.get("tag")
    if tag in {"A", "B", "main"}:
        return str(tag)
    if isinstance(node, tuple) and len(node) >= 2 and node[1] in {"A", "B"}:
        return str(node[1])
    return "main"


def _clone_trace(trace: go.BaseTraceType) -> go.Scatter:
    payload = trace.to_plotly_json()
    payload.pop("type", None)
    return go.Scatter(**payload)


def _subplot_ranges(recipe: dict) -> tuple[list[float], list[float]]:
    meta = recipe["meta"]
    x_range = [0.0, float(meta["L"])]
    y_abs = max(
        1.0,
        abs(float(meta.get("Y0", 1.0))) + abs(float(meta.get("amp_corr", 1.5))) + abs(float(meta.get("amp_loop", 0.7))),
    )
    return x_range, [-1.15 * y_abs, 1.15 * y_abs]


def plot_network(
    net: RiverNetworkNX,
    *,
    title: str | None = None,
    N: int = 600,
    cross_points: int = 60,
    show_nodes: bool = True,
    show_node_labels: bool = False,
) -> go.Figure:
    """Return a styled Plotly figure for a single synthetic network."""
    fig = net.plotly_figure(N=N, cross_points=cross_points)

    edge_rows = list(net.G.edges(keys=True, data=True))
    for trace, (_, _, _, data) in zip(fig.data, edge_rows):
        trace.update(line={**trace.line.to_plotly_json(), "color": _edge_color(data)})

    if show_nodes:
        xs = []
        ys = []
        texts = []
        customdata = []
        for node, data in net.G.nodes(data=True):
            x = float(data.get("x", 0.0))
            tag = _node_tag(node, data)
            xs.append(x)
            ys.append(_node_y(net, tag, x, N=N))
            texts.append(str(node) if show_node_labels else "")
            customdata.append([str(node), x, tag])

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if show_node_labels else "markers",
                text=texts,
                textposition="top center",
                marker=dict(size=7, color="#111827", line=dict(width=1, color="white")),
                customdata=customdata,
                hovertemplate=(
                    "node: %{customdata[0]}<br>"
                    "x: %{customdata[1]:.0f}<br>"
                    "tag: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    x_range, y_range = _subplot_ranges(net.to_recipe())
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=480,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
    )
    return fig


def plot_recipe(
    recipe: dict,
    *,
    network_id: int | None = None,
    title: str | None = None,
    N: int = 600,
    cross_points: int = 60,
    show_nodes: bool = True,
    show_node_labels: bool = False,
) -> go.Figure:
    """Build a network from a recipe and return its Plotly figure."""
    net = RiverNetworkNX.from_recipe(recipe)
    return plot_network(
        net,
        title=title or recipe_title(recipe, network_id=network_id),
        N=N,
        cross_points=cross_points,
        show_nodes=show_nodes,
        show_node_labels=show_node_labels,
    )


def plot_recipe_grid(
    recipes_gz_path: str | Path,
    *,
    network_ids: Sequence[int] | None = None,
    cols: int = 3,
    N: int = 500,
    cross_points: int = 60,
    show_nodes: bool = True,
    show_node_labels: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Plot a recipe file as a grid of subplot figures."""
    records = load_recipe_records(recipes_gz_path, network_ids=network_ids)
    if not records:
        raise ValueError(f"No recipes found in {recipes_gz_path}")

    cols = max(1, int(cols))
    rows = int(math.ceil(len(records) / cols))
    subplot_titles = [recipe_title(recipe, network_id=network_id) for network_id, recipe in records]
    subplot_titles.extend([""] * (rows * cols - len(subplot_titles)))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
    )

    for idx, (network_id, recipe) in enumerate(records):
        row = idx // cols + 1
        col = idx % cols + 1
        single = plot_recipe(
            recipe,
            network_id=network_id,
            title=None,
            N=N,
            cross_points=cross_points,
            show_nodes=show_nodes,
            show_node_labels=show_node_labels,
        )
        for trace in single.data:
            fig.add_trace(_clone_trace(trace), row=row, col=col)

        x_range, y_range = _subplot_ranges(recipe)
        fig.update_xaxes(range=x_range, row=row, col=col)
        fig.update_yaxes(range=y_range, row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(380, 320 * rows),
        width=max(700, 420 * cols),
        margin=dict(l=20, r=20, t=70 if title else 30, b=20),
        showlegend=False,
    )
    return fig


__all__ = [
    "iter_recipe_records",
    "load_recipe_records",
    "plot_network",
    "plot_recipe",
    "plot_recipe_grid",
    "recipe_break_summary",
    "recipe_summary_frame",
    "recipe_title",
]

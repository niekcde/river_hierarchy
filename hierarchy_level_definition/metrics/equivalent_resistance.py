from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.sparse import csr_matrix


NODE_ID_CANDIDATES = ("node_id", "id_node", "id")
EDGE_ID_CANDIDATES = ("edge_id", "id_edge", "id_link", "id")
EDGE_U_CANDIDATES = ("u", "id_us_node", "source", "from_node")
EDGE_V_CANDIDATES = ("v", "id_ds_node", "target", "to_node")
EDGE_K_CANDIDATES = ("K_e", "K", "edge_resistance", "resistance")
HEAD_VALUE_CANDIDATES = ("H", "head", "sword_wse", "wse", "h")
SOLVER_BACKENDS = (
    "dense_legacy",
    "sparse",
    "sparse_least_squares",
    "energy_sparse",
    "auto",
)
DEFAULT_SPARSE_MIN_INTERNAL_NODES = 64
SOLVER_ALGORITHM_VERSION = "equivalent-resistance-energy-sparse-v1"
SOLVER_CACHE_SCHEMA_VERSION = "equivalent-resistance-solve-cache-v2"


def _cache_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric.hex() if np.isfinite(numeric) else str(numeric)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _solver_cache_node_aliases(edge_frame: pd.DataFrame) -> dict[Any, Any]:
    aliases: dict[Any, Any] = {}
    for row in edge_frame.itertuples(index=False):
        aliases[row.u] = getattr(row, "solver_cache_u_alias", row.u)
        aliases[row.v] = getattr(row, "solver_cache_v_alias", row.v)
    return aliases


def _solver_cache_key(
    *,
    edge_frame: pd.DataFrame,
    upstream_node: Any,
    downstream_node: Any,
    normalized_head_guess: Mapping[Any, float],
    smooth: bool,
    eps: float,
    continuity_abs_tolerance: float,
    continuity_rel_tolerance: float,
    requested_solver_backend: str,
    resolved_solver_backend: str,
    sparse_min_internal_nodes: int,
) -> str:
    edge_records = [
        (
            _cache_scalar(
                getattr(row, "solver_cache_u_alias", row.u)
            ),
            _cache_scalar(
                getattr(row, "solver_cache_v_alias", row.v)
            ),
            _cache_scalar(
                getattr(row, "solver_cache_edge_alias", row.edge_id)
            ),
            float(row.K).hex(),
        )
        for row in edge_frame.itertuples(index=False)
    ]
    edge_records.sort(key=lambda record: tuple(str(value) for value in record))
    node_aliases = _solver_cache_node_aliases(edge_frame)
    head_records = [
        (
            _cache_scalar(node_aliases.get(node_id, node_id)),
            float(value).hex(),
        )
        for node_id, value in normalized_head_guess.items()
    ]
    head_records.sort(key=lambda record: tuple(str(value) for value in record))
    payload = {
        "schema_version": SOLVER_CACHE_SCHEMA_VERSION,
        "solver_algorithm_version": SOLVER_ALGORITHM_VERSION,
        "upstream_node": _cache_scalar(node_aliases.get(upstream_node, upstream_node)),
        "downstream_node": _cache_scalar(node_aliases.get(downstream_node, downstream_node)),
        "edges": edge_records,
        "normalized_head_guess": head_records,
        "smooth": bool(smooth),
        "eps": float(eps).hex(),
        "continuity_abs_tolerance": float(continuity_abs_tolerance).hex(),
        "continuity_rel_tolerance": float(continuity_rel_tolerance).hex(),
        "solver_backend_requested": requested_solver_backend,
        "solver_backend_resolved": resolved_solver_backend,
        "sparse_min_internal_nodes": int(sparse_min_internal_nodes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_solver_result(
    cache_path: Path,
    cache_key: str,
    *,
    edge_frame: pd.DataFrame,
    node_ids: Sequence[Any],
    upstream_node: Any,
    downstream_node: Any,
) -> dict[str, Any] | None:
    summary_path = cache_path / "summary.json"
    node_path = cache_path / "node_heads.csv"
    edge_path = cache_path / "edge_flows.csv"
    if not (summary_path.exists() and node_path.exists() and edge_path.exists()):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("cache_schema_version") != SOLVER_CACHE_SCHEMA_VERSION
            or summary.get("solver_cache_key") != cache_key
            or not bool(summary.get("solver_success", False))
        ):
            return None
        node_heads = pd.read_csv(node_path)
        edge_flows = pd.read_csv(edge_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if "solver_cache_edge_alias" in edge_frame.columns:
        if "solver_cache_edge_alias" not in edge_flows.columns:
            return None
        current_edges = edge_frame[
            ["edge_id", "u", "v", "solver_cache_edge_alias"]
        ].copy()
        if (
            current_edges["solver_cache_edge_alias"].duplicated().any()
            or edge_flows["solver_cache_edge_alias"].duplicated().any()
        ):
            return None
        current_lookup = current_edges.set_index("solver_cache_edge_alias")
        cached_aliases = edge_flows["solver_cache_edge_alias"]
        if not cached_aliases.isin(current_lookup.index).all():
            return None
        edge_flows["edge_id"] = cached_aliases.map(current_lookup["edge_id"])
        edge_flows["u"] = cached_aliases.map(current_lookup["u"])
        edge_flows["v"] = cached_aliases.map(current_lookup["v"])
        edge_flows["flow_from_node"] = np.where(
            edge_flows["q_unit"] >= 0.0, edge_flows["u"], edge_flows["v"]
        )
        edge_flows["flow_to_node"] = np.where(
            edge_flows["q_unit"] >= 0.0, edge_flows["v"], edge_flows["u"]
        )

    node_aliases = _solver_cache_node_aliases(edge_frame)
    if "solver_cache_node_alias" in node_heads.columns:
        alias_to_current: dict[Any, Any] = {}
        for node_id in node_ids:
            alias = node_aliases.get(node_id, node_id)
            if alias in alias_to_current and alias_to_current[alias] != node_id:
                return None
            alias_to_current[alias] = node_id
        if not node_heads["solver_cache_node_alias"].isin(alias_to_current).all():
            return None
        node_heads["node_id"] = node_heads["solver_cache_node_alias"].map(
            alias_to_current
        )

    summary["upstream_node"] = upstream_node
    summary["downstream_node"] = downstream_node
    summary["node_heads"] = node_heads
    summary["edge_flows"] = edge_flows
    summary["internal_heads"] = {
        row.node_id: float(row.head_norm)
        for row in node_heads.loc[~node_heads["is_boundary"].astype(bool)].itertuples(index=False)
    }
    summary["solver_cache_hit"] = True
    summary["solver_cache_path"] = str(cache_path.resolve())
    return summary


def _write_cached_solver_result(
    *,
    cache_path: Path,
    cache_key: str,
    result: Mapping[str, Any],
) -> None:
    if not bool(result.get("solver_success", False)):
        return
    cache_path.mkdir(parents=True, exist_ok=True)
    summary = {
        key: _cache_scalar(value)
        if isinstance(value, (np.integer, np.floating, np.bool_))
        else value
        for key, value in result.items()
        if key not in {"node_heads", "edge_flows", "internal_heads"}
    }
    summary.update(
        {
            "cache_schema_version": SOLVER_CACHE_SCHEMA_VERSION,
            "solver_cache_key": cache_key,
            "solver_cache_hit": False,
            "solver_cache_path": str(cache_path.resolve()),
        }
    )
    result["node_heads"].to_csv(cache_path / "node_heads.csv", index=False)
    result["edge_flows"].to_csv(cache_path / "edge_flows.csv", index=False)
    (cache_path / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )


def _to_numeric_array(value: object, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite numeric values.")
    return array


def _validate_positive(value: np.ndarray, *, field_name: str) -> None:
    if np.any(value <= 0):
        raise ValueError(f"{field_name} must be strictly positive.")


def _finalize_numeric_result(value: np.ndarray | float, *, templates: Sequence[object]) -> float | np.ndarray:
    if all(np.ndim(np.asarray(template)) == 0 for template in templates):
        return float(np.asarray(value))
    return np.asarray(value, dtype=float)


def compute_depth(width: float | Sequence[float], kb: float | Sequence[float]) -> float | np.ndarray:
    """Estimate flow depth from width and width-depth ratio."""
    width_array = _to_numeric_array(width, field_name="width")
    kb_array = _to_numeric_array(kb, field_name="kb")
    _validate_positive(width_array, field_name="width")
    _validate_positive(kb_array, field_name="kb")
    depth = width_array / kb_array
    return _finalize_numeric_result(depth, templates=(width, kb))


def compute_geometry(
    width: float | Sequence[float],
    kb: float | Sequence[float],
    *,
    method: str = "wide",
) -> dict[str, float | np.ndarray | str]:
    """Compute width-derived hydraulic geometry for the equivalent-resistance predictor.

    This geometry is intentionally discharge-free. Model discharge or hydrograph
    outputs should not be used inside the predictor calculation.
    """

    normalized_method = str(method or "wide").strip().lower()
    if normalized_method not in {"wide", "rectangular"}:
        raise ValueError("geometry method must be either 'wide' or 'rectangular'.")

    width_array = _to_numeric_array(width, field_name="width")
    depth_array = _to_numeric_array(compute_depth(width, kb), field_name="depth")
    area_array = width_array * depth_array
    wetted_perimeter = np.full_like(area_array, np.nan, dtype=float)

    if normalized_method == "wide":
        hydraulic_radius = depth_array
    else:
        wetted_perimeter = width_array + 2.0 * depth_array
        hydraulic_radius = area_array / wetted_perimeter

    return {
        "method": normalized_method,
        "d": _finalize_numeric_result(depth_array, templates=(width, kb)),
        "A": _finalize_numeric_result(area_array, templates=(width, kb)),
        "P": _finalize_numeric_result(wetted_perimeter, templates=(width, kb)),
        "R_h": _finalize_numeric_result(hydraulic_radius, templates=(width, kb)),
    }


def compute_K_manning(
    length: float | Sequence[float],
    width: float | Sequence[float],
    n: float | Sequence[float],
    kb: float | Sequence[float],
    geometry_method: str = "wide",
) -> float | np.ndarray:
    """Compute geometry-informed Manning resistance without using model discharge."""
    length_array = _to_numeric_array(length, field_name="length")
    n_array = _to_numeric_array(n, field_name="n")
    _validate_positive(length_array, field_name="length")
    _validate_positive(n_array, field_name="n")

    geometry = compute_geometry(width, kb, method=geometry_method)
    area_array = _to_numeric_array(geometry["A"], field_name="area")
    hydraulic_radius = _to_numeric_array(geometry["R_h"], field_name="hydraulic_radius")
    resistance = (n_array**2 * length_array) / (area_array**2 * hydraulic_radius ** (4.0 / 3.0))
    return _finalize_numeric_result(resistance, templates=(length, width, n, kb))


def compute_K_chezy(
    length: float | Sequence[float],
    width: float | Sequence[float],
    Cz: float | Sequence[float],
    kb: float | Sequence[float],
    geometry_method: str = "wide",
    *,
    g: float = 9.81,
) -> float | np.ndarray:
    """Compute geometry-informed Chezy resistance without using model discharge."""
    length_array = _to_numeric_array(length, field_name="length")
    cz_array = _to_numeric_array(Cz, field_name="Cz")
    _validate_positive(length_array, field_name="length")
    _validate_positive(cz_array, field_name="Cz")
    if g <= 0:
        raise ValueError("g must be positive.")

    geometry = compute_geometry(width, kb, method=geometry_method)
    area_array = _to_numeric_array(geometry["A"], field_name="area")
    hydraulic_radius = _to_numeric_array(geometry["R_h"], field_name="hydraulic_radius")
    resistance = length_array / (area_array**2 * cz_array**2 * float(g) * hydraulic_radius)
    return _finalize_numeric_result(resistance, templates=(length, width, Cz, kb))


def edge_flow(
    h_u: float | Sequence[float],
    h_v: float | Sequence[float],
    K: float | Sequence[float],
    *,
    smooth: bool = True,
    eps: float = 1e-10,
) -> float | np.ndarray:
    """Compute signed edge flow under a normalized head formulation."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    head_u = _to_numeric_array(h_u, field_name="h_u")
    head_v = _to_numeric_array(h_v, field_name="h_v")
    resistance = _to_numeric_array(K, field_name="K")
    _validate_positive(resistance, field_name="K")

    dh = head_u - head_v
    if smooth:
        flow = dh / (np.sqrt(resistance) * np.power(dh * dh + eps * eps, 0.25))
    else:
        flow = np.sign(dh) * np.sqrt(np.abs(dh) / resistance)
        flow = np.where(np.abs(dh) == 0.0, 0.0, flow)
    return _finalize_numeric_result(flow, templates=(h_u, h_v, K))


def reduce_simple_series(K_values: Sequence[float]) -> float:
    """Equivalent resistance for series branches."""
    resistance = _to_numeric_array(list(K_values), field_name="K_values")
    _validate_positive(resistance, field_name="K_values")
    return float(np.sum(resistance))


def reduce_simple_parallel(K_values: Sequence[float]) -> float:
    """Equivalent resistance for parallel branches under the quadratic head-flow law."""
    resistance = _to_numeric_array(list(K_values), field_name="K_values")
    _validate_positive(resistance, field_name="K_values")
    return float(1.0 / np.sum(1.0 / np.sqrt(resistance)) ** 2)


def compare_states(K_before: float | Sequence[float], K_after: float | Sequence[float]) -> dict[str, float | np.ndarray]:
    """Compare equivalent resistance states using resistance and conductance ratios."""
    before = _to_numeric_array(K_before, field_name="K_before")
    after = _to_numeric_array(K_after, field_name="K_after")
    _validate_positive(before, field_name="K_before")
    _validate_positive(after, field_name="K_after")

    lambda_k = after / before
    lambda_g = before / after
    pct_change_k = 100.0 * (lambda_k - 1.0)
    pct_change_g = 100.0 * (lambda_g - 1.0)
    return {
        "Lambda_K": _finalize_numeric_result(lambda_k, templates=(K_before, K_after)),
        "Lambda_G": _finalize_numeric_result(lambda_g, templates=(K_before, K_after)),
        "percent_change_K": _finalize_numeric_result(pct_change_k, templates=(K_before, K_after)),
        "percent_change_G": _finalize_numeric_result(pct_change_g, templates=(K_before, K_after)),
    }


def _as_frame(records: Any, *, label: str) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Mapping):
        return pd.DataFrame([dict(records)])
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        return pd.DataFrame.from_records(records)
    raise TypeError(f"Unsupported {label} input type: {type(records)!r}")


def _graph_to_edge_frame(graph: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, key, data in graph.edges(keys=True, data=True):
            record = dict(data)
            record.setdefault("edge_id", data.get("edge_id", key))
            record["u"] = u
            record["v"] = v
            records.append(record)
    else:
        for u, v, data in graph.edges(data=True):
            record = dict(data)
            record.setdefault("edge_id", data.get("edge_id", f"{u}->{v}"))
            record["u"] = u
            record["v"] = v
            records.append(record)
    return pd.DataFrame.from_records(records)


def _graph_to_node_frame(graph: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{"node_id": node_id, **dict(data)} for node_id, data in graph.nodes(data=True)]
    )


def _resolve_column(frame: pd.DataFrame, candidates: Sequence[str], *, label: str) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if label == "edge_id":
        return None
    raise ValueError(f"Could not resolve a {label} column from candidates: {list(candidates)}")


def _canonicalize_nodes_and_edges(
    nodes: Any,
    edges: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(edges, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        edge_frame = _graph_to_edge_frame(edges)
        node_frame = _graph_to_node_frame(edges) if nodes is None else _as_frame(nodes, label="nodes")
    else:
        edge_frame = _as_frame(edges, label="edges")
        node_frame = _as_frame(nodes, label="nodes")

    if edge_frame.empty:
        raise ValueError("At least one edge is required to solve equivalent resistance.")

    u_col = _resolve_column(edge_frame, EDGE_U_CANDIDATES, label="edge upstream/endpoint")
    v_col = _resolve_column(edge_frame, EDGE_V_CANDIDATES, label="edge downstream/endpoint")
    k_col = _resolve_column(edge_frame, EDGE_K_CANDIDATES, label="edge resistance")
    edge_id_col = _resolve_column(edge_frame, EDGE_ID_CANDIDATES, label="edge_id")

    canonical_edges = edge_frame.copy()
    canonical_edges["u"] = canonical_edges[u_col]
    canonical_edges["v"] = canonical_edges[v_col]
    canonical_edges["K"] = pd.to_numeric(canonical_edges[k_col], errors="coerce")
    canonical_edges["edge_id"] = (
        canonical_edges[edge_id_col]
        if edge_id_col is not None
        else pd.Series(range(1, len(canonical_edges) + 1), index=canonical_edges.index)
    )
    if canonical_edges["K"].isna().any():
        raise ValueError("All edges must provide finite resistance values in a K or K_e column.")
    _validate_positive(canonical_edges["K"].to_numpy(dtype=float), field_name="edge resistance")
    if "solver_cache_edge_alias" not in canonical_edges.columns:
        canonical_edges["solver_cache_edge_alias"] = canonical_edges["edge_id"]
    if "solver_cache_u_alias" not in canonical_edges.columns:
        canonical_edges["solver_cache_u_alias"] = canonical_edges["u"]
    if "solver_cache_v_alias" not in canonical_edges.columns:
        canonical_edges["solver_cache_v_alias"] = canonical_edges["v"]

    if node_frame.empty:
        endpoint_nodes = pd.Index(pd.unique(pd.concat([canonical_edges["u"], canonical_edges["v"]], ignore_index=True)))
        canonical_nodes = pd.DataFrame({"node_id": endpoint_nodes})
    else:
        node_id_col = _resolve_column(node_frame, NODE_ID_CANDIDATES, label="node id")
        canonical_nodes = node_frame.copy()
        canonical_nodes["node_id"] = canonical_nodes[node_id_col]

    return canonical_nodes.reset_index(drop=True), canonical_edges.reset_index(drop=True)


def _extract_head_guess_mapping(
    node_frame: pd.DataFrame,
    optional_H_guess: Any,
) -> dict[Any, float]:
    if optional_H_guess is None:
        for candidate in HEAD_VALUE_CANDIDATES:
            if candidate in node_frame.columns:
                source = node_frame[["node_id", candidate]].copy()
                source[candidate] = pd.to_numeric(source[candidate], errors="coerce")
                return {
                    row["node_id"]: float(row[candidate])
                    for row in source.to_dict("records")
                    if pd.notna(row[candidate])
                }
        return {}

    if isinstance(optional_H_guess, pd.Series):
        return {
            node_id: float(value)
            for node_id, value in optional_H_guess.items()
            if pd.notna(value)
        }
    if isinstance(optional_H_guess, Mapping):
        result: dict[Any, float] = {}
        for node_id, value in optional_H_guess.items():
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                result[node_id] = float(numeric)
        return result

    frame = _as_frame(optional_H_guess, label="optional_H_guess")
    node_col = _resolve_column(frame, NODE_ID_CANDIDATES, label="node id")
    head_col = _resolve_column(frame, HEAD_VALUE_CANDIDATES, label="head guess")
    source = frame[[node_col, head_col]].copy()
    source[head_col] = pd.to_numeric(source[head_col], errors="coerce")
    return {
        getattr(row, node_col): float(getattr(row, head_col))
        for row in source.itertuples(index=False)
        if pd.notna(getattr(row, head_col))
    }


def _normalize_head_guess_mapping(
    head_mapping: Mapping[Any, float],
    *,
    upstream_node: Any,
    downstream_node: Any,
) -> dict[Any, float]:
    if upstream_node not in head_mapping or downstream_node not in head_mapping:
        return {}
    h_up = float(head_mapping[upstream_node])
    h_down = float(head_mapping[downstream_node])
    if not np.isfinite(h_up) or not np.isfinite(h_down) or np.isclose(h_up, h_down):
        return {}
    scale = h_up - h_down
    return {
        node_id: float(np.clip((float(value) - h_down) / scale, 0.0, 1.0))
        for node_id, value in head_mapping.items()
        if np.isfinite(value)
    }


def _build_initial_head_guess(
    graph: nx.Graph,
    *,
    internal_nodes: list[Any],
    upstream_node: Any,
    downstream_node: Any,
    normalized_head_guess: Mapping[Any, float],
) -> np.ndarray:
    if not internal_nodes:
        return np.asarray([], dtype=float)

    dist_up = nx.single_source_dijkstra_path_length(graph, upstream_node, weight="weight")
    dist_down = nx.single_source_dijkstra_path_length(graph, downstream_node, weight="weight")

    guesses: list[float] = []
    for node_id in internal_nodes:
        if node_id in normalized_head_guess:
            guesses.append(float(normalized_head_guess[node_id]))
            continue
        if node_id in dist_up and node_id in dist_down:
            denominator = float(dist_up[node_id]) + float(dist_down[node_id])
            if denominator > 0:
                guesses.append(float(dist_down[node_id]) / denominator)
                continue
        guesses.append(0.5)
    return np.clip(np.asarray(guesses, dtype=float), 0.0, 1.0)


def solve_bubble_equivalent_K(
    nodes: pd.DataFrame | Sequence[Mapping[str, Any]] | nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph | None,
    edges: pd.DataFrame | Sequence[Mapping[str, Any]] | nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph,
    upstream_node: Any,
    downstream_node: Any,
    optional_H_guess: Any = None,
    *,
    smooth: bool = True,
    eps: float = 1e-10,
    continuity_abs_tolerance: float = 1e-8,
    continuity_rel_tolerance: float = 1e-3,
    solver_backend: str = "dense_legacy",
    sparse_min_internal_nodes: int = DEFAULT_SPARSE_MIN_INTERNAL_NODES,
    solver_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Solve a normalized unit-head equivalent resistance for a bubble subgraph.

    The solve fixes `h_up = 1` and `h_down = 0`, then solves for internal
    normalized heads using only edge resistance values derived from geometry.
    Model discharge Q, hydrograph outputs, and routed RAPID peaks must not be
    used inside this predictor calculation.
    """

    requested_solver_backend = str(solver_backend or "dense_legacy").strip().lower()
    if requested_solver_backend not in SOLVER_BACKENDS:
        raise ValueError(
            f"solver_backend must be one of {list(SOLVER_BACKENDS)}; "
            f"received {solver_backend!r}."
        )
    if int(sparse_min_internal_nodes) < 1:
        raise ValueError("sparse_min_internal_nodes must be at least 1.")

    node_frame, edge_frame = _canonicalize_nodes_and_edges(nodes, edges)
    edge_nodes = pd.Index(pd.unique(pd.concat([edge_frame["u"], edge_frame["v"]], ignore_index=True)))
    all_node_ids = list(dict.fromkeys(list(node_frame["node_id"]) + edge_nodes.tolist() + [upstream_node, downstream_node]))
    if upstream_node not in all_node_ids:
        raise ValueError("upstream_node is not present in the bubble node set.")
    if downstream_node not in all_node_ids:
        raise ValueError("downstream_node is not present in the bubble node set.")
    if upstream_node == downstream_node:
        raise ValueError("upstream_node and downstream_node must be distinct.")

    graph = nx.Graph()
    graph.add_nodes_from(all_node_ids)
    for row in edge_frame.itertuples(index=False):
        graph.add_edge(row.u, row.v, weight=float(row.K))

    if not graph.has_node(upstream_node) or not graph.has_node(downstream_node):
        raise ValueError("Boundary nodes must exist in the bubble graph.")
    if not nx.has_path(graph, upstream_node, downstream_node):
        raise ValueError("No path exists between the upstream and downstream boundary nodes.")

    connected_component = nx.node_connected_component(graph, upstream_node)
    disconnected = [node_id for node_id in all_node_ids if node_id not in connected_component]
    if disconnected:
        raise ValueError(
            "Bubble nodes must lie in the same connected component as the boundary nodes. "
            f"Disconnected nodes: {disconnected[:10]}"
        )

    node_ids = [node_id for node_id in all_node_ids if node_id in connected_component]
    internal_nodes = [node_id for node_id in node_ids if node_id not in {upstream_node, downstream_node}]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    internal_index = {node_id: idx for idx, node_id in enumerate(internal_nodes)}

    u_index = np.asarray([node_index[u] for u in edge_frame["u"]], dtype=int)
    v_index = np.asarray([node_index[v] for v in edge_frame["v"]], dtype=int)
    resistance = edge_frame["K"].to_numpy(dtype=float)

    incidence_internal = np.zeros((len(internal_nodes), len(edge_frame)), dtype=float)
    incidence_upstream = np.zeros(len(edge_frame), dtype=float)
    for edge_position, row in enumerate(edge_frame.itertuples(index=False)):
        if row.u in internal_index:
            incidence_internal[internal_index[row.u], edge_position] = 1.0
        if row.v in internal_index:
            incidence_internal[internal_index[row.v], edge_position] = -1.0
        if row.u == upstream_node:
            incidence_upstream[edge_position] = 1.0
        elif row.v == upstream_node:
            incidence_upstream[edge_position] = -1.0

    normalized_head_guess = _normalize_head_guess_mapping(
        _extract_head_guess_mapping(node_frame, optional_H_guess),
        upstream_node=upstream_node,
        downstream_node=downstream_node,
    )

    def assemble_heads(internal_head_values: np.ndarray) -> np.ndarray:
        heads = np.empty(len(node_ids), dtype=float)
        heads.fill(np.nan)
        heads[node_index[upstream_node]] = 1.0
        heads[node_index[downstream_node]] = 0.0
        for node_id, idx in internal_index.items():
            heads[node_index[node_id]] = float(internal_head_values[idx])
        return heads

    def residuals(internal_head_values: np.ndarray) -> np.ndarray:
        heads = assemble_heads(internal_head_values)
        flow = np.asarray(edge_flow(heads[u_index], heads[v_index], resistance, smooth=smooth, eps=eps), dtype=float)
        return incidence_internal @ flow

    def energy_and_gradient(
        internal_head_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return convex hydraulic energy and its continuity-residual gradient."""
        heads = assemble_heads(internal_head_values)
        dh = heads[u_index] - heads[v_index]
        if smooth:
            dh_squared_eps = dh * dh + eps * eps
            energy = np.sum(
                (2.0 / (3.0 * np.sqrt(resistance)))
                * (np.power(dh_squared_eps, 0.75) - eps**1.5)
            )
        else:
            energy = np.sum(
                (2.0 / (3.0 * np.sqrt(resistance)))
                * np.power(np.abs(dh), 1.5)
            )
        flow = np.asarray(
            edge_flow(
                heads[u_index], heads[v_index], resistance,
                smooth=smooth, eps=eps,
            ),
            dtype=float,
        )
        gradient = np.asarray(incidence_internal_sparse @ flow).ravel()
        return float(energy), gradient

    incidence_internal_sparse = csr_matrix(incidence_internal)

    def sparse_jacobian(internal_head_values: np.ndarray) -> csr_matrix:
        """Sparse derivative of internal-node continuity residuals."""
        heads = assemble_heads(internal_head_values)
        dh = heads[u_index] - heads[v_index]
        if smooth:
            dh_squared_eps = dh * dh + eps * eps
            flow_derivative = (
                (0.5 * dh * dh + eps * eps)
                / (np.sqrt(resistance) * np.power(dh_squared_eps, 1.25))
            )
        else:
            flow_derivative = 1.0 / (
                2.0 * np.sqrt(resistance) * np.sqrt(np.maximum(np.abs(dh), eps))
            )
        return (
            incidence_internal_sparse
            @ incidence_internal_sparse.multiply(flow_derivative).transpose()
        ).tocsr()

    if continuity_abs_tolerance < 0:
        raise ValueError("continuity_abs_tolerance cannot be negative.")
    if continuity_rel_tolerance < 0:
        raise ValueError("continuity_rel_tolerance cannot be negative.")

    if internal_nodes:
        if requested_solver_backend in {"sparse", "sparse_least_squares"}:
            resolved_solver_backend = "sparse_least_squares"
        elif requested_solver_backend == "energy_sparse":
            resolved_solver_backend = "energy_sparse"
        elif (
            requested_solver_backend == "auto"
            and len(internal_nodes) >= int(sparse_min_internal_nodes)
        ):
            resolved_solver_backend = "energy_sparse"
        else:
            resolved_solver_backend = "dense_legacy"
        x0 = _build_initial_head_guess(
            graph,
            internal_nodes=internal_nodes,
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            normalized_head_guess=normalized_head_guess,
        )
        solver_cache_key = _solver_cache_key(
            edge_frame=edge_frame,
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            normalized_head_guess=normalized_head_guess,
            smooth=smooth,
            eps=eps,
            continuity_abs_tolerance=continuity_abs_tolerance,
            continuity_rel_tolerance=continuity_rel_tolerance,
            requested_solver_backend=requested_solver_backend,
            resolved_solver_backend=resolved_solver_backend,
            sparse_min_internal_nodes=int(sparse_min_internal_nodes),
        )
        solver_cache_path = (
            Path(solver_cache_dir).expanduser() / solver_cache_key
            if solver_cache_dir is not None
            else None
        )
        if solver_cache_path is not None:
            cached = _load_cached_solver_result(
                solver_cache_path,
                solver_cache_key,
                edge_frame=edge_frame,
                node_ids=node_ids,
                upstream_node=upstream_node,
                downstream_node=downstream_node,
            )
            if cached is not None:
                return cached
        energy_success: bool | None = None
        energy_message = ""
        energy_nit = 0
        energy_nfev = 0
        energy_residual_norm = float("nan")
        energy_residual_relative = float("nan")
        polish_attempted = False
        polish_success: bool | None = None
        polish_message = ""
        polish_nfev = 0
        polish_residual_norm = float("nan")
        polish_residual_relative = float("nan")
        solver_solution_stage = "least_squares"

        if resolved_solver_backend == "energy_sparse":
            energy_result = minimize(
                energy_and_gradient,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * len(internal_nodes),
                options={
                    "ftol": 1e-15,
                    "gtol": 1e-10,
                    "maxiter": 20000,
                    "maxfun": 50000,
                    "maxls": 50,
                },
            )
            energy_success = bool(energy_result.success)
            energy_message = str(energy_result.message)
            energy_nit = int(energy_result.nit)
            energy_nfev = int(energy_result.nfev)
            energy_x = np.asarray(energy_result.x, dtype=float)
            energy_flow = np.asarray(
                edge_flow(
                    assemble_heads(energy_x)[u_index],
                    assemble_heads(energy_x)[v_index],
                    resistance,
                    smooth=smooth,
                    eps=eps,
                ),
                dtype=float,
            )
            energy_residual = residuals(energy_x)
            energy_q_unit = float(incidence_upstream @ energy_flow)
            energy_residual_norm = float(np.linalg.norm(energy_residual, ord=2))
            energy_scale = float(
                max(
                    np.max(np.abs(energy_flow)) if energy_flow.size else 0.0,
                    abs(energy_q_unit),
                    1.0,
                )
            )
            energy_residual_relative = float(energy_residual_norm / energy_scale)

            polish_attempted = True
            polish_result = least_squares(
                residuals,
                energy_x,
                bounds=(0.0, 1.0),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=4000,
                jac=sparse_jacobian,
                tr_solver="lsmr",
            )
            polish_success = bool(polish_result.success)
            polish_message = str(polish_result.message)
            polish_nfev = int(polish_result.nfev)
            polish_residual = residuals(polish_result.x)
            polish_flow = np.asarray(
                edge_flow(
                    assemble_heads(polish_result.x)[u_index],
                    assemble_heads(polish_result.x)[v_index],
                    resistance,
                    smooth=smooth,
                    eps=eps,
                ),
                dtype=float,
            )
            polish_q_unit = float(incidence_upstream @ polish_flow)
            polish_residual_norm = float(np.linalg.norm(polish_residual, ord=2))
            polish_scale = float(
                max(
                    np.max(np.abs(polish_flow)) if polish_flow.size else 0.0,
                    abs(polish_q_unit),
                    1.0,
                )
            )
            polish_residual_relative = float(polish_residual_norm / polish_scale)

            if (
                np.isfinite(polish_residual_relative)
                and (
                    not np.isfinite(energy_residual_relative)
                    or polish_residual_relative <= energy_residual_relative
                )
            ):
                selected_x = np.asarray(polish_result.x, dtype=float)
                solver_optimizer_success = polish_success
                solver_solution_stage = "sparse_least_squares_polish"
            else:
                selected_x = energy_x
                solver_optimizer_success = energy_success
                solver_solution_stage = "energy_minimization"
            solved_heads = assemble_heads(selected_x)
            continuity_residual = residuals(selected_x)
            solver_message = (
                f"Energy stage: {energy_message} "
                f"Sparse polish: {polish_message} "
                f"Selected stage: {solver_solution_stage}."
            )
            solver_method = "energy_lbfgsb_then_sparse_lsmr"
            solver_nfev = energy_nfev + polish_nfev
        else:
            solve_kwargs: dict[str, Any] = {}
            if resolved_solver_backend == "sparse_least_squares":
                solve_kwargs.update(jac=sparse_jacobian, tr_solver="lsmr")
            solve_result = least_squares(
                residuals,
                x0,
                bounds=(0.0, 1.0),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=4000,
                **solve_kwargs,
            )
            solved_heads = assemble_heads(solve_result.x)
            continuity_residual = residuals(solve_result.x)
            solver_optimizer_success = bool(solve_result.success)
            solver_message = str(solve_result.message)
            solver_method = (
                "least_squares_sparse_lsmr"
                if resolved_solver_backend == "sparse_least_squares"
                else "least_squares"
            )
            solver_nfev = int(solve_result.nfev)
    else:
        resolved_solver_backend = "boundary_only"
        solver_cache_key = ""
        solver_cache_path = None
        solved_heads = assemble_heads(np.asarray([], dtype=float))
        continuity_residual = np.asarray([], dtype=float)
        solver_optimizer_success = True
        solver_message = "No internal nodes; equivalent resistance evaluated from boundary-connected edges only."
        solver_method = "boundary_only"
        solver_nfev = 0
        energy_success = None
        energy_message = ""
        energy_nit = 0
        energy_nfev = 0
        energy_residual_norm = float("nan")
        energy_residual_relative = float("nan")
        polish_attempted = False
        polish_success = None
        polish_message = ""
        polish_nfev = 0
        polish_residual_norm = float("nan")
        polish_residual_relative = float("nan")
        solver_solution_stage = "boundary_only"

    flow = np.asarray(edge_flow(solved_heads[u_index], solved_heads[v_index], resistance, smooth=smooth, eps=eps), dtype=float)
    q_unit = float(incidence_upstream @ flow)
    residual_norm = float(np.linalg.norm(continuity_residual, ord=2)) if continuity_residual.size else 0.0
    continuity_scale = float(max(np.max(np.abs(flow)) if flow.size else 0.0, abs(q_unit), 1.0))
    residual_relative = float(residual_norm / continuity_scale)
    residual_acceptable = bool(
        residual_norm <= float(continuity_abs_tolerance)
        or residual_relative <= float(continuity_rel_tolerance)
    )

    if not np.isfinite(q_unit) or q_unit <= 0:
        k_eq = float("nan")
        g_eq = float("nan")
        solver_success = False
        solver_message = f"{solver_message} Upstream unit-head inflow was not positive."
    else:
        k_eq = float(1.0 / (q_unit**2))
        g_eq = float(1.0 / k_eq)
        solver_success = residual_acceptable
        if residual_acceptable and not solver_optimizer_success:
            solver_message = (
                f"{solver_message} Accepted by continuity residual tolerance "
                f"(abs={residual_norm:.3e}, rel={residual_relative:.3e})."
            )
        elif not residual_acceptable:
            solver_message = (
                f"{solver_message} Rejected by continuity residual tolerance "
                f"(abs={residual_norm:.3e}, rel={residual_relative:.3e})."
            )

    node_results = pd.DataFrame(
        {
            "node_id": node_ids,
            "head_norm": solved_heads,
            "is_boundary": [node_id in {upstream_node, downstream_node} for node_id in node_ids],
            "is_upstream_boundary": [node_id == upstream_node for node_id in node_ids],
            "is_downstream_boundary": [node_id == downstream_node for node_id in node_ids],
            "head_source": [
                "boundary_fixed" if node_id in {upstream_node, downstream_node} else "solved"
                for node_id in node_ids
            ],
            "continuity_residual": [
                np.nan if node_id not in internal_index else float(continuity_residual[internal_index[node_id]])
                for node_id in node_ids
            ],
        }
    )
    node_aliases = _solver_cache_node_aliases(edge_frame)
    node_results["solver_cache_node_alias"] = [
        node_aliases.get(node_id, node_id) for node_id in node_ids
    ]

    edge_results = edge_frame.copy()
    edge_results["K_e"] = resistance
    edge_results["G_e"] = 1.0 / resistance
    edge_results["dh"] = solved_heads[u_index] - solved_heads[v_index]
    edge_results["q_unit"] = flow
    edge_results["flow_from_node"] = np.where(edge_results["q_unit"] >= 0.0, edge_results["u"], edge_results["v"])
    edge_results["flow_to_node"] = np.where(edge_results["q_unit"] >= 0.0, edge_results["v"], edge_results["u"])

    result = {
        "upstream_node": upstream_node,
        "downstream_node": downstream_node,
        "n_nodes": int(len(node_ids)),
        "n_edges": int(len(edge_frame)),
        "solver_success": bool(solver_success and np.isfinite(residual_norm)),
        "solver_method": solver_method,
        "solver_algorithm_version": SOLVER_ALGORITHM_VERSION,
        "solver_backend_requested": requested_solver_backend,
        "solver_backend_resolved": resolved_solver_backend,
        "solver_optimizer_success": bool(solver_optimizer_success),
        "solver_solution_stage": solver_solution_stage,
        "energy_stage_success": energy_success,
        "energy_stage_message": energy_message,
        "energy_stage_nit": energy_nit,
        "energy_stage_nfev": energy_nfev,
        "energy_stage_residual_norm": energy_residual_norm,
        "energy_stage_residual_relative": energy_residual_relative,
        "polish_attempted": polish_attempted,
        "polish_success": polish_success,
        "polish_message": polish_message,
        "polish_nfev": polish_nfev,
        "polish_residual_norm": polish_residual_norm,
        "polish_residual_relative": polish_residual_relative,
        "sparse_min_internal_nodes": int(sparse_min_internal_nodes),
        "solver_cache_hit": False,
        "solver_cache_key": solver_cache_key,
        "solver_cache_path": (
            str(solver_cache_path.resolve()) if solver_cache_path is not None else ""
        ),
        "solver_message": solver_message,
        "solver_nfev": solver_nfev,
        "K_eq": k_eq,
        "G_eq": g_eq,
        "Q_unit": q_unit,
        "residual_norm": residual_norm,
        "residual_relative": residual_relative,
        "node_heads": node_results,
        "internal_heads": {
            row.node_id: float(row.head_norm)
            for row in node_results.loc[~node_results["is_boundary"]].itertuples(index=False)
        },
        "edge_flows": edge_results.reset_index(drop=True),
    }
    if solver_cache_path is not None:
        _write_cached_solver_result(
            cache_path=solver_cache_path,
            cache_key=solver_cache_key,
            result=result,
        )
    return result


__all__ = [
    "compare_states",
    "compute_K_chezy",
    "compute_K_manning",
    "compute_depth",
    "compute_geometry",
    "edge_flow",
    "reduce_simple_parallel",
    "reduce_simple_series",
    "solve_bubble_equivalent_K",
]

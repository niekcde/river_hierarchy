from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


NODE_ID_CANDIDATES = ("node_id", "id_node", "id")
EDGE_ID_CANDIDATES = ("edge_id", "id_edge", "id_link", "id")
EDGE_U_CANDIDATES = ("u", "id_us_node", "source", "from_node")
EDGE_V_CANDIDATES = ("v", "id_ds_node", "target", "to_node")
EDGE_K_CANDIDATES = ("K_e", "K", "edge_resistance", "resistance")
HEAD_VALUE_CANDIDATES = ("H", "head", "sword_wse", "wse", "h")


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
) -> dict[str, Any]:
    """Solve a normalized unit-head equivalent resistance for a bubble subgraph.

    The solve fixes `h_up = 1` and `h_down = 0`, then solves for internal
    normalized heads using only edge resistance values derived from geometry.
    Model discharge Q, hydrograph outputs, and routed RAPID peaks must not be
    used inside this predictor calculation.
    """

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

    if continuity_abs_tolerance < 0:
        raise ValueError("continuity_abs_tolerance cannot be negative.")
    if continuity_rel_tolerance < 0:
        raise ValueError("continuity_rel_tolerance cannot be negative.")

    if internal_nodes:
        x0 = _build_initial_head_guess(
            graph,
            internal_nodes=internal_nodes,
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            normalized_head_guess=normalized_head_guess,
        )
        solve_result = least_squares(
            residuals,
            x0,
            bounds=(0.0, 1.0),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=4000,
        )
        solved_heads = assemble_heads(solve_result.x)
        continuity_residual = residuals(solve_result.x)
        solver_success = bool(solve_result.success)
        solver_message = str(solve_result.message)
        solver_method = "least_squares"
        solver_nfev = int(solve_result.nfev)
    else:
        solve_result = None
        solved_heads = assemble_heads(np.asarray([], dtype=float))
        continuity_residual = np.asarray([], dtype=float)
        solver_success = True
        solver_message = "No internal nodes; equivalent resistance evaluated from boundary-connected edges only."
        solver_method = "boundary_only"
        solver_nfev = 0

    flow = np.asarray(edge_flow(solved_heads[u_index], solved_heads[v_index], resistance, smooth=smooth, eps=eps), dtype=float)
    q_unit = float(incidence_upstream @ flow)
    residual_norm = float(np.linalg.norm(continuity_residual, ord=2)) if continuity_residual.size else 0.0
    continuity_scale = float(max(np.max(np.abs(flow)) if flow.size else 0.0, abs(q_unit), 1.0))
    residual_relative = float(residual_norm / continuity_scale)

    if not np.isfinite(q_unit) or q_unit <= 0:
        k_eq = float("nan")
        g_eq = float("nan")
        solver_success = False
        solver_message = f"{solver_message} Upstream unit-head inflow was not positive."
    else:
        k_eq = float(1.0 / (q_unit**2))
        g_eq = float(1.0 / k_eq)
        if not solver_success and (
            residual_norm <= float(continuity_abs_tolerance)
            or residual_relative <= float(continuity_rel_tolerance)
        ):
            solver_success = True
            solver_message = (
                f"{solver_message} Accepted by continuity residual tolerance "
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

    edge_results = edge_frame.copy()
    edge_results["K_e"] = resistance
    edge_results["G_e"] = 1.0 / resistance
    edge_results["dh"] = solved_heads[u_index] - solved_heads[v_index]
    edge_results["q_unit"] = flow
    edge_results["flow_from_node"] = np.where(edge_results["q_unit"] >= 0.0, edge_results["u"], edge_results["v"])
    edge_results["flow_to_node"] = np.where(edge_results["q_unit"] >= 0.0, edge_results["v"], edge_results["u"])

    return {
        "upstream_node": upstream_node,
        "downstream_node": downstream_node,
        "n_nodes": int(len(node_ids)),
        "n_edges": int(len(edge_frame)),
        "solver_success": bool(solver_success and np.isfinite(residual_norm)),
        "solver_method": solver_method,
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

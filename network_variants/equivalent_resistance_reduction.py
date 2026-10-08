"""Exact structural reduction audit for two-terminal resistance domains.

The equivalent-resistance law used here is ``q = sign(dH) * sqrt(abs(dH) / K)``.
Consequently, serial edges combine by summing ``K`` and parallel edges combine by
adding ``1 / sqrt(K)``.  These operations are exact.  This module deliberately
does *not* approximate a general braided graph: it records only contractions
that preserve the two-terminal hydraulic solution exactly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from hierarchy_level_definition.metrics.equivalent_resistance import (
    reduce_simple_parallel,
    reduce_simple_series,
)


@dataclass(frozen=True)
class ResistanceReductionResult:
    """Products of an exact series/parallel reduction audit."""

    reduced_edges: pd.DataFrame
    operations: pd.DataFrame
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Resistance-edge input is missing required columns: {missing}")


def _edge_record(
    *,
    token: str,
    u: int,
    v: int,
    resistance: float,
    member_edge_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not np.isfinite(resistance) or resistance <= 0.0:
        raise ValueError("All resistance edges must have finite positive K values.")
    return {
        "reduced_edge_id": token,
        "u": int(u),
        "v": int(v),
        "K_e": float(resistance),
        "n_original_edges": int(len(member_edge_ids)),
        "original_edge_ids": ",".join(member_edge_ids),
    }


def _as_edge_records(edges: pd.DataFrame) -> list[dict[str, Any]]:
    _require_columns(edges, ("u", "v", "K_e"))
    edge_id_column = "edge_id" if "edge_id" in edges.columns else "id_link"
    _require_columns(edges, (edge_id_column,))
    records: list[dict[str, Any]] = []
    for index, row in edges.reset_index(drop=True).iterrows():
        token = f"E{index:06d}"
        records.append(
            _edge_record(
                token=token,
                u=int(row["u"]),
                v=int(row["v"]),
                resistance=float(row["K_e"]),
                member_edge_ids=(str(row[edge_id_column]),),
            )
        )
    return records


def _parallel_reduce_once(
    records: list[dict[str, Any]],
    *,
    token_counter: int,
    operation_counter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(sorted((int(record["u"]), int(record["v"]))))].append(record)

    reduced: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for pair in sorted(grouped):
        group = sorted(grouped[pair], key=lambda row: str(row["reduced_edge_id"]))
        if len(group) == 1:
            reduced.append(group[0])
            continue
        token_counter += 1
        operation_counter += 1
        members = tuple(
            item
            for record in group
            for item in str(record["original_edge_ids"]).split(",")
        )
        equivalent_k = reduce_simple_parallel([float(record["K_e"]) for record in group])
        replacement = _edge_record(
            token=f"P{token_counter:06d}",
            u=pair[0],
            v=pair[1],
            resistance=equivalent_k,
            member_edge_ids=members,
        )
        reduced.append(replacement)
        operations.append(
            {
                "operation_id": operation_counter,
                "operation": "parallel",
                "eliminated_node": pd.NA,
                "input_reduced_edge_ids": ",".join(
                    str(record["reduced_edge_id"]) for record in group
                ),
                "output_reduced_edge_id": replacement["reduced_edge_id"],
                "input_K_e": ",".join(f"{float(record['K_e']):.17g}" for record in group),
                "output_K_e": replacement["K_e"],
                "original_edge_ids": replacement["original_edge_ids"],
            }
        )
    return reduced, operations, token_counter, operation_counter


def _series_reduce_once(
    records: list[dict[str, Any]],
    *,
    source_node: int,
    sink_node: int,
    token_counter: int,
    operation_counter: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, int]:
    incident: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        incident[int(record["u"])].append(index)
        incident[int(record["v"])].append(index)

    candidate_nodes = sorted(
        node
        for node, indices in incident.items()
        if node not in {source_node, sink_node} and len(indices) == 2
    )
    for node in candidate_nodes:
        first_index, second_index = incident[node]
        first, second = records[first_index], records[second_index]
        first_other = int(first["v"]) if int(first["u"]) == node else int(first["u"])
        second_other = int(second["v"]) if int(second["u"]) == node else int(second["u"])
        # A degree-two loop does not form a serial source-to-sink connection;
        # retain it rather than silently discarding a potentially diagnostic edge.
        if first_other == second_other:
            continue
        token_counter += 1
        operation_counter += 1
        members = tuple(
            str(record["original_edge_ids"]).split(",")[member_index]
            for record in (first, second)
            for member_index in range(len(str(record["original_edge_ids"]).split(",")))
        )
        replacement = _edge_record(
            token=f"S{token_counter:06d}",
            u=first_other,
            v=second_other,
            resistance=reduce_simple_series([float(first["K_e"]), float(second["K_e"])]),
            member_edge_ids=members,
        )
        reduced = [
            record
            for index, record in enumerate(records)
            if index not in {first_index, second_index}
        ]
        reduced.append(replacement)
        operation = {
            "operation_id": operation_counter,
            "operation": "series",
            "eliminated_node": node,
            "input_reduced_edge_ids": f"{first['reduced_edge_id']},{second['reduced_edge_id']}",
            "output_reduced_edge_id": replacement["reduced_edge_id"],
            "input_K_e": f"{float(first['K_e']):.17g},{float(second['K_e']):.17g}",
            "output_K_e": replacement["K_e"],
            "original_edge_ids": replacement["original_edge_ids"],
        }
        return reduced, operation, token_counter, operation_counter
    return records, None, token_counter, operation_counter


def _graph_structure_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    graph = nx.Graph()
    graph.add_edges_from((int(row["u"]), int(row["v"])) for row in records)
    if not graph:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "n_articulation_nodes": 0,
            "n_biconnected_blocks": 0,
            "largest_biconnected_block_nodes": 0,
            "largest_biconnected_block_edges": 0,
        }
    blocks = list(nx.biconnected_components(graph))
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(len(records)),
        "n_articulation_nodes": int(len(list(nx.articulation_points(graph)))),
        "n_biconnected_blocks": int(len(blocks)),
        "largest_biconnected_block_nodes": int(max((len(block) for block in blocks), default=0)),
        "largest_biconnected_block_edges": int(
            max((graph.subgraph(block).number_of_edges() for block in blocks), default=0)
        ),
    }


def reduce_series_parallel_resistance_graph(
    edges: pd.DataFrame,
    *,
    source_node: int,
    sink_node: int,
) -> ResistanceReductionResult:
    """Apply exact serial and parallel contractions until no further step exists.

    Direction is intentionally ignored for the structural reduction because the
    unit-head resistance law is two-terminal and sign-symmetric.  The input
    domain must already have passed directed closure validation.
    """
    source_node, sink_node = int(source_node), int(sink_node)
    if source_node == sink_node:
        raise ValueError("source_node and sink_node must be distinct.")
    records = _as_edge_records(edges)
    original_structure = _graph_structure_summary(records)
    original_nodes = {int(row["u"]) for row in records} | {int(row["v"]) for row in records}
    if source_node not in original_nodes or sink_node not in original_nodes:
        raise ValueError("source_node and sink_node must occur in the edge domain.")

    operations: list[dict[str, Any]] = []
    token_counter = len(records)
    operation_counter = 0
    while True:
        records, parallel_operations, token_counter, operation_counter = _parallel_reduce_once(
            records,
            token_counter=token_counter,
            operation_counter=operation_counter,
        )
        operations.extend(parallel_operations)
        records, series_operation, token_counter, operation_counter = _series_reduce_once(
            records,
            source_node=source_node,
            sink_node=sink_node,
            token_counter=token_counter,
            operation_counter=operation_counter,
        )
        if series_operation is None:
            break
        operations.append(series_operation)

    reduced_edges = pd.DataFrame.from_records(records).sort_values(
        "reduced_edge_id", kind="stable"
    ).reset_index(drop=True)
    operations_frame = pd.DataFrame.from_records(
        operations,
        columns=(
            "operation_id", "operation", "eliminated_node", "input_reduced_edge_ids",
            "output_reduced_edge_id", "input_K_e", "output_K_e", "original_edge_ids",
        ),
    )
    reduced_structure = _graph_structure_summary(records)
    summary = {
        "schema_version": "exact-series-parallel-resistance-reduction-v1",
        "source_node": source_node,
        "sink_node": sink_node,
        "n_original_nodes": len(original_nodes),
        "n_original_edges": len(records) + int(sum(
            1 for operation in operations if operation["operation"] == "series"
        )),
        "n_reduced_nodes": reduced_structure["n_nodes"],
        "n_reduced_edges": len(records),
        "n_series_contractions": int(sum(
            operation["operation"] == "series" for operation in operations
        )),
        "n_parallel_contractions": int(sum(
            operation["operation"] == "parallel" for operation in operations
        )),
        "original_structure": original_structure,
        "reduced_structure": reduced_structure,
        "reduction_is_complete": True,
    }
    # ``n_original_edges`` is simply the original input size.  Keep it
    # explicit rather than infer it from reduction-operation counts.
    summary["n_original_edges"] = int(len(edges))
    return ResistanceReductionResult(
        reduced_edges=reduced_edges,
        operations=operations_frame,
        summary=summary,
    )


__all__ = [
    "ResistanceReductionResult",
    "reduce_series_parallel_resistance_graph",
]

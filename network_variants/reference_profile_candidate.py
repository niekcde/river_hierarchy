"""Non-destructive candidate hydraulic contexts from frozen reference profiles.

This module is deliberately separate from :mod:`network_variants.hydraulic_context`.
It is the first production implementation of the reviewed slope-refit method:

* a frozen raw-pass, stage-adjusted WSE profile supplies external heads;
* every standalone simple unit and assembled compound bubble is solved as a
  two-terminal, positive-drop, equal-per-link quadratic problem;
* links outside those uniquely owned domains retain their direct frozen-profile
  slope.  In particular, unassigned connectors are *not* altered by the
  retired nearest-anchor factor screen.

The outputs are candidate artifacts only.  They neither overwrite the current
hydraulic contexts nor trigger resistance/RAPID work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from network_variants.hydraulic_context import (
    SlopeConfig,
    _load_reference_section_support,
    _project_nodes_to_reference_section,
    _select_variant_lengths,
)


CANDIDATE_CONTEXT_SCHEMA_VERSION = "reference-profile-candidate-context-v3"

# Each realised link receives equal influence, regardless of its geometric
# length. This prevents a short link from becoming a cheap location at which
# to concentrate the boundary head loss solely because a length multiplier
# would otherwise down-weight its relative slope error.
DOMAIN_OBJECTIVE_NAME = "equal_per_link_relative_slope_error"
DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION = 0.01


@dataclass(frozen=True, slots=True)
class CandidateContextOutputs:
    output_dir: Path
    link_context_path: Path
    node_context_path: Path
    domain_audit_path: Path
    ownership_audit_path: Path
    manifest_path: Path
    summary: dict[str, object]


def _one_path(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {directory}; found {matches}")
    return matches[0]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_ids(value: object) -> set[int]:
    if value is None or pd.isna(value) or not str(value).strip():
        return set()
    return {int(float(token.strip())) for token in str(value).split(",") if token.strip()}


def _required_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _path_drop_audit(graph: nx.MultiDiGraph, drops_by_link: dict[int, float], source: int, sink: int) -> dict[str, float | int]:
    simple_graph = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(simple_graph):
        raise ValueError("A strictly decreasing scalar head field is impossible on a directed cycle.")
    if not nx.has_path(simple_graph, source, sink):
        raise ValueError("The domain has no source-to-sink path.")
    minimum = {int(node): np.inf for node in graph.nodes}
    maximum = {int(node): -np.inf for node in graph.nodes}
    n_paths = {int(node): 0 for node in graph.nodes}
    minimum[source] = maximum[source] = 0.0
    n_paths[source] = 1
    for node in nx.topological_sort(simple_graph):
        node = int(node)
        if n_paths[node] == 0:
            continue
        for _, target, key in graph.out_edges(node, keys=True):
            target = int(target)
            drop = float(drops_by_link[int(key)])
            minimum[target] = min(minimum[target], minimum[node] + drop)
            maximum[target] = max(maximum[target], maximum[node] + drop)
            n_paths[target] += n_paths[node]
    return {
        "n_paths": int(n_paths[sink]),
        "minimum_path_drop_m": float(minimum[sink]),
        "maximum_path_drop_m": float(maximum[sink]),
    }


def solve_boundary_conditioned_domain(
    selected_links: pd.DataFrame,
    *,
    source: int,
    sink: int,
    source_head_m: float,
    sink_head_m: float,
    reference_distance_m: float,
    numerical_slope_floor_m_per_m: float = 1e-10,
    minimum_reference_slope_fraction: float = DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION,
) -> tuple[pd.DataFrame, dict[str, object], dict[int, float]]:
    """Solve the reviewed positive, equal-per-link quadratic domain problem.

    The objective is
    ``sum_e [ ((H_u - H_d)/L_e - S_ref) / S_ref ]**2`` under fixed external
    profile heads and a lower bound of ``0.01 * S_ref`` on every realised
    directed-link slope. Equal link weights prevent short links from becoming
    cheap places to concentrate the domain's required head loss; the physical
    floor prevents a constrained optimum from creating effectively flat links.
    """
    _required_columns(selected_links, {"id_link", "id_us_node", "id_ds_node", "actual_length_m"}, label="Selected domain links")
    local = selected_links.copy()
    for column in ("id_link", "id_us_node", "id_ds_node"):
        local[column] = pd.to_numeric(local[column], errors="coerce").astype("Int64")
    local["actual_length_m"] = pd.to_numeric(local["actual_length_m"], errors="coerce")
    local = local.dropna(subset=["id_link", "id_us_node", "id_ds_node", "actual_length_m"]).copy()
    if local.empty or local["actual_length_m"].le(0.0).any():
        raise ValueError("A domain needs complete positive link lengths.")
    if local["id_link"].duplicated().any():
        raise ValueError("A domain cannot contain duplicate link IDs.")

    graph = nx.MultiDiGraph()
    for row in local.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link), length_m=float(row.actual_length_m))
    simple_graph = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(simple_graph) or not nx.has_path(simple_graph, int(source), int(sink)):
        raise ValueError("Selected domain must be an acyclic connected source-to-sink graph.")
    if not np.isfinite([source_head_m, sink_head_m, reference_distance_m]).all():
        raise ValueError("Domain boundary heads and distance must be finite.")
    total_drop = float(source_head_m - sink_head_m)
    if total_drop <= 0.0 or float(reference_distance_m) <= 0.0:
        raise ValueError("Boundary heads and reference distance must define a positive local slope.")
    reference_slope = total_drop / float(reference_distance_m)
    if not np.isfinite(minimum_reference_slope_fraction) or not 0.0 < float(minimum_reference_slope_fraction) < 1.0:
        raise ValueError("minimum_reference_slope_fraction must lie strictly between zero and one.")
    floor = max(
        float(numerical_slope_floor_m_per_m),
        float(minimum_reference_slope_fraction) * reference_slope,
    )

    nodes = set(int(value) for value in graph.nodes)
    if int(source) not in nodes or int(sink) not in nodes:
        raise ValueError("Both declared boundaries must be present in the selected domain.")
    internal = sorted(nodes.difference({int(source), int(sink)}))
    index = {node: position for position, node in enumerate(internal)}
    lengths = local["actual_length_m"].to_numpy(float)
    fixed_drop = np.zeros(len(local), dtype=float)
    incidence = np.zeros((len(local), len(internal)), dtype=float)
    for edge_index, row in enumerate(local.itertuples(index=False)):
        for node, sign in ((int(row.id_us_node), 1.0), (int(row.id_ds_node), -1.0)):
            if node in index:
                incidence[edge_index, index[node]] += sign
            elif node == int(source):
                fixed_drop[edge_index] += sign * float(source_head_m)
            elif node == int(sink):
                fixed_drop[edge_index] += sign * float(sink_head_m)
            else:
                raise ValueError("A domain contains an unclassified fixed node.")

    def drops(values: np.ndarray) -> np.ndarray:
        return fixed_drop + incidence @ values

    def objective(values: np.ndarray) -> float:
        residual = drops(values) / lengths - reference_slope
        return float(np.sum(np.square(residual / reference_slope)))

    def jacobian(values: np.ndarray) -> np.ndarray:
        return 2.0 * incidence.T @ (
            (drops(values) / lengths - reference_slope) / (lengths * reference_slope**2)
        )

    if internal:
        longest_to_sink = {int(sink): 0.0}
        for node in reversed(list(nx.topological_sort(simple_graph))):
            node = int(node)
            if node != int(sink):
                outgoing = list(graph.out_edges(node, keys=True, data=True))
                if not outgoing:
                    raise ValueError("A non-sink domain node has no downstream edge.")
                longest_to_sink[node] = max(float(data["length_m"]) + longest_to_sink[int(target)] for _, target, _, data in outgoing)
        source_longest = longest_to_sink[int(source)]
        if source_longest <= 0.0:
            raise ValueError("Domain source has zero downstream path length.")
        x0 = np.asarray([float(sink_head_m) + total_drop * longest_to_sink[node] / source_longest for node in internal], dtype=float)
        constraint = LinearConstraint(incidence, floor * lengths - fixed_drop, np.full(len(lengths), np.inf))
        optimized = minimize(objective, x0, jac=jacobian, method="SLSQP", constraints=[constraint], options={"ftol": 1e-12, "maxiter": 5_000})
        if not optimized.success:
            raise RuntimeError(f"Constrained domain solve failed: {optimized.message}")
        solved = np.asarray(optimized.x, dtype=float)
        optimizer_message = str(optimized.message)
    else:
        solved = np.asarray([], dtype=float)
        if np.any(drops(solved) < floor * lengths):
            raise ValueError("Direct boundary links do not satisfy the positive-drop constraint.")
        optimizer_message = "direct_boundary_solution"

    heads = {int(source): float(source_head_m), int(sink): float(sink_head_m), **{node: float(solved[position]) for node, position in index.items()}}
    candidate = local.copy()
    candidate["candidate_head_us_m"] = candidate["id_us_node"].astype(int).map(heads)
    candidate["candidate_head_ds_m"] = candidate["id_ds_node"].astype(int).map(heads)
    candidate["candidate_slope_m_per_m"] = (candidate["candidate_head_us_m"] - candidate["candidate_head_ds_m"]) / candidate["actual_length_m"]
    drop_by_link = dict(zip(candidate["id_link"].astype(int), candidate["candidate_slope_m_per_m"] * candidate["actual_length_m"], strict=True))
    audit = _path_drop_audit(graph, drop_by_link, int(source), int(sink))
    audit.update({
        "boundary_head_loss_m": total_drop,
        "local_reference_slope_m_per_m": reference_slope,
        "max_abs_path_residual_m": max(abs(float(audit["minimum_path_drop_m"]) - total_drop), abs(float(audit["maximum_path_drop_m"]) - total_drop)),
        "objective_value": objective(solved),
        "objective_name": DOMAIN_OBJECTIVE_NAME,
        "numerical_slope_floor_m_per_m": floor,
        "minimum_reference_slope_fraction": float(minimum_reference_slope_fraction),
        "minimum_domain_slope_m_per_m": floor,
        "optimizer_message": optimizer_message,
    })
    return candidate, audit, heads


def _candidate_ownership(
    links: gpd.GeoDataFrame,
    unit_summary: pd.DataFrame,
    bubble_summary: pd.DataFrame,
    legacy_context: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Assign every base link once, using the workflow's current bubble record."""
    _required_columns(links, {"id_link", "unit_ids"}, label="Directed base links")
    _required_columns(
        unit_summary,
        {"unit_id", "compound_bubble_role", "is_compound", "bifurcation", "confluence"},
        label="Unit summary",
    )
    _required_columns(
        bubble_summary,
        {"compound_bubble_id", "upstream_boundary_node", "downstream_boundary_node"},
        label="Bubble summary",
    )
    _required_columns(legacy_context, {"id_link", "compound_bubble_id_link"}, label="Existing link context")
    ownership = links[["id_link", "unit_ids"]].copy().merge(
        legacy_context[["id_link", "compound_bubble_id_link"]], on="id_link", how="left", validate="1:1"
    )
    ownership["id_link"] = pd.to_numeric(ownership["id_link"], errors="coerce").astype("Int64")
    ownership["compound_bubble_id_link"] = pd.to_numeric(
        ownership["compound_bubble_id_link"], errors="coerce"
    ).astype("Int64")
    standalone = unit_summary.loc[
        unit_summary["compound_bubble_role"].astype(str).eq("standalone")
        & ~unit_summary["is_compound"].map(_as_bool)
    ].copy()
    standalone["unit_id"] = pd.to_numeric(standalone["unit_id"], errors="coerce").astype("Int64")
    standalone_ids = set(standalone["unit_id"].dropna().astype(int))
    ownership["standalone_unit_candidates"] = ownership["unit_ids"].map(_parse_ids).map(
        lambda values: sorted(values & standalone_ids)
    )
    ownership["ownership_kind"] = np.select(
        [
            ownership["compound_bubble_id_link"].notna(),
            ownership["standalone_unit_candidates"].map(len).eq(1),
            ownership["standalone_unit_candidates"].map(len).gt(1),
            ownership["unit_ids"].map(_parse_ids).map(len).eq(0),
        ],
        [
            "compound_bubble",
            "simple_unit",
            "ambiguous_simple_membership",
            "direct_profile_unassigned_or_boundary",
        ],
        default="direct_profile_nonrefit_membership",
    )
    ownership["ownership_domain_id"] = pd.NA
    bubble_mask = ownership["ownership_kind"].eq("compound_bubble")
    ownership.loc[bubble_mask, "ownership_domain_id"] = (
        ownership.loc[bubble_mask, "compound_bubble_id_link"].astype(int).astype(str)
    )
    simple_mask = ownership["ownership_kind"].eq("simple_unit")
    ownership.loc[simple_mask, "ownership_domain_id"] = ownership.loc[
        simple_mask, "standalone_unit_candidates"
    ].map(lambda ids: str(ids[0]))

    domains: list[dict[str, object]] = []
    standalone_lookup = standalone.dropna(subset=["unit_id"]).set_index("unit_id")
    for unit_id in sorted(
        pd.to_numeric(ownership.loc[simple_mask, "ownership_domain_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    ):
        row = standalone_lookup.loc[int(unit_id)]
        link_ids = set(
            ownership.loc[
                ownership["ownership_kind"].eq("simple_unit")
                & ownership["ownership_domain_id"].eq(str(unit_id)),
                "id_link",
            ]
            .dropna()
            .astype(int)
        )
        domains.append(
            {
                "domain_kind": "simple_unit",
                "domain_id": int(unit_id),
                "source": int(row["bifurcation"]),
                "sink": int(row["confluence"]),
                "link_ids": link_ids,
            }
        )

    bubbles = bubble_summary.copy()
    for column in ("compound_bubble_id", "upstream_boundary_node", "downstream_boundary_node"):
        bubbles[column] = pd.to_numeric(bubbles[column], errors="coerce").astype("Int64")
    bubble_lookup = bubbles.dropna(subset=["compound_bubble_id"]).set_index("compound_bubble_id")
    for bubble_id in sorted(
        pd.to_numeric(ownership.loc[bubble_mask, "ownership_domain_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    ):
        link_ids = set(
            ownership.loc[
                ownership["ownership_kind"].eq("compound_bubble")
                & ownership["ownership_domain_id"].eq(str(bubble_id)),
                "id_link",
            ]
            .dropna()
            .astype(int)
        )
        if int(bubble_id) not in bubble_lookup.index:
            ownership.loc[
                ownership["ownership_domain_id"].eq(str(bubble_id)), "ownership_kind"
            ] = "direct_profile_missing_bubble_boundary"
            continue
        row = bubble_lookup.loc[int(bubble_id)]
        if pd.isna(row["upstream_boundary_node"]) or pd.isna(row["downstream_boundary_node"]):
            ownership.loc[
                ownership["ownership_domain_id"].eq(str(bubble_id)), "ownership_kind"
            ] = "direct_profile_missing_bubble_boundary"
            continue
        domains.append(
            {
                "domain_kind": "compound_bubble",
                "domain_id": int(bubble_id),
                "source": int(row["upstream_boundary_node"]),
                "sink": int(row["downstream_boundary_node"]),
                "link_ids": link_ids,
            }
        )
    return ownership, domains


def _read_profile(profile_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    profile_path = profile_dir / "reference_profile.parquet"
    selection_path = profile_dir / "reference_profile_selection.json"
    audit_path = profile_dir / "reference_profile_audit.json"
    if not profile_path.exists() or not selection_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            "Reference profile directory is incomplete; expected profile parquet, selection JSON, and audit JSON."
        )
    profile = pd.read_parquet(profile_path)
    _required_columns(
        profile,
        {"reference_section_dist_out_m", "fitted_wse_m"},
        label="Frozen reference profile",
    )
    profile = (
        profile[["reference_section_dist_out_m", "fitted_wse_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .drop_duplicates("reference_section_dist_out_m")
        .sort_values("reference_section_dist_out_m")
    )
    if len(profile) < 2 or profile["reference_section_dist_out_m"].diff().dropna().le(0.0).any():
        raise ValueError("Frozen reference profile needs at least two unique increasing coordinates.")
    return profile, {
        "selection": json.loads(selection_path.read_text(encoding="utf-8")),
        "audit": json.loads(audit_path.read_text(encoding="utf-8")),
        "profile_path": str(profile_path),
    }


def generate_reference_profile_candidate_context(
    experiment_dir: str | Path,
    profile_dir: str | Path,
    output_dir: str | Path,
) -> CandidateContextOutputs:
    """Create a fully audited base-state candidate context without mutation."""
    experiment = Path(experiment_dir).expanduser().resolve()
    profile_root = Path(profile_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Candidate context output must be empty: {output}")
    state = experiment / "states" / "S000_base"
    directed = state / "variant" / "directed"
    hierarchy = state / "hierarchy"
    legacy_context_path = state / "hydraulic_context" / "link_hydraulic_context.csv"
    reference_summary = experiment / "reference_section" / "reference_section_summary.json"
    for required in (state, directed, hierarchy, legacy_context_path, reference_summary):
        if not required.exists():
            raise FileNotFoundError(f"Required base input was not found: {required}")

    links = gpd.read_file(_one_path(directed, "*directed_links.gpkg"))
    nodes = gpd.read_file(_one_path(directed, "*directed_nodes.gpkg"))
    units = pd.read_csv(hierarchy / "unit_summary.csv")
    bubbles = pd.read_csv(hierarchy / "bubble_equivalent_metrics.csv")
    legacy = pd.read_csv(legacy_context_path)
    profile, profile_metadata = _read_profile(profile_root)
    lengths, length_source = _select_variant_lengths(links, preferred_length_field="len")
    support = _load_reference_section_support(str(reference_summary))
    distances, distance_source = _project_nodes_to_reference_section(
        nodes,
        support=support,
        config=SlopeConfig(reference_section_summary_path=str(reference_summary)),
    )
    node_context = nodes[["id_node"]].copy()
    node_context["id_node"] = pd.to_numeric(node_context["id_node"], errors="coerce").astype("Int64")
    node_context["reference_section_dist_out_m"] = distances
    node_context["reference_section_dist_source"] = distance_source
    if node_context["reference_section_dist_out_m"].isna().any():
        missing = (
            node_context.loc[node_context["reference_section_dist_out_m"].isna(), "id_node"]
            .dropna()
            .astype(int)
            .tolist()
        )
        raise ValueError(f"Base nodes missing reference-section coordinates: {missing}")
    xp = profile["reference_section_dist_out_m"].to_numpy(float)
    yp = profile["fitted_wse_m"].to_numpy(float)
    node_context["profile_head_m"] = np.interp(
        node_context["reference_section_dist_out_m"].to_numpy(float), xp, yp
    )
    node_context["candidate_head_m"] = node_context["profile_head_m"]
    node_context["candidate_head_source_method"] = "frozen_reference_profile_direct"

    link_frame = links[["id_link", "id_us_node", "id_ds_node"]].copy()
    for column in ("id_link", "id_us_node", "id_ds_node"):
        link_frame[column] = pd.to_numeric(link_frame[column], errors="coerce").astype("Int64")
    link_frame["actual_length_m"] = lengths
    link_frame["actual_length_source"] = length_source
    link_frame = link_frame.merge(
        node_context[["id_node", "profile_head_m"]].rename(
            columns={"id_node": "id_us_node", "profile_head_m": "profile_head_us_m"}
        ),
        on="id_us_node",
        how="left",
        validate="m:1",
    ).merge(
        node_context[["id_node", "profile_head_m"]].rename(
            columns={"id_node": "id_ds_node", "profile_head_m": "profile_head_ds_m"}
        ),
        on="id_ds_node",
        how="left",
        validate="m:1",
    )
    link_frame["profile_direct_slope_m_per_m"] = (
        link_frame["profile_head_us_m"] - link_frame["profile_head_ds_m"]
    ) / link_frame["actual_length_m"]

    ownership, domains = _candidate_ownership(links, units, bubbles, legacy)
    link_frame = link_frame.merge(
        ownership[["id_link", "ownership_kind", "ownership_domain_id"]],
        on="id_link",
        how="left",
        validate="1:1",
    )
    link_frame["candidate_head_us_m"] = link_frame["profile_head_us_m"]
    link_frame["candidate_head_ds_m"] = link_frame["profile_head_ds_m"]
    link_frame["candidate_slope_m_per_m"] = link_frame["profile_direct_slope_m_per_m"]
    link_frame["candidate_slope_source_method"] = "frozen_profile_direct_connector_or_boundary"
    link_frame["candidate_domain_kind"] = "direct_profile"
    link_frame["candidate_domain_id"] = pd.NA
    link_frame["candidate_numerical_floor_active"] = False
    link_frame["candidate_domain_path_residual_m"] = np.nan

    profile_heads = node_context.set_index("id_node")["profile_head_m"]
    profile_distances = node_context.set_index("id_node")["reference_section_dist_out_m"]
    domain_audits: list[dict[str, object]] = []
    assigned_internal_heads: dict[int, float] = {}
    for domain in domains:
        kind, identifier = str(domain["domain_kind"]), int(domain["domain_id"])
        label = f"{kind}:{identifier}"
        selected_ids = set(domain["link_ids"])
        selected = link_frame.loc[link_frame["id_link"].isin(selected_ids)].copy()
        audit_row: dict[str, object] = {
            "domain_kind": kind,
            "domain_id": identifier,
            "domain_label": label,
            "source_node": int(domain["source"]),
            "sink_node": int(domain["sink"]),
            "n_links": int(len(selected)),
            "status": "",
            "error": "",
        }
        try:
            source, sink = int(domain["source"]), int(domain["sink"])
            if source not in profile_heads.index or sink not in profile_heads.index:
                raise ValueError("one or both domain boundary heads are unavailable")
            reference_distance = float(profile_distances.loc[source] - profile_distances.loc[sink])
            solved, audit, heads = solve_boundary_conditioned_domain(
                selected,
                source=source,
                sink=sink,
                source_head_m=float(profile_heads.loc[source]),
                sink_head_m=float(profile_heads.loc[sink]),
                reference_distance_m=reference_distance,
            )
            if solved["id_link"].duplicated().any() or set(solved["id_link"].astype(int)) != selected_ids:
                raise RuntimeError("domain solver did not return exactly its owned links")
            solved_by_id = solved.set_index("id_link")
            mask = link_frame["id_link"].isin(selected_ids)
            link_frame.loc[mask, "candidate_head_us_m"] = link_frame.loc[mask, "id_link"].map(
                solved_by_id["candidate_head_us_m"]
            )
            link_frame.loc[mask, "candidate_head_ds_m"] = link_frame.loc[mask, "id_link"].map(
                solved_by_id["candidate_head_ds_m"]
            )
            link_frame.loc[mask, "candidate_slope_m_per_m"] = link_frame.loc[mask, "id_link"].map(
                solved_by_id["candidate_slope_m_per_m"]
            )
            link_frame.loc[mask, "candidate_slope_source_method"] = "profile_boundary_conditioned_domain"
            link_frame.loc[mask, "candidate_domain_kind"] = kind
            link_frame.loc[mask, "candidate_domain_id"] = identifier
            link_frame.loc[mask, "candidate_numerical_floor_active"] = link_frame.loc[
                mask, "candidate_slope_m_per_m"
            ].le(float(audit["numerical_slope_floor_m_per_m"]) + 1e-14)
            link_frame.loc[mask, "candidate_domain_path_residual_m"] = float(
                audit["max_abs_path_residual_m"]
            )
            for node_id, head in heads.items():
                if node_id in {source, sink}:
                    continue
                existing = assigned_internal_heads.get(node_id)
                if existing is not None and not np.isclose(existing, float(head), rtol=0.0, atol=1e-9):
                    raise ValueError(f"Internal node {node_id} received incompatible domain head assignments.")
                assigned_internal_heads[node_id] = float(head)
            audit_row.update(audit)
            audit_row["status"] = "solved"
        except Exception as exc:
            # Fail closed: do not create an arbitrary fallback. The prefilled
            # direct profile values remain, with a durable review record.
            audit_row["status"] = "not_solved_profile_direct_retained"
            audit_row["error"] = f"{type(exc).__name__}: {exc}"
        domain_audits.append(audit_row)

    for node_id, head in assigned_internal_heads.items():
        mask = node_context["id_node"].eq(node_id)
        node_context.loc[mask, "candidate_head_m"] = head
        node_context.loc[mask, "candidate_head_source_method"] = (
            "profile_boundary_conditioned_domain_internal"
        )
    baseline_fields = [column for column in ("id_link", "slope_used", "slope_source_method") if column in legacy.columns]
    link_frame = link_frame.merge(legacy[baseline_fields], on="id_link", how="left", validate="1:1")
    link_frame["candidate_to_current_slope_ratio"] = link_frame["candidate_slope_m_per_m"] / pd.to_numeric(
        link_frame.get("slope_used"), errors="coerce"
    )
    invalid_final = link_frame["candidate_slope_m_per_m"].isna() | link_frame[
        "candidate_slope_m_per_m"
    ].le(0.0)
    if invalid_final.any():
        invalid = link_frame.loc[invalid_final, ["id_link", "ownership_kind", "candidate_slope_source_method"]].copy()
        raise ValueError(
            "Candidate context contains missing or non-positive final slopes. "
            "A direct-profile link whose projected endpoints cannot supply a positive drop must be "
            f"resolved as an explicit domain, not repaired implicitly: {invalid.to_dict(orient='records')}"
        )

    output.mkdir(parents=True, exist_ok=False)
    link_path = output / "candidate_link_hydraulic_context.parquet"
    node_path = output / "candidate_node_hydraulic_context.parquet"
    domain_path = output / "candidate_domain_head_loss_audit.csv"
    ownership_path = output / "candidate_link_ownership_audit.csv"
    manifest_path = output / "candidate_context_manifest.json"
    domain_audit = pd.DataFrame(domain_audits)
    link_frame.to_parquet(link_path, index=False)
    node_context.to_parquet(node_path, index=False)
    domain_audit.to_csv(domain_path, index=False)
    ownership.to_csv(ownership_path, index=False)
    solved_domain_rows = domain_audit.loc[domain_audit.get("status", pd.Series(dtype=str)).eq("solved")]
    summary: dict[str, object] = {
        "schema_version": CANDIDATE_CONTEXT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment),
        "profile_dir": str(profile_root),
        "profile_metadata": profile_metadata,
        "n_links": int(len(link_frame)),
        "n_nodes": int(len(node_context)),
        "n_domains": int(len(domain_audit)),
        "n_domains_solved": int(domain_audit.get("status", pd.Series(dtype=str)).eq("solved").sum()),
        "n_domains_not_solved": int(domain_audit.get("status", pd.Series(dtype=str)).eq("not_solved_profile_direct_retained").sum()),
        "max_solved_domain_path_residual_m": float(pd.to_numeric(solved_domain_rows.get("max_abs_path_residual_m"), errors="coerce").max()) if not solved_domain_rows.empty else float("nan"),
        "n_profile_direct_links": int(link_frame["candidate_slope_source_method"].eq("frozen_profile_direct_connector_or_boundary").sum()),
        "n_domain_solved_links": int(link_frame["candidate_slope_source_method"].eq("profile_boundary_conditioned_domain").sum()),
        "length_policy": "len preferred through existing _select_variant_lengths; len_adj only if len is unavailable or invalid",
        "domain_objective": DOMAIN_OBJECTIVE_NAME,
        "domain_minimum_reference_slope_fraction": DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION,
        "connector_policy": "direct frozen-profile heads/slopes; no retired nearest-anchor factor or three-span correction",
        "outputs": {
            "link_context": str(link_path),
            "node_context": str(node_path),
            "domain_head_loss_audit": str(domain_path),
            "link_ownership_audit": str(ownership_path),
        },
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CandidateContextOutputs(
        output_dir=output,
        link_context_path=link_path,
        node_context_path=node_path,
        domain_audit_path=domain_path,
        ownership_audit_path=ownership_path,
        manifest_path=manifest_path,
        summary=summary,
    )


__all__ = [
    "CANDIDATE_CONTEXT_SCHEMA_VERSION",
    "DOMAIN_OBJECTIVE_NAME",
    "DOMAIN_MINIMUM_REFERENCE_SLOPE_FRACTION",
    "CandidateContextOutputs",
    "generate_reference_profile_candidate_context",
    "solve_boundary_conditioned_domain",
]

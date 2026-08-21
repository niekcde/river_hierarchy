"""Contracts for topology-defined local collapse patches.

This module intentionally contains no raster processing or RivGraph execution.
It defines the boundary between the future local-patch planner, local network
generator, and conservative parent-child merger.

Outer collars and splice anchors define the safe raster domain. The anchored
scientific merge instead preserves every interface where an unselected parent
link meets the selected link set. A closed unit normally has one upstream and
one downstream interface; an open compound member may have several.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import pandas as pd

from network_variants.equivalent_transition_domain import (
    select_smallest_closed_parent_domain,
)


LOCAL_PATCH_CONTRACT_VERSION = "topology-local-patch-v3"

DomainType = Literal[
    "closed_unit",
    "containing_compound",
    "selected_compound",
    "selected_group",
]
SelectionType = Literal["unit", "compound_bubble", "group"]
CollarRole = Literal["upstream", "downstream"]


def _unique_int_tuple(values: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must contain unique IDs.")
    return resolved


@dataclass(frozen=True, slots=True)
class LocalPatchCollar:
    """One external base link used to establish a local-patch interface.

    ``domain_boundary_node_id`` is the endpoint adjacent to the selected core.
    ``outer_splice_anchor_node_id`` is the protected node away from the core.
    The base link is expected to match, but it is explicitly allowed to change
    if strict parent-child matching demonstrates that the local topology differs.
    """

    role: CollarRole
    base_link_id: int
    base_upstream_node_id: int
    base_downstream_node_id: int
    domain_boundary_node_id: int
    outer_splice_anchor_node_id: int
    unit_membership_count: int = 0
    match_expectation: str = "prefer_strict_unchanged_1to1"
    change_policy: str = "allow_retain_replace_split_or_merge"

    def __post_init__(self) -> None:
        if self.role not in {"upstream", "downstream"}:
            raise ValueError("Collar role must be 'upstream' or 'downstream'.")
        if int(self.base_upstream_node_id) == int(self.base_downstream_node_id):
            raise ValueError("A collar link must connect two distinct nodes.")
        endpoints = {int(self.base_upstream_node_id), int(self.base_downstream_node_id)}
        if {
            int(self.domain_boundary_node_id),
            int(self.outer_splice_anchor_node_id),
        } != endpoints:
            raise ValueError(
                "Collar boundary and outer-anchor nodes must be the two endpoints "
                "of the base collar link."
            )
        if self.role == "upstream" and (
            int(self.outer_splice_anchor_node_id) != int(self.base_upstream_node_id)
            or int(self.domain_boundary_node_id) != int(self.base_downstream_node_id)
        ):
            raise ValueError(
                "An upstream collar must point from the outer splice anchor toward "
                "the selected domain."
            )
        if self.role == "downstream" and (
            int(self.domain_boundary_node_id) != int(self.base_upstream_node_id)
            or int(self.outer_splice_anchor_node_id) != int(self.base_downstream_node_id)
        ):
            raise ValueError(
                "A downstream collar must point from the selected domain toward "
                "the outer splice anchor."
            )
        if int(self.unit_membership_count) < 0:
            raise ValueError("Collar unit-membership counts must be non-negative.")
        if self.change_policy != "allow_retain_replace_split_or_merge":
            raise ValueError(
                "Collars are mutable matching context; their change policy may not "
                "require an unchanged match."
            )


@dataclass(frozen=True, slots=True)
class LocalPatchRasterPolicy:
    """Safety contract for the future cropped-raster builder."""

    minimum_halo_m: float = 500.0
    halo_expansion_step_m: float = 250.0
    maximum_halo_m: float = 5000.0
    main_bank_selection: str = "two_largest_global_land_components"
    cut_geometry: str = "nearest_main_banks_via_reference_centerline_context"
    minimum_collar_context_m: float = 1000.0
    bank_overlap_pixels: float = 4.0
    cut_barrier_half_width_pixels: float = 1.5
    main_water_selection: str = "components_seeded_from_core_after_bank_cuts"
    require_complete_cuts_inside_crop: bool = True
    expected_unauthorized_boundary_sections: int = 0
    unsafe_plan_action: str = "fall_back_to_full_raster"

    def __post_init__(self) -> None:
        if float(self.minimum_halo_m) < 0:
            raise ValueError("minimum_halo_m must be non-negative.")
        if float(self.halo_expansion_step_m) <= 0:
            raise ValueError("halo_expansion_step_m must be positive.")
        if float(self.maximum_halo_m) < float(self.minimum_halo_m):
            raise ValueError("maximum_halo_m must be at least minimum_halo_m.")
        if self.main_bank_selection != "two_largest_global_land_components":
            raise ValueError("Unsupported local-patch main-bank selection rule.")
        if self.cut_geometry != "nearest_main_banks_via_reference_centerline_context":
            raise ValueError("Unsupported local-patch cut geometry.")
        if float(self.minimum_collar_context_m) <= 0:
            raise ValueError("minimum_collar_context_m must be positive.")
        if float(self.bank_overlap_pixels) <= 0:
            raise ValueError("bank_overlap_pixels must be positive.")
        if float(self.cut_barrier_half_width_pixels) <= 0:
            raise ValueError("cut_barrier_half_width_pixels must be positive.")
        if int(self.expected_unauthorized_boundary_sections) != 0:
            raise ValueError("A safe cut crop may have no unprescribed edge contacts.")


@dataclass(frozen=True, slots=True)
class LocalPatchMergePolicy:
    """Conservative merge contract: the base wins only after a strict match."""

    outside_splice_domain_action: str = "retain_base"
    exact_match_action: str = "retain_base"
    unmatched_local_action: str = "insert_local"
    conflicting_base_action: str = "remove_or_replace_base"
    collar_match_required: bool = False
    collar_change_policy: str = "allow_retain_replace_split_or_merge"
    protected_interface: str = "anchored_inner_boundary_nodes"
    strict_match_dimensions: tuple[str, ...] = (
        "geometry",
        "directed_endpoints",
        "topology",
    )

    def __post_init__(self) -> None:
        if self.collar_match_required:
            raise ValueError(
                "A collar match may be preferred but cannot be required: collar "
                "topology is allowed to change near the patch edge."
            )
        if self.collar_change_policy != "allow_retain_replace_split_or_merge":
            raise ValueError("The merge policy must allow changed collar links.")
        required = {"geometry", "directed_endpoints", "topology"}
        if not required.issubset(self.strict_match_dimensions):
            raise ValueError(
                "Strict unchanged-link matching must include geometry, directed "
                "endpoints, and topology."
            )


@dataclass(frozen=True, slots=True)
class LocalPatchContract:
    """Planner output consumed by future crop, RivGraph, and merge stages."""

    example_id: str
    selection_type: SelectionType
    selected_unit_ids: tuple[int, ...]
    domain_type: DomainType
    core_link_ids: tuple[int, ...]
    upstream_collar: LocalPatchCollar
    downstream_collar: LocalPatchCollar
    selected_link_ids: tuple[int, ...] = ()
    anchored_source_node_id: int | None = None
    anchored_sink_node_id: int | None = None
    anchored_upstream_boundary_node_ids: tuple[int, ...] = ()
    anchored_downstream_boundary_node_ids: tuple[int, ...] = ()
    upstream_connector_link_ids: tuple[int, ...] = ()
    downstream_connector_link_ids: tuple[int, ...] = ()
    selected_compound_bubble_id: int | None = None
    computational_context_unit_id: int | None = None
    raster_policy: LocalPatchRasterPolicy = field(default_factory=LocalPatchRasterPolicy)
    merge_policy: LocalPatchMergePolicy = field(default_factory=LocalPatchMergePolicy)
    contract_version: str = LOCAL_PATCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not str(self.example_id).strip():
            raise ValueError("example_id must not be empty.")
        if self.selection_type not in {"unit", "compound_bubble", "group"}:
            raise ValueError(f"Unsupported selection_type: {self.selection_type!r}.")
        if self.domain_type not in {
            "closed_unit",
            "containing_compound",
            "selected_compound",
            "selected_group",
        }:
            raise ValueError(f"Unsupported domain_type: {self.domain_type!r}.")
        selected_unit_ids = _unique_int_tuple(
            self.selected_unit_ids, field_name="selected_unit_ids"
        )
        core_link_ids = _unique_int_tuple(self.core_link_ids, field_name="core_link_ids")
        selected_link_ids = (
            _unique_int_tuple(self.selected_link_ids, field_name="selected_link_ids")
            if self.selected_link_ids
            else core_link_ids
        )
        object.__setattr__(self, "selected_unit_ids", selected_unit_ids)
        object.__setattr__(self, "core_link_ids", core_link_ids)
        object.__setattr__(self, "selected_link_ids", selected_link_ids)
        if not set(selected_link_ids).issubset(core_link_ids):
            raise ValueError("selected_link_ids must be a subset of core_link_ids.")
        anchored_source = (
            int(self.anchored_source_node_id)
            if self.anchored_source_node_id is not None
            else int(self.upstream_collar.domain_boundary_node_id)
        )
        anchored_sink = (
            int(self.anchored_sink_node_id)
            if self.anchored_sink_node_id is not None
            else int(self.downstream_collar.domain_boundary_node_id)
        )
        if anchored_source == anchored_sink:
            raise ValueError("Anchored source and sink nodes must be distinct.")
        object.__setattr__(self, "anchored_source_node_id", anchored_source)
        object.__setattr__(self, "anchored_sink_node_id", anchored_sink)
        upstream_boundaries = (
            _unique_int_tuple(
                self.anchored_upstream_boundary_node_ids,
                field_name="anchored_upstream_boundary_node_ids",
            )
            if self.anchored_upstream_boundary_node_ids
            else (anchored_source,)
        )
        downstream_boundaries = (
            _unique_int_tuple(
                self.anchored_downstream_boundary_node_ids,
                field_name="anchored_downstream_boundary_node_ids",
            )
            if self.anchored_downstream_boundary_node_ids
            else (anchored_sink,)
        )
        object.__setattr__(self, "anchored_upstream_boundary_node_ids", upstream_boundaries)
        object.__setattr__(self, "anchored_downstream_boundary_node_ids", downstream_boundaries)
        upstream_connectors = tuple(int(value) for value in self.upstream_connector_link_ids)
        downstream_connectors = tuple(int(value) for value in self.downstream_connector_link_ids)
        if len(set(upstream_connectors)) != len(upstream_connectors):
            raise ValueError("upstream_connector_link_ids must contain unique IDs.")
        if len(set(downstream_connectors)) != len(downstream_connectors):
            raise ValueError("downstream_connector_link_ids must contain unique IDs.")
        object.__setattr__(self, "upstream_connector_link_ids", upstream_connectors)
        object.__setattr__(self, "downstream_connector_link_ids", downstream_connectors)

        if self.upstream_collar.role != "upstream":
            raise ValueError("upstream_collar must have the upstream role.")
        if self.downstream_collar.role != "downstream":
            raise ValueError("downstream_collar must have the downstream role.")
        collar_ids = {
            int(self.upstream_collar.base_link_id),
            int(self.downstream_collar.base_link_id),
        }
        if len(collar_ids) != 2:
            raise ValueError("Upstream and downstream collars must be distinct links.")
        if collar_ids.intersection(core_link_ids):
            raise ValueError("External collar links must not be core-domain links.")
        connector_ids = set(upstream_connectors).union(downstream_connectors)
        if connector_ids.intersection(core_link_ids):
            raise ValueError("Connector links must be outside the selected core domain.")
        if connector_ids.intersection(collar_ids):
            raise ValueError("Connector and collar links must be distinct.")
        if len(connector_ids) != len(upstream_connectors) + len(downstream_connectors):
            raise ValueError("Upstream and downstream connector links must be disjoint.")
        if self.source_splice_anchor_node_id == self.sink_splice_anchor_node_id:
            raise ValueError("Source and sink splice anchors must be distinct nodes.")
        if self.domain_type in {"containing_compound", "selected_compound"}:
            if (
                self.selected_compound_bubble_id is None
                and self.computational_context_unit_id is None
            ):
                raise ValueError(
                    "Compound local-patch domains require a compound bubble or "
                    "an explicit computational context unit."
                )
        elif self.selected_compound_bubble_id is not None:
            raise ValueError(
                "selected_compound_bubble_id is only valid for a compound domain."
            )
        if self.contract_version != LOCAL_PATCH_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported local-patch contract version: {self.contract_version!r}."
            )

    @property
    def source_splice_anchor_node_id(self) -> int:
        return int(self.upstream_collar.outer_splice_anchor_node_id)

    @property
    def sink_splice_anchor_node_id(self) -> int:
        return int(self.downstream_collar.outer_splice_anchor_node_id)

    @property
    def collar_link_ids(self) -> tuple[int, int]:
        return (
            int(self.upstream_collar.base_link_id),
            int(self.downstream_collar.base_link_id),
        )

    @property
    def local_context_link_ids(self) -> tuple[int, ...]:
        """All links whose geometry must be represented in the local raster."""
        return tuple(
            dict.fromkeys(
                (
                    *self.core_link_ids,
                    *self.upstream_connector_link_ids,
                    *self.downstream_connector_link_ids,
                    *self.collar_link_ids,
                )
            )
        )

    @property
    def anchored_replacement_link_ids(self) -> tuple[int, ...]:
        """Parent links scientifically selected for anchored replacement."""
        return self.selected_link_ids

    @property
    def computational_domain_link_ids(self) -> tuple[int, ...]:
        """Complete hydraulically closed link set regenerated by RivGraph."""
        return self.core_link_ids

    def to_manifest(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation of the contract."""
        payload = asdict(self)
        payload["source_splice_anchor_node_id"] = self.source_splice_anchor_node_id
        payload["sink_splice_anchor_node_id"] = self.sink_splice_anchor_node_id
        payload["collar_link_ids"] = list(self.collar_link_ids)
        payload["local_context_link_ids"] = list(self.local_context_link_ids)
        payload["computational_domain_link_ids"] = list(
            self.computational_domain_link_ids
        )
        payload["anchored_replacement_link_ids"] = list(self.anchored_replacement_link_ids)
        return payload


@dataclass(frozen=True, slots=True)
class LocalPatchPlanningResult:
    """Safe planner output or an auditable request for full-raster fallback."""

    status: Literal["safe", "fallback_full_raster"]
    contract: LocalPatchContract | None
    fallback_code: str = ""
    fallback_reason: str = ""
    audit: Mapping[str, Any] = field(default_factory=dict)
    planner_version: str = "topology-local-patch-planner-v2"

    def __post_init__(self) -> None:
        if self.status not in {"safe", "fallback_full_raster"}:
            raise ValueError(f"Unsupported planning status: {self.status!r}.")
        if self.status == "safe" and self.contract is None:
            raise ValueError("A safe planning result requires a local-patch contract.")
        if self.status == "fallback_full_raster" and self.contract is not None:
            raise ValueError("A fallback planning result cannot contain a safe contract.")
        if self.status == "fallback_full_raster" and not self.fallback_code:
            raise ValueError("A fallback planning result requires a fallback code.")

    @property
    def is_safe(self) -> bool:
        return self.status == "safe"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "planner_version": self.planner_version,
            "status": self.status,
            "fallback_code": self.fallback_code,
            "fallback_reason": self.fallback_reason,
            "audit": dict(self.audit),
            "contract": self.contract.to_manifest() if self.contract is not None else None,
        }


class _CollarPlanningError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _required_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _membership_counts(unit_link_membership: pd.DataFrame) -> dict[int, int]:
    _required_columns(
        unit_link_membership,
        {"unit_id", "id_link"},
        name="unit_link_membership",
    )
    membership = unit_link_membership[["unit_id", "id_link"]].copy()
    membership["unit_id"] = pd.to_numeric(membership["unit_id"], errors="coerce")
    membership["id_link"] = pd.to_numeric(membership["id_link"], errors="coerce")
    membership = membership.dropna().astype({"unit_id": int, "id_link": int})
    return (
        membership.drop_duplicates(["unit_id", "id_link"])
        .groupby("id_link")["unit_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def _immediate_external_collar(
    *,
    full_links: pd.DataFrame,
    selected_link_ids: set[int],
    boundary_node: int,
    role: CollarRole,
    membership_count_by_link: Mapping[int, int],
) -> LocalPatchCollar:
    """Return the first directed link outside the closed computational domain.

    The caller supplies the complete closed-domain link set.  The returned
    collar is temporary computational context outside that domain and is
    removed before the selected morphology is merged back into the parent.
    """
    if role == "upstream":
        candidates = full_links.loc[
            pd.to_numeric(full_links["id_ds_node"], errors="coerce").eq(int(boundary_node))
        ].copy()
    else:
        candidates = full_links.loc[
            pd.to_numeric(full_links["id_us_node"], errors="coerce").eq(int(boundary_node))
        ].copy()
    candidate_ids = pd.to_numeric(candidates["id_link"], errors="coerce")
    candidates = candidates.loc[
        candidate_ids.notna()
        & ~candidate_ids.fillna(-1).astype(int).isin(selected_link_ids)
    ].copy()
    if len(candidates) != 1:
        ids = sorted(
            pd.to_numeric(candidates.get("id_link"), errors="coerce")
            .dropna().astype(int).tolist()
        )
        code = f"missing_{role}_collar" if candidates.empty else f"ambiguous_{role}_interface"
        raise _CollarPlanningError(
            code,
            f"Expected one immediate external {role} continuation at node "
            f"{int(boundary_node)}; found links {ids}.",
        )
    row = candidates.iloc[0]
    link_id = int(row["id_link"])
    upstream_node = int(row["id_us_node"])
    downstream_node = int(row["id_ds_node"])
    if role == "upstream":
        domain_node, outer_node = downstream_node, upstream_node
    else:
        domain_node, outer_node = upstream_node, downstream_node
    return LocalPatchCollar(
        role=role,
        base_link_id=link_id,
        base_upstream_node_id=upstream_node,
        base_downstream_node_id=downstream_node,
        domain_boundary_node_id=domain_node,
        outer_splice_anchor_node_id=outer_node,
        unit_membership_count=int(membership_count_by_link.get(link_id, 0)),
    )


def _contract_domain_type(
    *,
    selection_type: SelectionType,
    hydraulic_domain_type: str,
) -> DomainType:
    if selection_type == "compound_bubble":
        return "selected_compound"
    if selection_type == "group":
        return "selected_group"
    if hydraulic_domain_type == "containing_compound_context":
        return "containing_compound"
    return "closed_unit"


def plan_local_patch(
    *,
    example_id: str,
    full_links: pd.DataFrame,
    unit_summary: pd.DataFrame,
    unit_link_membership: pd.DataFrame,
    bubble_edge_resistance: pd.DataFrame,
    selected_unit_ids: Sequence[int],
    selection_type: SelectionType = "unit",
    selected_compound_bubble_id: int | None = None,
    computational_context_unit_id: int | None = None,
    raster_policy: LocalPatchRasterPolicy | None = None,
    merge_policy: LocalPatchMergePolicy | None = None,
    max_collar_search_links: int = 64,
) -> LocalPatchPlanningResult:
    """Plan a topology-defined patch without reading or writing raster data.

    Unsafe or ambiguous interfaces produce an explicit full-raster fallback
    result.  Invalid input schemas still raise because they cannot be repaired
    by changing execution mode.
    """
    _required_columns(
        full_links,
        {"id_link", "id_us_node", "id_ds_node"},
        name="full_links",
    )
    if int(max_collar_search_links) <= 0:
        raise ValueError("max_collar_search_links must be positive.")
    selected_ids = tuple(dict.fromkeys(int(value) for value in selected_unit_ids))
    if not selected_ids:
        raise ValueError("selected_unit_ids must not be empty.")

    try:
        domain_selection_ids = (
            (int(computational_context_unit_id),)
            if computational_context_unit_id is not None
            else selected_ids
        )
        domain = select_smallest_closed_parent_domain(
            full_links=full_links,
            unit_summary=unit_summary,
            unit_link_membership=unit_link_membership,
            bubble_edge_resistance=bubble_edge_resistance,
            selected_unit_ids=domain_selection_ids,
            selected_compound_bubble_id=(
                None if computational_context_unit_id is not None else selected_compound_bubble_id
            ),
        )
    except (ValueError, KeyError) as exc:
        return LocalPatchPlanningResult(
            status="fallback_full_raster",
            contract=None,
            fallback_code="domain_selection_failed",
            fallback_reason=str(exc),
            audit={
                "selected_unit_ids": list(selected_ids),
                "computational_context_unit_id": computational_context_unit_id,
            },
        )

    selected_membership = unit_link_membership.loc[
        pd.to_numeric(unit_link_membership["unit_id"], errors="coerce").isin(selected_ids)
    ]
    selected_link_ids = tuple(
        pd.to_numeric(selected_membership["id_link"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    if not selected_link_ids:
        raise ValueError(f"No link membership was found for selected units {selected_ids}.")
    # RivGraph must see a complete hydraulically closed computational domain.
    # The final anchored merge still replaces only ``selected_link_ids``.
    core_link_ids = tuple(
        pd.to_numeric(domain.links["id_link"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    if not core_link_ids:
        raise ValueError("The selected hydraulic domain contains no links.")
    selected_links = full_links.loc[
        pd.to_numeric(full_links["id_link"], errors="coerce").isin(selected_link_ids)
    ]
    selected_graph = nx.MultiDiGraph()
    for row in selected_links.itertuples(index=False):
        selected_graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link))
    anchored_sources = sorted(
        int(node_id) for node_id in selected_graph
        if selected_graph.in_degree(node_id) == 0 and selected_graph.out_degree(node_id) > 0
    )
    anchored_sinks = sorted(
        int(node_id) for node_id in selected_graph
        if selected_graph.out_degree(node_id) == 0 and selected_graph.in_degree(node_id) > 0
    )
    if len(anchored_sources) != 1 or len(anchored_sinks) != 1:
        raise ValueError(
            "Selected links do not define one anchored source and sink: "
            f"sources={anchored_sources}, sinks={anchored_sinks}."
        )
    selected_set = set(selected_link_ids)
    selected_nodes = set(selected_graph.nodes)
    upstream_interface_nodes: set[int] = set()
    downstream_interface_nodes: set[int] = set()
    for row in full_links.itertuples(index=False):
        if int(row.id_link) in selected_set:
            continue
        upstream, downstream = int(row.id_us_node), int(row.id_ds_node)
        if downstream in selected_nodes and selected_graph.out_degree(downstream) > 0:
            upstream_interface_nodes.add(downstream)
        if upstream in selected_nodes and selected_graph.in_degree(upstream) > 0:
            downstream_interface_nodes.add(upstream)
    if not upstream_interface_nodes:
        upstream_interface_nodes.add(anchored_sources[0])
    if not downstream_interface_nodes:
        downstream_interface_nodes.add(anchored_sinks[0])
    membership_counts = _membership_counts(unit_link_membership)
    try:
        upstream_collar = _immediate_external_collar(
            full_links=full_links,
            selected_link_ids=set(core_link_ids),
            boundary_node=int(domain.source_node),
            role="upstream",
            membership_count_by_link=membership_counts,
        )
        downstream_collar = _immediate_external_collar(
            full_links=full_links,
            selected_link_ids=set(core_link_ids),
            boundary_node=int(domain.sink_node),
            role="downstream",
            membership_count_by_link=membership_counts,
        )
        upstream_connectors: tuple[int, ...] = ()
        downstream_connectors: tuple[int, ...] = ()
    except _CollarPlanningError as exc:
        return LocalPatchPlanningResult(
            status="fallback_full_raster",
            contract=None,
            fallback_code=exc.code,
            fallback_reason=str(exc),
            audit={
                "selected_unit_ids": list(selected_ids),
                "hydraulic_domain_type": domain.domain_type,
                "domain_source_node_id": int(domain.source_node),
                "domain_sink_node_id": int(domain.sink_node),
                "core_link_count": len(core_link_ids),
            },
        )

    domain_type = _contract_domain_type(
        selection_type=selection_type,
        hydraulic_domain_type=domain.domain_type,
    )
    if computational_context_unit_id is not None:
        domain_type = "containing_compound"
    resolved_compound_id = (
        int(domain.compound_bubble_id)
        if domain_type in {"containing_compound", "selected_compound"}
        and domain.compound_bubble_id is not None
        else None
    )
    try:
        contract = LocalPatchContract(
            example_id=str(example_id),
            selection_type=selection_type,
            selected_unit_ids=selected_ids,
            domain_type=domain_type,
            core_link_ids=core_link_ids,
            upstream_collar=upstream_collar,
            downstream_collar=downstream_collar,
            selected_link_ids=selected_link_ids,
            anchored_source_node_id=anchored_sources[0],
            anchored_sink_node_id=anchored_sinks[0],
            anchored_upstream_boundary_node_ids=tuple(sorted(upstream_interface_nodes)),
            anchored_downstream_boundary_node_ids=tuple(sorted(downstream_interface_nodes)),
            upstream_connector_link_ids=upstream_connectors,
            downstream_connector_link_ids=downstream_connectors,
            selected_compound_bubble_id=resolved_compound_id,
            computational_context_unit_id=computational_context_unit_id,
            raster_policy=raster_policy or LocalPatchRasterPolicy(),
            merge_policy=merge_policy or LocalPatchMergePolicy(),
        )
    except ValueError as exc:
        return LocalPatchPlanningResult(
            status="fallback_full_raster",
            contract=None,
            fallback_code="contract_validation_failed",
            fallback_reason=str(exc),
            audit={
                "selected_unit_ids": list(selected_ids),
                "hydraulic_domain_type": domain.domain_type,
                "core_link_count": len(core_link_ids),
            },
        )

    return LocalPatchPlanningResult(
        status="safe",
        contract=contract,
        audit={
            "selected_unit_ids": list(selected_ids),
            "computational_context_unit_id": computational_context_unit_id,
            "selected_unit_is_closed": bool(domain.selected_unit_is_closed),
            "selected_unit_violating_link_ids": list(
                domain.selected_unit_violating_link_ids
            ),
            "hydraulic_domain_type": domain.domain_type,
            "domain_source_node_id": int(domain.source_node),
            "domain_sink_node_id": int(domain.sink_node),
            "computational_domain_link_count": len(core_link_ids),
            "selected_replacement_link_count": len(selected_link_ids),
            "core_link_count": len(core_link_ids),
            "upstream_connector_link_count": len(upstream_connectors),
            "downstream_connector_link_count": len(downstream_connectors),
        },
    )


def write_local_patch_contract(
    contract: LocalPatchContract,
    output_path: str | Path,
) -> Path:
    """Write a planner contract without creating any raster or graph outputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(contract.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_local_patch_planning_result(
    result: LocalPatchPlanningResult,
    output_path: str | Path,
) -> Path:
    """Write either a safe contract or a full-raster fallback audit."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "LOCAL_PATCH_CONTRACT_VERSION",
    "LocalPatchCollar",
    "LocalPatchContract",
    "LocalPatchMergePolicy",
    "LocalPatchPlanningResult",
    "LocalPatchRasterPolicy",
    "plan_local_patch",
    "write_local_patch_contract",
    "write_local_patch_planning_result",
]

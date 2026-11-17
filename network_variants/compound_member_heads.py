"""Selected-member adapter for hydraulic regeneration using canonical node IDs.

This is the opt-in replacement for the containing-compound slope solve. It never
falls back to that larger domain and never modifies geometry or hydraulic fields
other than heads and slopes. Reviewed repairs supply explicit lineage contracts.
"""
import numpy as np
from .boundary_contract_repair import derive_contract
from .local_head_reconstruction import LocalHeadContract, reconstruct_local_heads


def member_contract(
    child,
    reference,
    selection,
    *,
    exterior_paths=None,
    allow_boundary_flexibility=False,
):
    """Use the existing selected region; verify its boundary and exterior lineage.

    Single-edge exterior paths are supported. Ambiguous, missing or subdivided
    paths require explicit review, rather than silent selection by proximity.
    """
    selected = set(map(int, selection["selected_parent_links"]))
    if not selection.get("valid") or not selection.get("target_ok"):
        raise ValueError("Saved selected-member contract is invalid")
    canonical = reference[["id_link", "u", "v"]].rename(
        columns={"u": "id_us_node", "v": "id_ds_node"}
    )
    boundary = set(derive_contract(canonical, selected)["boundary_nodes"])
    if boundary != set(selection["boundary_nodes"]):
        raise ValueError("Saved boundaries disagree with selected-member incidence")
    outside = reference[~reference.id_link.isin(selected)]
    matches = {}
    if exterior_paths is not None:
        if len(exterior_paths) != len(outside) or {
            int(p["parent_link"]) for p in exterior_paths
        } != set(outside.id_link):
            raise ValueError(
                "Saved exterior path coverage disagrees with selected member"
            )
        for p in exterior_paths:
            chain = p["child_links"]
            if not p.get("path_unique", True) or len(chain) != 1:
                raise ValueError(f"Exterior path needs review: {p['parent_link']}")
            if chain[0] in matches:
                raise ValueError("Exterior child reused")
            matches[int(chain[0])] = int(p["parent_link"])
    else:
        groups = {}
        for r in child.itertuples():
            groups.setdefault((r.u, r.v), []).append(r)
        for p in outside.itertuples():
            hits = [
                r
                for r in groups.get((p.u, p.v), [])
                if np.isclose(r.link_length_m, p.link_length_m, rtol=1e-8, atol=1e-5)
            ]
            if len(hits) > 1:
                # Stable IDs may resolve otherwise identical parallel reaches.
                hits = [r for r in hits if r.id_link == p.id_link]
            if len(hits) != 1 or hits[0].id_link in matches:
                raise ValueError(f"Exterior correspondence needs review: {p.id_link}")
            matches[int(hits[0].id_link)] = int(p.id_link)
    return LocalHeadContract(
        frozenset(selected),
        frozenset(boundary),
        matches,
        flexible_nodes=frozenset(boundary)
        if allow_boundary_flexibility
        else frozenset(),
    )


def reconstruct_compound_member_heads(
    child,
    reference,
    reference_floors,
    selection,
    *,
    exterior_paths=None,
    reviewed_contract=None,
    allow_boundary_flexibility=False,
):
    """Regenerate selected-member slopes without changing containing-compound heads.

    Set allow_boundary_flexibility explicitly to apply the agreed minimal-relative-
    drop policy to necessary boundary conflicts. A reviewed contract supplies the
    exact approved region, reversed connectors and permissible releases instead.
    """
    contract = reviewed_contract or member_contract(
        child,
        reference,
        selection,
        exterior_paths=exterior_paths,
        allow_boundary_flexibility=allow_boundary_flexibility,
    )
    result, audit = reconstruct_local_heads(
        child, reference, reference_floors, contract
    )
    return result, audit, contract

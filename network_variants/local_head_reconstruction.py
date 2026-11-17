"""Opt-in selected-scope head reconstruction; no geometry or routing mutations.

Inputs use canonical reference node IDs. Exterior lineage must be one-to-one;
subdivided paths and unresolved matches are rejected. Flexibility is explicit,
never inferred as permission to widen the selected region.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .boundary_contract_repair import derive_contract
from ._local_head_feasibility import head_series, path_constraints
from ._local_head_qp import fit_internal, fit_primary


@dataclass(frozen=True)
class LocalHeadContract:
    selected_parent_links: frozenset[int]
    boundary_nodes: frozenset[int]
    exterior_parent_by_child: Mapping[int, int]
    reversed_exterior_children: frozenset[int] = frozenset()
    flexible_nodes: frozenset[int] = frozenset()
    # Absolute lower bounds saved with the original reference, never recomputed.
    reference_floor_fraction: float = 0.01


def reconstruct_local_heads(links, reference, reference_floors, contract):
    """Return copied links and audit, or raise on an unresolved contract/conflict.

    First preserve reference exterior heads. If necessary and authorised, minimise
    squared relative exterior drop changes, with minimum head displacement only
    on the primary optimal face. Then fit equal-weight relative replacement slopes.
    Existing exterior floor residuals are grandfathered explicitly in the audit.
    No widths, reference flows, depths or routing coefficients are recalculated.
    """
    f = links.copy().reset_index(drop=True)
    if f.empty or f.id_link.duplicated().any() or reference.id_link.duplicated().any():
        raise ValueError("Nonempty unique link IDs required")
    if not np.isfinite(f.link_length_m).all() or (f.link_length_m <= 0).any():
        raise ValueError("Positive finite lengths required")
    scope = set(contract.selected_parent_links)
    parent = reference.set_index("id_link")
    canonical = reference[["id_link", "u", "v"]].rename(
        columns={"u": "id_us_node", "v": "id_ds_node"}
    )
    derived = derive_contract(canonical, scope)["boundary_nodes"]
    if set(derived) != set(contract.boundary_nodes):
        raise ValueError(
            "Declared boundaries disagree with selected reference incidence"
        )
    outside = dict(contract.exterior_parent_by_child)
    if set(outside.values()) != set(parent.index) - scope or len(
        set(outside.values())
    ) != len(outside):
        raise ValueError(
            "Exterior correspondence must cover every exterior parent exactly once"
        )
    if not set(outside) <= set(f.id_link):
        raise ValueError("Exterior child missing")
    reversed_ids = set(contract.reversed_exterior_children)
    if not reversed_ids <= set(outside):
        raise ValueError("Reversal lacks exterior lineage")
    fi = f.set_index("id_link")
    for child, original in outside.items():
        cr, pr = fi.loc[child], parent.loc[original]
        expected = (pr.v, pr.u) if child in reversed_ids else (pr.u, pr.v)
        if (cr.u, cr.v) != expected:
            raise ValueError(f"Unapproved endpoint/direction change on child {child}")
    allowed = set(contract.boundary_nodes)
    for child in reversed_ids:
        allowed.update([int(fi.loc[child].u), int(fi.loc[child].v)])
    if not set(contract.flexible_nodes) <= allowed:
        raise ValueError(
            "Flexibility extends beyond boundaries/approved reversal endpoints"
        )
    f["parent_link"] = f.id_link.map(outside)
    f["role"] = f.id_link.map(
        lambda c: "reversed_exterior"
        if c in reversed_ids
        else "exterior"
        if c in outside
        else "selected_replacement"
    )
    patch = f[f.role.eq("selected_replacement")]
    if patch.empty:
        raise ValueError("No replacement reaches")
    floor_values = reference_floors.loc[sorted(scope)].dropna().unique()
    if (
        len(floor_values) != 1
        or not np.isfinite(floor_values[0])
        or floor_values[0] <= 0
    ):
        raise ValueError("Selected region requires one positive frozen reference floor")
    fraction = contract.reference_floor_fraction
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid original reference floor fraction")
    floor = float(floor_values[0])
    target = floor / fraction
    refheads = head_series(reference)
    if not np.isfinite(refheads).all():
        raise ValueError("Nonfinite reference heads")
    ext = reference[~reference.id_link.isin(scope)]
    fixed = set(ext.u) | set(ext.v) | set(derived)
    if not fixed <= (set(f.u) | set(f.v)):
        raise ValueError("Reference exterior/boundary node missing")
    floors = []
    inherited = []
    for row in f.itertuples():
        if row.id_link not in outside:
            floors.append(floor)
            continue
        pid = outside[row.id_link]
        saved = reference_floors.loc[pid]
        effective = min(saved, parent.loc[pid, "slope"]) if np.isfinite(saved) else 0.0
        if np.isfinite(saved) and effective < saved:
            inherited.append(
                dict(
                    parent_link=int(pid),
                    saved_floor=float(saved),
                    effective_floor=float(effective),
                )
            )
        floors.append(effective)
    drops = np.maximum(np.asarray(floors) * f.link_length_m.to_numpy(), 1e-10)
    paths = path_constraints(f, refheads.loc[sorted(fixed)], drops)
    bad = [p for p in paths if p["available_drop_m"] < p["path_weight"] - 1e-9]
    conflict_nodes = {v for p in bad for v in (p["upstream"], p["downstream"])}
    candidates = sorted(conflict_nodes & set(contract.flexible_nodes))
    blocked_paths = [
        p for p in bad if not {p["upstream"], p["downstream"]} & set(candidates)
    ]
    if blocked_paths:
        raise ValueError(
            f"Fixed-head infeasibility requires explicit review; no permitted endpoint on paths={blocked_paths}"
        )
    newfixed = refheads.loc[sorted(fixed)].to_dict()
    primary = dict(candidates=candidates, strict_fixed_feasible=not bad, conflicts=bad)
    if candidates:
        ix = {v: i for i, v in enumerate(candidates)}
        A = []
        b = []
        for p in paths:
            row = np.zeros(len(candidates))
            for v, sign in ((p["upstream"], 1), (p["downstream"], -1)):
                if v in ix:
                    row[ix[v]] += sign
            rhs = p["path_weight"] - p["available_drop_m"]
            if np.any(row):
                A.append(row)
                b.append(rhs)
            elif rhs > 1e-8:
                raise ValueError("Unrelated exterior floor conflict")
        affected = f[
            f.role.eq("exterior") & (f.u.isin(candidates) | f.v.isin(candidates))
        ]
        C = np.zeros((len(affected), len(candidates)))
        D = []
        for j, row in enumerate(affected.itertuples()):
            for v, sign in ((row.u, 1), (row.v, -1)):
                if v in ix:
                    C[j, ix[v]] += sign
            pr = parent.loc[outside[row.id_link]]
            D.append(pr.head_us - pr.head_ds)
        D = np.asarray(D)
        if np.any(D <= 0):
            raise ValueError("Nonpositive exterior reference drop")
        scale = max(max(b), 0.001)
        delta, certificate = fit_primary(C, 1 / D, np.asarray(A), np.asarray(b) / scale)
        delta *= scale
        for v, d in zip(candidates, delta):
            newfixed[v] += float(d)
        primary.update(
            head_changes={int(v): float(d) for v, d in zip(candidates, delta)},
            certificate=certificate,
            objective=float(np.sum((C @ delta / D) ** 2)),
        )
    patchfixed = {v: newfixed[v] for v in (set(patch.u) | set(patch.v)) & set(newfixed)}
    heads, secondary = fit_internal(
        patch,
        patchfixed,
        target,
        np.maximum(patch.link_length_m.to_numpy() * floor, 1e-10),
    )
    heads.update(newfixed)
    f["head_us"] = f.u.map(heads)
    f["head_ds"] = f.v.map(heads)
    f["slope"] = (f.head_us - f.head_ds) / f.link_length_m
    f["floor"] = floors
    deficit = drops - (f.head_us - f.head_ds).to_numpy()
    if not np.isfinite(f.slope).all() or (f.slope <= 0).any() or max(deficit) > 1e-8:
        raise ValueError("Final positive slope/floor certificate failed")
    return f, dict(
        policy="selected_scope_relative_exterior_drop_v1",
        primary=primary,
        secondary=secondary,
        frozen_floor=floor,
        reference_slope=target,
        inherited_floor_residuals=inherited,
        max_floor_deficit_m=max(0.0, float(max(deficit))),
    )

"""Stage and independently verify slopes for the complete compound-member inventory.

No routing, flow allocation, depth recalculation, RF fitting or production writes.
python -m network_variants.run_compound_member_heads --output NEW_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import pyogrio
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .reviewed_geometry import ReviewedGeometry
from .compound_member_heads import reconstruct_compound_member_heads
from .local_head_reconstruction import LocalHeadContract
from ._local_head_feasibility import head_series

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT
    / "RAPID/outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet"
)
ACCEPTED = (
    ROOT / "RAPID/outputs/accepted_boundary_completion_20261028/candidate_index.json"
)
BASE = ROOT / "RAPID/assets/bounded_robust_slopes_20261002"


def encode(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(type(x).__name__)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=encode))


def lp_feasible(edges, fixed_heads, drops):
    """Independent full incidence-matrix LP, not the solver's path algorithm."""
    free = sorted((set(edges.u) | set(edges.v)) - set(fixed_heads))
    ix = {v: i for i, v in enumerate(free)}
    A = lil_matrix((len(edges), len(free)))
    b = -np.asarray(drops).copy()
    for j, r in enumerate(edges.itertuples()):
        for v, sign in ((r.u, 1), (r.v, -1)):
            if v in ix:
                A[j, ix[v]] = -sign
            else:
                b[j] += sign * fixed_heads[v]
    if not free:
        return bool(np.min(b) >= -1e-9)
    # Fixed-fixed rows are checked in metres before the LP, avoiding its tolerance.
    active = np.asarray(A.getnnz(axis=1) > 0)
    if np.any(b[~active] < -1e-9):
        return False
    res = linprog(
        np.zeros(len(free)),
        A_ub=A.tocsr()[active],
        b_ub=b[active],
        bounds=[(None, None)] * len(free),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9},
    )
    if res.success and np.max(A @ res.x - b) > 1e-8:
        raise ValueError("Independent LP residual failed")
    return bool(res.success)


def verify(result, original, parent, floors, contract, audit):
    """Check physical constraints/locality and measure old versus new exterior slopes."""
    ph = head_series(parent)
    h = head_series(result)
    pi = parent.set_index("id_link")
    before = original.set_index("id_link")
    outside = dict(contract.exterior_parent_by_child)
    scope = set(contract.selected_parent_links)
    ext = parent[~parent.id_link.isin(scope)]
    fixed = set(ext.u) | set(ext.v) | set(contract.boundary_nodes)
    released = set(audit["primary"]["candidates"])
    unchanged = fixed - released
    if not released <= set(contract.flexible_nodes):
        raise ValueError("Unauthorised release")
    if (
        np.max(
            abs(h.loc[sorted(unchanged)] - ph.loc[sorted(unchanged)]).to_numpy(),
            initial=0,
        )
        > 1e-9
    ):
        raise ValueError("Exterior head moved without release")
    np.testing.assert_allclose(
        result.slope,
        (result.head_us - result.head_ds) / result.link_length_m,
        rtol=1e-12,
        atol=1e-14,
    )
    required = []
    comparisons = []
    for r in result.itertuples():
        if r.id_link in outside:
            pid = outside[r.id_link]
            p = pi.loc[pid]
            floor = floors.loc[pid]
            floor = min(floor, p.slope) if np.isfinite(floor) else 0.0
            reversed_edge = r.id_link in contract.reversed_exterior_children
            du = r.head_us - ph.loc[r.u]
            dv = r.head_ds - ph.loc[r.v]
            length_changed = abs(r.link_length_m - p.link_length_m) > 1e-6
            head_changed = max(abs(du), abs(dv)) > 1e-8
            old = (
                float(before.loc[r.id_link, "slope"])
                if r.id_link in before.index
                else None
            )
            comparisons.append(
                dict(
                    parent_link=int(pid),
                    child_link=int(r.id_link),
                    reference_slope=float(p.slope),
                    existing_slope=old,
                    corrected_slope=float(r.slope),
                    relative_change=float(r.slope / p.slope - 1),
                    old_relative_change=float(old / p.slope - 1)
                    if old is not None
                    else None,
                    length_changed=length_changed,
                    head_changed=head_changed,
                    reversed=reversed_edge,
                    head_us_change_m=float(du),
                    head_ds_change_m=float(dv),
                )
            )
            if not reversed_edge and not length_changed and not head_changed:
                if not np.isclose(r.slope, p.slope, rtol=1e-8, atol=1e-13):
                    raise ValueError("Unexplained exterior slope change")
        else:
            floor = float(floors.loc[sorted(scope)].dropna().unique()[0])
        required.append(max(floor * r.link_length_m, 1e-10))
    drops = result.head_us - result.head_ds
    if (
        not np.isfinite(result.slope).all()
        or (result.slope <= 0).any()
        or max(required - drops) > 1e-8
    ):
        raise ValueError("Floor/positivity check failed")
    strict_lp = lp_feasible(result, ph.loc[sorted(fixed)].to_dict(), required)
    if strict_lp != audit["primary"]["strict_fixed_feasible"]:
        raise ValueError("Path feasibility disagrees with independent LP")
    if not lp_feasible(result, h.loc[sorted(fixed)].to_dict(), required):
        raise ValueError("Corrected boundary heads fail independent LP")
    return comparisons, dict(
        strict_lp_feasible=strict_lp,
        final_lp_feasible=True,
        shared_heads=True,
        unreleased_exterior_heads_fixed=True,
        max_floor_deficit_m=max(0.0, float(max(required - drops))),
        changed_boundary_heads=int(
            np.sum(abs(h.loc[sorted(fixed)] - ph.loc[sorted(fixed)]) > 1e-8)
        ),
        max_boundary_head_change_m=float(
            np.max(
                abs(h.loc[sorted(fixed)] - ph.loc[sorted(fixed)]).to_numpy(), initial=0
            )
        ),
    )


def run(output, limit=None, examples=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    rows = []
    all_comp = []
    cache = {}

    def track(p):
        p = Path(p)
        hashes.setdefault(str(p), hashlib.sha256(p.read_bytes()).hexdigest())
        return p

    def js(p):
        return json.loads(track(p).read_text())

    def read(p):
        return pd.read_parquet(track(p))

    def vectors(p, columns):
        return pyogrio.read_dataframe(track(p), columns=columns)

    inv = read(INVENTORY)
    inv = inv[inv.edit_type.eq("Compound member")].copy()
    if len(inv) != 1467 or inv[["example", "state"]].duplicated().any():
        raise ValueError("Compound-member inventory changed or duplicated")
    accepted = {r["case"]: r for r in js(ACCEPTED)}
    release = ReviewedGeometry()
    track(release.manifest)
    # Freeze implementation fingerprints before the run as well as its inputs.
    for p in list(Path(__file__).parent.glob("*local_head*.py")) + [
        Path(__file__),
        Path(__file__).with_name("compound_member_heads.py"),
        Path(__file__).with_name("boundary_contract_repair.py"),
    ]:
        track(p)
    if examples:
        inv = inv[inv.example.astype(str).str.zfill(2).isin(examples)]
    if limit:
        inv = inv.iloc[:limit]
    for path in Path(__file__).parent.glob("*compound_member_heads*.py"):
        track(path)
    code = out / "source_code"
    code.mkdir()
    for path in list(hashes):
        if Path(path).parent == Path(__file__).parent and path.endswith(".py"):
            (code / Path(path).name).write_bytes(Path(path).read_bytes())
    for i, z in enumerate(inv.itertuples(), 1):
        ex = str(z.example).zfill(2)
        state = z.state
        key = f"{ex}/{state}"
        dest = out / ex / state
        dest.mkdir(parents=True)
        row = dict(
            case=key, example=ex, state=state, accepted_candidate=key in accepted
        )
        try:
            if ex not in cache:
                br = release.lookup(ex, "indep", "S000_base")
                parent = read(br["original_frame"])
                old = read(BASE / ex / "floor_0.05/old/links.parquet").set_index(
                    "id_link"
                )
                positions = (
                    read(BASE / ex / "floor_0.05/old/nodes.parquet")
                    .set_index("id_node")
                    .reference_section_dist_out_m
                )
                units = pd.read_csv(track(BASE / ex / "floor_0.05/old/units.csv"))
                membership = pd.read_csv(
                    track(
                        Path(br["links"]).parents[2]
                        / "variant/summary/unit_link_membership.csv"
                    )
                )
                pn = vectors(br["nodes"], ["id_node"]).set_index("id_node")
                rawmap = dict(zip(parent.u, parent.id_us_node)) | dict(
                    zip(parent.v, parent.id_ds_node)
                )
                points = {v: pn.loc[raw, "geometry"] for v, raw in rawmap.items()}
                cache[ex] = parent, old, positions, units, membership, points
            parent, old, positions, units, membership, points = cache[ex]
            record = release.lookup(ex, "indep", state)
            selection = js(record["contract_path"])
            if hashes[str(record["contract_path"])] != record["contract_sha256"]:
                raise ValueError("Published contract checksum mismatch")
            original = read(record["original_frame"])
            unit_ids = set(map(int, z.selected_unit_ids))
            if set(selection["selected_unit_ids"]) != unit_ids or set(
                membership[membership.unit_id.isin(unit_ids)].id_link
            ) != set(selection["selected_parent_links"]):
                raise ValueError("Unit membership differs from saved selection")
            for p, expected in record.get("retained_source_checksums", {}).items():
                track(p)
                if hashes[str(p)] != expected:
                    raise ValueError("Published source checksum mismatch: " + p)
            approved = None
            paths = None
            if key in accepted:
                a = accepted[key]
                folder = Path(a["candidate_folder"])
                gpath = track(folder / "geometry_candidate.parquet")
                spath = track(folder / "slope_candidate.parquet")
                if (
                    hashes[str(gpath)] != a["geometry_sha256"]
                    or hashes[str(spath)] != a["slope_sha256"]
                ):
                    raise ValueError("Accepted candidate fingerprint changed")
                child = read(gpath)
                spec = js(folder / "boundary_contract.json")
                primary = js(folder / "primary_head_fit.json")
                approved = LocalHeadContract(
                    frozenset(spec["approved_scope"]),
                    frozenset(spec["boundary_nodes"]),
                    {
                        int(r.id_link): int(r.parent_link)
                        for r in child.itertuples()
                        if pd.notna(r.parent_link)
                    },
                    frozenset(map(int, spec["accepted_reversed_connectors"])),
                    frozenset(primary["candidates"]),
                )
                selection_scope = set(spec["approved_scope"])
            else:
                if any(k.startswith("reviewed_") for k in selection):
                    raise ValueError(
                        "Manual boundary exception lacks accepted candidate"
                    )
                child = original[["id_link", "u", "v", "link_length_m"]].copy()
                selection_scope = set(selection["selected_parent_links"])
                # Verify that canonical identities really are the published ones.
                pairs = pd.concat(
                    [
                        original[["id_us_node", "u"]].rename(
                            columns={"id_us_node": "raw", "u": "canonical"}
                        ),
                        original[["id_ds_node", "v"]].rename(
                            columns={"id_ds_node": "raw", "v": "canonical"}
                        ),
                    ]
                ).drop_duplicates()
                if pairs.raw.duplicated().any() or pairs.canonical.duplicated().any():
                    raise ValueError("Node correspondence is not injective")
                mapping = dict(zip(pairs.raw, pairs.canonical))
                inverse = {v: k for k, v in mapping.items()}
                explicit = (
                    {
                        int(k): int(v)
                        for k, v in js(record["validated_correspondences_path"]).items()
                    }
                    if record.get("validated_correspondences_path")
                    else {}
                )
                if any(mapping.get(k) != v for k, v in explicit.items()):
                    raise ValueError("Published node correspondence disagrees")
                nodes = vectors(record["nodes"], ["id_node"]).set_index("id_node")
                for v in selection["boundary_nodes"]:
                    raw = inverse.get(v)
                    if raw is None or (
                        nodes.loc[raw, "geometry"].distance(points[v]) > 1e-5
                        and explicit.get(raw) != v
                    ):
                        raise ValueError("Unverified boundary identity: " + str(v))
                geometry = (
                    vectors(record["links"], ["id_link"])
                    .set_index("id_link")
                    .loc[child.id_link]
                )
                lengths = geometry.geometry.length.to_numpy()
                np.testing.assert_allclose(
                    lengths, child.link_length_m, rtol=1e-9, atol=1e-6
                )
                pathfile = (
                    Path(record.get("source_bundle") or "") / "exterior_paths.json"
                )
                if record.get("source_bundle") and pathfile.exists():
                    paths = js(pathfile)
            # Independently establish frozen-floor origin from the containing reference compound.
            selected = old.loc[sorted(selection_scope)]
            domains = selected[
                ["ownership_kind", "ownership_domain_id"]
            ].drop_duplicates()
            if (
                len(domains) != 1
                or selected.lower_bound.isna().any()
                or selected.lower_bound.nunique() != 1
            ):
                raise ValueError(
                    "Selection crosses incompatible reference floor domains"
                )
            d = domains.iloc[0]
            unit = units[
                (units.kind == d.ownership_kind)
                & (units.domain == d.ownership_domain_id)
            ]
            if len(unit) != 1:
                raise ValueError("Reference containing compound unresolved")
            u = unit.iloc[0]
            ph = head_series(parent)
            target = (ph.loc[u.source] - ph.loc[u.sink]) / (
                positions.loc[u.source] - positions.loc[u.sink]
            )
            np.testing.assert_allclose(
                selected.lower_bound.iloc[0], 0.01 * target, rtol=1e-7, atol=1e-12
            )
            result, audit, contract = reconstruct_compound_member_heads(
                child,
                parent,
                old.lower_bound,
                selection,
                exterior_paths=paths,
                reviewed_contract=approved,
                allow_boundary_flexibility=True,
            )
            np.testing.assert_array_equal(
                result[["id_link", "u", "v", "link_length_m"]].to_numpy(),
                child[["id_link", "u", "v", "link_length_m"]].to_numpy(),
            )
            if key in accepted:
                expected = read(spath).set_index("id_link").loc[result.id_link]
                np.testing.assert_allclose(
                    result[["head_us", "head_ds", "slope"]].to_numpy(),
                    expected[["head_us", "head_ds", "slope"]].to_numpy(),
                    rtol=1e-10,
                    atol=1e-10,
                )
            comparisons, checks = verify(
                result, original, parent, old.lower_bound, contract, audit
            )
            # Keep hydraulic fields out of this slope-only result, even if present in a reviewed frame.
            columns = [
                "id_link",
                "u",
                "v",
                "link_length_m",
                "head_us",
                "head_ds",
                "slope",
                "floor",
                "parent_link",
                "role",
            ]
            result[columns].to_parquet(dest / "slopes.parquet", index=False)
            save(dest / "audit.json", dict(solver=audit, independent_checks=checks))
            save(
                dest / "contract.json",
                dict(
                    selected_parent_links=contract.selected_parent_links,
                    boundary_nodes=contract.boundary_nodes,
                    exterior_parent_by_child=contract.exterior_parent_by_child,
                    reversed_exterior_children=contract.reversed_exterior_children,
                    flexible_nodes=contract.flexible_nodes,
                ),
            )
            for c in comparisons:
                c["case"] = key
            all_comp.extend(comparisons)
            comparable = [
                c
                for c in comparisons
                if not c["length_changed"]
                and not c["reversed"]
                and c["existing_slope"] is not None
            ]
            row.update(
                status="verified_fixed"
                if audit["primary"]["strict_fixed_feasible"]
                else "verified_boundary_flex",
                replacement_links=int(result.role.eq("selected_replacement").sum()),
                exterior_links=len(comparisons),
                exterior_slopes_changed=sum(
                    abs(c["relative_change"]) > 1e-8 for c in comparisons
                ),
                comparable_exterior_links=len(comparable),
                old_comparable_changes=sum(
                    abs(c["old_relative_change"]) > 1e-8 for c in comparable
                ),
                new_comparable_changes=sum(
                    abs(c["relative_change"]) > 1e-8 for c in comparable
                ),
                exterior_length_changes=sum(c["length_changed"] for c in comparisons),
                reversed_exterior_links=sum(c["reversed"] for c in comparisons),
                max_relative_exterior_slope_change=max(
                    abs(c["relative_change"]) for c in comparisons
                ),
                released_nodes=audit["primary"]["candidates"],
                **checks,
            )
        except Exception as e:
            row.update(status="blocked_explicitly", reason=f"{type(e).__name__}: {e}")
            (dest / "failure.txt").write_text(traceback.format_exc())
        rows.append(row)
        if i % 25 == 0 or row["status"] == "blocked_explicitly":
            print(
                i,
                "/",
                len(inv),
                key,
                row["status"],
                row.get("reason", "")[:180],
                flush=True,
            )
            save(out / "progress.json", rows)
    for p, h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError("Source changed during run: " + p)
    pd.DataFrame(all_comp).to_parquet(out / "exterior_comparison.parquet", index=False)
    save(out / "case_results.json", rows)
    pd.DataFrame(rows).to_csv(out / "case_results.csv", index=False)
    summary = dict(
        cases=len(rows),
        networks=int(inv.example.nunique()),
        status_counts=pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
        production_modified=False,
        routing_runs=0,
        rf_training_runs=0,
        source_hashes=hashes,
    )
    save(out / "manifest.json", summary)
    print({k: v for k, v in summary.items() if k != "source_hashes"}, flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.workers > 1:
        if a.limit:
            p.error("--limit is supported only with one worker")
        from .run_compound_member_heads_parallel import run_parallel

        run_parallel(a.output, a.workers)
    else:
        run(a.output, a.limit)

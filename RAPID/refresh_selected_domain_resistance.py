"""Refresh existing selected-domain comparisons with anchored local depth.

Geometry/domain membership is preserved; heads come from the reconstructed
contexts. Results and solver caches are isolated from historical experiments.
Missing selected domains are reported, never replaced by whole-network values.
"""
from pathlib import Path
import argparse
import json
from dataclasses import asdict
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"RAPID"), str(ROOT/"RAPID/src")]
import pandas as pd
from run_hydraulic_policy_refresh import INVENTORY, DEFAULT_OUTPUT, reference_summary
from hierarchy_level_definition.metrics.bubble_equivalent_metrics import BubbleEquivalentResistanceConfig
from network_variants.equivalent_transition_domain import (HydraulicDomain, solve_hydraulic_domain,
    select_smallest_closed_parent_domain, build_matched_child_domain, complete_anchor_map_from_unchanged_lineage)



def island_domain_membership(example, state, base_record, parent, child):
    suffix = "_direction130" if example == "15" else ""
    source = ROOT/f"network_variants/output/{example}_island{suffix}__anchored"
    islands = pd.read_csv(source/"island_analysis_manifest_strict.csv")
    selected = islands[islands.island_id.eq(state)]
    if len(selected) != 1 or pd.isna(selected.iloc[0].source_unit_id):
        raise ValueError("Island lacks an unambiguous source unit")
    base_source = Path(base_record.manifest).parent.parent
    parent_domain = select_smallest_closed_parent_domain(full_links=parent,
        unit_summary=pd.read_csv(base_source/"hierarchy/unit_summary.csv"),
        unit_link_membership=pd.read_csv(base_source/"variant/summary/unit_link_membership.csv"),
        bubble_edge_resistance=pd.read_csv(base_source/"hierarchy/bubble_edge_resistance.csv"),
        selected_unit_ids=[int(selected.iloc[0].source_unit_id)])
    mapping = dict(zip(child.u.astype(int), child.id_us_node.astype(int)))
    mapping.update(zip(child.v.astype(int), child.id_ds_node.astype(int)))
    lineage_path = Path(selected.iloc[0].analysis_variant_dir)/"matching/link_lineage.csv"
    anchor_audit = []
    if lineage_path.exists():
        mapping, anchor_audit = complete_anchor_map_from_unchanged_lineage(
            parent_links=parent, child_links=child, parent_to_child_node=mapping,
            lineage=pd.read_csv(lineage_path), anchors=[parent_domain.source_node, parent_domain.sink_node])
    child_domain = build_matched_child_domain(parent_domain=parent_domain, child_links=child,
        parent_to_child_node=mapping)
    domain = pd.concat([parent_domain.links.assign(domain_role="parent"),
                       child_domain.links.assign(domain_role="child")], ignore_index=True)
    domain.attrs["anchor_recovery"] = anchor_audit
    return domain


def run(example, output, *, states=None, domain_overrides=None):
    inventory = pd.read_csv(INVENTORY/example/"inventory.csv", dtype={"example": str})
    dest = output/example/"selected_domain_resistance"
    dest.mkdir(parents=True, exist_ok=True)
    config = BubbleEquivalentResistanceConfig(resistance_law="chezy", depth_width_exponent=.65,
        kb_mode="based_reference_section", reference_kb_summary=reference_summary(example),
        solver_backend="energy_sparse", solver_cache_dir=str(dest/"solve_cache"))
    base = inventory[(inventory.family == "indep") & (inventory.state == "S000_base")].iloc[0]
    base_frame = pd.read_parquet(base.frame)
    selected_rows = inventory[inventory.status.eq("ready") & ~inventory.state.eq("S000_base")]
    results = []
    if states is not None:
        selected_rows = selected_rows[selected_rows.state.isin(states)]
        if selected_rows.empty:
            raise ValueError("No requested comparison states found")
        existing = pd.read_csv(dest/"comparisons.csv", dtype={"example": str})
        results = existing[~existing.state.isin(states)].to_dict("records")
    for r in selected_rows.itertuples():
        source = Path(r.manifest).parent.parent
        domain_path = source/"equivalent_resistance/selected_hydraulic_domain_links.csv"
        using_override = False
        if domain_overrides is not None:
            replacement = Path(domain_overrides)/r.family/r.state/domain_path.name
            if replacement.exists():
                domain_path = replacement
                using_override = True
        row = dict(example=example, family=r.family, state=r.state, fingerprint=r.fingerprint)
        if not domain_path.exists() and r.family != "island":
            row.update(status="missing_selected_domain", reason="No saved selected-domain membership; requires domain reconstruction.")
        else:
            try:
                child = pd.read_parquet(r.frame)
                domain = (pd.read_csv(domain_path) if domain_path.exists() else
                    island_domain_membership(example, r.state, base, base_frame, child))
                row["domain_source"] = "staged_membership" if using_override else ("saved_membership" if domain_path.exists() else "reconstructed_closed_island_domain")
                domain_dest = dest/r.family/r.state
                domain_dest.mkdir(parents=True, exist_ok=True)
                domain.to_csv(domain_dest/"domain_links.csv", index=False)
                (domain_dest/"anchor_recovery.json").write_text(json.dumps(domain.attrs.get("anchor_recovery", []), indent=2))
                values = {}
                for role, frame in (("parent", base_frame), ("child", child)):
                    ids = domain[domain.domain_role.eq(role)].id_link
                    links = frame[frame.id_link.isin(ids)].copy()
                    if set(links.id_link) != set(ids):
                        raise ValueError("Domain IDs do not match audited hydraulic frame")
                    sources = set(links.id_us_node) - set(links.id_ds_node)
                    sinks = set(links.id_ds_node) - set(links.id_us_node)
                    if len(sources) != 1 or len(sinks) != 1:
                        raise ValueError("Saved domain lacks a unique source/sink")
                    links["len"] = links.link_length_m
                    nodes = pd.concat([
                        links[["id_us_node", "head_us"]].rename(columns={"id_us_node":"id_node", "head_us":"H"}),
                        links[["id_ds_node", "head_ds"]].rename(columns={"id_ds_node":"id_node", "head_ds":"H"})]).drop_duplicates("id_node")
                    d = HydraulicDomain("saved_selected_domain", int(next(iter(sources))), int(next(iter(sinks))), links, (), None, True, ())
                    value, _ = solve_hydraulic_domain(domain=d, nodes=nodes, config=config)
                    if not value["solver_success"]:
                        raise ValueError(f"{role} solve failed: {value.get('solver_message')}")
                    values[role] = value
                    row[f"{role}_K"] = value["K_eq"]
                    row[f"{role}_residual_relative"] = value.get("residual_relative")
                row.update(status="recomputed", Lambda_K=row["child_K"]/row["parent_K"])
            except Exception as error:
                row.update(status="failed", reason=str(error))
        results.append(row)
        pd.DataFrame(results).to_csv(dest/"comparisons.csv", index=False)
        print(example, r.family, r.state, row["status"], flush=True)
    (dest/"manifest.json").write_text(json.dumps(dict(config=asdict(config),
        input_inventory=str(INVENTORY/example/"inventory.csv"),
        counts=pd.DataFrame(results).status.value_counts().to_dict(),
        promoted_to_historical_experiment=False), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("examples", nargs="+")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--states", nargs="+")
    p.add_argument("--domain-overrides", type=Path, help="Use staged domain CSVs without changing source experiments.")
    args = p.parse_args()
    for example in args.examples:
        run(example, args.output, states=args.states, domain_overrides=args.domain_overrides)

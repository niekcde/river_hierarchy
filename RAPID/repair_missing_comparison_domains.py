"""Stage auditable domain-membership repairs; leave source experiments untouched."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"RAPID"), str(ROOT/"RAPID/src")]
import pandas as pd
from run_hydraulic_policy_refresh import INVENTORY
from network_variants.equivalent_resistance_experiment import _parse_int_list
from network_variants.equivalent_transition_domain import (
    select_smallest_closed_parent_domain, complete_anchor_map_from_unchanged_lineage,
    enclose_domain_at_surviving_anchors,
)


def reconstruct(record, base, parent, child):
    source = Path(record.manifest).parent.parent
    if record.family == "island":
        suffix = "_direction130" if str(record.example) == "15" else ""
        island_root = ROOT/f"network_variants/output/{record.example}_island{suffix}__anchored"
        registry = pd.read_csv(island_root/"island_analysis_manifest_strict.csv")
        selected = registry[registry.island_id.eq(record.state)]
        if len(selected) != 1 or pd.isna(selected.iloc[0].source_unit_id):
            raise ValueError("Island repair requires one recorded source unit.")
        selection = selected.iloc[0]
        unit_ids = [int(selection.source_unit_id)]
        compound = None
        variant_dir = Path(selection.analysis_variant_dir)
    else:
        registry = pd.read_csv(source.parents[1]/"state_registry.csv")
        selected = registry[registry.state_id.eq(record.state)]
        if len(selected) != 1 or selected.iloc[0].parent_state_id != "S000_base":
            raise ValueError("Repair requires one recorded selection with the audited base parent.")
        selection = selected.iloc[0]
        unit_ids = _parse_int_list(selection.selected_unit_ids)
        compound = int(selection.compound_bubble_id) if pd.notna(selection.compound_bubble_id) else None
        variant_dir = Path(selection.variant_output_dir)
    base_source = Path(base.manifest).parent.parent
    original = select_smallest_closed_parent_domain(full_links=parent,
        unit_summary=pd.read_csv(base_source/"hierarchy/unit_summary.csv"),
        unit_link_membership=pd.read_csv(base_source/"variant/summary/unit_link_membership.csv"),
        bubble_edge_resistance=pd.read_csv(base_source/"hierarchy/bubble_edge_resistance.csv"),
        selected_unit_ids=unit_ids, selected_compound_bubble_id=compound)
    mapping = dict(zip(child.u.astype(int), child.id_us_node.astype(int)))
    mapping.update(zip(child.v.astype(int), child.id_ds_node.astype(int)))
    mapping, recovered = complete_anchor_map_from_unchanged_lineage(parent_links=parent,
        child_links=child, parent_to_child_node=mapping,
        lineage=pd.read_csv(variant_dir/"matching/link_lineage.csv"),
        anchors=[original.source_node, original.sink_node])
    pdomain, cdomain, audit = enclose_domain_at_surviving_anchors(parent_domain=original,
        parent_links=parent, child_links=child, parent_to_child_node=mapping)
    columns = ["id_link", "id_us_node", "id_ds_node", "link_length_m", "wid_adj_routing"]
    domain = pd.concat([
        pdomain.links[columns].assign(domain_role="parent", state_id="S000_base"),
        cdomain.links[columns].assign(domain_role="child", state_id=record.state)], ignore_index=True)
    domain["len"] = domain.link_length_m
    domain["comparison_domain_type"] = pdomain.domain_type
    audit.update(anchor_recovery=recovered, selected_unit_ids=list(original.selected_unit_ids),
                 family=record.family, state=record.state, fingerprint=record.fingerprint,
                 hydraulic_inputs_changed=False)
    return domain, audit


def run(example, output):
    inv = pd.read_csv(INVENTORY/example/"inventory.csv", dtype={"example": str})
    base = inv[(inv.family == "indep") & (inv.state == "S000_base")].iloc[0]
    parent = pd.read_parquet(base.frame)
    rows = []
    for record in inv[inv.status.eq("ready") & ~inv.state.eq("S000_base")].itertuples():
        path = Path(record.manifest).parent.parent/"equivalent_resistance/selected_hydraulic_domain_links.csv"
        if path.exists() and path.read_text().strip():
            continue
        domain, audit = reconstruct(record, base, parent, pd.read_parquet(record.frame))
        if record.family == "island" and not audit["expanded"]:
            continue
        dest = output/record.family/record.state
        dest.mkdir(parents=True, exist_ok=True)
        domain.to_csv(dest/path.name, index=False)
        audit.update(source_path=str(path), original_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            base_frame_sha256=hashlib.sha256(Path(base.frame).read_bytes()).hexdigest(),
            child_frame_sha256=hashlib.sha256(Path(record.frame).read_bytes()).hexdigest())
        (dest/"recovery_audit.json").write_text(json.dumps(audit, indent=2))
        rows.append(dict(example=example, family=record.family, state=record.state,
                         source=str(path), staged=str(dest/path.name), **{k:v for k,v in audit.items() if k in
                         ["original_parent_source", "original_parent_sink", "parent_source", "parent_sink", "parent_n_links", "child_n_links"]}))
        print(record.state, audit, flush=True)
    pd.DataFrame(rows).to_csv(output/"repairs.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("example")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.example, args.output)

"""Audit saved BASED slopes on geometrically retained links in all 15 networks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from manning_beta1_network import ROOT, OUT, digest

DEST=OUT/'retained_slope_audit_all_networks_20261027'

def match(base,child):
    # u/v are mapped reference-node IDs. Raw link IDs can be regenerated.
    m=base.merge(child,on=['u','v'],suffixes=('_base','_child'))
    keep=np.ones(len(m),bool)
    for k in ['link_length_m','width']:
        keep &= np.isclose(m[k+'_base'],m[k+'_child'],rtol=1e-10,atol=1e-7)
    m=m[keep].copy()
    # Preserve stable IDs when parallel links have identical geometry attributes.
    stable=m[m.id_link_base.eq(m.id_link_child)]
    assert stable.id_link_base.is_unique and stable.id_link_child.is_unique
    rest=m[~m.id_link_base.isin(stable.id_link_base)&~m.id_link_child.isin(stable.id_link_child)]
    ambiguous=rest.id_link_base.duplicated(False)|rest.id_link_child.duplicated(False)
    matched=pd.concat([stable,rest[~ambiguous]],ignore_index=True)
    assert matched.id_link_base.is_unique and matched.id_link_child.is_unique
    return matched,int(ambiguous.sum())

def main():
    DEST.mkdir(exist_ok=True);sources={}
    def read(p):
        sources[str(p)]=digest(p)
        return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
    old=read(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json')
    ext=read(ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json')
    new=read(OUT/'routing_plan.json')
    jobs={}
    for doc in [old,ext,new]:
        for c in doc['cases']:
            for j in c['jobs']:jobs[c['example'],j['state_id']]=j
    inv=read(OUT/'validation_group_inventory_20261023/records.parquet')
    rows=[];changed_links=[]
    for example,g in inv.groupby('example'):
        base=read(Path(jobs[example,'S000_base']['parameters']))
        for r in g.itertuples():
            child=read(Path(jobs[example,r.state]['parameters']))
            m,ambiguous=match(base,child)
            relative=(m.slope_child/m.slope_base-1).abs()
            changed=~np.isclose(m.slope_base,m.slope_child,rtol=1e-8,atol=1e-10)
            outside=~m.id_link_base.isin(r.parent_footprint_links)
            rows.append(dict(example=example,state=r.state,edit_type=r.edit_type,
                parent_links=len(base),child_links=len(child),matched_retained_links=len(m),ambiguous_candidate_pairs=ambiguous,
                changed_retained_links=int(changed.sum()),changed_outside_saved_footprint=int((changed&outside).sum()),
                changed_over_1pct=int((relative>.01).sum()),changed_over_1pct_outside=int(((relative>.01)&outside).sum()),
                max_relative_slope_change=float(relative.max()) if len(m) else 0))
            selected=m[changed].copy()
            changed_links.extend(dict(example=example,state=r.state,parent_link=int(x.id_link_base),child_link=int(x.id_link_child),
                slope_base=float(x.slope_base),slope_child=float(x.slope_child),outside_saved_footprint=bool(x.id_link_base not in r.parent_footprint_links))
                for x in selected.itertuples())
        print('Audited',example,len(g),'states',flush=True)
    f=pd.DataFrame(rows);f.to_csv(DEST/'state_summary.csv',index=False)
    pd.DataFrame(changed_links).to_parquet(DEST/'changed_retained_links.parquet',index=False)
    def summary(g):
        return pd.Series(dict(cases=len(g),cases_with_retained_slope_changes=int(g.changed_retained_links.gt(0).sum()),
            cases_with_changes_outside_footprint=int(g.changed_outside_saved_footprint.gt(0).sum()),
            cases_with_over1pct_changes=int(g.changed_over_1pct.gt(0).sum()),
            cases_with_over1pct_changes_outside=int(g.changed_over_1pct_outside.gt(0).sum()),
            cases_with_no_retained_matches=int(g.matched_retained_links.eq(0).sum()),ambiguous_pairs=int(g.ambiguous_candidate_pairs.sum())))
    bynet=f.groupby('example').apply(summary,include_groups=False)
    bytype=f.groupby('edit_type').apply(summary,include_groups=False)
    bynet.to_csv(DEST/'by_network.csv');bytype.to_csv(DEST/'by_type.csv')
    for p in [Path(__file__),ROOT/'network_variants/shared_head_reconstruction.py']:sources[str(p)]=digest(p)
    for p,h in sources.items():assert digest(p)==h
    manifest=dict(complete=True,networks=15,cases=len(f),totals=summary(f).to_dict(),
        scope='2185 accepted main collapse comparisons including 15 segmented full-network states; excludes additional combination augmentation and separate single-centerline controls.',
        matching='Reference-mapped u/v, length and width; rtol=1e-10, atol=1e-7; stable link IDs resolve parallel matches where available; remaining ambiguous matches omitted.',
        slope_threshold='not isclose(parent,child,rtol=1e-8,atol=1e-10); additional >1% relative threshold reported',
        footprint_caveat='Saved source/anchor membership, conservative for islands. Unmatched or ambiguous links are not assumed unchanged.',
        hard_constraint='Reference/profile nodes and containing-domain boundary heads. All other interior heads in affected owned domains are free by default.',
        objective='Minimize squared departures from a reference relative slope, with a positive slope floor. No penalty for departures from original individual heads/slopes; parent heads are not soft preservation targets.',
        source_hashes=sources,production_modified=False)
    (DEST/'audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(bynet.to_string());print(bytype.to_string());print('TOTAL',summary(f).to_dict())

if __name__=='__main__':main()

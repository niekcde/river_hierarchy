"""Run isolated reviewed fixtures; all outputs stay under RAPID/outputs."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.ops import unary_union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping, repair, topology

OUT = review.ROOT / 'RAPID/outputs/anchor_repair_trial_20261008'
CASES = [
 ('41','S016_unit_4',[5,7],{}),
 ('02','S023_unit_12',[27],{}),
 ('02','S026_unit_16',[32],{}),
 ('13','S039_unit_33',[64],{64:66}),
 ('15','S135_unit_144',[281],{}),
 ('02','S005_unit_25',[],{56:50}),
 ('02','S004_unit_5',[],{58:7}),
 ('02','CB001_bubble_5',[],{}),
 ('02','CB005_bubble_6',[],{}),
]


def exterior_rows(base, selected, child):
    rows = []
    for p in base[~base.id_link.isin(selected)].itertuples():
        candidates = child[((child.u==p.id_us_node)&(child.v==p.id_ds_node)) |
                           ((child.v==p.id_us_node)&(child.u==p.id_ds_node))]
        rows.append(dict(parent_link=int(p.id_link),endpoint_pair_present=bool(len(candidates)),
                         best_hausdorff_m=min((p.geometry.hausdorff_distance(g) for g in candidates.geometry),default=np.nan)))
    return pd.DataFrame(rows)


def plot(before, after, parent, nodes_before, nodes_after, parent_nodes, required, selected, title, **view_options):
    # Shared viewer: parent-only panel, explicit ID namespaces and rectangular extent.
    from rapid_tools.boundary_contract_review import plot_comparison
    return plot_comparison(before, after, parent, nodes_before, nodes_after, parent_nodes,
                           required, selected, title, **view_options)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[]; provenance=[]
    for ex,state,cuts,matches in CASES:
        row=review.case_row(ex,state)
        paths=[row.child_links_path,row.child_nodes_path,row.base_links_path,row.base_nodes_path,row.frame_path]
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
        links=review.vectors(row.child_links_path,'links').copy()
        nodes=review.vectors(row.child_nodes_path,'nodes').copy()
        parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes')
        frame=pd.read_parquet(row.frame_path);mapping=node_mapping(frame)
        # Hydraulic source supplies the directed incidence; geometry is oriented by repair().
        links=links[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node','u','v']],on='id_link',validate='one_to_one')
        required=review.int_ids(row.required_boundary_nodes);selected=review.int_ids(row.selected_parent_link_ids)
        after,an,actions=repair(links,nodes,pn,mapping,required,cuts=cuts,correspondences=matches)
        nodes['parent_identity']=nodes.id_node.map(mapping)
        bt,at=topology(links),topology(after)
        for key in ['components','cycle_rank','dag','inlets','outlets']:
            assert bt[key]==at[key],(ex,state,key,bt,at)
        assert at['nodes']-bt['nodes']==len(cuts) and at['links']-bt['links']==len(cuts)
        lineage=[]
        for lid,g in after.groupby('source_link'):
            source=links[links.id_link==lid].iloc[0]
            old=source.geometry
            chain=nx.DiGraph(list(zip(g.id_us_node,g.id_ds_node)))
            assert nx.is_directed_acyclic_graph(chain)
            assert nx.has_path(chain,source.id_us_node,source.id_ds_node)
            assert chain.number_of_edges()==len(g)
            assert all(chain.in_degree(v)==chain.out_degree(v)==1 for v in chain if v not in [source.id_us_node,source.id_ds_node])
            delta=abs(g.length.sum()-old.length);distance=old.hausdorff_distance(unary_union(g.geometry))
            assert delta<1e-5 and distance<1e-5
            lineage.append(dict(source_link=int(lid),fragments=len(g),length_difference_m=delta,hausdorff_m=distance))
        bef=exterior_rows(parent,selected,links);aft=exterior_rows(parent,selected,after)
        ext=bef.merge(aft,on='parent_link',suffixes=('_before','_after'))
        dest=OUT/f'{ex}_{state}';dest.mkdir(exist_ok=True)
        after.to_parquet(dest/'trial_links.parquet',index=False);an.to_parquet(dest/'trial_nodes.parquet',index=False)
        pd.DataFrame(actions).to_json(dest/'actions.json',orient='records',indent=2)
        pd.DataFrame(lineage).to_csv(dest/'geometry_lineage.csv',index=False);ext.to_csv(dest/'exterior_comparison.csv',index=False)
        present_before=set(mapping.values());present_after=set(an.parent_identity)
        residual=[]
        for p in required:
            if p not in present_after:
                near=an.distance(pn.set_index('id_node').geometry.loc[p]).idxmin()
                residual.append(dict(parent_node=p,nearest_raw_child=int(an.loc[near,'id_node']),distance_m=float(an.loc[near,'geometry'].distance(pn.set_index('id_node').geometry.loc[p]))))
        summary=dict(example=ex,state=state,inserted_cuts=len(cuts),identity_changes=sum(a['action']=='identity_restored' for a in actions),
                     missing_identity_before=sorted(set(required)-present_before),missing_identity_after=sorted(set(required)-present_after),
                     remaining_unreviewed_correspondences=residual,topology_before=bt,topology_after=at,
                     exterior_missing_endpoint_pairs_before=int((~bef.endpoint_pair_present).sum()),
                     exterior_missing_endpoint_pairs_after=int((~aft.endpoint_pair_present).sum()),
                     max_geometry_length_error_m=max(v['length_difference_m'] for v in lineage),
                     max_geometry_hausdorff_m=max(v['hausdorff_m'] for v in lineage))
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        plot(links,after,parent,nodes,an,pn,required,selected,f'{ex} — {state} | Geometry-only repair trial').write_html(dest/'comparison.html',include_plotlyjs='directory')
        assert hashes=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'Source modified'
        provenance.append(dict(example=ex,state=state,fingerprint=row.fingerprint,source_sha256=hashes))
        summaries.append(summary)
        print(ex,state,'cuts',len(cuts),'identities',summary['identity_changes'],'remaining',summary['missing_identity_after'],flush=True)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    intro='''<h1>Reviewed anchoring repair trial</h1><p>Explicit prescribed-node cuts and user-reviewed identity mappings only. Source files are unchanged (SHA-256 checked). Six inserted degree-two nodes preserve centerline geometry, connectivity and cycle rank. Displaced reviewed nodes retain their coordinates.</p><p><b>This is not routing-ready:</b> fragment widths, heads, slopes, K and X have not been regenerated. Exterior connectivity recovery does not certify unchanged exterior geometry or hydraulic properties. No island creation or duplicate-selection policy is modified.</p><p>The exterior metric below tests whether each parent exterior endpoint pair has a direct child reach (in either direction). An absent direct pair can also reflect subdivision: this is a diagnostic, not a count of disconnected paths. Per-link geometric distances are in exterior_comparison.csv.</p><p><b>41 / unit 4:</b> the merged 3181.30 m link becomes 4→5 (878.82 m), 5→7 (665.14 m), and 7→6 (1637.34 m). Both exterior fragments reproduce their parent geometry exactly. The middle fragment needs its own width measurement; copying the old merged-link width would be inappropriate.</p><p><b>Remaining scope:</b> 02–12 still has a different exterior representation around parent reach 48 (31→32); 02–16 also retains that issue and unreviewed displaced identities P18/P20. 15–144 retains unreviewed P291. The P7 control in 02–5 stays correct, but P9/P12 remain unreviewed. These are not silently accepted or fixed.</p>'''
    table='<table border="1" cellpadding="8"><tr><th>Example / state</th><th>New cuts</th><th>Identity changes</th><th>Missing boundary identities before → after</th><th>Exterior missing endpoint pairs before → after</th></tr>'
    for s in summaries:
        name=f"{s['example']}_{s['state']}"
        table+=f'''<tr><td><a href="{name}/comparison.html">{name}</a></td><td>{s['inserted_cuts']}</td><td>{s['identity_changes']}</td><td>{s['missing_identity_before']} → {s['missing_identity_after']}</td><td>{s['exterior_missing_endpoint_pairs_before']} → {s['exterior_missing_endpoint_pairs_after']}</td></tr>'''
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:40px;max-width:1300px}td,th{text-align:left}p{max-width:1000px}</style>'+intro+table+'</table><p>Unreviewed displaced boundary correspondences are listed in each summary.json. No automatic nearest-node acceptance is performed.</p>')

if __name__=='__main__':run()

"""Scalable legacy-equivalent path features for every distinct parent-child comparison."""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
import networkx as nx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT
from rapid_tools.path_geometry import path_geometry
from rapid_tools.comparison_population import COMPARISON_KEY
SOURCE=ROOT/'RAPID/validation/deduplicated_response_benchmark_20261005'
OUT=ROOT/'RAPID/validation/all_pairs_lag_20261005'

def main():
    OUT.mkdir(parents=True,exist_ok=True);d=pd.read_parquet(SOURCE/'canonical_all.parquet');d['omega']=d.len_fraction_width_weighted_log_lambda;d['resistance_log']=d.neg_log10_lambda;records=[];audit=[]
    for ex,group in d.groupby('example_id'):
        inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv');ready=inv[inv.status.eq('ready')].drop_duplicates('fingerprint').set_index('fingerprint');base=ready[ready.state.eq('S000_base')].iloc[0];parent=pd.read_parquet(base.frame)
        graph=nx.MultiDiGraph()
        for r in parent.itertuples():graph.add_edge(r.id_us_node,r.id_ds_node,key=r.id_link,length=r.link_length_m)
        inlet=next(n for n in graph if graph.in_degree(n)==0);outlet=next(n for n in graph if graph.out_degree(n)==0)
        up=nx.single_source_dijkstra_path_length(graph,inlet,weight='length');down=nx.single_source_dijkstra_path_length(graph.reverse(),outlet,weight='length');pc={};cc={};frames={}
        for r in group.itertuples():
            if r.matched_base_link_ids not in pc:pc[r.matched_base_link_ids]=path_geometry(parent,[int(i) for i in r.matched_base_link_ids.split(',')])
            if r.fingerprint not in frames:frames[r.fingerprint]=pd.read_parquet(ready.loc[r.fingerprint,'frame'])
            key=(r.fingerprint,r.child_link_ids)
            if key not in cc:cc[key]=path_geometry(frames[r.fingerprint],[int(i) for i in r.child_link_ids.split(',')])
            p,u,v=pc[r.matched_base_link_ids];c,_,_=cc[key];row=r._asdict();row.update({f'parent_{k}':value for k,value in p.items()});row.update({f'child_{k}':value for k,value in c.items()})
            row.update(signed_length_width_association=p['association'],domain_reference_slope=p['slope'],upstream_shortest_length_m=up[u],downstream_shortest_length_m=down[v],log_equivalent_length_ratio=np.log(c['equivalent_length']/p['equivalent_length']),log_elongation_ratio=np.log(c['elongation']/p['elongation']),change_path_length_cv=c['path_length_cv']-p['path_length_cv'],change_width_dominance=c['dominant_width_fraction']-p['dominant_width_fraction']);records.append(row)
        print(ex,len(group),'complete',flush=True)
    f=pd.DataFrame(records);old=pd.read_parquet(SOURCE/'common_features.parquet');compare=f.merge(old,on=COMPARISON_KEY,suffixes=('_new','_old'),validate='one_to_one')
    cols=['parent_equivalent_length','parent_elongation','parent_dominant_width_fraction','parent_path_length_cv','signed_length_width_association','domain_reference_slope','log_equivalent_length_ratio','log_elongation_ratio','change_path_length_cv','change_width_dominance','upstream_shortest_length_m','downstream_shortest_length_m']
    for col in cols:
        a=compare[col+'_new'].to_numpy();b=compare[col+'_old'].to_numpy();np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
        audit.append(dict(feature=col,n=len(a),maximum_absolute_error=float(abs(a-b).max())))
    assert len(f)==2214 and len(compare)==1128 and f.example_id.nunique()==16
    f.to_parquet(OUT/'features.parquet',index=False);pd.DataFrame(audit).to_csv(OUT/'enumeration_parity.csv',index=False)
    (OUT/'geometry_provenance.json').write_text(json.dumps(dict(rows=len(f),rivers=16,previously_filtered_pairs_restored=562,exclusions=0,max_parent_paths=float(f.parent_paths.max()),max_child_paths=float(f.child_paths.max()),max_quadrature_relative_difference=float(f[['parent_quadrature_relative_error','child_quadrature_relative_error']].max().max()),method='Uniform-path DAG dynamic programming plus checked Laplace-integral quadrature for harmonic width moments and fractional optimization for maximum width. Original metric definitions retained; no path sampling, no path-count limit, no imputation.',parity_rows=len(compare),parity_rtol=1e-8,parity_atol=1e-10),indent=2))
    print('Complete',len(f),'max paths',f.parent_paths.max(),flush=True)

if __name__=='__main__':main()

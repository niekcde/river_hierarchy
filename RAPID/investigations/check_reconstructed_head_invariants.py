"""Independent hydraulic and frozen-boundary checks on all new candidates."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_child_heads import OUT,BASE,EXAMPLES
import numpy as np
import pandas as pd
import networkx as nx


def main():
    inventory=pd.concat([pd.read_csv(OUT/ex/'inventory.csv',dtype={'example':str}) for ex in EXAMPLES],ignore_index=True)
    inventory.to_csv(OUT/'inventory.csv',index=False)
    rows=[];bases=[]
    for ex in EXAMPLES:
        old=pd.read_csv(BASE/'full_signed_validation'/ex/'inventory.csv');oldbase=old[(old.family=='indep')&(old.state=='S000_base')].iloc[0]
        inv=inventory[inventory.example==ex];r=inv[(inv.family=='indep')&(inv.state=='S000_base')].iloc[0]
        before=pd.read_parquet(oldbase.frame).sort_values('id_link');after=pd.read_parquet(r.frame).sort_values('id_link')
        cols=['slope','width','depth_m','rapid_k']
        assert np.array_equal(before[cols].to_numpy(),after[cols].to_numpy())
        bases.append(dict(example=ex,identical_base_hydraulics=True))
        baseheads=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node')['head']
        for r in inv[inv.status=='ready'].drop_duplicates('fingerprint').itertuples():
            f=pd.read_parquet(r.frame);qa=json.loads((Path(r.frame).with_name(r.fingerprint+'_heads.json')).read_text())
            hs=pd.concat([f[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),f[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})])
            spread=hs.groupby('node')['head'].agg(lambda x:float(x.max()-x.min())).max()
            assert spread<1e-10 and np.isfinite(f.slope).all() and f.slope.min()>0
            nh=hs.drop_duplicates('node').set_index('node')['head'];anchor_error=0.;floor_error=0.
            for patch in qa['patches']:
                anchors=patch['boundary_nodes'];anchor_error=max(anchor_error,float(abs(nh.loc[anchors]-baseheads.loc[anchors]).max()))
            assert anchor_error<1e-10
            G=nx.DiGraph();G.add_edges_from(zip(f.u,f.v));assert nx.is_directed_acyclic_graph(G)
            src=[v for v in G if G.in_degree(v)==0];snk=[v for v in G if G.out_degree(v)==0];assert len(src)==len(snk)==1
            inlet_outlet_error=abs((nh[src[0]]-nh[snk[0]])-(baseheads[src[0]]-baseheads[snk[0]]))
            assert inlet_outlet_error<1e-10
            context=pd.read_csv(Path(r.candidate_manifest).parent/'link_hydraulic_context.csv');assert len(context)==len(f)
            assert np.allclose(context.slope_used,(context.candidate_head_us_m-context.candidate_head_ds_m)/context.link_length_m,rtol=1e-9,atol=1e-12)
            rows.append(dict(example=ex,state=r.state,fingerprint=r.fingerprint,n_links=len(f),min_slope=f.slope.min(),max_slope=f.slope.max(),min_raw_celerity=f.rapid_celerity_mps_raw.min(),max_raw_celerity=f.rapid_celerity_mps_raw.max(),low_cap_links=int((f.rapid_celerity_mps_raw<.05).sum()),high_cap_links=int((f.rapid_celerity_mps_raw>5).sum()),n_patches=len(qa['patches']),released_anchors=len(qa['released_interior_anchors']),coincident_ports=len(qa['node_matching']['coincident_ports']),max_anchor_error_m=anchor_error,inlet_outlet_error_m=inlet_outlet_error,max_shared_head_error_m=spread))
    pd.DataFrame(rows).to_csv(OUT/'hydraulic_invariants.csv',index=False)
    pd.DataFrame(bases).to_csv(OUT/'base_unchanged.csv',index=False)
    print(inventory.groupby(['family','status']).size().to_string());print(pd.DataFrame(rows).drop(columns=['example','state','fingerprint']).agg(['min','max']).to_string())

if __name__=='__main__':main()

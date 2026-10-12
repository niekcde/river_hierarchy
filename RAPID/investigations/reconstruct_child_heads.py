"""Inventory and materialize all available comparison geometries for validation.

Strict node matching and explicitly constrained new-node head allocation:
never silently reuse stale child slopes or nearest-node substitute heads. Production inputs remain read-only.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import pyogrio

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_explicit_celerity_regularization import BASE,longest_travel
OUT=BASE/'child_head_reconstruction_20261003'
sys.path.insert(0,str(ROOT))
from network_variants.shared_head_reconstruction import reconstruct_shared_heads, match_child_nodes
EXAMPLES=['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']



def source_inventory(ex):
    suffix='_direction130' if ex=='15' else ''
    rows=[]
    for family in ['indep','compound_bubbles']:
        exp=ROOT/'network_variants/output'/f'{ex}_{family}{suffix}__anchored'
        parents={}
        if (exp/'transition_registry.csv').exists():
            t=pd.read_csv(exp/'transition_registry.csv')
            parents=dict(zip(t.child_state_id,t.parent_state_id))
        # Official registry membership excludes temporary promotion directories.
        state_ids=pd.read_csv(exp/'state_registry.csv').state_id.astype(str)
        for state_id in sorted(state_ids):
            state=exp/'states'/state_id
            if not state.is_dir():raise FileNotFoundError(f'Registered state directory missing: {state}')
            rows.append(dict(example=ex,family=family,state=state.name,
                parent_state=parents.get(state.name,'S000_base'),
                manifest=str(state/'hydraulic_context/hydraulic_context_manifest.json')))
    exp=ROOT/'network_variants/output'/f'{ex}_island{suffix}__anchored'
    reg=exp/'island_hydraulic_context_registry.csv'
    if reg.exists():
        for r in pd.read_csv(reg).itertuples():
            rows.append(dict(example=ex,family='island',state=r.island_id,parent_state='S000_base',
                manifest=str(r.hydraulic_context_manifest_path),source_status=r.hydraulic_context_status))
    return rows


def materialize(ex):
    dest=OUT/ex;dest.mkdir(parents=True,exist_ok=True)
    suffix='_direction130' if ex=='15' else ''
    exp=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored'
    m=json.loads((exp/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json').read_text())
    bn=pyogrio.read_dataframe(m['input_paths']['directed_nodes'])
    base_nodes=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet')
    heads=base_nodes.set_index('id_node')['head']
    base_links=pd.read_parquet(BASE/ex/'floor_0.05/old/links.parquet')
    ref=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example').loc[ex]
    cache={};records=[];configs=[]
    for family in ['indep','compound_bubbles','island']:
        ep=ROOT/'network_variants/output'/f'{ex}_{family}{suffix}__anchored'
        pm=ep/('island_rapid_prep_manifest.json' if family=='island' else 'rapid_prep_manifest.json')
        em=ep/'equivalent_resistance/equivalent_resistance_manifest.json'
        pc=json.loads(pm.read_text()).get('prep_config',{}) if pm.exists() else {}
        ec=json.loads(em.read_text()).get('equivalent_resistance_config',{}) if em.exists() else {}
        configs.append(dict(example=ex,family=family,prep_manifest=str(pm),prep_exists=pm.exists(),
            saved_n=pc.get('n_manning'),saved_cap_min=pc.get('min_celerity_mps'),saved_cap_max=pc.get('max_celerity_mps'),
            saved_length_correction=pc.get('use_stability_effective_length_cap'),saved_kb_mode=pc.get('kb_mode'),
            saved_depth_alpha=pc.get('depth_width_exponent',pc.get('kb_width_exponent',None)),
            equivalent_manifest=str(em),equivalent_exists=em.exists(),equivalent_law=ec.get('resistance_law'),
            equivalent_default_n=ec.get('default_manning_n'),equivalent_default_chezy=ec.get('default_chezy_cz'),
            equivalent_kb_mode=ec.get('kb_mode')))
    for i,row in enumerate(source_inventory(ex)):
        import os
        if os.environ.get('HEAD_CHECK_STATES') and row['state'] not in os.environ['HEAD_CHECK_STATES'].split(','):continue
        if row.get('source_status')=='excluded_from_primary_analysis':
            records.append(dict(row,status='excluded_topology_review',reason='Existing primary-analysis exclusion',context_alias=False));continue
        mp=Path(row['manifest']);key=str(mp)
        if key in cache:
            records.append(dict(row,**cache[key],context_alias=True));continue
        f=None
        try:
            cm=json.loads(mp.read_text())
            nodes=pyogrio.read_dataframe(cm['input_paths']['directed_nodes'])
            if nodes.crs!=bn.crs:raise ValueError('Child/base CRS mismatch')
            h=pd.read_csv(mp.parent/'link_hydraulic_context.csv')
            widths=pyogrio.read_dataframe(cm['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False)
            f=h[['id_link','id_us_node','id_ds_node','link_length_m','slope_used']].merge(widths,on='id_link',validate='one_to_one')
            if len(f)!=len(h):raise ValueError('Incomplete width match')
            mapping,unknown,match_audit=match_child_nodes(bn,nodes,base_links,f)
            unmatched=len(unknown)
            f['u']=f.id_us_node.map(mapping);f['v']=f.id_ds_node.map(mapping);f['width']=f.wid_adj_routing
            f,head_audit=reconstruct_shared_heads(f,base_links,base_nodes,heads)
            head_audit['node_matching']=match_audit
            a=f[['width','link_length_m','slope']].to_numpy()
            if not np.isfinite(a).all() or (a<=0).any():raise ValueError('Nonpositive/nonfinite geometry or mapped-head slope')
            f['count']=np.maximum(1,np.rint(f.link_length_m/200)).astype(int)
            f['rapid_effective_length_m']=np.maximum(200,f.link_length_m/f['count'])
            f['depth_m']=ref.stored_depth_m*(f.width/ref.reference_width_m)**.65
            f['kb_local']=f.width/f.depth_m
            f['rapid_celerity_mps_raw']=5/(3*.035)*f.depth_m**(2/3)*np.sqrt(f.slope)
            f['rapid_celerity_mps']=f.rapid_celerity_mps_raw.clip(.05,5)
            f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps
            G=nx.DiGraph();G.add_edges_from(zip(f.u,f.v))
            if not nx.is_directed_acyclic_graph(G):raise ValueError('Routing graph has a cycle')
            if sum(G.in_degree(v)==0 for v in G)!=1 or sum(G.out_degree(v)==0 for v in G)!=1:
                raise ValueError('Expected one inlet and outlet')
            physical=f[['u','v','link_length_m','width','slope']].sort_values(['u','v','link_length_m','width']).to_numpy()
            fingerprint=hashlib.sha256(physical.tobytes()).hexdigest()[:20]
            output=dest/f'{fingerprint}.parquet'
            f.to_parquet(output,index=False)
            (dest/f'{fingerprint}_heads.json').write_text(json.dumps(head_audit,indent=2))
            context_key=hashlib.sha256(str(mp.resolve()).encode()).hexdigest()[:20]
            context_dir=dest/'contexts'/context_key;context_dir.mkdir(parents=True,exist_ok=True)
            context=f[['id_link','id_us_node','id_ds_node','link_length_m','head_us','head_ds','slope','depth_m','kb_local']].copy()
            context=context.rename(columns={'head_us':'candidate_head_us_m','head_ds':'candidate_head_ds_m','slope':'slope_used','kb_local':'k_b'})
            context['slope_source_method']='shared_head_domain_reconstruction'
            context['slope_adjusted']=True
            context.to_csv(context_dir/'link_hydraulic_context.csv',index=False)
            node_heads=pd.concat([f[['id_us_node','head_us']].rename(columns={'id_us_node':'id_node','head_us':'candidate_head_m'}),f[['id_ds_node','head_ds']].rename(columns={'id_ds_node':'id_node','head_ds':'candidate_head_m'})])
            assert node_heads.groupby('id_node').candidate_head_m.nunique().max()==1
            node_heads.drop_duplicates('id_node').sort_values('id_node').to_csv(context_dir/'node_hydraulic_context.csv',index=False)
            candidate_manifest=dict(state_id=row['state'],input_paths={k:v for k,v in cm['input_paths'].items() if k in ['directed_links','directed_nodes','hierarchy_output_dir']},source_input_paths=cm['input_paths'],source_manifest=str(mp),
                method='shared_head_domain_reconstruction',profile_unchanged=True,
                base_nodes=str(BASE/ex/'floor_0.05/old/nodes.parquet'),
                head_audit=head_audit,n_manning=.035,depth_width_exponent=.65,celerity_caps=[.05,5.],
                production_promoted=False,output_paths={'link_hydraulic_context':str(context_dir/'link_hydraulic_context.csv'),'node_hydraulic_context':str(context_dir/'node_hydraulic_context.csv')})
            (context_dir/'hydraulic_context_manifest.json').write_text(json.dumps(candidate_manifest,indent=2))
            rec=dict(candidate_manifest=str(context_dir/'hydraulic_context_manifest.json'),status='ready',reason='',frame=str(output),fingerprint=fingerprint,n_links=len(f),n_segments=int(f['count'].sum()),
                max_travel_seconds=longest_travel(f),upper_dt_seconds=float(1.8*f.rapid_k.min()),
                high_cap_links=int((f.rapid_celerity_mps_raw>5).sum()),
                new_nodes_allocated=unmatched,max_nearest_base_node_distance_m=match_audit['max_nearest_distance_m'],
                reconstructed_free_nodes=head_audit['free_nodes'],
                inherited_internal_nodes_reallocated=head_audit['free_nodes']-unmatched,
                saved_slopes_differ=int((~np.isclose(f.slope,f.slope_used,rtol=1e-6,atol=1e-12)).sum()),
                context_generated_at=cm.get('generated_at'),context_sha256=hashlib.sha256((mp.parent/'link_hydraulic_context.csv').read_bytes()).hexdigest())
        except Exception as err:
            if f is not None and 'slope' in f:
                bad=f[(~np.isfinite(f.slope))|(f.slope<=0)]
                if len(bad):bad.to_csv(dest/f"blocked_{row['family']}_{row['state']}_links.csv",index=False)
            rec=dict(status='blocked_input',reason=str(err),frame='',fingerprint='')
        cache[key]=rec;records.append(dict(row,**rec,context_alias=False))
        if i%50==0:print(ex,'materialized',i,flush=True)
    pd.DataFrame(records).to_csv(dest/'inventory.csv',index=False)
    pd.DataFrame(configs).to_csv(dest/'saved_settings.csv',index=False)
    print('INVENTORY DONE',ex,pd.Series([r['status'] for r in records]).value_counts().to_dict(),flush=True)
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(materialize,sys.argv[1:] or EXAMPLES))
    inv=pd.concat([pd.read_csv(p,dtype={'example':str}) for p in OUT.glob('*/inventory.csv')],ignore_index=True)
    inv.to_csv(OUT/'inventory.csv',index=False)
    pd.concat([pd.read_csv(p,dtype={'example':str}) for p in OUT.glob('*/saved_settings.csv')],ignore_index=True).to_csv(OUT/'saved_settings.csv',index=False)
    print(inv.groupby(['family','status']).size().to_string())


if __name__=='__main__':main()

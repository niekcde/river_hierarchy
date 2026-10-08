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
from scipy.spatial import cKDTree
from scipy.optimize import minimize,LinearConstraint,linprog

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_explicit_celerity_regularization import BASE,longest_travel
OUT=BASE/'full_signed_validation'
EXAMPLES=['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']


def allocate_new_heads(f,heads,unknown):
    """Original equal-per-link slope objective, with every inherited head fixed.

    Newly created vertices have no base head to inherit. Solve only their
    connected patches. The target and 1% floor use patch boundary drop divided
    by the patch's longest physical path. No nearest-node head substitution.
    """
    heads=heads.copy();G=nx.Graph()
    for r in f.itertuples():
        if r.u in unknown and r.v in unknown:G.add_edge(r.u,r.v)
    G.add_nodes_from(unknown)
    for component in nx.connected_components(G):
        patch=f[f.u.isin(component)|f.v.isin(component)]
        boundary=(set(patch.u)|set(patch.v))-component
        if len(boundary)<2:raise ValueError('New-node patch lacks two inherited boundary heads')
        H=nx.DiGraph()
        for r in patch.itertuples():H.add_edge(r.u,r.v,weight=r.link_length_m)
        if not nx.is_directed_acyclic_graph(H):raise ValueError('Cyclic new-node patch')
        span=float(heads.loc[list(boundary)].max()-heads.loc[list(boundary)].min())
        path=nx.dag_longest_path_length(H,weight='weight')
        if span<=0 or path<=0:raise ValueError('Nonpositive new-node patch reference drop')
        ref=span/path;ids=sorted(component);idx={v:i for i,v in enumerate(ids)}
        A=np.zeros((len(patch),len(ids)));b=np.zeros(len(patch));center=heads.loc[list(boundary)].min()
        scale=ref*patch.link_length_m.to_numpy()/span
        for i,r in enumerate(patch.itertuples()):
            for v,sign in [(r.u,1),(r.v,-1)]:
                if v in idx:A[i,idx[v]]+=sign/scale[i]
                else:b[i]+=sign*(heads.loc[v]-center)/span/scale[i]
        initial=np.linalg.lstsq(A,1-b,rcond=None)[0]
        if (A@initial+b>=.01-1e-10).all():solution=initial
        else:
            feasible=linprog(np.zeros(len(ids)),A_ub=-A,b_ub=b-.01,bounds=[(None,None)]*len(ids),method='highs')
            if not feasible.success:raise ValueError('Infeasible fixed boundary heads for new-node slope floor')
            initial=feasible.x
            objective=lambda z:float(np.sum((A@z+b-1)**2))
            jac=lambda z:2*A.T@(A@z+b-1)
            result=minimize(objective,initial,jac=jac,constraints=[LinearConstraint(A,.01-b,np.inf)],
                            method='SLSQP',options={'ftol':1e-12,'maxiter':200})
            if not result.success or (A@result.x+b<.01-1e-7).any():
                raise ValueError('New-node constrained head allocation failed: '+result.message)
            solution=result.x
        for v,z in zip(ids,solution):heads.loc[v]=center+span*z
    return heads


def source_inventory(ex):
    suffix='_direction130' if ex=='15' else ''
    rows=[]
    for family in ['indep','compound_bubbles']:
        exp=ROOT/'network_variants/output'/f'{ex}_{family}{suffix}__anchored'
        parents={}
        if (exp/'transition_registry.csv').exists():
            t=pd.read_csv(exp/'transition_registry.csv')
            parents=dict(zip(t.child_state_id,t.parent_state_id))
        for state in sorted((exp/'states').iterdir()):
            if not state.is_dir():continue
            rows.append(dict(example=ex,family=family,state=state.name,
                parent_state=parents.get(state.name,'S000_base'),
                manifest=str(state/'hydraulic_context/hydraulic_context_manifest.json')))
    exp=ROOT/'network_variants/output'/f'{ex}_island{suffix}__anchored'
    reg=exp/'island_hydraulic_context_registry.csv'
    if reg.exists():
        for r in pd.read_csv(reg).itertuples():
            rows.append(dict(example=ex,family='island',state=r.island_id,parent_state='S000_base',
                manifest=str(r.hydraulic_context_manifest_path)))
    return rows


def materialize(ex):
    dest=OUT/ex;dest.mkdir(parents=True,exist_ok=True)
    suffix='_direction130' if ex=='15' else ''
    exp=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored'
    m=json.loads((exp/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json').read_text())
    bn=pyogrio.read_dataframe(m['input_paths']['directed_nodes'])
    xy=np.c_[bn.geometry.x,bn.geometry.y];tree=cKDTree(xy)
    heads=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node')['head']
    base_links=pd.read_parquet(BASE/ex/'floor_0.05/old/links.parquet')
    unit_audit=pd.read_csv(BASE/ex/'floor_0.05/old/units.csv')
    protected=set(unit_audit.source)|set(unit_audit.sink)
    direct=base_links[~base_links.ownership_kind.isin(['simple_unit','compound_bubble'])]
    protected |= set(direct.id_us_node)|set(direct.id_ds_node)
    interiors=[]
    for _,group in base_links[base_links.ownership_kind.isin(['simple_unit','compound_bubble'])].groupby(['ownership_kind','ownership_domain_id']):
        interiors.append((set(group.id_us_node)|set(group.id_ds_node))-protected)
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
        mp=Path(row['manifest']);key=str(mp)
        if key in cache:
            records.append(dict(row,**cache[key],context_alias=True));continue
        f=None
        try:
            cm=json.loads(mp.read_text())
            nodes=pyogrio.read_dataframe(cm['input_paths']['directed_nodes'])
            if nodes.crs!=bn.crs:raise ValueError('Child/base CRS mismatch')
            distance,ix=tree.query(np.c_[nodes.geometry.x,nodes.geometry.y])
            unmatched=int((distance>1e-5).sum())
            mapping=dict(zip(nodes.id_node.astype(int),bn.id_node.to_numpy()[ix].astype(int)))
            unknown=set();next_id=int(heads.index.max())+1
            for child_id in nodes.id_node.to_numpy()[distance>1e-5]:
                mapping[int(child_id)]=next_id;unknown.add(next_id);next_id+=1
            h=pd.read_csv(mp.parent/'link_hydraulic_context.csv')
            widths=pyogrio.read_dataframe(cm['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False)
            f=h[['id_link','id_us_node','id_ds_node','link_length_m','slope_used']].merge(widths,on='id_link',validate='one_to_one')
            if len(f)!=len(h):raise ValueError('Incomplete width match')
            f['u']=f.id_us_node.map(mapping);f['v']=f.id_ds_node.map(mapping);f['width']=f.wid_adj_routing
            state_heads=allocate_new_heads(f,heads,unknown) if unknown else heads
            f['head_us']=f.u.map(state_heads);f['head_ds']=f.v.map(state_heads)
            f['slope']=(f.head_us-f.head_ds)/f.link_length_m
            reallocated=set()
            bad=f[f.slope<=0]
            if len(bad):
                affected=set(bad.u)|set(bad.v)
                for internal in interiors:
                    if affected&internal:reallocated|=internal
                reallocated &= set(f.u)|set(f.v)
                if reallocated:
                    state_heads=allocate_new_heads(f,heads,unknown|reallocated)
                    f['head_us']=f.u.map(state_heads);f['head_ds']=f.v.map(state_heads)
                    f['slope']=(f.head_us-f.head_ds)/f.link_length_m
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
            if not output.exists():f.to_parquet(output,index=False)
            rec=dict(status='ready',reason='',frame=str(output),fingerprint=fingerprint,n_links=len(f),n_segments=int(f['count'].sum()),
                max_travel_seconds=longest_travel(f),upper_dt_seconds=float(1.8*f.rapid_k.min()),
                high_cap_links=int((f.rapid_celerity_mps_raw>5).sum()),
                new_nodes_allocated=unmatched,max_nearest_base_node_distance_m=float(distance.max()),
                inherited_internal_nodes_reallocated=len(reallocated),
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
    OUT.mkdir(exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(materialize,sys.argv[1:] or EXAMPLES))
    inv=pd.concat([pd.read_csv(p,dtype={'example':str}) for p in OUT.glob('*/inventory.csv')],ignore_index=True)
    inv.to_csv(OUT/'inventory.csv',index=False)
    pd.concat([pd.read_csv(p,dtype={'example':str}) for p in OUT.glob('*/saved_settings.csv')],ignore_index=True).to_csv(OUT/'saved_settings.csv',index=False)
    print(inv.groupby(['family','status']).size().to_string())


if __name__=='__main__':main()

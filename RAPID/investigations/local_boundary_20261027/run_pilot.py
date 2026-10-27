"""Read-only source audit and three-case local boundary experiment."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import numpy as np
import pandas as pd
from network_variants.reviewed_geometry import ReviewedGeometry
from local_contract import exterior_map,correct,ContractError,restore_named_exterior,port_path_replacement
from rapid_tools.cunge import route
from rapid_tools.k_values import compute_dt_bounds
OUT=ROOT/'RAPID/outputs/local_boundary_20261027_v2'
SOURCE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    audit=json.loads((SOURCE/'retained_slope_audit_all_networks_20261027/audit.json').read_text())
    changed=[]
    for p,h in audit['source_hashes'].items():
        if digest(p)!=h:changed.append(p)
    summary=pd.read_csv(SOURCE/'retained_slope_audit_all_networks_20261027/state_summary.csv',dtype={'example':str})
    totals=dict(cases=len(summary),networks=summary.example.nunique(),changed=int(summary.changed_retained_links.gt(0).sum()),outside=int(summary.changed_outside_saved_footprint.gt(0).sum()),over1pct=int(summary.changed_over_1pct.gt(0).sum()))
    (OUT/'audit_verification.json').write_text(json.dumps(dict(totals=totals,source_files_checked=len(audit['source_hashes']),changed_sources=changed,by_type=summary.groupby('edit_type').agg(cases=('state','size'),changed=('changed_retained_links',lambda v:int(v.gt(0).sum()))).to_dict('index')),indent=2))
    jobs={}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json',SOURCE/'routing_plan.json']:
        for c in json.loads(path.read_text())['cases']:
            for j in c['jobs']:jobs[c['example'],j['state_id']]=j
    inv=pd.read_parquet(SOURCE/'validation_group_inventory_20261023/records.parquet')
    parent=pd.read_parquet(jobs['22','S000_base']['parameters']);r=ReviewedGeometry();pg,_,_,_,_=r.load('22','indep','S000_base')
    records=[];successful={};hashes={}
    for state,family in [('S013_unit_8','indep'),('S296_unit_30','indep'),('island_22_018','island'),('island_22_041','island')]:
        job=jobs['22',state];path=job['parameters'];hashes[path]=digest(path);child=pd.read_parquet(path)
        row=inv[inv.example.eq('22')&inv.state.eq(state)].iloc[0]
        cg,_,contract,_,_=r.load('22',family,state)
        selected=list(map(int,contract.get('selected_parent_links',row.parent_footprint_links)))
        rec=dict(state=state,edit_type=row.edit_type,selected_count=len(selected),scope_source='reviewed contract' if contract else 'conservative inventory footprint',source_parameters=path)
        folder=OUT/state;folder.mkdir();(folder/'selected_parent_links.json').write_text(json.dumps(selected))
        try:
            mapping=exterior_map(parent,child,pg,cg,selected)
            corrected,qa=correct(parent,child,mapping,selected)
            corrected.to_parquet(folder/'parameters.parquet',index=False)
            (folder/'exterior_mapping.json').write_text(json.dumps(mapping))
            cp=child.set_index('id_link').loc[list(mapping)];pp=parent.set_index('id_link').loc[list(mapping.values())]
            for k in ['slope','reference_discharge_cms','width','depth_m','rapid_k','rapid_x','split_fraction','count']:
                rec['existing_changed_exterior_'+k]=int((~np.isclose(cp[k],pp[k],rtol=1e-8,atol=1e-10)).sum())
                rec['corrected_changed_exterior_'+k]=0
            rec.update(status='passed',**qa);successful[state]=(child,corrected)
        except (ValueError,AssertionError) as e:rec.update(status='blocked',reason=str(e))
        records.append(rec);print(json.dumps(rec),flush=True)
        if state in ['S296_unit_30','island_22_041']:
            restored={1010:208,1012:225} if state=='S296_unit_30' else {1008:148}
            try:
                trial,geometry,actions=restore_named_exterior(parent,child,pg,cg,restored)
                (folder/'restoration_actions.json').write_text(json.dumps(actions,indent=2))
                mm=exterior_map(parent,trial,pg,geometry,selected)
                ff,qa=correct(parent,trial,mm,selected)
                records.append(dict(state=state,variant='restored_saved_topology',status='passed',**qa))
            except (ValueError,AssertionError) as e:
                records.append(dict(state=state,variant='restored_saved_topology',status='blocked',reason=str(e)))
            trial,geometry,lineage=port_path_replacement(parent,pg,selected)
            mm=exterior_map(parent,trial,pg,geometry,selected)
            ff,qa=correct(parent,trial,mm,selected)
            ff.to_parquet(folder/'port_path_parameters.parquet',index=False)
            geometry.to_parquet(folder/'port_path_geometry.parquet',index=False)
            (folder/'port_path_lineage.json').write_text(json.dumps(lineage,indent=2))
            records.append(dict(state=state,variant='port_path_reduction',status='passed',original_selected_links=len(selected),**qa))
            successful[state]=(child,ff)
            print('Port-path reduction passed',state,qa,flush=True)
    (OUT/'pilot.json').write_text(json.dumps(records,indent=2))
    # A controlled pulse isolates routing response from event-window choices.
    frames={'parent':parent}
    for state,(old,new) in successful.items():frames[state+'_existing']=old;frames[state+'_local']=new
    results=[]
    if successful:
        step=300.;upper=min(compute_dt_bounds(f.rapid_k,x_value=f.rapid_x,coefficient_policy='signed_inflow')[1] for f in frames.values())
        nsub=int(np.ceil(step/min(15.,.99*upper)));dt=step/nsub
        # Long common tail derived from maximum path mean and variance.
        import networkx as nx
        def tail(f):
            g=nx.DiGraph();g.add_edges_from(zip(f.u,f.v));travel={v:0. for v in g};variance=travel.copy()
            for u in nx.topological_sort(g):
                for z in f[f.u.eq(u)].itertuples():
                    travel[z.v]=max(travel[z.v],travel[u]+z.count*z.rapid_k)
                    variance[z.v]=max(variance[z.v],variance[u]+z.count*z.rapid_k**2*(1-2*z.rapid_x))
            return max(travel.values())+12*np.sqrt(max(variance.values()))
        duration=48*3600+max(tail(f) for f in frames.values());t=np.arange(int(np.ceil(duration/step))+1)*step
        flow=100+100*np.maximum(0,1-abs(t-24*3600)/(12*3600))
        signals=pd.DataFrame({'time_seconds':t,'inflow_cms':flow})
        for name,f in frames.items():
            for refinement in [1,2]:
                q,a=route(f,flow,nsub*refinement,dt/refinement)
                excess=q-100;mass=np.sum(excess)*step;centroid=float(np.sum(t*excess)/np.sum(excess))
                results.append(dict(name=name,refinement=refinement,dt=dt/refinement,peak_cms=float(q.max()),peak_seconds=float(t[q.argmax()]),centroid_seconds=centroid,volume_m3=float(mass),relative_volume_error=float(mass/(np.sum(flow-100)*step)-1),tail_excess_cms=float(excess[-1]),**a))
                if refinement==1:signals[name]=q
            print('Routed',name,flush=True)
        signals.to_parquet(OUT/'pulse_hydrographs.parquet',index=False)
        pd.DataFrame(results).to_csv(OUT/'routing_responses.csv',index=False)
    assert all(digest(p)==h for p,h in hashes.items())
    (OUT/'manifest.json').write_text(json.dumps(dict(complete=True,production_modified=False,rf_fits=0,source_hashes=hashes,geometry_release=r.release,source_audit_mismatches=changed),indent=2))

if __name__=='__main__':main()

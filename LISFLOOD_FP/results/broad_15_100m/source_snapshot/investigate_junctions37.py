"""Isolated diagnostics; never changes the broad-study equations or results.
The alternative junction widths are a sensitivity hypothesis, not a validated fix.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import broad_study as b
from check_combined_junction import network

OUT=b.OUT/'investigation';OUT.mkdir(exist_ok=True)

def partition_widths(cells,m):
    f=cells.reset_index(drop=True);changes=[]
    for node in set(f.u)&set(f.v):
        incoming=np.flatnonzero(f.v.to_numpy()==node);outgoing=np.flatnonzero(f.u.to_numpy()==node)
        if len(incoming)<2 or len(outgoing)<2:continue
        wu=float(f.width.iloc[incoming].sum());wd=float(f.width.iloc[outgoing].sum())
        mask=np.isin(m['a'],incoming)&np.isin(m['b'],outgoing)
        for j in np.flatnonzero(mask):
            old=m['wf'][j];new=m['w'][m['a'][j]]*m['w'][m['b'][j]]/max(wu,wd)
            changes.append(dict(node=int(node),face=int(j),old_width=float(old),partitioned_width=float(new),total_incoming_width=wu,total_outgoing_width=wd))
            m['wf'][j]=new
    return changes

def load_initial(r):
    with np.load(Path(r['folder'])/'initial.npz') as d:return d['h'].copy(),d['q'].copy()

def static_checks():
    rows=[];summaries=[];reaches=[]
    for ex in ['15','22','48','37']:
        records=[('mc_roughness',json.loads((b.OUT/ex/'baseline.json').read_text()))]
        cal=json.loads((b.OUT/ex/'calibration.json').read_text())
        if cal.get('final',{}).get('valid'):records.append(('selected_roughness',cal['final']))
        if ex=='37':records += [('selected_child',r) for r in json.loads((b.OUT/ex/'transfer.json').read_text())['runs']]
        for label,r in records:
            f=b.frame(ex,r['state']);cells=b.v.grid(f,'max100').reset_index(drop=True);m=b.build(cells,r['n']);h,q=load_initial(r);N=len(cells);eta=m['z'][:N]+h
            incoming_count=cells.groupby('v').size();outgoing_count=cells.groupby('u').size()
            interior=m['b']<N;js=np.flatnonzero(interior);a=m['a'][js];z=m['b'][js]
            hf=np.maximum(np.maximum(eta[a],eta[z])-np.maximum(m['z'][a],m['z'][z]),1e-12);slope=(eta[a]-eta[z])/m['dxf'][js]
            froude=abs(q[js])/(hf*np.sqrt(9.80665*hf));freefr=hf**(1/6)*np.sqrt(abs(slope))/(r['n']*np.sqrt(9.80665))
            for k in np.flatnonzero(froude>.98):
                j=js[k];u=int(a[k]);v=int(z[k]);node=cells.v.iloc[u]
                rows.append(dict(example=ex,state=r['state'],setting=label,n=r['n'],face=int(j),up_reach=int(cells.id_link.iloc[u]),down_reach=int(cells.id_link.iloc[v]),same_reach=bool(cells.id_link.iloc[u]==cells.id_link.iloc[v]),combined_junction=bool(incoming_count.get(node,0)>1 and outgoing_count.get(node,0)>1),up_width=m['w'][u],down_width=m['w'][v],face_width=m['wf'][j],up_depth=h[u],down_depth=h[v],bed_slope=(m['z'][u]-m['z'][v])/m['dxf'][j],water_surface_slope=slope[k],actual_face_froude=froude[k],unlimited_equilibrium_face_froude=freefr[k]))
            summary=dict(example=ex,state=r['state'],setting=label,n=r['n'],storage_m3=float(np.sum(h*m['w']*m['dx'])),Q_initial=float(b.forcing(ex).forcing_cms.iloc[0]),max_depth=float(h.max()),max_face_froude=float(froude.max()),reverse_faces=int((q[js]<-1e-8).sum()))
            summaries.append(summary)
            if ex=='37' and label!='mc_roughness':
                Q=summary['Q_initial'];outq=np.bincount(m['a'],weights=q*m['wf'],minlength=N)
                for reach,group in cells.groupby('id_link',sort=False):
                    ids=group.index.to_numpy();last=ids[-1];ref=f[f.id_link.eq(reach)].iloc[0]
                    reaches.append(dict(state=r['state'],id_link=int(reach),storage_m3=float(np.sum((h*m['w']*m['dx'])[ids])),length_m=float(group.link_length_m.sum()),width=float(ref.width),head_us=float(ref.head_us),head_ds=float(ref.head_ds),outflow_cms=float(outq[last]),local_flow_fraction=float(outq[last]/Q),mc_reference_flow_fraction=float(ref.reference_flow_fraction),mean_depth=float(np.average(h[ids],weights=m['dx'][ids]))))
    pd.DataFrame(rows).to_csv(OUT/'limiter_face_diagnosis.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'initial_storage_diagnosis.csv',index=False);pd.DataFrame(reaches).to_csv(OUT/'37_initial_reach_hydraulics.csv',index=False)
    # Independent equal-channel benchmark for the hypothesized interface partition.
    f=network(True);cells=b.v.grid(f,'max100');m=b.build(cells,.025);changes=partition_widths(cells,m);depth=2.;unit=depth**(5/3)*np.sqrt(.001)/.025;Q=200*unit
    result=b.v.solve(m,[0,86400],[Q,Q],(np.full(len(cells),depth),np.full(len(m['a']),unit)),cfl=.35,max_dt=150)
    b.save(OUT/'partitioned_junction_uniform_check.json',dict(changes=changes,depth_error_m=float(abs(result['h']-depth).max()),audit=result['audit'],interpretation='Width partition restores the symmetric benchmark; this alone does not validate asymmetric or real junction hydraulics.'))
    print('STATIC AND SYNTHETIC CHECKS COMPLETE',flush=True)

def variant(state,ex='37',bed_source=None,partition=True):
    folder=OUT/(ex+('_partitioned_' if partition else '_current_beds_')+state);folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    if (folder/'wave.parquet').exists():
        from recover_hydraulic_metadata import recover
        return recover(folder,ex,state,bed_source,partition)
    start=time.monotonic();cal=json.loads((b.OUT/ex/'calibration.json').read_text());record=cal['final'] if state=='S000_base' else next(r for r in json.loads((b.OUT/ex/'transfer.json').read_text())['runs'] if r['state']==state)
    f=pd.read_parquet(bed_source) if bed_source else b.frame(ex,state)
    cells=b.v.grid(f,'max100');m=b.build(cells,cal['selected_n']);changes=partition_widths(cells,m) if partition else [];initial=load_initial(record)
    assert initial[0].shape==(len(cells),) and initial[1].shape==m['a'].shape
    # Preserve volumetric interface flows in the initial guess after width changes.
    for c in changes:initial[1][c['face']]*=c['old_width']/c['partitioned_width']
    Q=float(b.forcing(ex).forcing_cms.iloc[0])
    try:
        for day in range(1,61):
            spun=b.v.solve(m,[0,86400],[Q,Q],initial,cfl=.35,max_dt=150);initial=(spun['h'],spun['q'])
            if spun['audit']['final_node_imbalance_cms']/max(Q,1)<1e-8:break
        else:raise RuntimeError('Steady initialization did not converge')
        np.savez(folder/'initial.npz',h=initial[0],q=initial[1]);forcing=b.forcing(ex);origin=forcing.interval_start_utc.iloc[0]
        t=np.r_[(forcing.interval_start_utc-origin).dt.total_seconds(),(forcing.interval_end_utc.iloc[-1]-origin).total_seconds()];vals=np.r_[forcing.forcing_cms,forcing.forcing_cms.iloc[-1]]
        m['abort_on_invalid']=True;sample=np.unique(np.linspace(0,len(t)-1,100,dtype=int));result=b.v.solve(m,t,vals,initial,cfl=.35,max_dt=150,samples=sample)
        means=np.diff(result['wave'][:,4])/60;w=forcing[['time_utc','phase']].copy();w['q_outlet_cms']=means;w.to_parquet(folder/'wave.parquet',index=False)
        np.savez_compressed(folder/'sampled_fields.npz',time_seconds=t[sample],depth=result['depth'],flow=result['flow'],reach_ids=cells.id_link.to_numpy())
        event=forcing[forcing.phase.eq('input_event')];seconds=(forcing.time_utc-origin).dt.total_seconds().to_numpy();baseline=np.interp(seconds,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[event.forcing_cms.iloc[0],event.forcing_cms.iloc[-1]])
        relative=float(result['audit']['maximum_mass_residual_m3']/result['wave'][-1,3])
        r=dict(example=ex,state=state,valid=result['audit']['limited_face_steps']==0 and result['audit']['clipping_volume_m3']<1e-6 and relative<1e-9,n=cal['selected_n'],bed_source=str(bed_source) if bed_source else None,bed_source_sha256=b.sha(bed_source) if bed_source else None,partitioned_junction=partition,spinup_days=day,relative_mass_residual=relative,changes=changes,seconds=time.monotonic()-start,folder=str(folder),**b.v.metrics(seconds,means,baseline),**result['audit'])
    except Exception as exc:r=dict(state=state,valid=False,error=str(exc),seconds=time.monotonic()-start,folder=str(folder))
    b.save(folder/'result.json',r);print('JUNCTION VARIANT',state,r['valid'],r['seconds'],flush=True);return r

if __name__=='__main__':
    b.v.OUT=OUT
    static_checks()
    records=[variant(s) for s in ['S000_base','S274_unit_280']]
    b.save(OUT/'37_partitioned_junction_results.json',dict(runs=records,interpretation='Frozen n; only combined-junction interface widths change. Diagnostic sensitivity, not an accepted production correction.'))

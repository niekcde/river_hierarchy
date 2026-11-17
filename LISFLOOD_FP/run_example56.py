"""Reproducible cached example-56 pilot; all writes stay in LISFLOOD_FP/results."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from diffusive_network import Network

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/56'
FORCING=ROOT/'RAPID/outputs/master_workflow_verification_20261009/calibration/bounded_beta_20261009/56/observed_upstream.parquet'
STATES={'base':'c2578749d698d194b631','collapse_S014_unit_10':'c3894ba4eb9cbdccfc79'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dx',type=float,default=200.)
    p.add_argument('--max-step',type=float,default=600.)
    p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--n',type=float,default=.035)
    p.add_argument('--tag',default='dx200')
    args=p.parse_args()
    out=ROOT/'LISFLOOD_FP/results/example56'/args.tag
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_parquet(FORCING).sort_values('time')
    if raw.time.duplicated().any() or not np.isfinite(raw.discharge_cms).all() or (raw.discharge_cms<=0).any():
        raise ValueError('Invalid forcing samples')
    origin=raw.time.iloc[0]
    observed_t=(raw.time-origin).dt.total_seconds().to_numpy()
    observed_q=raw.discharge_cms.to_numpy()
    end=observed_t[-1]+4*86400.
    times=np.unique(np.r_[np.arange(0,end,300.),end])
    inflow=lambda t:np.interp(t,observed_t,observed_q)
    baseline=np.interp(times,[0,observed_t[-1]],[observed_q[0],observed_q[-1]])
    hydro=pd.DataFrame({'time_seconds':times,'time_utc':origin+pd.to_timedelta(times,unit='s'),
                        'inflow_cms':inflow(times),'metric_baseline_cms':baseline})
    frames={name:pd.read_parquet(CACHE/(key+'.parquet')) for name,key in STATES.items()}
    keys=['u','v','link_length_m','width','head_us','head_ds']
    base,child=frames.values()
    unchanged=base.merge(child,on=keys)
    removed=base.merge(child[keys],on=keys,how='left',indicator=True).query('_merge=="left_only"')
    added=child.merge(base[keys],on=keys,how='left',indicator=True).query('_merge=="left_only"')
    if len(unchanged)!=41 or len(removed)!=5 or len(added)!=1:
        raise ValueError('Cached collapse does not match expected controlled comparison')
    removed[keys].to_csv(out/'removed_links.csv',index=False)
    added[keys].to_csv(out/'replacement_link.csv',index=False)
    manifest=dict(settings=vars(args),solver='custom diffusive-wave graph finite volume + scipy BDF',
        geometry_status='Historical local cache; active master drive unavailable; not certified latest production geometry',
        bed='cached reconstructed water-surface heads used directly as fixed bed elevations',
        cross_section='rectangular, fixed width, unlimited banks, no floodplain or lateral inputs',
        boundary='observed upstream Q, normal-flow outlet with unchanged terminal width/slope',
        forcing=dict(path=str(FORCING.relative_to(ROOT)),sha256=sha(FORCING),origin_utc=str(origin),
            end_observed_utc=str(raw.time.iloc[-1]),interpolation='piecewise linear point discharge',
            maximum_gap_seconds=float(np.diff(observed_t).max()),gaps_over_15min=int((np.diff(observed_t)>900).sum()),
            tail_hours=96,tail='hold final observed Q',initial_discharge_cms=float(observed_q[0]),
            peak_discharge_cms=float(observed_q.max()),peak_utc=str(raw.time.iloc[np.argmax(observed_q)])),
        unchanged_links=41,collapsed_domain_nodes=[20,19,18,17],states={},
        source_hashes={str((CACHE/(key+'.parquet')).relative_to(ROOT)):sha(CACHE/(key+'.parquet')) for key in STATES.values()},
        code_hashes={name:sha(Path(__file__).with_name(name)) for name in ['diffusive_network.py','run_example56.py']})
    for name,f in frames.items():
        start=time.perf_counter()
        m=Network.from_frame(f,dx=args.dx,roughness=args.n)
        print(f'{args.tag} {name}: {len(f)} links, {len(m.bed)} nodes, spinup',flush=True)
        initial,spin=m.spinup(float(observed_q[0]))
        print(f'{args.tag} {name}: initialized after {spin["days"]} days; routing',flush=True)
        sol=m.integrate(initial,times,inflow,max_step=args.max_step,rtol=args.rtol)
        h=sol.y[:-2]
        flow=np.array([m.fluxes(x)[0] for x in h.T])
        outlet=np.array([m.fluxes(x)[1] for x in h.T])
        storage=m.storage_area@h
        residual=storage-storage[0]-(sol.y[-2]-sol.y[-1])
        hydro[name+'_cms']=outlet
        hydro[name+'_storage_m3']=storage
        hydro[name+'_mass_residual_m3']=residual
        excess=np.maximum(outlet-baseline,0.)
        volume=np.trapz(excess,times)
        peak=int(np.argmax(outlet))
        area=m.width[None,:]*.5*(h[m.a].T+h[m.b].T)
        froude=abs(flow/area)/np.sqrt(9.81*.5*(h[m.a].T+h[m.b].T))
        # Endpoint flows for each original link: preserve actual dynamic junction allocation.
        traces=[]
        for link,r in f.reset_index(drop=True).iterrows():
            ids=np.flatnonzero(m.faces.link.to_numpy()==link)
            traces.append(pd.DataFrame({'time_seconds':times,'link':link,'u':int(r.u),'v':int(r.v),
                'upstream_cms':flow[:,ids[0]],'downstream_cms':flow[:,ids[-1]]}))
        pd.concat(traces,ignore_index=True).to_parquet(out/(name+'_link_flows.parquet'),index=False)
        m.faces.to_csv(out/(name+'_mesh.csv'),index=False)
        pd.DataFrame({'node_id':m.node_ids,'bed_m':m.bed,'storage_area_m2':m.storage_area,
            'initial_depth_m':initial,'minimum_depth_m':h.min(axis=1),'maximum_depth_m':h.max(axis=1),
            'final_depth_m':h[:,-1]}).to_csv(out/(name+'_nodes.csv'),index=False)
        state=dict(links=len(f),mesh_faces=len(m.faces),mesh_nodes=len(m.bed),spinup=spin,
            peak_cms=float(outlet[peak]),peak_seconds=float(times[peak]),peak_utc=str(hydro.time_utc.iloc[peak]),
            peak_lag_from_inflow_hours=float((times[peak]-observed_t[np.argmax(observed_q)])/3600),
            centroid_seconds=float(np.trapz(times*excess,times)/volume),
            centroid_definition='positive excess above straight line from first to last observed inflow; constant over tail',
            positive_excess_volume_m3=float(volume),cumulative_inflow_m3=float(sol.y[-2,-1]),
            cumulative_outflow_m3=float(sol.y[-1,-1]),storage_change_m3=float(storage[-1]-storage[0]),
            maximum_mass_residual_m3=float(abs(residual).max()),
            maximum_mass_residual_fraction=float(abs(residual).max()/sol.y[-2,-1]),
            minimum_depth_m=float(h.min()),maximum_depth_m=float(h.max()),maximum_froude=float(froude.max()),
            minimum_face_discharge_cms=float(flow.min()),
            final_outlet_cms=float(outlet[-1]),final_inflow_cms=float(observed_q[-1]),
            final_relative_node_imbalance=float(np.max(abs(m.rhs(end,sol.y[:,-1],inflow)[:-2]*m.storage_area))/observed_q[-1]),
            nfev=sol.nfev,njev=sol.njev,nlu=sol.nlu,elapsed_seconds=time.perf_counter()-start)
        manifest['states'][name]=state
        hydro.to_csv(out/'hydrographs.csv',index=False)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    a,b=manifest['states'].values()
    manifest['comparison']=dict(peak_change_cms=b['peak_cms']-a['peak_cms'],
        peak_change_percent=100*(b['peak_cms']/a['peak_cms']-1),
        peak_time_shift_minutes=(b['peak_seconds']-a['peak_seconds'])/60,
        positive_excess_centroid_shift_minutes=(b['centroid_seconds']-a['centroid_seconds'])/60,
        maximum_absolute_hydrograph_difference_cms=float(abs(hydro.iloc[:,0]*0+hydro.collapse_S014_unit_10_cms-hydro.base_cms).max()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['comparison']),flush=True)


if __name__=='__main__':main()

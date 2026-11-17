"""Local-inertial experiment using audited MC cells, roughness and forcing."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from local_inertial_pilot56 import geometry,route,self_check
from diffusive_network import normal_depth
from calibrate_boundaries56 import ROOT,peak

OUT=ROOT/'LISFLOOD_FP/results/example56/local_inertial_mc_grid'
SAVED=ROOT/'RAPID/outputs/event_window_runs_20261013/56/tail_216h'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(f):
    rows=[];virtual=int(min(f.u.min(),f.v.min(),0))-1
    for r in f.itertuples():
        count=int(r.count);start=int(r.u)
        assert np.isclose(r.rapid_effective_length_m*count,r.link_length_m)
        for j in range(count):
            end=int(r.v) if j==count-1 else virtual
            if j<count-1:virtual-=1
            rows.append(dict(id_link=int(r.id_link),subsection=j+1,u=start,v=end,
                link_length_m=float(r.rapid_effective_length_m),width=float(r.width),
                head_us=float(r.head_us+(r.head_ds-r.head_us)*j/count),
                head_ds=float(r.head_us+(r.head_ds-r.head_us)*(j+1)/count)))
            start=end
    cells=pd.DataFrame(rows)
    assert len(cells)==int(f['count'].sum())
    for r in f.itertuples():
        c=cells[cells.id_link==r.id_link]
        assert np.isclose(c.link_length_m.sum(),r.link_length_m)
        assert (c.width==r.width).all()
        assert c.iloc[0].u==r.u and c.iloc[-1].v==r.v
        assert c.iloc[0].head_us==r.head_us and np.isclose(c.iloc[-1].head_ds,r.head_ds)
    assert np.isclose((cells.width*cells.link_length_m).sum(),(f.width*f.link_length_m).sum())
    return cells


def main():
    OUT.mkdir(exist_ok=True);self_check()
    plan=json.loads((ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
    case=next(c for c in plan['cases'] if c['example']=='56')
    forcing=pd.read_parquet(SAVED/'forcing.parquet')
    receipt=json.loads((SAVED.parent/'completion.json').read_text())
    assert sha(SAVED/'forcing.parquet')==receipt['files'][str(SAVED/'forcing.parquet')]
    origin=forcing.interval_start_utc.iloc[0]
    times=np.r_[(forcing.interval_start_utc-origin).dt.total_seconds(),(forcing.interval_end_utc.iloc[-1]-origin).total_seconds()]
    assert np.all(np.diff(times)==60)
    vals=np.r_[forcing.forcing_cms.to_numpy(),forcing.forcing_cms.iloc[-1]]
    event=forcing[forcing.phase.eq('input_event')]
    baseline=np.interp((forcing.time_utc-origin).dt.total_seconds(),
        [(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],
        [vals[0],vals[-1]])
    summaries=[];sources={};stage=None
    for state in ['S000_base','S014_unit_10']:
        job=next(j for j in case['jobs'] if j['state_id']==state)
        path=Path(job['parameters']);assert sha(path)==case['source_sha256'][str(path)]
        f=pd.read_parquet(path);f.to_parquet(OUT/(state+'_mc_parameters.parquet'),index=False)
        sources[state]=dict(path=str(path),sha256=sha(path),physical_reaches=len(f),sections=int(f['count'].sum()))
        n=float(f.n_manning.iloc[0]);assert np.allclose(f.n_manning,n)
        cells=expand(f);cells.to_csv(OUT/(state+'_matched_cells.csv'),index=False)
        assert len(cells)==job['n_segments']
        if stage is None:
            sink=next(iter(set(f.v)-set(f.u)));r=f[f.v==sink].iloc[0]
            stage=float(r.head_ds+normal_depth(vals[0],r.width,(r.head_us-r.head_ds)/r.link_length_m,n))
        for grid,refinement in [('mc',False),('physical',False),('mc',True)]:
            tag=state+'_'+grid+('_refined' if refinement else '')
            record=OUT/(tag+'.json')
            if record.exists():summaries.append(json.loads(record.read_text()));continue
            geom=cells if grid=='mc' else f
            m=geometry(geom,stage,n);dt=150 if refinement else 300;cfl=.35 if refinement else .7
            initial=None
            for day in range(1,61):
                _,_,initial,audit=route(m,np.array([0.,86400.]),np.array([0.,86400.]),np.full(2,vals[0]),initial,max_dt=dt,cfl=cfl)
                if audit['final_node_imbalance_cms']/vals[0]<1e-7:break
            else:raise RuntimeError('Spinup failed')
            print(tag,'spinup days',day,flush=True)
            wave,depth,final,audit=route(m,times,times,vals,initial,max_dt=dt,cfl=cfl,forcing_kind='hold')
            means=np.diff(wave.outflow_volume_m3)/60
            w=forcing[['interval_start_utc','interval_end_utc','time_utc','phase']].copy();w['q_outlet_cms']=means
            w.to_parquet(OUT/(tag+'.parquet'),index=False)
            pd.DataFrame(dict(cell=np.arange(len(geom)),maximum_depth_m=depth.max(axis=0),minimum_depth_m=depth.min(axis=0))).to_csv(OUT/(tag+'_depths.csv'),index=False)
            t=(w.time_utc-origin).dt.total_seconds().to_numpy();tp,qp=peak(t,means)
            excess=np.maximum(means-baseline,0);centroid=float(np.trapz(t*excess,t)/np.trapz(excess,t))
            error=float(wave.inflow_volume_m3.iloc[-1]-60*forcing.forcing_cms.sum())
            assert abs(error)/(60*forcing.forcing_cms.sum())<1e-10
            assert audit['maximum_mass_residual_m3']<1 and audit['clipping_volume_m3']<1e-4
            result=dict(state=state,grid=grid,refined=refinement,cells=len(geom),n=n,fixed_stage_m=stage,spinup_days=day,
                peak_cms=qp,peak_seconds=tp,peak_utc=str(origin+pd.to_timedelta(tp,unit='s')),
                centroid_seconds=centroid,inflow_volume_error_m3=error,**audit)
            record.write_text(json.dumps(result,indent=2));summaries.append(result);print(result,flush=True)
        mc=pd.read_parquet(SAVED/state/'outlet.parquet')
        assert mc.time_utc.equals(forcing.time_utc)
        t=(mc.time_utc-origin).dt.total_seconds().to_numpy();q=mc.q_outlet_cms.to_numpy();tp,qp=peak(t,q)
        excess=np.maximum(q-baseline,0)
        summaries.append(dict(state=state,grid='MC_reference',refined=False,cells=len(cells),n=n,
            peak_cms=qp,peak_seconds=tp,peak_utc=str(origin+pd.to_timedelta(tp,unit='s')),
            centroid_seconds=float(np.trapz(t*excess,t)/np.trapz(excess,t))))
    meta=dict(engine='Python translation, not executed in MATLAB',parameter_sources=sources,
        forcing_path=str(SAVED/'forcing.parquet'),forcing_sha256=sha(SAVED/'forcing.parquet'),
        forcing='exact held 60-second MC interval means; same buffered window; output is 60-second integrated mean',
        origin_utc=str(origin),fixed_stage_m=stage,runs=summaries)
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()

"""Isolated fixed-n/beta rectangular-radius sensitivity; production stays immutable."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/cap_investigation_20261020')]
import run_three_runs_control_fix as r
from rapid_tools.event_window_batch import settling_hours
SOURCE=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')/r.RUNS[2]
DEST=ROOT/'RAPID/outputs/uncapped_event_window_20261020/rectangular_sensitivity_20261029'
SELECTION={'22':['S000_base','island_22_293','S122_unit_216'],'13':['S000_base','S001_unit_23']}

def rectangular(source):
    f=source.copy(); depths=[]
    for x in f.itertuples():
        def discharge(h): return x.width*h/x.n_manning*(x.width*h/(x.width+2*h))**(2/3)*np.sqrt(x.slope)
        hi=2*x.depth_m
        while discharge(hi)<x.reference_discharge_cms: hi*=2
        h=brentq(lambda h: discharge(h)-x.reference_discharge_cms,x.depth_m*.999999,hi,xtol=1e-12)
        depths.append(h)
    h=np.array(depths); B=f.width.to_numpy(); Q=f.reference_discharge_cms.to_numpy()
    f['depth_m']=h
    # Exact derivative of Q=(Bh/n)[Bh/(B+2h)]^(2/3) sqrt(S), divided by dA/dh=B.
    c=Q/(B*h)*(1+(2/3)*B/(B+2*h))
    f['rapid_celerity_mps_raw']=c;f['rapid_celerity_mps']=c;f['celerity_times_n']=c*f.n_manning
    f['rapid_celerity_capped']=False
    f['hydraulic_length_scale_m']=Q/(B*f.slope*c)
    f['rapid_k']=f.rapid_effective_length_m/c
    f['rapid_x']=.5-f.diffusion_multiplier*Q/(2*B*f.slope*c*f.rapid_effective_length_m)
    f['depth_policy']='full_rectangular_Manning_frozen_midpoint_Q'
    np.testing.assert_allclose(B*h/f.n_manning*(B*h/(B+2*h))**(2/3)*np.sqrt(f.slope),Q,rtol=1e-10)
    eps=h*1e-5
    def qfun(d): return B*d/f.n_manning*(B*d/(B+2*d))**(2/3)*np.sqrt(f.slope)
    np.testing.assert_allclose((qfun(h+eps)-qfun(h-eps))/(2*eps*B),c,rtol=1e-8)
    for col in ['width','slope','count','link_length_m','split_fraction','reference_discharge_cms','n_manning','diffusion_multiplier']:
        np.testing.assert_array_equal(f[col],source[col])
    assert (h>=source.depth_m-1e-10).all()
    assert (f.rapid_effective_length_m<=f.cunge_alpha*f.hydraulic_length_scale_m*(1+1e-12)).all()
    return f

def main():
    DEST.mkdir(parents=True,exist_ok=True);r.init_worker(DEST,'isolated_sensitivity')
    metrics=[];auditrows=[];depths=[];hashes={}
    for ex,states in SELECTION.items():
      case=r.CTX['cases'][ex]
      for state in states:
        src=SOURCE/'states'/ex/state;path=src/'parameters.parquet'
        receipt=json.loads((src/'completion.json').read_text())
        for name in ['parameters.parquet','metrics.parquet','outlet.parquet']:
            p=src/name;sha=r.digest(p);assert sha==receipt['files'][str(p)];hashes[str(p)]=sha
        wide=pd.read_parquet(path);exact=rectangular(wide);dt=receipt['native_dt_seconds']
        folder=DEST/ex/state;folder.mkdir(parents=True,exist_ok=True);exact.to_parquet(folder/'rectangular_parameters.parquet',index=False)
        d=wide[['id_link','u','v','width','slope','reference_discharge_cms']].copy()
        d['example']=ex;d['state']=state;d['wide_depth_m']=wide.depth_m;d['rectangular_depth_m']=exact.depth_m
        d['wide_celerity_mps']=wide.rapid_celerity_mps;d['rectangular_celerity_mps']=exact.rapid_celerity_mps;depths.append(d)
        # Match each completed run's time support, including its settled tail.
        outlet=pd.read_parquet(src/'outlet.parquet');force=pd.read_parquet(case['forcing'])
        force=outlet.merge(force[['time_utc','forcing_cms']],on='time_utc',how='left',validate='one_to_one')
        missing=force.forcing_cms.isna();assert force.loc[missing,'phase'].eq('settling').all()
        force.loc[missing,'forcing_cms']=case['final_cms']
        nb=int(force.phase.eq('spinup').sum());initial=float(force.forcing_cms.iloc[0])
        hold=force.loc[force.phase.eq('settling'),'interval_start_utc'].iloc[0]
        job=next(j for j in case['jobs'] if j['state']==state)
        native=pd.read_parquet(src/'metrics.parquet').assign(model='wide',clock='native');metrics.append(native)
        for model,frame,clock in [('rectangular',exact,1),('wide',wide,2),('rectangular',exact,2)]:
            print('Routing',ex,state,model,'clock divisor',clock,flush=True)
            q,audit=r.solve_wave(frame,force.forcing_cms.to_numpy()[nb:],dt/clock,initial)
            required,settled=settling_hours(force.time_utc.iloc[nb:],q,hold,case['final_cms'],case['tolerance_cms'])
            assert settled,('Tail must be extended',ex,state,model)
            q=np.r_[np.full(nb,initial),q]
            m=r.metrics_for(ex,state,job['aliases'],force,q).assign(model=model,clock='native' if clock==1 else 'half_dt');metrics.append(m)
            wave=force[['time_utc']].copy();wave['q_outlet_cms']=q;wave.to_parquet(folder/f'{model}_clock{clock}_outlet.parquet',index=False)
            # Endpoint trapezoidal inlet integral, matching the router's held endpoint forcing.
            flow=force.forcing_cms.to_numpy()[nb:];invol=60*flow.sum()+dt/clock/2*(initial-flow[-1])
            outvol=60*q[nb:].sum()
            equilibrium_storage_change=float((frame['count']*frame.rapid_k*frame.reference_flow_fraction).sum()*(case['final_cms']-initial))
            excessvol=60*(flow-initial).sum()
            auditrows.append(dict(example=ex,state=state,model=model,clock=clock,dt_seconds=dt/clock,settled=settled,
                equilibrium_endpoint_balance_residual_pct=100*(invol-outvol-equilibrium_storage_change)/excessvol,
                balance_note='Uses settled equilibrium storage at end; not an exact transient-storage audit',**audit))
        pd.concat(metrics).to_csv(DEST/'metrics.csv',index=False)
        pd.DataFrame(auditrows).to_csv(DEST/'routing_audit.csv',index=False)
        pd.concat(depths).to_csv(DEST/'reach_depth_changes.csv',index=False)
    m=pd.concat(metrics);responses=[]
    for (ex,model,clock,window),g in m.groupby(['example','model','clock','window']):
        parent=g[g.state.eq('S000_base')].iloc[0]
        for child in g[g.state.ne('S000_base')].itertuples():
            responses.append(dict(example=ex,state=child.state,model=model,clock=clock,window=window,
                centroid_shift_min=(child.centroid_seconds-parent.centroid_seconds)/60,
                peak_shift_min=(child.peak_seconds-parent.peak_seconds)/60,
                peak_excess_change_pct=100*(child.peak_excess_cms/parent.peak_excess_cms-1),
                duration80_change_pct=100*(child.duration80_hours/parent.duration80_hours-1)))
    pd.DataFrame(responses).to_csv(DEST/'paired_responses.csv',index=False)
    # Check inherited geometries outside replacements for unchanged hydraulic inputs/depths.
    matches=[]
    for ex,states in SELECTION.items():
        base=pd.read_parquet(DEST/ex/'S000_base'/'rectangular_parameters.parquet')
        keys=['u','v','link_length_m','width']
        for state in states[1:]:
            f=pd.read_parquet(DEST/ex/state/'rectangular_parameters.parquet')
            z=base.merge(f,on=keys,suffixes=('_parent','_child'),validate='one_to_one')
            changed=np.zeros(len(z),bool)
            for col in ['slope','reference_discharge_cms','n_manning','diffusion_multiplier']:
                changed|=~np.isclose(z[col+'_parent'],z[col+'_child'],rtol=1e-12,atol=1e-12)
            delta=abs(z.depth_m_parent-z.depth_m_child)
            assert (delta[~changed]<1e-9).all()
            matches.append(dict(example=ex,state=state,shared_geometries=len(z),shared_with_changed_hydraulic_inputs=int(changed.sum()),unchanged_inputs_max_depth_difference_m=float(delta[~changed].max()) if (~changed).any() else None))
    r.write_json(DEST/'completion.json',dict(selection=SELECTION,source_hashes=hashes,script_sha256=r.digest(__file__),shared_geometry_checks=matches,
        restrictions=['No recalibration','No slope/geometry/flow-allocation changes','Same spatial grids and production native timesteps; half-dt diagnostics only','Three selected collapses cannot establish all-network robustness']))
    print(pd.DataFrame(responses).query("window == '0.5%' and clock == 'native'").to_string(index=False),flush=True)

if __name__=='__main__':main()

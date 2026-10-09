"""Conditional n calibration at beta=.25; time primary, height assessed afterwards."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
    dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
    out=dest/'46_diffusion_multiplier';freeze=dest/'baseline'
    comp=pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
    row=comp[(comp.example=='46')&(comp.method=='cunge')].iloc[0]
    grid=pd.read_parquet(freeze/Path(row.output_dir).relative_to(ROOT)/'frozen_grid.parquet')
    data=pd.read_parquet(out/'waves.parquet')
    meta=next(x for x in json.loads((dest/'46/60s.json').read_text()) if x['method']=='cunge')
    target=pd.Timestamp(meta['target_utc']);q0=meta['initial_cms'];dt=meta['dt_s'];beta=.25
    valid=data.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid)
    obs=data.downstream_cms.to_numpy();shift=q0-obs[ids[0]]
    score=valid&~data.downstream_gap.to_numpy(bool)
    records=[];cache={}
    def run(n):
        if n in cache:return cache[n]
        f=cunge.parameters(grid,n,method='cunge');f['rapid_x']=.5-beta*(.5-f.rapid_x)
        q,audit=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/dt)),dt,initial_flow=q0)
        assert audit['minimum_native_flow_cms']>=0
        pk=ids[np.argmax(q[ids])];err=(data.time_utc.iloc[pk]-target).total_seconds()/60
        rec=dict(n_manning=n,beta=beta,peak_utc=str(data.time_utc.iloc[pk]),peak_error_minutes=err,peak_cms=q[pk],observed_peak_cms=np.max(obs[ids]),peak_error_pct=100*(q[pk]/np.max(obs[ids])-1),raw_rmse_cms=np.sqrt(np.mean((q[score]-obs[score])**2)),display_shifted_rmse_cms=np.sqrt(np.mean((q[score]-obs[score]-shift)**2)))
        records.append(rec);cache[n]=(err,q,rec)
        print(f'n={n:.10f} error={err:.1f}min peak={q[pk]:.2f}',flush=True)
        return cache[n]
    # Coarse scan protects against assuming a globally monotonic peak operator.
    lo=float(grid.grid_n_min.iloc[0]);hi=float(grid.grid_n_max.iloc[0])
    samples=np.unique(np.r_[np.linspace(lo,hi,13),meta['n_manning']])
    for n in samples:run(float(n))
    brackets=[(float(a),float(b)) for a,b in zip(samples[:-1],samples[1:]) if run(float(a))[0]*run(float(b))[0]<0]
    for a,b in brackets:
        for _ in range(24):
            mid=(a+b)/2;e=run(mid)[0]
            if abs(e)<=.5:break
            if run(a)[0]*e<=0:b=mid
            else:a=mid
    # Timing only selects the trial. Height is not used to select among candidates.
    best=min(records,key=lambda r:abs(r['peak_error_minutes']))
    err,q,_=run(best['n_manning'])
    result=data[['interval_start_utc','interval_end_utc','time_utc','upstream_cms','downstream_cms','forcing_cms','cunge_cms']].copy()
    result['beta025_original_n_cms']=data.beta_0_25_base_cms if 'beta_0_25_base_cms' in data else data['beta_0.25_base_cms']
    result['beta025_refitted_n_cms']=q
    # Check selected fit against half native dt and half cell lengths at equal dt.
    checks=[];half=None
    for name,mult in [('half_dt',1),('half_dx_half_dt',2)]:
        g=grid.copy();g['count']*=mult;g['rapid_effective_length_m']=g.link_length_m/g['count']
        f=cunge.parameters(g,best['n_manning'],method='cunge');f['rapid_x']=.5-beta*(.5-f.rapid_x)
        qc,audit=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(120/dt)),dt/2,initial_flow=q0)
        assert audit['minimum_native_flow_cms']>=0
        ref=q if half is None else half
        checks.append(dict(control=name,max_difference_cms=np.max(abs(qc[valid]-ref[valid])),peak_shift_minutes=(data.time_utc.iloc[ids[np.argmax(qc[ids])]]-data.time_utc.iloc[ids[np.argmax(ref[ids])]]).total_seconds()/60))
        if half is None:half=qc
    fig=make_subplots(rows=2,cols=1,subplot_titles=['Complete event','Crest detail'],vertical_spacing=.15)
    traces=[('Upstream',data.upstream_cms.to_numpy(),0,'#2580a5','dash'),('Downstream — shifted',obs,shift,'#17212b','solid'),('β=1, original n',data.cunge_cms.to_numpy(),0,'#ce3d45','solid'),('β=0.25, original n',result.beta025_original_n_cms.to_numpy(),0,'#dc8a16','dot'),(f'β=0.25, refitted n={best["n_manning"]:.6f}',q,0,'#25965a','solid')]
    for name,values,offset,color,dash in traces:
        for panel in [1,2]:
            fig.add_trace(go.Scatter(x=data.time_utc[valid],y=(values[valid]+offset)/q0,customdata=values[valid],name=name,legendgroup=name,showlegend=panel==1,line=dict(color=color,dash=dash),hovertemplate='%{x}<br>Original Q: %{customdata:.2f} m³/s<br>Shared normalized Q: %{y:.4f}<extra>'+name+'</extra>'),row=panel,col=1)
    for panel in [1,2]:
        fig.add_vline(x=target.timestamp()*1000,line_dash='dot',line_color='#8353ad',row=panel,col=1)
        fig.update_yaxes(title_text='Q / initial forcing; downstream shifted',row=panel,col=1)
        fig.update_xaxes(title_text='UTC; target 3 Sep 12:15',row=panel,col=1)
    fig.update_xaxes(range=[target-pd.Timedelta(days=2),target+pd.Timedelta(days=3)],row=2,col=1)
    fig.update_layout(title='46: β=0.25 with timing-only recalibration of Manning n<br><sup>Same equilibrium start and forcing; height assessed using unshifted observations. Exploratory fit, not production adoption.</sup>',height=1100,template='plotly_white',legend=dict(orientation='h',y=1.06))
    fig.write_html(out/'46_beta025_refitted_n.html',include_plotlyjs=True)
    result.to_parquet(out/'beta025_refitted_n_waves.parquet',index=False)
    pd.DataFrame(records).sort_values('n_manning').to_csv(out/'beta025_n_search.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'beta025_refit_numerical_controls.csv',index=False)
    (out/'beta025_refit.json').write_text(json.dumps(dict(selected=best,original_n=meta['n_manning'],n_bounds=[lo,hi],brackets=len(brackets),selection='Minimum absolute peak timing error among sampled trials; height not used for selection. 13-point coarse scan plus original n, then sign-bracket bisection; not an exhaustive global search.',height_role='Soft assessment; original observed maximum, not baseline-shifted display.',initialization='Unchanged equilibrium at current event start; final antecedent assessment still outstanding.',production_modified=False,numerical_controls=checks,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(best,indent=2));print(checks)
if __name__=='__main__':main()

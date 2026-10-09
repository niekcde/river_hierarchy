"""All-example X diagnostics; frozen n/K, no refitting or production changes."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
    dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
    freeze=dest/'baseline'
    comp=pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
    rows=comp[(comp.method=='cunge')&(comp.example!='11')].sort_values('example')
    records=[]; reaches=[]
    fig=make_subplots(rows=5,cols=3,subplot_titles=list(rows.example),vertical_spacing=.055)
    for panel,r in enumerate(rows.itertuples()):
        ex=r.example
        folder=freeze/Path(r.output_dir).relative_to(ROOT)
        grid=pd.read_parquet(folder/'frozen_grid.parquet')
        f=cunge.parameters(grid,r.n_manning,method='cunge')
        data=pd.read_parquet(dest/ex/'60s.parquet')
        meta=next(m for m in json.loads((dest/ex/'60s.json').read_text()) if m['method']=='cunge')
        dt=meta['dt_s']; initial=meta['initial_cms']; ns=int(round(60/dt))
        valid=data.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid)
        base=data.cunge_cms.to_numpy(); bp=ids[np.argmax(base[ids])]
        qobs=data.downstream_cms.to_numpy(); shift=initial-qobs[ids[0]]
        amp=meta['inlet_amplitude_cms']
        coef=(2*f.rapid_k*(1-f.rapid_x)-dt)/(2*f.rapid_k*(1-f.rapid_x)+dt)
        f['local_decay_days']=-dt/np.log(abs(coef))/86400; f['example']=ex
        reaches.append(f)
        rec=dict(example=ex,n_manning=r.n_manning,reach_count=len(f),x_min=f.rapid_x.min(),x_median=f.rapid_x.median(),x_max=f.rapid_x.max(),pct_x_below_minus10=100*(f.rapid_x < -10).mean(),pct_x_below_minus100=100*(f.rapid_x < -100).mean(),max_local_decay_days=f.local_decay_days.max(),base_peak_cms=base[bp],observed_shifted_peak_cms=np.max(qobs[ids]+shift),base_peak_minus_observed_pct_inlet_amplitude=(base[bp]-np.max(qobs[ids]+shift))/amp*100)
        curves={'Cunge':base,'Downstream (shifted)':qobs+shift}
        for name,mask in [('all_x01',np.ones(len(f),bool)),('x_below_minus100_to01',(f.rapid_x < -100).to_numpy()),('x_below_minus10_to01',(f.rapid_x < -10).to_numpy())]:
            params=f.copy();params.loc[mask,'rapid_x']=.1
            q,audit=route_interval_means(params,data.forcing_cms.to_numpy(),ns,dt,initial_flow=initial)
            pk=ids[np.argmax(q[ids])]
            rec[name+'_peak_cms']=q[pk]
            rec[name+'_peak_increase_pct_inlet_amplitude']=(q[pk]-base[bp])/amp*100
            rec[name+'_max_difference_pct_inlet_amplitude']=np.max(abs(q[ids]-base[ids]))/amp*100
            rec[name+'_peak_shift_minutes']=(data.time_utc.iloc[pk]-data.time_utc.iloc[bp]).total_seconds()/60
            curves[name]=q;data[name+'_cms']=q
            assert audit['minimum_native_flow_cms']>=0
        rec['all_x01_peak_increase_percent_of_base']=100*(rec['all_x01_peak_cms']/rec['base_peak_cms']-1)
        records.append(rec)
        data.to_parquet(dest/ex/'x_audit_controls.parquet',index=False)
        # Display every fifth supported minute plus each curve's peak. Full results in Parquet.
        selected=np.unique(np.r_[ids[::5],ids[-1],*[ids[np.nanargmax(q[ids])] for q in curves.values()]])
        for j,(name,q) in enumerate(curves.items()):
            fig.add_trace(go.Scatter(x=data.time_utc.iloc[selected],y=q[selected]/initial,name=name,legendgroup=name,showlegend=panel==0,line=dict(color=['#ce3d45','#18212b','#2580a5','#dc8a16','#8353ad'][j],width=1.4)),row=panel//3+1,col=panel%3+1)
        print(ex,'done',flush=True)
    pd.DataFrame(records).to_csv(dest/'x_audit_summary.csv',index=False)
    pd.concat(reaches).to_parquet(dest/'x_audit_reaches.parquet',index=False)
    fig.update_layout(title='X sensitivity at unchanged Cunge n and K — diagnostic replacements, not fitted alternatives<br><sup>Shared initial-forcing scale; downstream shifted. Display every 5th minute plus peaks; full minute data saved separately.</sup>',height=2050,template='plotly_white',legend=dict(orientation='h',y=1.04))
    fig.write_html(dest/'15_panel_x_audit.html',include_plotlyjs=True)
    (dest/'x_audit_protocol.json').write_text(json.dumps(dict(examples=list(rows.example),production_modified=False,threshold_note='X < -100 and X < -10 are descriptive screening thresholds, not an admissibility bound. Replacement by 0.1 is a causal diagnostic, not a recommended fix.',held_fixed=['n','K','geometry','reference flow','forcing','initial state','native timestep'],peak_note='Observed-window global maxima only; these do not replace reviewed feature targets. Extreme-only effects are not additive contributions.',display='every fifth minute plus curve peaks; full one-minute controls in per-example Parquet'),indent=2))
    print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()

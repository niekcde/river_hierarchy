"""46 diffusion sensitivity at fixed n and hydraulics, with numerical controls."""
from pathlib import Path
import hashlib,json,sys
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
    freeze=dest/'baseline'; out=dest/'46_diffusion_multiplier';out.mkdir(exist_ok=True)
    comp=pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
    row=comp[(comp.example=='46')&(comp.method=='cunge')].iloc[0]
    folder=freeze/Path(row.output_dir).relative_to(ROOT)
    grid=pd.read_parquet(folder/'frozen_grid.parquet')
    source=dest/'46/60s.parquet';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    data=pd.read_parquet(source)
    meta=next(m for m in json.loads((dest/'46/60s.json').read_text()) if m['method']=='cunge')
    initial=meta['initial_cms']; dt=meta['dt_s']; n=meta['n_manning'];target=pd.Timestamp(meta['target_utc'])
    valid=data.downstream_cms.notna().to_numpy();ids=np.flatnonzero(valid)
    score=valid & ~data.downstream_gap.to_numpy(bool)
    obs=data.downstream_cms.to_numpy();shift=initial-obs[ids[0]];times=data.time_utc
    records=[];checks=[];curves={}
    for beta in [1.,.75,.5,.25]:
        key=f'beta_{beta:g}'
        for variant,subdivision,timestep in [('base',1,dt),('half_dt',1,dt/2),('half_dx_half_dt',2,dt/2)]:
            g=grid.copy()
            g['count']=g['count']*subdivision
            g['rapid_effective_length_m']=g.link_length_m/g['count']
            f=cunge.parameters(g,n,method='cunge')
            original_x=f.rapid_x.copy()
            f['rapid_x']=.5-beta*(.5-original_x)
            np.testing.assert_allclose(f.rapid_celerity_mps*f.rapid_effective_length_m*(.5-f.rapid_x),beta*f.hydraulic_diffusivity_m2_s,rtol=1e-12)
            q,audit=route_interval_means(f,data.forcing_cms.to_numpy(),int(round(60/timestep)),timestep,initial_flow=initial)
            assert audit['minimum_native_flow_cms']>=0
            data[key+'_'+variant+'_cms']=q
            if variant=='base':
                curves[key]=q;pk=ids[np.argmax(q[ids])]
                if beta==1:np.testing.assert_allclose(q,data.cunge_cms,rtol=1e-12)
                rec=dict(beta=beta,n_manning=n,dt_s=timestep,peak_cms=q[pk],peak_utc=str(times.iloc[pk]),peak_error_minutes=(times.iloc[pk]-target).total_seconds()/60,observed_peak_cms=np.max(obs[ids]),shifted_observed_peak_cms=np.max(obs[ids]+shift),peak_minus_raw_observed_cms=q[pk]-np.max(obs[ids]),peak_minus_shifted_observed_cms=q[pk]-np.max(obs[ids]+shift),raw_rmse_cms=np.sqrt(np.mean((q[score]-obs[score])**2)),display_shifted_rmse_cms=np.sqrt(np.mean((q[score]-obs[score]-shift)**2)),outlet_volume_observed_window_m3=np.sum(q[valid])*60,input_volume_observed_window_m3=np.sum(data.forcing_cms.to_numpy()[valid])*60,minimum_native_flow_cms=audit['minimum_native_flow_cms'])
                for part,mask in [('before_target',score&(times<target).to_numpy()),('after_target',score&(times>=target).to_numpy())]:
                    rec[part+'_raw_rmse_cms']=np.sqrt(np.mean((q[mask]-obs[mask])**2))
                records.append(rec)
            else:
                reference=curves[key] if variant=='half_dt' else data[key+'_half_dt_cms'].to_numpy()
                pk=ids[np.argmax(q[ids])];rpk=ids[np.argmax(reference[ids])]
                checks.append(dict(beta=beta,control=variant,comparison='base' if variant=='half_dt' else 'half_dt',max_event_difference_cms=np.max(abs(q[valid]-reference[valid])),max_event_difference_pct_inlet_amplitude=np.max(abs(q[valid]-reference[valid]))/meta['inlet_amplitude_cms']*100,peak_shift_minutes=(times.iloc[pk]-times.iloc[rpk]).total_seconds()/60))
        print(key,'complete',flush=True)
    fig=make_subplots(rows=2,cols=1,subplot_titles=['Complete observed event','Crest comparison (same curves and scale)'],vertical_spacing=.15)
    traces=[('Upstream',data.upstream_cms.to_numpy(),0,'#2580a5','dash'),('Downstream — baseline shifted',obs,shift,'#17212b','solid')]+[(f'β = {b:g}',curves[f'beta_{b:g}'],0,col,'solid') for b,col in zip([1,.75,.5,.25],['#ce3d45','#dc8a16','#8353ad','#25965a'])]
    for name,q,offset,color,dash in traces:
        for panel in [1,2]:
            fig.add_trace(go.Scatter(x=times[valid],y=(q[valid]+offset)/initial,customdata=q[valid],name=name,legendgroup=name,showlegend=panel==1,line=dict(color=color,dash=dash,width=1.7),hovertemplate='%{x}<br>Original Q: %{customdata:.2f} m³/s<br>Shared normalized Q: %{y:.4f}<extra>'+name+'</extra>'),row=panel,col=1)
    for panel in [1,2]:
        fig.add_vline(x=target.timestamp()*1000,line_color='#8353ad',line_dash='dot',row=panel,col=1)
        fig.update_yaxes(title_text='Q / initial forcing (downstream shifted)',row=panel,col=1)
        fig.update_xaxes(title_text='UTC — dotted line: calibration target, 3 Sep 12:15',row=panel,col=1)
    fig.update_xaxes(range=[target-pd.Timedelta(days=2),target+pd.Timedelta(days=4)],row=2,col=1)
    fig.update_layout(title=f'46: reduce diffusion, keep n = {n:.8f} and K fixed<br><sup>Equilibrium at current event start; 60 s means. Downstream display shift +{shift:.2f} m³/s; hover retains original Q. No refit.</sup>',template='plotly_white',height=1100,legend=dict(orientation='h',y=1.06),hovermode='closest')
    fig.write_html(out/'46_diffusion_multiplier.html',include_plotlyjs=True)
    data.to_parquet(out/'waves.parquet',index=False)
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'numerical_controls.csv',index=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    pd.testing.assert_frame_equal(data[pd.read_parquet(source).columns],pd.read_parquet(source))
    (out/'protocol.json').write_text(json.dumps(dict(source=str(source),source_sha256=source_hash,betas=[1,.75,.5,.25],formula='X_adjusted = 0.5 - beta*(0.5-X_Cunge)',n_fixed=n,initial_state='Equilibrium at current event start, identical for all beta; no antecedent substitution',initial_cms=initial,downstream_display_offset_cms=shift,numerical_controls='Half dt at fixed grid; doubled subdivision compared with original grid at same half dt. Physical properties and diffusivity preserved under subdivision.',score='Physical unshifted discharge primary; baseline-shifted score diagnostic only. Global maximum compared with unchanged target.',volume_note='Observed-window outlet volume is not a mass-conservation residual: storage differs and event tail is incomplete.',production_modified=False,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(records).to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))
if __name__=='__main__':main()

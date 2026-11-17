"""Align saved current MC predictor targets with local-inertial collapse runs.
No fitting or hydraulic rerouting. Preserve both historical and matched metrics.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import broad_study as b

OUT=b.OUT/'investigation';OUT.mkdir(exist_ok=True)
MC=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028/03_manning_midpoint')
if (OUT/'current_mc/snapshot.json').exists():MC=OUT/'current_mc'
PRED=b.ROOT/'RAPID/outputs/entropy_complexity_g_ridge_20261104/predictions.parquet'
DIAG=b.ROOT/'RAPID/outputs/uncapped_event_window_20261020/three_run_review_20261029/incremental_storage/group_investigation/population_diagnostics.parquet'

def moment(t,q):
    x=t-t[0];dt=np.diff(x);a=q[:-1];z=q[1:];v=dt*(a+z)/2
    if q.min() < -1e-7 or v.sum()<=0:return np.nan,float(v.sum()),float(q.min())
    return float(np.sum(x[:-1]*v+dt*dt*(a/2+(z-a)/3))/v.sum()),float(v.sum()),float(q.min())

def window_pair(parent,child,start,end,baseline):
    tp=parent.time_utc.astype('int64').to_numpy()/1e9;tc=child.time_utc.astype('int64').to_numpy()/1e9
    assert max(tp[0],tc[0])<=start and min(tp[-1],tc[-1])>=end
    t=np.unique(np.r_[start,end,tp[(tp>start)&(tp<end)],tc[(tc>start)&(tc<end)]])
    a,va,mina=moment(t,np.interp(t,tp,parent.q_outlet_cms)-baseline)
    z,vz,minz=moment(t,np.interp(t,tc,child.q_outlet_cms)-baseline)
    return dict(centroid_minutes=(z-a)/60,parent_volume_m3=va,child_volume_m3=vz,minimum_excess_cms=min(mina,minz))

def main():
    pred=pd.read_parquet(PRED);pred=pred[pred.estimator.eq('entropy_complexity_G_ridge')].copy()
    diag=pd.read_parquet(DIAG).set_index(['example','state']);rows=[];geometry=[];sources={str(PRED):b.sha(PRED),str(DIAG):b.sha(DIAG)}
    for ex in b.CASES:
        cp=b.OUT/ex/'calibration.json';tr=b.OUT/ex/'transfer.json'
        if not cp.exists() or not tr.exists():continue
        cal=json.loads(cp.read_text());transfer=json.loads(tr.read_text());base=cal.get('final',{})
        if not base.get('valid') or transfer.get('n')!=cal.get('selected_n'):continue
        local_base=pd.read_parquet(Path(base['folder'])/'wave.parquet')
        sources[str(Path(base['folder'])/'wave.parquet')]=b.sha(Path(base['folder'])/'wave.parquet')
        parent_path=MC/'states'/ex/'S000_base'/'outlet.parquet';current_base=pd.read_parquet(parent_path);sources[str(parent_path)]=b.sha(parent_path)
        old_base=pd.read_parquet(b.OUT/'inputs'/ex/'mc_base.parquet')
        for r in transfer['runs']:
            if not r['valid']:continue
            state=r['state'];key=(ex,state)
            original_r=r;corrected=False
            correction=OUT/(ex+'_current_beds_'+state)/'result.json'
            if correction.exists():
                candidate=json.loads(correction.read_text())
                if candidate.get('valid') and candidate.get('n')==cal['selected_n'] and candidate.get('bed_source_sha256')==b.sha(MC/'states'/ex/state/'parameters.parquet'):
                    r=candidate;corrected=True
            if key not in diag.index:continue
            d=diag.loc[key];assert isinstance(d,pd.Series)
            start=pd.Timestamp(d.onset_utc).timestamp();end=pd.Timestamp(d.end_utc).timestamp();baseline=d.baseline_cms
            wave=MC/'states'/ex/state/'outlet.parquet';current=pd.read_parquet(wave);sources[str(wave)]=b.sha(wave)
            local=pd.read_parquet(Path(r['folder'])/'wave.parquet');old=pd.read_parquet(b.OUT/'inputs'/ex/(state+'_mc.parquet'))
            sources[str(Path(r['folder'])/'wave.parquet')]=b.sha(Path(r['folder'])/'wave.parquet')
            matched=window_pair(current_base,current,start,end,baseline);li=window_pair(local_base,local,start,end,baseline);legacy=window_pair(old_base,old,start,end,baseline)
            target=float(pred[pred.example.eq(ex)&pred.state.eq(state)].centroid_shift_hours.iloc[0])*60
            assert abs(matched['centroid_minutes']-target)<1e-4,(key,matched,target)
            for st in ['S000_base',state]:
                if any(x['example']==ex and x['state']==st for x in geometry):continue
                a=b.frame(ex,st).sort_values('id_link').reset_index(drop=True);path=MC/'states'/ex/st/'parameters.parquet';z=pd.read_parquet(path).sort_values('id_link').reset_index(drop=True);sources[str(path)]=b.sha(path)
                cols=['id_link','u','v','width','link_length_m','head_us','head_ds','slope']
                identical=len(a)==len(z) and all(np.allclose(a[c],z[c],rtol=1e-12,atol=1e-12) for c in cols)
                geometry.append(dict(example=ex,state=st,physical_geometry_identical=identical,legacy_mc_n=float(a.n_manning.iloc[0]),current_mc_n=float(z.n_manning.iloc[0]),legacy_depth_max=float(a.depth_m.max()),current_depth_max=float(z.depth_m.max()),legacy_cells=int(a['count'].sum()),current_cells=int(z['count'].sum())))
            rows.append(dict(example=ex,state=state,calibration_status=cal['status'],local_n=cal['selected_n'],corrected_bed_run=corrected,onset_utc=d.onset_utc,end_utc=d.end_utc,baseline_cms=baseline,current_mc_minutes=target,local_minutes=li['centroid_minutes'],legacy_mc_matched_window_minutes=legacy['centroid_minutes'],local_broad_centroid_minutes=(r['centroid_seconds']-base['centroid_seconds'])/60,legacy_bed_local_broad_centroid_minutes=(original_r['centroid_seconds']-base['centroid_seconds'])/60,local_window_volume_change_percent=100*(li['child_volume_m3']/li['parent_volume_m3']-1),local_minimum_excess_cms=li['minimum_excess_cms']))
    cases=pd.DataFrame(rows);geom=pd.DataFrame(geometry).set_index(['example','state']);cases['physical_geometry_identical']=[bool(geom.loc[(e,'S000_base'),'physical_geometry_identical'] and geom.loc[(e,s),'physical_geometry_identical']) for e,s in zip(cases.example,cases.state)]
    cases['legacy_geometry_identical']=cases.physical_geometry_identical
    cases['physical_geometry_identical']=cases.physical_geometry_identical|cases.corrected_bed_run
    cases['reduced_magnitude']=cases.local_minutes.abs()<cases.current_mc_minutes.abs();cases['same_direction']=cases.local_minutes*cases.current_mc_minutes>0
    joined=cases.merge(pred[['example','state','strategy','repeat','predicted_hours']],on=['example','state'],validate='one_to_many');joined['prediction_minutes']=60*joined.predicted_hours
    joined['prediction_same_direction']=joined.prediction_minutes*joined.current_mc_minutes>0
    joined['prediction_underestimated_magnitude']=joined.prediction_same_direction&(joined.prediction_minutes.abs()<joined.current_mc_minutes.abs())
    joined['prediction_error_vs_mc_minutes']=joined.prediction_minutes-joined.current_mc_minutes
    joined['prediction_error_vs_local_minutes']=joined.prediction_minutes-joined.local_minutes
    cases.to_csv(OUT/'matched_current_mc_cases.csv',index=False);joined.to_parquet(OUT/'matched_predictor_repeats.parquet',index=False);pd.DataFrame(geometry).to_csv(OUT/'mc_reference_geometry_audit.csv',index=False)
    summary=[]
    good=joined[joined.calibration_status.eq('timing_calibrated')&joined.physical_geometry_identical&joined.local_minutes.notna()]
    for (strategy,repeat),d in good.groupby(['strategy','repeat']):
        smaller=d[d.reduced_magnitude&d.same_direction]
        summary.append(dict(strategy=strategy,repeat=int(repeat),cases=len(d),smaller_same_direction=len(smaller),underpredicted_among_smaller=int(smaller.prediction_underestimated_magnitude.sum()),wrong_prediction_direction_among_smaller=int((~smaller.prediction_same_direction).sum()),prediction_closer_to_local=int((d.prediction_error_vs_local_minutes.abs()<d.prediction_error_vs_mc_minutes.abs()).sum()),mae_vs_mc=float(d.prediction_error_vs_mc_minutes.abs().mean()),mae_vs_local=float(d.prediction_error_vs_local_minutes.abs().mean())))
    pd.DataFrame(summary).to_csv(OUT/'matched_predictor_summary.csv',index=False)
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Current MC versus local inertia: matched centroid window','Predictor error versus current MC'])
    cases['plot_status']=np.where(~cases.physical_geometry_identical,'Bed profiles differ — exclude',cases.calibration_status)
    for status,d in cases.groupby('plot_status'):
        fig.add_trace(go.Scatter(x=d.current_mc_minutes,y=d.local_minutes,mode='markers',name=status,customdata=d[['example','state']].to_numpy(),hovertemplate='%{customdata[0]} %{customdata[1]}<br>MC=%{x:.3f} min<br>Local=%{y:.3f} min<extra></extra>'),row=1,col=1)
    lon=joined[joined.strategy.eq('leave_one_network_out')]
    fig.add_trace(go.Scatter(x=lon.current_mc_minutes,y=lon.prediction_error_vs_mc_minutes,mode='markers',name='Leave-river-out prediction error',customdata=lon[['example','state']].to_numpy(),hovertemplate='%{customdata[0]} %{customdata[1]}<br>MC=%{x:.3f}<br>Prediction − MC=%{y:.3f} min<extra></extra>'),row=1,col=2)
    bounds=[float(np.nanmin(np.r_[cases.current_mc_minutes,cases.local_minutes,0]))-1,float(np.nanmax(np.r_[cases.current_mc_minutes,cases.local_minutes,0]))+1]
    fig.add_trace(go.Scatter(x=bounds,y=bounds,mode='lines',line=dict(dash='dot',color='gray'),name='Equal collapse effect'),row=1,col=1)
    fig.update_xaxes(title_text='Current MC centroid effect (min)');fig.update_yaxes(title_text='Local-inertial centroid effect (min)',row=1,col=1);fig.update_yaxes(title_text='Prediction − current MC (min)',row=1,col=2)
    fig.update_layout(title='Matched current-MC comparison — provisional selected cases',template='plotly_white',height=650,legend=dict(orientation='h',y=-.22),margin=dict(b=140));fig.write_html(OUT/'matched_current_mc_plotly.html',include_plotlyjs=True)
    alignment=pd.read_csv(OUT/'forcing_alignment.csv');assert (alignment.maximum_shared_forcing_difference==0).all() and (alignment.shared_rows==alignment.legacy_rows).all() and (alignment.shared_rows==alignment.current_rows).all()
    sources[str(OUT/'forcing_alignment.csv')]=b.sha(OUT/'forcing_alignment.csv')
    b.save(OUT/'matched_predictor_provenance.json',dict(sources=sources,script_sha256=b.sha(__file__),primary='Saved six-input quadratic ridge with G; all held-out repeats retained, no refitting.',metric='Exact piecewise-linear excess-flow centroid in the existing fixed 0.5% window, constant saved baseline, no clipping; current MC targets independently reproduced to <1e-4 min.',forcing='All 15 input forcings identical by timestamps and values to the current MC run specification.',caveat='Selected cases only. Different hydraulic models are not observations; improved agreement does not establish physical accuracy. Bed-mismatched states excluded until independently rerun on current bed profiles.'))
    print(cases.to_string(index=False));print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()

"""Post-hoc RF errors and actual saved hydrographs; no fitting or routing."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/investigations/three_run_review_20261029'))
from investigate_j_outlier_groups import read_saved,SOURCES
from incremental_storage_patterns import moment
from plot_j_outlier_groups import display_indices
BASE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
SOURCE=BASE/'hydrography_rf_compound_20261029'
OUT=SOURCE/'outlier_audit_20261030'
OUT.mkdir(exist_ok=True)
def read(path):
    SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path)
p=read(SOURCE/'predictions.parquet');p=p[p.model.eq('geometry_normalized_9')].copy()
p['actual_min']=p.centroid_shift_hours*60;p['prediction_min']=p.predicted_hours*60
p['error_min']=p.prediction_min-p.actual_min
p['abs_error_min']=p.error_min.abs()
for (cohort,repeat),g in p.groupby(['cohort','repeat']):
    w=1/g.groupby('example').example.transform('size')
    mu=np.average(g.error_min,weights=w);sd=np.sqrt(np.average((g.error_min-mu)**2,weights=w))
    p.loc[g.index,'residual_z']=(g.error_min-mu)/sd
    p.loc[g.index,'weighted_sse']=w*g.error_min**2
    p.loc[g.index,'weighted_absolute_error']=w*g.abs_error_min
p['z3']=p.residual_z.abs()>3
p['missed_delay']=(p.actual_min>1+1e-6)&(p.prediction_min<=1+1e-6)
p['false_delay']=(p.actual_min<=1+1e-6)&(p.prediction_min>1+1e-6)
p['wrong_direction_material']=(p.actual_min*p.prediction_min<0)&(p.actual_min.abs()>1)&(p.prediction_min.abs()>1)
p.to_parquet(OUT/'repeat_errors.parquet',index=False)
a=p.groupby(['cohort','example','state','edit_type']).agg(actual_min=('actual_min','first'),pred_median_min=('prediction_min','median'),pred_min_min=('prediction_min','min'),pred_max_min=('prediction_min','max'),mae_min=('abs_error_min','mean'),min_abs_error_min=('abs_error_min','min'),max_abs_z=('residual_z',lambda x:x.abs().max()),z3_repeats=('z3','sum'),missed_delay_repeats=('missed_delay','sum'),false_delay_repeats=('false_delay','sum'),wrong_direction_repeats=('wrong_direction_material','sum'),weighted_sse=('weighted_sse','sum'),weighted_absolute_error=('weighted_absolute_error','sum')).reset_index()
assert len(a)==2185
for cohort,g in a.groupby('cohort'):
    a.loc[g.index,'sse_share_pct']=100*g.weighted_sse/g.weighted_sse.sum()
    a.loc[g.index,'absolute_error_share_pct']=100*g.weighted_absolute_error/g.weighted_absolute_error.sum()
a=a.sort_values('mae_min',ascending=False)
a.to_csv(OUT/'case_errors.csv',index=False)
for col in ['example','edit_type']:
    rows=[]
    for (cohort,label),g in a.groupby(['cohort',col]):
        rows.append(dict(cohort=cohort,**{col:label},cases=len(g),sse_share_pct=g.sse_share_pct.sum(),absolute_error_share_pct=g.absolute_error_share_pct.sum(),persistent_z3=int(g.z3_repeats.eq(5).sum()),persistent_wrong_direction=int(g.wrong_direction_repeats.eq(5).sum()),persistent_missed_delay=int(g.missed_delay_repeats.eq(5).sum())))
    pd.DataFrame(rows).to_csv(OUT/f'errors_by_{col}.csv',index=False)
local=a[a.cohort=='original'];full=a[a.cohort=='segmented']
selections={}
def select(frame,reason,n):
    for r in frame.head(n).itertuples():selections.setdefault((r.example,r.state),[]).append(reason)
select(local,'largest mean absolute error',10)
select(local[local.wrong_direction_repeats.eq(5)],'material wrong direction in all five repeats',5)
select(local[local.missed_delay_repeats.eq(5)],'missed >1min delay in all five repeats',5)
select(full,'largest full-network error (separate cohort)',3)
selected=a.merge(pd.DataFrame([dict(example=ex,state=st,selection_reason='; '.join(why)) for (ex,st),why in selections.items()]),on=['example','state'],validate='one_to_one')
features=read(SOURCE/'input.parquet')
design=json.loads((SOURCE/'design.json').read_text());cols=design['panels']['geometry_normalized_9']
selected=selected.merge(features[['example','state']+cols],on=['example','state'],validate='one_to_one')
# Hydraulic diagnostics are joined only after selection; never used as predictors.
diagnostic=read(BASE/'three_run_review_20261029/incremental_storage/group_investigation/population_diagnostics.parquet')
keep=['example','state','onset_utc','end_utc','baseline_cms','parent_centroid_seconds','child_centroid_seconds','centroid_shift_minutes','delta_J_minutes','residual_z','is_outlier','min_slope','max_depth_m','parent_max_depth_m','transfer_sd_hours','parent_transfer_sd_hours','reaches','parent_reaches']
selected=selected.merge(diagnostic[keep].rename(columns={'residual_z':'J_fit_residual_z','is_outlier':'J_fit_outlier'}),on=['example','state'],validate='one_to_one')
np.testing.assert_allclose(selected.actual_min,selected.centroid_shift_minutes,atol=1e-8)
# Context: percentile positions within the local/full cohort, descriptive only.
for col in cols:
    ranks=features.groupby('cohort')[col].rank(pct=True)
    selected=selected.merge(features[['example','state']].assign(**{col+'_percentile':ranks*100}),on=['example','state'],validate='one_to_one')
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08)
buttons=[];waves=[];cache={}
for i,r in enumerate(selected.itertuples()):
    key=(r.example,'S000_base')
    if key not in cache:cache[key]=read_saved(*key,'outlet.parquet')
    parent=cache[key];child=read_saved(r.example,r.state,'outlet.parquet')
    tp=parent.time_utc.astype('int64').to_numpy()/1e9;tc=child.time_utc.astype('int64').to_numpy()/1e9
    qp=parent.q_outlet_cms.to_numpy();qc=child.q_outlet_cms.to_numpy()
    start=pd.Timestamp(r.onset_utc).timestamp();end=pd.Timestamp(r.end_utc).timestamp()
    knots=np.unique(np.r_[start,end,tp[(tp>start)&(tp<end)],tc[(tc>start)&(tc<end)]])
    mp,vp=moment(knots,np.interp(knots,tp,qp)-r.baseline_cms);mc,vc=moment(knots,np.interp(knots,tc,qc)-r.baseline_cms)
    shift=(mc-mp)/60
    assert abs(shift-r.actual_min)<1e-4,(r.example,r.state,shift,r.actual_min)
    common_start=max(tp[0],tc[0]);common_end=min(tp[-1],tc[-1])
    whole=np.unique(np.r_[common_start,common_end,tp[(tp>common_start)&(tp<common_end)],tc[(tc>common_start)&(tc<common_end)]])
    mpp,_=moment(whole,np.interp(whole,tp,qp)-r.baseline_cms);mcc,_=moment(whole,np.interp(whole,tc,qc)-r.baseline_cms)
    waves.append(dict(example=r.example,state=r.state,recomputed_window_shift_min=shift,full_saved_record_shift_min=(mcc-mpp)/60,window_volume_change_pct=100*(vc/vp-1),parent_final_minus_baseline_cms=qp[-1]-r.baseline_cms,child_final_minus_baseline_cms=qc[-1]-r.baseline_cms,saved_tail_hours=(common_end-end)/3600))
    xp=(tp-start)/3600;xc=(tc-start)/3600
    for x,q,label,color in [(xp,qp,'Parent','#333333'),(xc,qc,'Child','#1978b5')]:
        idx=display_indices(x,q)
        fig.add_trace(go.Scatter(x=x[idx],y=q[idx],name=label,line=dict(color=color),visible=i==0),row=1,col=1)
    mask=(tc>=tp[0])&(tc<=tp[-1]);delta=qc[mask]-np.interp(tc[mask],tp,qp);x=xc[mask];idx=display_indices(x,delta)
    fig.add_trace(go.Scatter(x=x[idx],y=delta[idx],name='Child − parent',line=dict(color='#1978b5'),visible=i==0),row=2,col=1)
    duration=(end-start)/3600
    shapes=[dict(type='rect',xref='x',yref='paper',x0=0,x1=duration,y0=0,y1=1,fillcolor='#eeeeee',opacity=.4,line_width=0,layer='below')]
    title=f'{r.example} / {r.state} — modelled {r.actual_min:+.2f} min; RF median {r.pred_median_min:+.2f} min<br><sup>RF range {r.pred_min_min:+.2f} to {r.pred_max_min:+.2f} min; {r.selection_reason}. Grey = finite event window.</sup>'
    layout={'title.text':title,'shapes':shapes,'xaxis.range':[-6,duration+6],'xaxis2.range':[-6,duration+6],'yaxis.autorange':True,'yaxis2.autorange':True}
    visible=[False]*(3*len(selected));visible[3*i:3*i+3]=[True]*3
    buttons.append(dict(label=f'{r.example} / {r.state}',method='update',args=[{'visible':visible},layout]))
    if i==0:fig.update_layout(**layout)
wave=pd.DataFrame(waves);selected=selected.merge(wave,on=['example','state'],validate='one_to_one');selected.to_csv(OUT/'selected_case_diagnostics.csv',index=False)
fig.update_layout(template='plotly_white',height=780,margin=dict(t=165),legend=dict(orientation='h',y=1.04),updatemenus=[dict(buttons=buttons,x=0,y=1.24)])
fig.update_yaxes(title_text='Outlet discharge (m³/s)',row=1,col=1);fig.update_yaxes(title_text='Child − parent (m³/s)',row=2,col=1);fig.update_xaxes(title_text='Hours since event-window onset',row=2,col=1)
fig.write_html(OUT/'outlier_hydrographs.html',include_plotlyjs=True)
net=local.groupby('example').sse_share_pct.sum().sort_values(ascending=False)
summary=dict(local_cases=len(local),full_network_cases=len(full),local_z3_any=int((local.z3_repeats>0).sum()),local_z3_all5=int(local.z3_repeats.eq(5).sum()),actual_delays=int((local.actual_min>1+1e-6).sum()),missed_delays_all5=int(local.missed_delay_repeats.eq(5).sum()),wrong_material_direction_all5=int(local.wrong_direction_repeats.eq(5).sum()),top10_absolute_error_cases_sse_share_pct=float(local.head(10).sse_share_pct.sum()),network_sse_share_pct=net.to_dict(),selected_hydrographs=len(selected),max_centroid_reproduction_error_min=float((selected.recomputed_window_shift_min-selected.actual_min).abs().max()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'provenance.json').write_text(json.dumps(dict(sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),definition='Residual=RF minus simulated centroid. Residual z is centered/scaled within each repeat and cohort with equal network weights; descriptive screening, not significance testing. Cases counted once; five repeat errors retained. Scores not recomputed after exclusions.',selection='Top10 local MAE; top5 persistent wrong-direction and missed-delay cases ranked by MAE; top3 full-network MAE. Union, no handpicked replacements.',hydraulics='Existing hydraulic/DeltaJ diagnostics joined only for post-hoc explanation; absent from model inputs. Full-record centroid is still finite and is not necessarily complete isolated-pulse timing.',display='10-minute bins retain first/min/max original samples; calculations use full records.',refitting=False,rerouting=False),indent=2)+'\n')
print(json.dumps(summary,indent=2));print(selected[['example','state','actual_min','pred_median_min','mae_min','J_fit_outlier','full_saved_record_shift_min','min_slope','max_depth_m']].to_string(index=False))

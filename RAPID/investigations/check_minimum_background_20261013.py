"""Quick sensitivity check: common minimum background, unchanged accepted results."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_response import load,distribution,METRICS,digest,utc
OUT=ROOT/'RAPID/outputs/minimum_background_check_20261013';OUT.mkdir(exist_ok=True)


def main():
    states,accepted=load();rows=[];summary=[];hashes={};figure=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['46: saved model hydrographs and two reference levels','46: direct discharge differences from base (no background subtraction)'])
    for ex,g in states.groupby('example',sort=True):
        waves={}
        for r in g.itertuples():
            f=pd.read_parquet(r.outlet);t=pd.to_datetime(f.time_utc,utc=True).astype('int64').to_numpy()/1e9;q=f.q_outlet_cms.to_numpy(float)
            waves[r.state]=(t,q);hashes[r.outlet]=digest(r.outlet)
        minimum_state=min(waves,key=lambda k:waves[k][1].min());mt,mq=waves[minimum_state];mi=int(mq.argmin());level=float(mq[mi])
        lo=max(t[0] for t,q in waves.values());hi=min(t[-1] for t,q in waves.values())
        windows={'full_common_record':(lo,hi),'selected_event_envelope':(max(lo,float(g.rise_seconds.min())),min(hi,float(g.fall_seconds.max())))}
        info=dict(example=ex,accepted_threshold_cms=float(g.threshold_cms.iloc[0]),minimum_baseline_cms=level,minimum_state=minimum_state,minimum_utc=pd.Timestamp(mt[mi],unit='s',tz='UTC').isoformat(),full_start_utc=pd.Timestamp(lo,unit='s',tz='UTC').isoformat(),full_end_utc=pd.Timestamp(hi,unit='s',tz='UTC').isoformat())
        for mode,(a,b) in windows.items():
            metrics={}
            for r in g.itertuples():
                t,q=waves[r.state];tt=np.r_[a,t[(t>a)&(t<b)],b];qq=np.interp(tt,t,q);excess=qq-level
                assert excess.min()>=0
                m=distribution(tt,excess);m.update(peak_seconds=r.peak_seconds,peak_excess_cms=r.peak_cms-level,start_excess_cms=float(excess[0]),end_excess_cms=float(excess[-1]))
                metrics[r.state]=m
            p=metrics['S000_base'];info[mode+'_hours']=(b-a)/3600;info[mode+'_parent_start_excess_cms']=p['start_excess_cms'];info[mode+'_parent_end_excess_cms']=p['end_excess_cms']
            for r in g[g.state.ne('S000_base')].itertuples():
                c=metrics[r.state];row=dict(example=ex,state=r.state,aliases=r.aliases,scope=mode,baseline_cms=level,start_seconds=a,end_seconds=b,
                    peak_shift_hours=(c['peak_seconds']-p['peak_seconds'])/3600,peak_excess_change_pct=100*(c['peak_excess_cms']/p['peak_excess_cms']-1),
                    centroid_shift_hours=(c['centroid_seconds']-p['centroid_seconds'])/3600,duration80_change_pct=100*(c['duration80_hours']/p['duration80_hours']-1),
                    asymmetry_change=c['asymmetry']-p['asymmetry'],volume_change_pct=100*(c['volume_m3']/p['volume_m3']-1),
                    parent_volume_m3=p['volume_m3'],child_volume_m3=c['volume_m3'],parent_asymmetry=p['asymmetry'],child_asymmetry=c['asymmetry'],
                    parent_start_excess_cms=p['start_excess_cms'],parent_end_excess_cms=p['end_excess_cms'],child_start_excess_cms=c['start_excess_cms'],child_end_excess_cms=c['end_excess_cms'])
                rows.append(row)
        summary.append(info)
        if ex=='46':
            bubble=g[g.aliases.str.contains('CB006_bubble_101',regex=False)].iloc[0].state
            for state,name,color in [('S000_base','Base','#ce3d45'),(bubble,'CB006_bubble_101','#27855c'),('fully_collapsed','Single centerline','#e38b15')]:
                t,q=waves[state];use=(t>=lo)&(t<=hi)
                figure.add_trace(go.Scatter(x=utc(t[use]),y=q[use],name=name,line=dict(color=color)),row=1,col=1)
                if state!='S000_base':
                    pt,pq=waves['S000_base'];figure.add_trace(go.Scatter(x=utc(t[use]),y=q[use]-np.interp(t[use],pt,pq),name=name+' − base',line=dict(color=color)),row=2,col=1)
            figure.add_hline(y=level,line_dash='dash',line_color='#168a65',annotation_text=f'Common minimum {level:.3f}',row=1,col=1)
            figure.add_hline(y=info['accepted_threshold_cms'],line_dash='dash',line_color='#9155b5',annotation_text='Accepted threshold',row=1,col=1)
        print(ex,round(level,6),minimum_state,flush=True)
    result=pd.DataFrame(rows);assert len(result)==4374 and np.isfinite(result[list(METRICS)]).all().all()
    result.to_csv(OUT/'responses.csv',index=False);result.to_parquet(OUT/'responses.parquet',index=False);pd.DataFrame(summary).to_csv(OUT/'baseline_summary.csv',index=False)
    full=result[result.scope.eq('full_common_record')];joined=full.merge(accepted[['example','state',*METRICS]],on=['example','state'],suffixes=('_minimum','_accepted'));joined.to_csv(OUT/'comparison_to_accepted.csv',index=False)
    focus=accepted[(accepted.example=='46')&(accepted.aliases.str.contains('CB006_bubble_101',regex=False)|accepted.state.eq('fully_collapsed'))][['example','state','aliases',*METRICS]].copy();focus['scope']='accepted_selected_pulse'
    focus=pd.concat([focus,result[(result.example=='46')&(result.aliases.str.contains('CB006_bubble_101',regex=False)|result.state.eq('fully_collapsed'))][['example','state','aliases','scope',*METRICS]]],ignore_index=True)
    focus.to_csv(OUT/'46_comparison.csv',index=False)
    figure.update_layout(template='plotly_white',height=850,title='Minimum-reference sensitivity: full common saved record',legend=dict(orientation='h'));figure.update_yaxes(title_text='Discharge (m³/s)',row=1,col=1);figure.update_yaxes(title_text='Child − base (m³/s)',row=2,col=1);figure.update_xaxes(title_text='UTC',row=2,col=1)
    header='<h1>Common-minimum background sensitivity</h1><p>Diagnostic only; accepted thresholds and response tables are unchanged. The baseline is the minimum over all saved model versions per example, excluding gauges. Distribution metrics use one common window per example: either the full overlapping record or the envelope of accepted pulses. Peak metrics retain the previously selected event peaks (not a new whole-record maximum). These windows can contain positive excess at their boundaries, other events, and artificial held-forcing tails.</p>'
    (OUT/'46_minimum_background.html').write_text(header+figure.to_html(full_html=False,include_plotlyjs=True)+focus.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'<h2>Baselines for all examples</h2>'+pd.DataFrame(summary)[['example','accepted_threshold_cms','minimum_baseline_cms','minimum_state','minimum_utc']].to_html(index=False))
    (OUT/'manifest.json').write_text(json.dumps(dict(status='Sensitivity experiment, not adopted',script_sha256=digest(__file__),inputs=hashes,baselines='One minimum across every saved base, partial-collapse and single-centerline model hydrograph per example; gauges excluded',windows=list(windows),peak_definition='Retained accepted event maxima to avoid substituting a different event in multi-event records',comparisons_per_window=2187),indent=2))
    print(focus[['aliases','scope',*METRICS]].to_string(index=False),flush=True)
if __name__=='__main__':main()

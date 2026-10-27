"""Shared-window minimum-baseline hydrograph responses and Plotly review helpers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / 'RAPID/outputs/completed_event_review_20261014'
OUT = ROOT / 'RAPID/outputs/hydrograph_response'
METRICS = {
    'peak_shift_hours': ('Peak-time shift', 'hours'),
    'peak_excess_change_pct': ('Peak-excess change', '%'),
    'centroid_shift_hours': ('Excess-volume centroid shift', 'hours'),
    'duration80_change_pct': ('Central 80% volume-duration change', '%'),
    'asymmetry_change': ('Volume-quantile asymmetry change', 'dimensionless'),
    'volume_change_pct': ('Excess-volume change', '%'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(t, q):
    """Exact moments and quantiles of nonnegative piecewise-linear discharge.

    Times in seconds; shift to the first time internally to avoid epoch-scale
    cancellation. No finite negative discharge is silently clipped.
    """
    t, q = np.asarray(t, float), np.asarray(q, float)
    if len(t)<2 or not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(np.diff(t)<=0) or np.any(q<0):
        raise ValueError('Finite, increasing times and nonnegative discharge required')
    x=t-t[0];h=np.diff(x);a,b=q[:-1],q[1:];mass=h*(a+b)/2;volume=float(mass.sum())
    if volume<=0:raise ValueError('Positive excess volume required')
    mu=float(np.sum(x[:-1]*mass+h*h*(a/2+(b-a)/3))/volume)
    u=x[:-1]-mu
    variance=float(np.sum(u*u*mass+2*u*h*h*(a/2+(b-a)/3)+h**3*(a/3+(b-a)/4))/volume)
    cumulative=np.r_[0.,np.cumsum(mass)]
    def quantile(p):
        i=min(np.searchsorted(cumulative,p*volume,side='right')-1,len(h)-1)
        remaining=p*volume-cumulative[i];slope=(b[i]-a[i])/h[i]
        root=np.sqrt(max(0.,a[i]**2+2*slope*remaining))
        offset=2*remaining/(a[i]+root) if remaining else 0.
        return float(t[0]+x[i]+offset)
    t10,t50,t90=[quantile(p) for p in [.1,.5,.9]]
    return dict(volume_m3=volume,centroid_seconds=float(t[0]+mu),temporal_sd_seconds=float(np.sqrt(max(0.,variance))),
                t10_seconds=t10,t50_seconds=t50,t90_seconds=t90,duration80_hours=(t90-t10)/3600,
                asymmetry=((t90-t50)-(t50-t10))/(t90-t10))


def pulse(row):
    """Shared finite analysis window; endpoint excess is retained, never zeroed."""
    if 'outlet_sha256' in row and digest(row['outlet']) != row['outlet_sha256']:
        raise ValueError('Routed outlet changed since calculation')
    w=pd.read_parquet(row['outlet'])
    t=pd.to_datetime(w.time_utc,utc=True).astype('int64').to_numpy()/1e9
    q=w.q_outlet_cms.to_numpy(float)
    a,b=float(row['rise_seconds']),float(row['fall_seconds'])
    if not t[0]<=a<b<=t[-1]:raise ValueError('Window outside saved support')
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    excess=np.interp(tt,t,q)-float(row['baseline_cms'])
    if excess.min()<0:raise ValueError('Negative excess: baseline must be reviewed, not clipped')
    return tt,excess,0.


def response_rows(states):
    responses=[]
    for (window,ex),g in states.groupby(['window','example'],sort=False):
        p=g[g.state.eq('S000_base')].iloc[0]
        assert g.rise_seconds.nunique()==g.fall_seconds.nunique()==g.baseline_cms.nunique()==1
        for _,c in g[~g.state.eq('S000_base')].iterrows():
            r=dict(example=ex,state=c.state,aliases=c.aliases,window=window,
                comparison='single centerline' if c.state=='fully_collapsed' else 'partial collapse',
                baseline_cms=c.baseline_cms,threshold_cms=c.baseline_cms,
                onset_utc=c.rise_utc,end_utc=c.fall_utc,threshold_fraction=c.threshold_fraction)
            for k in ['peak_seconds','peak_cms','peak_excess_cms','centroid_seconds','duration80_hours','asymmetry','volume_m3','t50_seconds','duration_hours','temporal_sd_seconds']:
                r['parent_'+k]=p[k];r['child_'+k]=c[k]
            r.update(peak_shift_hours=(c.peak_seconds-p.peak_seconds)/3600,
                peak_excess_change_pct=100*(c.peak_excess_cms/p.peak_excess_cms-1),
                total_peak_change_pct=100*(c.peak_cms/p.peak_cms-1),
                centroid_shift_hours=(c.centroid_seconds-p.centroid_seconds)/3600,
                duration80_change_pct=100*(c.duration80_hours/p.duration80_hours-1),
                duration80_change_hours=c.duration80_hours-p.duration80_hours,
                asymmetry_change=c.asymmetry-p.asymmetry,volume_change_pct=100*(c.volume_m3/p.volume_m3-1),
                median_shift_hours=(c.t50_seconds-p.t50_seconds)/3600)
            responses.append(r)
    return pd.DataFrame(responses)


def calculate():
    """Recalculate completed reruns at shared 0.5%, 1% and 2% windows."""
    import shutil
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        old=json.loads((OUT/'manifest.json').read_text())
        if old.get('schema')!='shared-window-minimum-v1':
            archive=OUT/'historical_horizontal_threshold_20261013';archive.mkdir(exist_ok=True)
            for name in ['manifest.json','state_metrics.csv','state_metrics.parquet','responses.csv','responses.parquet']:
                if (OUT/name).exists() and not (archive/name).exists():shutil.copy2(OUT/name,archive/name)
    plan_path=ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'
    plan=json.loads(plan_path.read_text())
    windows_path=REVIEW/'threshold_window_locations.csv';baseline_path=REVIEW/'baseline_proposal.csv'
    windows=pd.read_csv(windows_path,dtype={'example':str});baselines=pd.read_csv(baseline_path,dtype={'example':str}).set_index('example')
    inputs={str(p):digest(p) for p in [plan_path,windows_path,baseline_path]};rows=[];hashes={}
    for case in plan['cases']:
        ex=case['example'];receipt=ROOT/f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json'
        done=json.loads(receipt.read_text());assert done['complete'] and done['plan_sha256']==digest(plan_path)
        inputs[str(receipt)]=digest(receipt);folder=Path(done['final_directory'])
        forcing=pd.read_parquet(folder/'forcing.parquet');input_ids=np.flatnonzero(forcing.phase.eq('input_event'))
        base=float(baselines.loc[ex,'baseline_cms']);w=windows[windows.example.eq(ex)]
        assert set(w.threshold)=={'0.5%','1%','2%'} and len(w)==3
        checked={k:dict(start=np.inf,end=-np.inf) for k in w.threshold}
        minima=min(case['initial_cms'],case['final_cms'])
        for job in case['jobs']:
            path=folder/job['state_id']/'outlet.parquet';h=digest(path)
            assert h==done['files'][str(path)];hashes[str(path)]=h
            frame=pd.read_parquet(path);assert frame.time_utc.equals(forcing.time_utc)
            t=frame.time_utc.astype('int64').to_numpy()/1e9;q=frame.q_outlet_cms.to_numpy(float)
            minima=min(minima,float(q[input_ids[0]:].min()))
            state='fully_collapsed' if job['state_id']=='single_centerline' else job['state_id']
            aliases=' | '.join(dict.fromkeys([state,*[a.split('/')[-1] for a in job['aliases']]]))
            for win in w.itertuples():
                a=pd.Timestamp(win.start_utc).timestamp();b=pd.Timestamp(win.end_utc).timestamp()
                ia=int(np.searchsorted(t,a));ib=int(np.searchsorted(t,b));assert t[ia]==a and t[ib]==b
                # Independently verify the reviewed window rule against this rerun.
                tol=float(win.tolerance_cms)
                depart=abs(q[input_ids[0]:]-case['initial_cms'])>tol
                cs=np.r_[0,np.cumsum(depart)];run=np.flatnonzero(cs[30:]-cs[:-30]==30)
                assert len(run)
                onset=input_ids[0]+int(run[0]);outside=np.flatnonzero((np.arange(len(q))>input_ids[-1])&(abs(q-case['final_cms'])>tol))
                end=int(outside[-1]+1) if len(outside) else input_ids[-1]+1
                checked[win.threshold]['start']=min(checked[win.threshold]['start'],t[onset])
                checked[win.threshold]['end']=max(checked[win.threshold]['end'],t[end])
                tt=t[ia:ib+1];e=q[ia:ib+1]-base
                m=distribution(tt,e);peak=ia+int(np.argmax(q[ia:ib+1]))
                # Assert the selected maximum equals the event-and-response maximum.
                assert np.isclose(q[peak],q[input_ids[0]:].max(),rtol=0,atol=1e-8)
                rows.append(dict(example=ex,state=state,aliases=aliases,window=win.threshold,
                    threshold_fraction=float(win.threshold.strip('%'))/100,
                    tolerance_cms=tol,baseline_cms=base,threshold_cms=base,
                    rise_seconds=a,fall_seconds=b,rise_utc=win.start_utc,fall_utc=win.end_utc,
                    outlet=str(path),outlet_sha256=h,peak_seconds=float(t[peak]),peak_cms=float(q[peak]),
                    peak_excess_cms=float(q[peak]-base),duration_hours=(b-a)/3600,
                    start_excess_cms=float(e[0]),end_excess_cms=float(e[-1]),**m))
        assert abs(minima-base)<1e-8
        for win in w.itertuples():
            assert checked[win.threshold]==dict(start=pd.Timestamp(win.start_utc).timestamp(),end=pd.Timestamp(win.end_utc).timestamp())
        print(ex,len(case['jobs']),'states × 3 windows checked and calculated',flush=True)
    all_states=pd.DataFrame(rows);all_responses=response_rows(all_states)
    assert len(all_states)==6606 and len(all_responses)==6561
    assert np.isfinite(all_responses[list(METRICS)]).all().all()
    states=all_states[all_states.window.eq('0.5%')].copy();responses=all_responses[all_responses.window.eq('0.5%')].copy()
    for name,d in [('state_metrics',states),('responses',responses),('sensitivity_state_metrics',all_states),('sensitivity_responses',all_responses)]:
        d.to_parquet(OUT/f'{name}.parquet',index=False);d.to_csv(OUT/f'{name}.csv',index=False)
    manifest=dict(schema='shared-window-minimum-v1',status='Recalculated completed reruns: primary shared 0.5% window',
        states=len(states),comparisons=len(responses),partial_collapses=2172,single_centerlines=15,
        primary_window='0.5%',sensitivity_windows=['1%','2%'],source_sha256=digest(__file__),input_sha256=inputs,outlet_sha256=hashes,
        baseline='One fixed minimum per example across all completed model curves after input start and initial/final equilibrium levels; held fixed for sensitivity.',
        convention='Shared earliest 30-minute sustained departure from initial equilibrium and latest post-input return within tolerance of final equilibrium across all states; tolerance is fraction of input range. No confirmation padding.',
        peak='First saved maximum inside the shared analysis window; verified against maximum from input start through full routed response.',
        integration='Exact integrals of piecewise-linear reconstruction between saved interval means on the common window. Endpoint excess retained; no clipping or zeroing.',
        limitations=['Operational finite-window excess, not physical baseflow separation','All four distribution metrics can depend on baseline/window','54 double-crest timing can switch lobes','21/22 source clocks remain provisional','Input interpolation flags remain as in completed routing'])
    manifest['table_sha256']={str(OUT/f'{n}.parquet'):digest(OUT/f'{n}.parquet') for n in ['state_metrics','responses','sensitivity_state_metrics','sensitivity_responses']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return states,responses


def load(window='0.5%'):
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest.get('schema')!='shared-window-minimum-v1':raise ValueError('Legacy response tables: recalculate completed reruns')
    if manifest['source_sha256']!=digest(__file__):raise ValueError('Calculation source changed: recalculate')
    for path,h in {**manifest['input_sha256'],**manifest['table_sha256']}.items():
        if digest(path)!=h:raise ValueError('Response input/table changed: '+path)
    if window not in ['0.5%','1%','2%']:raise ValueError('Choose 0.5%, 1% or 2%')
    prefix='' if window=='0.5%' else 'sensitivity_'
    states=pd.read_parquet(OUT/f'{prefix}state_metrics.parquet');responses=pd.read_parquet(OUT/f'{prefix}responses.parquet')
    return states[states.window.eq(window)].copy(),responses[responses.window.eq(window)].copy()


def sensitivity_figure(example,child):
    all_responses=pd.read_parquet(OUT/'sensitivity_responses.parquet')
    states,_=load();_,c=select_pair(states,example,child)
    g=all_responses[(all_responses.example==c.example)&(all_responses.state==c.state)].set_index('window').loc[['0.5%','1%','2%']]
    fig=make_subplots(rows=3,cols=2,subplot_titles=[v[0] for v in METRICS.values()])
    for i,(key,(_,unit)) in enumerate(METRICS.items()):
        fig.add_scatter(x=g.index,y=g[key],mode='lines+markers',showlegend=False,row=i//2+1,col=i%2+1)
        fig.update_yaxes(title_text=unit,row=i//2+1,col=i%2+1)
    fig.update_layout(template='plotly_white',height=850,title=f'{example}: {c.aliases} — shared-window sensitivity (baseline fixed)')
    return g.reset_index()[['window',*METRICS]],fig


def select_pair(states,example,child='fully_collapsed'):
    g=states[states.example.eq(str(example).zfill(2))]
    matches=g[g.state.eq(child)|g.aliases.map(lambda a:child in a.split(' | '))]
    if len(matches)!=1 or matches.iloc[0].state=='S000_base':raise ValueError('Choose one child state or alias from the state table')
    return g[g.state.eq('S000_base')].iloc[0],matches.iloc[0]


def utc(seconds):return pd.to_datetime(np.asarray(seconds),unit='s',utc=True).tz_localize(None).to_numpy()


def threshold_figure(states,example='46',child='fully_collapsed',view='event',gauges=True):
    p,c=select_pair(states,example,child);ex=p.example;g=states[states.example.eq(ex)]
    center=g[g.state.eq('fully_collapsed')].iloc[0];fig=go.Figure()
    curves=[('Base',p,'#ce3d45'),('Single centerline',center,'#e38b15')]
    if c.state!='fully_collapsed':curves.append((c.aliases,c,'#27855c'))
    times=[]
    for label,row,color in curves:
        w=pd.read_parquet(row.outlet);t=pd.to_datetime(w.time_utc,utc=True).dt.tz_localize(None);times.extend([t.iloc[0],t.iloc[-1]])
        fig.add_scatter(x=t,y=w.q_outlet_cms,name=label,line=dict(color=color),hovertemplate='%{x}<br>%{y:.4f} m³/s<extra>%{fullData.name}</extra>')
        fig.add_scatter(x=utc([row.rise_seconds,row.fall_seconds]),y=np.interp([row.rise_seconds,row.fall_seconds],pd.to_datetime(w.time_utc,utc=True).astype('int64').to_numpy()/1e9,w.q_outlet_cms),name=label+' window endpoints',mode='markers',marker=dict(color=color,size=9),showlegend=False)
    if gauges:
        w=pd.read_parquet(ROOT/f'RAPID/outputs/bounded_beta_calibration_20261009/{ex}/wave.parquet');t=pd.to_datetime(w.time_utc,utc=True).dt.tz_localize(None)
        for side,color in [('upstream','#2580a5'),('downstream','#343a40')]:
            q=w[side+'_cms'];flags=w[side+'_gap'].fillna(False).to_numpy(bool)
            fig.add_scatter(x=t,y=q.mask(flags),name=side.title()+' gauge (unshifted)',line=dict(color=color,width=1),connectgaps=False)
            if flags.any():
                near=flags.copy();near[1:]|=flags[:-1];near[:-1]|=flags[1:]
                fig.add_scatter(x=t,y=q.where(near),name=side.title()+' gauge: long sampling interval',line=dict(color=color,dash='dot',width=1),showlegend=False,connectgaps=False)
    start,end=float(g.rise_seconds.min()),float(g.fall_seconds.max())
    if view=='event':pad=(end-start)*.05;limits=utc([start-pad,end+pad])
    elif view=='peak':
        a=min(r.peak_seconds for _,r,_ in curves);b=max(r.peak_seconds for _,r,_ in curves);pad=max((end-start)*.08,3600);limits=utc([a-pad,b+pad])
    elif view=='full':limits=[min(times),max(times)]
    else:raise ValueError('view must be full, event or peak')
    fig.add_hline(y=p.threshold_cms,line_dash='dash',line_color='#9155b5',annotation_text=f'Shared minimum baseline {p.baseline_cms:.6f} m³/s')
    for boundary in [start,end]:fig.add_vline(x=pd.Timestamp(boundary,unit='s',tz='UTC').tz_localize(None),line_dash='dot',line_color='#168a65')
    fig.update_layout(template='plotly_white',height=550,title=f'{ex}: shared {p.window} window and minimum baseline',xaxis_title='Time (UTC)',yaxis_title='Discharge (m³/s)',legend=dict(orientation='h',y=1.16),margin=dict(t=110))
    fig.update_xaxes(range=limits)
    return fig


def metric_figure(states,example,child,metric):
    p,c=select_pair(states,example,child);fig=go.Figure()
    origin=min(p.rise_seconds,c.rise_seconds)
    for label,row,color in [('Parent',p,'#ce3d45'),('Child',c,'#27855c')]:
        t,e,_=pulse(row);x=(t-origin)/3600
        if metric=='peak_excess_change_pct':
            fig.add_bar(x=[label],y=[row.peak_excess_cms],name=label,marker_color=color,customdata=[row.peak_cms],hovertemplate='Peak excess: %{y:.4f}<br>Total peak: %{customdata:.4f} m³/s<extra>%{fullData.name}</extra>')
        elif metric=='asymmetry_change':
            fig.add_bar(x=['Early half: t50 − t10','Late half: t90 − t50'],y=[(row.t50_seconds-row.t10_seconds)/3600,(row.t90_seconds-row.t50_seconds)/3600],name=f'{label}: A={row.asymmetry:.4f}',marker_color=color)
        elif metric=='duration80_change_pct':
            mass=np.diff(t)*(e[:-1]+e[1:])/2;cdf=np.r_[0,np.cumsum(mass)]/row.volume_m3
            fig.add_scatter(x=x,y=cdf,name=label,line=dict(color=color))
            fig.add_scatter(x=([row.t10_seconds,row.t90_seconds]-np.array(origin))/3600,y=[.1,.9],mode='markers',marker=dict(color=color,size=10),showlegend=False)
        else:
            fig.add_scatter(x=x,y=e,name=label,line=dict(color=color),fill='tozeroy' if metric=='volume_change_pct' else None,opacity=.7 if metric=='volume_change_pct' else 1)
            if metric in ['peak_shift_hours','centroid_shift_hours']:
                at=row.peak_seconds if metric=='peak_shift_hours' else row.centroid_seconds
                fig.add_vline(x=(at-origin)/3600,line_color=color,line_dash='dash')
    ytitle='Excess discharge (m³/s)'
    if metric=='asymmetry_change':ytitle='Duration (hours)'
    if metric=='duration80_change_pct':ytitle='Fraction of excess volume passed';fig.add_hline(y=.1,line_dash='dot');fig.add_hline(y=.9,line_dash='dot')
    fig.update_layout(template='plotly_white',height=380,title=METRICS[metric][0],barmode='group',yaxis_title=ytitle,xaxis_title='' if metric in ['asymmetry_change','peak_excess_change_pct'] else 'Hours from shared analysis start',legend=dict(orientation='h'))
    return fig


def comparison_table(responses,states,example,child):
    _,c=select_pair(states,example,child);r=responses[(responses.example==c.example)&(responses.state==c.state)].iloc[0]
    keys=[('peak_shift_hours','peak_seconds'),('peak_excess_change_pct','peak_excess_cms'),('centroid_shift_hours','centroid_seconds'),('duration80_change_pct','duration80_hours'),('asymmetry_change','asymmetry'),('volume_change_pct','volume_m3')]
    out=[]
    for response,k in keys:
        a,b=r['parent_'+k],r['child_'+k]
        if k.endswith('_seconds'):a=pd.Timestamp(a,unit='s',tz='UTC').isoformat();b=pd.Timestamp(b,unit='s',tz='UTC').isoformat()
        out.append(dict(response=METRICS[response][0],parent=a,child=b,change=r[response],change_unit=METRICS[response][1]))
    return pd.DataFrame(out)


def overview_figure(responses):
    fig=make_subplots(rows=3,cols=2,subplot_titles=[v[0] for v in METRICS.values()])
    for i,(key,(_,unit)) in enumerate(METRICS.items()):
        for name,g in responses.groupby('comparison'):
            fig.add_trace(go.Scatter(x=g.example,y=g[key],mode='markers',name=name,legendgroup=name,showlegend=i==0,
                marker=dict(size=9 if name=='single centerline' else 4,symbol='diamond' if name=='single centerline' else 'circle',opacity=.9 if name=='single centerline' else .35,color='#e38b15' if name=='single centerline' else '#2580a5'),
                customdata=g[['aliases']].to_numpy(),hovertemplate='%{x} · %{customdata[0]}<br>%{y:.6g}<extra>%{fullData.name}</extra>'),row=i//2+1,col=i%2+1)
        fig.add_hline(y=0,line_color='#aaa',row=i//2+1,col=i%2+1);fig.update_yaxes(title_text=unit,row=i//2+1,col=i%2+1)
        fig.update_xaxes(type='category',categoryorder='array',categoryarray=sorted(responses.example.unique()),row=i//2+1,col=i%2+1)
    fig.update_layout(template='plotly_white',height=1050,title='Signed responses relative to each example’s base',legend=dict(orientation='h',y=1.06));return fig

if __name__=='__main__':calculate()

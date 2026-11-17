"""Compare the translated MATLAB-style pilot with saved routing runs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from local_inertial_pilot56 import OUT
from calibrate_boundaries56 import ROOT, DATA, OUT as DIFF, peak


def main():
    meta=json.loads((OUT/'manifest.json').read_text());rows=[];waves=[]
    for state in ['base','S014_unit_10']:
        r=next(r for r in meta['runs'] if r['state']==state and r['tag']=='standard')
        fine=next(r for r in meta['runs'] if r['state']==state and r['tag']=='refined')
        w=pd.read_parquet(OUT/(state+'_standard.parquet'));f=pd.read_parquet(OUT/(state+'_refined.parquet'))
        row=dict(model='Local-inertial translation',state=state,peak_cms=r['peak_cms'],peak_utc=r['peak_utc'],
                 peak_seconds=r['peak_seconds'],centroid_seconds=r['centroid_seconds'],maximum_depth_m=r['maximum_depth_m'],
                 refinement_peak_change_minutes=(fine['peak_seconds']-r['peak_seconds'])/60,
                 refinement_max_flow_difference_cms=float(abs(w.outlet_cms-f.outlet_cms).max()))
        rows.append(row);waves.append((row['model'],state,w.time_utc,w.outlet_cms))
        records=[json.loads(p.read_text()) for p in (DIFF/'A_normal/runs').glob('*/result.json')]
        d=next(d for d in records if d['state']==state and d['dx']==200 and d['extension_m']==0)
        dw=pd.read_parquet(ROOT/d['wave']);rows.append(dict(model='Diffusive',state=state,peak_cms=d['peak_cms'],
            peak_utc=d['peak_utc'],peak_seconds=d['peak_seconds'],centroid_seconds=d['centroid_seconds'],maximum_depth_m=d['maximum_depth_m']))
        waves.append(('Diffusive',state,dw.time_utc,dw.gauge_cms))
        saved='S000_base' if state=='base' else state
        mc=pd.read_parquet(ROOT/'RAPID/outputs/event_window_runs_20261013/56/tail_216h'/saved/'outlet.parquet')
        t=(mc.time_utc-pd.Timestamp('2025-06-21T00:00Z')).dt.total_seconds().to_numpy();tp,qp=peak(t,mc.q_outlet_cms.to_numpy())
        rows.append(dict(model='MC saved',state=state,peak_cms=qp,peak_utc=str(pd.Timestamp('2025-06-21T00:00Z')+pd.to_timedelta(tp,unit='s')),peak_seconds=tp))
        # MC plotting only: retain five-minute extrema and exact peak samples.
        groups=mc.groupby(mc.time_utc.dt.floor('5min')).q_outlet_cms
        keep=sorted(set(groups.idxmin())|set(groups.idxmax()));mc=mc.loc[keep]
        waves.append(('MC saved',state,mc.time_utc,mc.q_outlet_cms))
    table=pd.DataFrame(rows)
    for model in table.model.unique():
        idx=table.model.eq(model);parent=table[idx & table.state.eq('base')].iloc[0]
        table.loc[idx,'collapse_peak_shift_min']=(table.loc[idx,'peak_seconds']-parent.peak_seconds)/60
        table.loc[idx,'collapse_centroid_shift_min']=(table.loc[idx,'centroid_seconds']-parent.centroid_seconds)/60
    table.to_csv(OUT/'comparison.csv',index=False)
    colors={'Local-inertial translation':'#b64b30','Diffusive':'#176d9c','MC saved':'#54823b'}
    obs=pd.read_parquet(DATA/'observed_downstream.parquet')
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    interactive=make_subplots(rows=2,cols=1,subplot_titles=['Event hydrographs','Around the main peak'],vertical_spacing=.17)
    for j,ax in enumerate(axes):
        ax.plot(obs.time,obs.discharge_cms,color='#999999',lw=.8,label='Observed downstream')
        interactive.add_trace(go.Scatter(x=obs.time.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=obs.discharge_cms.tolist(),mode='lines',name='Observed downstream',legendgroup='obs',showlegend=j==0,line=dict(color='#999999',width=1)),j+1,1)
        for model,state,t,q in waves:
            name=model+' · '+('original' if state=='base' else 'S014 collapse')
            ax.plot(t,q,color=colors[model],ls='-' if state=='base' else '--',lw=1.5,label=name)
            interactive.add_trace(go.Scatter(x=t.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=q.tolist(),mode='lines',name=name,legendgroup=name,showlegend=j==0,line=dict(color=colors[model],width=2,dash='solid' if state=='base' else 'dash')),j+1,1)
        ax.set_ylabel('Outlet discharge (m³/s)');ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    axes[0].set_xlim(pd.Timestamp('2025-06-25',tz='UTC'),pd.Timestamp('2025-07-02',tz='UTC'));axes[0].set_ylim(650,2050)
    axes[1].set_xlim(pd.Timestamp('2025-06-27T06:00Z'),pd.Timestamp('2025-06-27T20:00Z'));axes[1].set_ylim(1800,1950)
    axes[0].legend(fontsize=8,ncol=2);axes[1].set_xlabel('Time (UTC)')
    fig.suptitle('Example 56 · MATLAB-style local-inertial pilot (executed in Python)')
    fig.text(.07,.015,'Local-inertial: 46 / 42 cells, fixed downstream stage, same n as diffusive; no recalibration.\nMC uses different saved forcing preprocessing. Differences are not attributable to inertia alone.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(OUT/'hydrographs.png',dpi=160);plt.close(fig)
    interactive.update_xaxes(type='date',range=['2025-06-25','2025-07-02'],row=1,col=1)
    interactive.update_xaxes(type='date',range=['2025-06-27 06:00','2025-06-27 20:00'],row=2,col=1)
    interactive.update_yaxes(type='linear',range=[650,2050],autorange=False,title='Discharge (m³/s)',row=1,col=1)
    interactive.update_yaxes(type='linear',range=[1800,1950],autorange=False,title='Discharge (m³/s)',row=2,col=1)
    interactive.update_layout(template='plotly_white',height=1000,title='Example 56 · local-inertial pilot executed in Python',hovermode='x unified',margin=dict(t=100,b=180),legend=dict(orientation='h',y=-.13))
    note='<p>MATLAB could not run: license server error -96. These are Python-translated local-inertial results, not verified MATLAB output. Solid = original; dashed = S014. Local-inertial uses fixed downstream stage and one cell per reach; diffusive uses a normal outlet and finer mesh. Same n, different hydraulics. MC is a saved contextual reference with different forcing preprocessing.</p>'
    html=interactive.to_html(include_plotlyjs=True,full_html=True,config=dict(responsive=True,displaylogo=False)).replace('</body>',note+'</body>')
    (OUT/'hydrographs_plotly.html').write_text(html)
    report='MATLAB-style local-inertial pilot; executed in Python because MATLAB license server was unavailable.\n\n'+json.dumps(meta,indent=2)+'\n\n'+table.to_string(index=False)+'\n\nNo recalibration. Bed proxy is mean endpoint elevation. Fixed stage differs from diffusive normal outlet. Coarse reach cells change numerical resolution under collapse. No equation-only attribution; MC forcing preprocessing differs.\n'
    (OUT/'REPORT.txt').write_text(report)
    print(table.to_string(index=False))


if __name__=='__main__':main()

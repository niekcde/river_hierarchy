"""Report local-inertial / MC comparison with matching cells and input means."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from match_mc_grid56 import OUT,SAVED


def main():
    meta=json.loads((OUT/'manifest.json').read_text());table=pd.DataFrame(meta['runs'])
    table['peak_shift_min']=0.;table['centroid_shift_min']=0.
    for (grid,refined),g in table.groupby(['grid','refined']):
        parent=g[g.state.eq('S000_base')].iloc[0]
        table.loc[g.index,'peak_shift_min']=(g.peak_seconds-parent.peak_seconds)/60
        table.loc[g.index,'centroid_shift_min']=(g.centroid_seconds-parent.centroid_seconds)/60
    table.to_csv(OUT/'comparison.csv',index=False)
    checks=[]
    for state in ['S000_base','S014_unit_10']:
        a=table[table.state.eq(state)&table.grid.eq('mc')&~table.refined].iloc[0]
        b=table[table.state.eq(state)&table.grid.eq('mc')&table.refined].iloc[0]
        wa=pd.read_parquet(OUT/(state+'_mc.parquet'));wb=pd.read_parquet(OUT/(state+'_mc_refined.parquet'))
        checks.append(dict(state=state,refinement_peak_shift_minutes=(b.peak_seconds-a.peak_seconds)/60,
            max_flow_change_cms=float(abs(wa.q_outlet_cms-wb.q_outlet_cms).max())))
    pd.DataFrame(checks).to_csv(OUT/'timestep_checks.csv',index=False)
    waves=[];colors={'MC_reference':'#54823b','mc':'#b64b30','physical':'#176d9c'}
    names={'MC_reference':'MC reference','mc':'Local-inertial · MC grid','physical':'Local-inertial · unsplit control'}
    for grid in ['MC_reference','mc','physical']:
        for state in ['S000_base','S014_unit_10']:
            path=SAVED/state/'outlet.parquet' if grid=='MC_reference' else OUT/(state+'_'+grid+'.parquet')
            w=pd.read_parquet(path)
            groups=w.groupby(w.time_utc.dt.floor('5min')).q_outlet_cms
            w=w.loc[sorted(set(groups.idxmin())|set(groups.idxmax()))]
            waves.append((grid,state,w))
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    p=make_subplots(rows=2,cols=1,vertical_spacing=.18,subplot_titles=['Event hydrographs','Peak detail'])
    for grid,state,w in waves:
        label=names[grid]+' · '+('original' if state=='S000_base' else 'S014')
        for j,ax in enumerate(axes):
            ax.plot(w.time_utc,w.q_outlet_cms,color=colors[grid],ls='-' if state=='S000_base' else '--',lw=1.6,label=label)
            p.add_trace(go.Scatter(x=w.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=w.q_outlet_cms.tolist(),mode='lines',name=label,legendgroup=label,showlegend=j==0,line=dict(color=colors[grid],dash='solid' if state=='S000_base' else 'dash',width=2)),j+1,1)
    for j,ax in enumerate(axes):
        bounds=['2025-06-25','2025-07-02'] if j==0 else ['2025-06-27 06:00','2025-06-27 20:00']
        yr=[650,1950] if j==0 else [1800,1870]
        ax.set_xlim(*[pd.Timestamp(t,tz='UTC') for t in bounds]);ax.set_ylim(*yr)
        ax.set_ylabel('Outlet discharge (m³/s)');ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
        p.update_xaxes(type='date',range=bounds,row=j+1,col=1)
        p.update_yaxes(type='linear',range=yr,autorange=False,title='Discharge (m³/s)',row=j+1,col=1)
    axes[0].legend(ncol=2,fontsize=8);axes[1].set_xlabel('Time (UTC)')
    fig.suptitle('Example 56 · matching MC subdivision, roughness and forcing')
    fig.text(.07,.015,'58 / 53 matched cells; 46 / 42 unsplit controls. All n = 0.020446875; identical held inflow and one-minute output means.\nLocal-inertial model executed in Python, not verified against MATLAB. Fixed stage and evolving depth differ from MC.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(OUT/'hydrographs.png',dpi=160);plt.close(fig)
    p.update_layout(template='plotly_white',height=1000,title='Example 56 · matching MC cells, n and input',hovermode='x unified',margin=dict(t=100,b=180),legend=dict(orientation='h',y=-.13))
    note='<p>Matched physical reaches, widths, lengths, MC subdivisions, Manning n, forcing window and held one-minute inflow means. Outlet series are one-minute means. Solid = original; dashed = S014. Local-inertial is a Python translation, not MATLAB-verified; fixed stage and evolving depth/dynamic splits still differ from MC. Plot thinning retains extrema of five-minute bins; metrics use the full one-minute series. Unsplit controls use the SAME new forcing and roughness, not the earlier pilot settings.</p>'
    (OUT/'hydrographs_plotly.html').write_text(p.to_html(include_plotlyjs=True,full_html=True,config=dict(responsive=True,displaylogo=False)).replace('</body>',note+'</body>'))
    report='Example 56: local-inertial runs on exact MC subdivision.\n\n'+table.to_string(index=False)+'\n\nTIMESTEP CHECKS\n'+pd.DataFrame(checks).to_string(index=False)+'\n\n'+json.dumps(meta,indent=2)+'\n\nMatched: physical reaches, section counts/lengths/widths/connectivity, n, forcing/window/interval means. Bed-proxy endpoint profile is interpolated to each cell center. MC has fixed reference depths, fixed splits and no prescribed downstream stage; these remain hydraulic-model differences. Thus this is not an inertia-only experiment. Previous pilot used a different n and forcing; the unsplit controls here isolate subdivision.\n'
    (OUT/'REPORT.txt').write_text(report)
    print(table[['state','grid','refined','cells','peak_cms','peak_utc','peak_shift_min','centroid_shift_min','maximum_depth_m']].to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__=='__main__':main()

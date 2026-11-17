"""Summarize verification evidence; exclude clipped runs from main comparisons."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from verify_inertial56 import OUT,PREVIOUS,SAVED,ROOT

METRICS=['peak','centroid','rising50','falling50']
def main():
    meta=json.loads((OUT/'manifest.json').read_text());table=pd.DataFrame(meta['runs'])
    table['clock']=table.halfdt.map(lambda x:'quarter' if x=='quarter' else 'half' if x else 'standard')
    table['valid_event']=table.clipping_volume_m3.lt(1e-6)&table.limited_face_steps.eq(0)&table.relative_mass_residual.lt(1e-9)
    for _,g in table.groupby(['grid','slope_factor','clock']):
        base=g[g.state.eq('S000_base')].iloc[0]
        for key in METRICS:table.loc[g.index,key+'_shift_min']=(g[key+'_seconds']-base[key+'_seconds'])/60
    table.to_csv(OUT/'all_results.csv',index=False)
    mc=pd.read_csv(OUT/'mc_resolution.csv')
    for _,g in mc.groupby(['factor','dt']):
        base=g[g.state.eq('S000_base')].iloc[0]
        for key in METRICS:mc.loc[g.index,key+'_shift_min']=(g[key+'_seconds']-base[key+'_seconds'])/60
    mc.to_csv(OUT/'mc_resolution_summary.csv',index=False)
    grids=['physical','mc','floor200','max1000','max500','max250','max125','max62.5']
    selected=[]
    for grid in grids:
        clock='quarter' if grid=='max62.5' else 'half' if grid=='max125' else 'standard'
        selected.append(table[table.grid.eq(grid)&table.slope_factor.eq(1)&table.clock.eq(clock)])
    selected=pd.concat(selected);assert selected.valid_event.all()
    selected.to_csv(OUT/'main_comparison.csv',index=False)
    temporal=[]
    for (state,grid),g in table[table.slope_factor.eq(1)].groupby(['state','grid']):
        for ca,cb in [('standard','half'),('half','quarter')]:
            if {ca,cb}.issubset(set(g.clock)):
                a=g[g.clock.eq(ca)].iloc[0];b=g[g.clock.eq(cb)].iloc[0]
                qa=pd.read_parquet(OUT/(a.tag+'.parquet')).q_outlet_cms;qb=pd.read_parquet(OUT/(b.tag+'.parquet')).q_outlet_cms
                temporal.append(dict(state=state,grid=grid,coarse_clock=ca,fine_clock=cb,coarse_valid=bool(a.valid_event),fine_valid=bool(b.valid_event),peak_difference_min=(b.peak_seconds-a.peak_seconds)/60,max_flow_difference_cms=float(abs(qb-qa).max()),max_depth_difference_m=b.maximum_depth_m-a.maximum_depth_m))
    pd.DataFrame(temporal).to_csv(OUT/'timestep_checks.csv',index=False)
    # Profiles only at saved snapshots; max depth and Fr audits cover every step.
    diagnostic=[]
    for r in selected.itertuples():
        with np.load(OUT/(r.tag+'_diagnostics.npz')) as d:
            N=len(d['bed'])-1;a=d['a'];b=d['b'];interior=b<N;aa=a[interior];bb=b[interior];distance=d['dxf'][interior]
            bed_slope=(d['bed'][aa]-d['bed'][bb])/distance
            for k,time in enumerate(d['times']):
                depth=d['depth'][k];eta=d['bed'][:-1]+depth
                water_slope=(eta[aa]-eta[bb])/distance;positive=bed_slope>1e-10
                ratio=water_slope[positive]/bed_slope[positive]
                diagnostic.append(dict(state=r.state,grid=r.grid,time_seconds=time,positive_bed_faces=int(positive.sum()),face_fraction_water_slope_below_90pct_bed=float(np.mean(ratio<.9)),minimum_water_bed_slope_ratio=float(ratio.min()),reverse_flow_faces=int(np.sum(d['flow'][k,interior]<-1e-8))))
            cells=pd.read_csv(OUT/(r.tag+'_cells.csv'))
            # Junction branch fractions on outgoing faces, once per snapshot.
            rows=[]
            for ai in np.unique(a):
                js=np.flatnonzero(a==ai)
                if len(js)<2:continue
                for k,time in enumerate(d['times']):
                    total=d['flow'][k,js].sum()
                    for j in js:rows.append(dict(time_seconds=time,upstream_link=int(cells.id_link.iloc[ai]),downstream_link=int(cells.id_link.iloc[b[j]]),flow_cms=float(d['flow'][k,j]),fraction=float(d['flow'][k,j]/total)))
            pd.DataFrame(rows).to_csv(OUT/(r.tag+'_branch_splits.csv'),index=False)
    pd.DataFrame(diagnostic).to_csv(OUT/'hydraulic_profile_diagnostics.csv',index=False)
    # Spatial error measures: convergence of the collapse effect, not baseline only.
    spatial=[]
    for ga,gb in [('max1000','max500'),('max500','max250'),('max250','max125'),('max125','max62.5')]:
        a=selected[selected.grid.eq(ga)&selected.state.eq('S014_unit_10')].iloc[0];b=selected[selected.grid.eq(gb)&selected.state.eq('S014_unit_10')].iloc[0]
        row=dict(coarse_grid=ga,fine_grid=gb)
        for key in METRICS:row[key+'_collapse_change_min']=b[key+'_shift_min']-a[key+'_shift_min']
        spatial.append(row)
    pd.DataFrame(spatial).to_csv(OUT/'spatial_checks.csv',index=False)
    labels={'physical':'Unsegmented','mc':'MC grid','floor200':'200 m floor','max1000':'Max 1,000 m','max500':'Max 500 m','max250':'Max 250 m','max125':'Max 125 m','max62.5':'Max 62.5 m','MC_reference':'MC reference'}
    colors=dict(zip(grids+['MC_reference'],['#999999','#d95f02','#a6761d','#e7298a','#7570b3','#1b9e77','#66a61e','#0072b2','#111111']))
    p=make_subplots(rows=3,cols=1,vertical_spacing=.09,subplot_titles=['Event hydrographs','Peak detail','Collapse minus original discharge'])
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    waves={}
    for grid in grids+['MC_reference']:
        for state in ['S000_base','S014_unit_10']:
            if grid=='MC_reference':path=SAVED/state/'outlet.parquet'
            else:path=OUT/(selected[selected.grid.eq(grid)&selected.state.eq(state)].iloc[0].tag+'.parquet')
            w=pd.read_parquet(path);waves[grid,state]=w
            thin=w.iloc[::5];label=labels[grid]+' · '+('original' if state=='S000_base' else 'S014 collapse')
            for row in [1,2]:
                p.add_trace(go.Scatter(x=thin.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=thin.q_outlet_cms.tolist(),name=label,legendgroup=grid,showlegend=row==1,mode='lines',line=dict(color=colors[grid],dash='solid' if state=='S000_base' else 'dash',width=1.8)),row,1)
            if grid in ['physical','mc','max62.5','MC_reference']:
                for ax in axes:ax.plot(w.time_utc,w.q_outlet_cms,color=colors[grid],ls='-' if state=='S000_base' else '--',lw=1.4,label=label)
        base=waves[grid,'S000_base'];child=waves[grid,'S014_unit_10'];diff=child.q_outlet_cms-base.q_outlet_cms
        p.add_trace(go.Scatter(x=base.time_utc.iloc[::5].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=diff.iloc[::5].tolist(),name=labels[grid]+' · collapse difference',legendgroup=grid,showlegend=False,line=dict(color=colors[grid])),3,1)
    for row,bounds,yr in [(1,['2025-06-25','2025-07-02'],[650,1950]),(2,['2025-06-27 09:00','2025-06-27 16:00'],[1835,1857]),(3,['2025-06-25','2025-07-02'],None)]:
        p.update_xaxes(type='date',range=bounds,row=row,col=1);p.update_yaxes(title='ΔQ (m³/s)' if row==3 else 'Q (m³/s)',range=yr,row=row,col=1)
        if row<3:
            ax=axes[row-1];ax.set_xlim(*[pd.Timestamp(x,tz='UTC') for x in bounds]);ax.set_ylim(*yr);ax.set_ylabel('Discharge (m³/s)');ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    axes[0].legend(ncol=2,fontsize=8);fig.suptitle('Example 56: normal-flow outlet, fixed n; solid = original, dashed = S014')
    fig.tight_layout();fig.savefig(OUT/'hydrographs.png',dpi=150);plt.close(fig)
    p.update_layout(title='Example 56 · normal flow · numerical verification',template='plotly_white',height=1350,hovermode='x unified',legend=dict(orientation='h',y=-.12),margin=dict(b=220))
    summary=selected[selected.state.eq('S014_unit_10')][['grid','cells','peak_shift_min','centroid_shift_min','rising50_shift_min','falling50_shift_min','maximum_depth_m']]
    note='<p>All local-inertial runs use n=0.020446875, identical held inflow, and a normal-flow outlet. Solid = original; dashed = S014 collapse. The 125 m grid uses CFL 0.35 and 62.5 m uses CFL 0.175. Invalid CFL 0.7 / 125 m runs are excluded. MC references retain frozen-depth hydraulics. Click a legend item to toggle a model group. All times UTC; metrics use full one-minute means. Python translation; MATLAB comparison pending.</p>'+summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    html=p.to_html(include_plotlyjs=True,full_html=True,config=dict(responsive=True,displaylogo=False)).replace('</body>',note+'</body>')
    (OUT/'hydrographs_plotly.html').write_text(html)
    # Boundary results remain separate from spatial refinement.
    boundary=table[table.grid.eq('max250')&((table.slope_factor!=1)|table.clock.eq('half'))]
    boundary.to_csv(OUT/'boundary_sensitivity.csv',index=False)
    invalid=table[~table.valid_event]
    checks=json.loads((OUT/'hydraulic_checks.json').read_text())
    report='EXAMPLE 56 NORMAL-FLOW VERIFICATION\n\nPython CPU translation; no MATLAB numerical comparison has been run.\n'
    report+='Fixed n=0.020446875; measured widths and existing endpoint bed proxies; same held one-minute MC forcing and interval-mean outputs. No calibration.\n'
    report+='Main normal-flow boundary uses terminal-cell depth and physical terminal-reach slope (wide rectangular Manning). It is a zero depth-gradient closure, not a fixed stage.\n'
    report+='200 m floor is applied to unsplit physical reaches only; endpoints held fixed, so storage, distance and slope change. It is not a geometry-preserving refinement.\n'
    report+='Temporal acceptance: no depth clipping/limiter activation, relative mass residual <1e-9; compare peak timing and full wave. Spatial screening target: successive collapse timing changes below 1 minute for peak/centroid/50% limbs, alongside depth/flux diagnostics; this is a study choice, not a universal literature tolerance.\n'
    report+='50% limbs cross halfway between each run initial flow and its own peak; centroid uses common baseline and positive excess over full buffered window.\n\nHYDRAULIC CHECKS\n'+json.dumps(checks,indent=2)
    report+='\n\nMAIN COMPARISON (negative timing shift = earlier after collapse)\n'+selected[['state','grid','cells','peak_utc','peak_cms','peak_shift_min','centroid_shift_min','rising50_shift_min','falling50_shift_min','maximum_depth_m']].to_string(index=False)
    report+='\n\nSPATIAL REFINEMENT\n'+pd.DataFrame(spatial).to_string(index=False)
    report+='\n\nTIMESTEP CHECKS\n'+pd.DataFrame(temporal).to_string(index=False)
    report+='\n\nINVALID EVENT RUNS (do not use for physical inference)\n'+invalid[['tag','clipping_volume_m3','limited_face_steps','maximum_depth_m']].to_string(index=False)
    report+='\n\nBOUNDARY SENSITIVITY\n'+boundary[['state','slope_factor','clock','peak_utc','peak_shift_min','centroid_shift_min','maximum_depth_m']].to_string(index=False)
    report+='\n\nMC REFINEMENT\n'+mc.to_string(index=False)
    report+='\n\nINDEPENDENT BACKWATER ODE / CHANNEL PULSE BENCHMARKS\n'+(OUT/'additional_benchmarks.json').read_text()
    report+='\n\nINTERPRETATION LIMITS\nLake-at-rest, steady Manning and conservation tests verify specific properties, not the full network junction approximation. Pairwise connections and minimum interface widths are inherited from the supplied code. No independently verified network hydraulic solution is available. Spatial convergence within this formulation does not establish field accuracy. Bed elevations remain agreed proxies.\n'
    report+='Reduced water-surface slopes relative to bed slopes identify backwater-like profiles, not proof that backwater causes the MC timing difference. Inertia, changing depth, branch fractions, outlet closure and cross-section friction conventions all differ.\n'
    (OUT/'REPORT.txt').write_text(report)
    print(summary.to_string(index=False));print(pd.DataFrame(spatial).to_string(index=False));print('Invalid runs:',invalid.tag.tolist())

if __name__=='__main__':main()

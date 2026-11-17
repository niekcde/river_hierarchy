"""Evidence and interactive plots for grid, outlet-extension, MATLAB comparisons."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from extended_convergence56 import OUT,BASE
from verify_inertial56 import SAVED,PREVIOUS


def main():
    records=[]
    for p in OUT.glob('S*.json'):
        try:records.append(json.loads(p.read_text()))
        except json.JSONDecodeError:pass
    table=pd.DataFrame(records)
    geometry_checks=[]
    for r in table.itertuples():
        cells=pd.read_csv(OUT/r.tag/'cells.csv')
        physical=pd.read_parquet(PREVIOUS/(r.state+'_mc_parameters.parquet'))
        terminal=physical[~physical.v.isin(physical.u)].iloc[0]
        expected_length=physical.link_length_m.sum()+r.extension_m
        expected_area=(physical.width*physical.link_length_m).sum()+r.extension_m*terminal.width
        length_error=float(cells.link_length_m.sum()-expected_length)
        area_error=float((cells.width*cells.link_length_m).sum()-expected_area)
        passed=abs(length_error)<1e-6 and abs(area_error)/expected_area<1e-12 and cells.link_length_m.max()<=r.length*(1+1e-12)
        geometry_checks.append(dict(case=r.tag,minimum_cell_length_m=float(cells.link_length_m.min()),maximum_cell_length_m=float(cells.link_length_m.max()),total_length_error_m=length_error,storage_area_error_m2=area_error,passed=bool(passed)))
    pd.DataFrame(geometry_checks).to_csv(OUT/'geometry_checks.csv',index=False)
    for _,g in table.groupby(['length','extension_m','cfl']):
        if set(g.state)!={'S000_base','S014_unit_10'}:continue
        a=g[g.state.eq('S000_base')].iloc[0]
        for key in ['peak','centroid','rising50','falling50']:
            table.loc[g.index,key+'_shift_min']=(g[key+'_seconds']-a[key+'_seconds'])/60
    table.to_csv(OUT/'all_results.csv',index=False)
    grids=table[table.extension_m.eq(0)&table.cfl.eq(.35)].copy()
    grids.to_csv(OUT/'grid_comparison.csv',index=False)
    finest_checks=[]
    for state,g in grids.groupby('state'):
        if not g.length.eq(10).any():continue
        ref=g[g.length.eq(10)].iloc[0]
        qref=pd.read_parquet(OUT/ref.tag/'wave.parquet').q_outlet_cms.to_numpy()
        for r in g.itertuples():
            q=pd.read_parquet(OUT/r.tag/'wave.parquet').q_outlet_cms.to_numpy()
            row=dict(state=state,length=r.length,maximum_flow_difference_cms=float(abs(q-qref).max()),rms_flow_difference_cms=float(np.sqrt(np.mean((q-qref)**2))),maximum_depth_difference_m=r.maximum_depth_m-ref.maximum_depth_m)
            for key in ['peak','centroid','rising50','falling50']:
                row[key+'_absolute_difference_seconds']=getattr(r,key+'_seconds')-ref[key+'_seconds']
                row[key+'_collapse_effect_difference_seconds']=60*(getattr(r,key+'_shift_min')-ref[key+'_shift_min'])
            finest_checks.append(row)
    pd.DataFrame(finest_checks).to_csv(OUT/'differences_from_10m.csv',index=False)
    changes=[]
    for state,g in grids.groupby('state'):
        g=g.sort_values('length',ascending=False)
        for (_,a),(_,b) in zip(g.iloc[:-1].iterrows(),g.iloc[1:].iterrows()):
            row=dict(state=state,coarse_max_length_m=a.length,fine_max_length_m=b.length)
            for key in ['peak','centroid','rising50','falling50']:
                row[key+'_absolute_change_min']=(b[key+'_seconds']-a[key+'_seconds'])/60
                row[key+'_collapse_effect_change_min']=b[key+'_shift_min']-a[key+'_shift_min']
            qa=pd.read_parquet(OUT/a.tag/'wave.parquet').q_outlet_cms.to_numpy();qb=pd.read_parquet(OUT/b.tag/'wave.parquet').q_outlet_cms.to_numpy()
            row['maximum_hydrograph_change_cms']=float(abs(qb-qa).max());row['rms_hydrograph_change_cms']=float(np.sqrt(np.mean((qb-qa)**2)));changes.append(row)
    pd.DataFrame(changes).to_csv(OUT/'spatial_changes.csv',index=False)
    temporal=[]
    for r in table[table.cfl.eq(.175)].itertuples():
        a=table[table.state.eq(r.state)&table.length.eq(r.length)&table.extension_m.eq(r.extension_m)&table.cfl.eq(.35)].iloc[0]
        qa=pd.read_parquet(OUT/a.tag/'wave.parquet').q_outlet_cms;qb=pd.read_parquet(OUT/r.tag/'wave.parquet').q_outlet_cms
        temporal.append(dict(state=r.state,length=r.length,peak_change_seconds=r.peak_seconds-a.peak_seconds,centroid_change_seconds=r.centroid_seconds-a.centroid_seconds,max_flow_change_cms=float(abs(qb-qa).max()),coarse_valid=bool(a.valid),fine_valid=bool(r.valid)))
    pd.DataFrame(temporal).to_csv(OUT/'timestep_checks.csv',index=False)
    extensions=table[table.length.isin([125,50])&table.cfl.eq(.35)].copy()
    # Independently audit the upstream control volume at the monitoring face.
    monitor_checks=[]
    for r in extensions.itertuples():
        cells=pd.read_csv(OUT/r.tag/'cells.csv')
        original=pd.read_parquet(PREVIOUS/(r.state+'_mc_parameters.parquet'))
        mask=cells.id_link.isin(original.id_link).to_numpy()
        d=np.load(OUT/r.tag/'result.npz');inp=loadmat(OUT/r.tag/'input.mat')
        sample=inp['sample_indices'].ravel().astype(int)
        area=(cells.width*cells.link_length_m).to_numpy()[mask]
        storage=d['depth'][:,mask]@area;wave=d['wave'][sample]
        residual=storage-storage[0]-wave[:,3]+wave[:,8]
        error=float(abs(residual).max())
        monitor_checks.append(dict(case=r.tag,upstream_control_volume_mass_residual_m3=error,passed=error<.1))
    pd.DataFrame(monitor_checks).to_csv(OUT/'monitor_control_volume_checks.csv',index=False)
    for (state,length),g in extensions.groupby(['state','length']):
        a=g[g.extension_m.eq(0)].iloc[0]
        for key in ['peak','centroid','rising50','falling50']:extensions.loc[g.index,key+'_extension_change_min']=(g[key+'_seconds']-a[key+'_seconds'])/60
        extensions.loc[g.index,'depth_extension_change_m']=g.maximum_depth_m-a.maximum_depth_m
    extensions.to_csv(OUT/'extension_comparison.csv',index=False)
    parity=[]
    for p in OUT.glob('*/matlab_result.mat'):
        d=loadmat(p);r=np.load(p.parent/'result.npz')
        qm=np.diff(d['wave'][:,8])/60;qp=np.diff(r['wave'][:,8])/60
        row=dict(case=p.parent.name,maximum_interval_mean_flow_difference_cms=float(abs(qm-qp).max()),maximum_instantaneous_monitor_flow_difference_cms=float(abs(d['wave'][:,7]-r['wave'][:,7]).max()),maximum_saved_depth_difference_m=float(abs(d['depth']-r['depth']).max()),maximum_final_depth_difference_m=float(abs(d['h'].ravel()-r['h']).max()),maximum_saved_face_flow_difference_cms=float(abs(d['flow']-r['flow']).max()),matlab_steps=float(d['audit'].ravel()[0]),python_steps=float(json.loads((OUT/(p.parent.name+'.json')).read_text())['steps']),matlab_clipping_volume_m3=float(d['audit'].ravel()[2]))
        row['passed']=row['maximum_interval_mean_flow_difference_cms']<1e-6 and row['maximum_saved_depth_difference_m']<1e-8 and row['matlab_clipping_volume_m3']==0
        parity.append(row)
    pd.DataFrame(parity).to_csv(OUT/'matlab_comparison.csv',index=False)
    # Hydrographs: all requested lengths plus prior 125m reference and MC reference.
    lengths=[1000,500,250,200,150,125,100,50,25,10]
    colors=['#999999','#a6761d','#e7298a','#7570b3','#66a61e','#1b9e77','#d95f02','#e6ab02','#0072b2','#000080']
    fig=make_subplots(rows=3,cols=1,vertical_spacing=.09,subplot_titles=['Hydrographs at the existing outlet','Peak detail','Collapse timing change versus maximum cell length'])
    for length,color in zip(lengths,colors):
        for state in ['S000_base','S014_unit_10']:
            sel=grids[grids.length.eq(length)&grids.state.eq(state)]
            if sel.empty:continue
            r=sel.iloc[0];w=pd.read_parquet(OUT/r.tag/'wave.parquet').iloc[::5]
            for row in [1,2]:fig.add_trace(go.Scatter(x=w.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=w.q_outlet_cms.tolist(),name=f'Max {length} m · '+('original' if state=='S000_base' else 'S014'),legendgroup=str(length),showlegend=row==1,line=dict(color=color,dash='solid' if state=='S000_base' else 'dash')),row,1)
    for state in ['S000_base','S014_unit_10']:
        w=pd.read_parquet(SAVED/state/'outlet.parquet').iloc[::5]
        for row in [1,2]:fig.add_trace(go.Scatter(x=w.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=w.q_outlet_cms.tolist(),name='MC reference · '+('original' if state=='S000_base' else 'S014'),legendgroup='MC',showlegend=row==1,line=dict(color='black',dash='solid' if state=='S000_base' else 'dash')),row,1)
    child=grids[grids.state.eq('S014_unit_10')].sort_values('length')
    for key,color in [('peak','#0072b2'),('centroid','#d95f02'),('rising50','#1b9e77'),('falling50','#7570b3')]:fig.add_trace(go.Scatter(x=child.length.tolist(),y=child[key+'_shift_min'].tolist(),name=key+' shift',mode='lines+markers',line=dict(color=color)),3,1)
    fig.update_xaxes(type='date',range=['2025-06-25','2025-07-02'],row=1,col=1);fig.update_yaxes(range=[650,1950],title='Q (m³/s)',row=1,col=1)
    fig.update_xaxes(type='date',range=['2025-06-27 09:00','2025-06-27 16:00'],row=2,col=1);fig.update_yaxes(range=[1835,1857],title='Q (m³/s)',row=2,col=1)
    fig.update_xaxes(type='log',autorange='reversed',title='Maximum cell length (m); finer →',row=3,col=1);fig.update_yaxes(title='Collapse change (minutes)',row=3,col=1)
    fig.update_layout(template='plotly_white',height=1400,title='Example 56 · requested grid convergence · normal-flow outlet',hovermode='x unified',legend=dict(orientation='h',y=-.12),margin=dict(b=240))
    note='<p>All local-inertial lines: CFL 0.35, normal flow, identical roughness and held forcing. Solid=original, dashed=S014 collapse. Actual lengths preserve physical reaches and are at or below the named maximum. MC is the saved reference. Metrics use full one-minute means; plotted hydrographs sampled every five minutes.</p>'
    (OUT/'grid_convergence_plotly.html').write_text(fig.to_html(include_plotlyjs=True,full_html=True).replace('</body>',note+child[['length','cells','peak_shift_min','centroid_shift_min','rising50_shift_min','falling50_shift_min']].to_html(index=False)+'</body>'))
    ef=make_subplots(rows=2,cols=1,subplot_titles=['Peak at original outlet: 50 m grid','Boundary-extension effect on collapse timing'])
    for extension,color in zip([0,5000,10000,20000],['black','#0072b2','#d95f02','#1b9e77']):
        for state in ['S000_base','S014_unit_10']:
            sel=extensions[extensions.length.eq(50)&extensions.extension_m.eq(extension)&extensions.state.eq(state)]
            if sel.empty:continue
            r=sel.iloc[0];w=pd.read_parquet(OUT/r.tag/'wave.parquet').iloc[::5]
            ef.add_trace(go.Scatter(x=w.time_utc.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),y=w.q_outlet_cms.tolist(),name=f'{extension/1000:g} km extension · '+('original' if state=='S000_base' else 'S014'),line=dict(color=color,dash='solid' if state=='S000_base' else 'dash')),1,1)
    for length in [125,50]:
        g=extensions[extensions.length.eq(length)&extensions.state.eq('S014_unit_10')].sort_values('extension_m')
        for key in ['peak','centroid']:ef.add_trace(go.Scatter(x=(g.extension_m/1000).tolist(),y=g[key+'_shift_min'].tolist(),name=f'{length} m grid · {key}',mode='lines+markers'),2,1)
    ef.update_xaxes(type='date',range=['2025-06-27 10:00','2025-06-27 14:00'],row=1,col=1);ef.update_yaxes(range=[1852,1855],title='Q at original outlet (m³/s)',row=1,col=1)
    ef.update_xaxes(title='Extension (km)',row=2,col=1);ef.update_yaxes(title='Collapse timing change (min)',row=2,col=1)
    ef.update_layout(template='plotly_white',height=950,title='Normal-flow boundary moved downstream; original measurement point retained',legend=dict(orientation='h',y=-.2),margin=dict(b=180))
    (OUT/'outlet_extension_plotly.html').write_text(ef.to_html(include_plotlyjs=True,full_html=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    png,axes=plt.subplots(1,2,figsize=(12,4.6))
    for key,color in [('peak','#0072b2'),('centroid','#d95f02'),('rising50','#1b9e77'),('falling50','#7570b3')]:
        axes[0].plot(child.length,child[key+'_shift_min'],'o-',color=color,label=key)
    axes[0].set_xscale('log');axes[0].invert_xaxis();axes[0].set_xlabel('Maximum cell length (m); finer →');axes[0].set_ylabel('Collapse timing change (minutes)');axes[0].set_title('Spatial convergence, no extension');axes[0].legend()
    for length in [125,50]:
        g=extensions[extensions.length.eq(length)&extensions.state.eq('S014_unit_10')].sort_values('extension_m')
        axes[1].plot(g.extension_m/1000,g.peak_shift_min,'o-',label=f'{length} m grid: peak')
    axes[1].set_xlabel('Downstream extension (km)');axes[1].set_ylabel('Collapse peak-time change (minutes)');axes[1].set_title('Measured at original outlet');axes[1].legend()
    for ax in axes:ax.grid(alpha=.25)
    png.tight_layout();png.savefig(OUT/'convergence.png',dpi=160);plt.close(png)
    report='EXTENDED EXAMPLE 56 CONVERGENCE / OUTLET / MATLAB STUDY\n\n'
    report+='No calibration; n=0.020446875; existing bed proxies and physical junctions preserved. The requested lengths are maximum computational-cell lengths, not altered physical reaches. Added extension uses terminal width, n and bed slope. It is an idealized continuation, not surveyed geometry.\n'
    report+='Each physical reach is divided into ceil(L/Lmax) equal cells. Actual cell lengths need not be round numbers; physical total length and storage area remain unchanged. Geometry audit: '+str(all(r['passed'] for r in geometry_checks))+'.\n'
    report+='All extension metrics are measured at the ORIGINAL outlet face, using integrated interface flux. Remote-boundary discharge is separately saved.\n'
    report+='Times are UTC. Peak uses local quadratic interpolation of one-minute means. Centroid uses positive excess over the same baseline as previous runs. Rising/falling times cross halfway between each run initial flow and its own peak. Negative collapse shifts mean earlier arrival.\n'
    report+='MATLAB comparison uses a CPU adaptation of the supplied momentum formulation with corrected time accounting, held forcing and adopted normal-flow boundary. Shared steady initial state and geometry isolate numerical language agreement. It is not execution/validation of the untouched supplied GPU program and not independent physical validation.\n\nGRID RESULTS\n'+grids[['state','length','cells','valid','peak_utc','peak_cms','peak_shift_min','centroid_shift_min','rising50_shift_min','falling50_shift_min','maximum_depth_m']].sort_values(['length','state'],ascending=[False,True]).to_string(index=False)
    report+='\n\nSUCCESSIVE SPATIAL CHANGES\n'+pd.DataFrame(changes).to_string(index=False)
    report+='\n\nDIFFERENCES FROM 10m REFERENCE\n'+pd.DataFrame(finest_checks).to_string(index=False)
    report+='\n\nTIMESTEP CHECKS\n'+pd.DataFrame(temporal).to_string(index=False)
    report+='\n\nEXTENSION RESULTS\n'+extensions[['state','length','extension_m','valid','peak_extension_change_min','centroid_extension_change_min','peak_shift_min','centroid_shift_min','maximum_depth_m']].sort_values(['length','extension_m','state']).to_string(index=False)
    report+='\n\nMATLAB PARITY\n'+pd.DataFrame(parity).to_string(index=False)
    report+='\n\nORIGINAL OUTLET MONITOR CONTROL-VOLUME CHECK\n'+pd.DataFrame(monitor_checks).to_string(index=False)
    report+='\n\nActual numerical convergence must be assessed from both absolute hydrographs and collapse-minus-original metrics. Smaller cells add no new bathymetric information. Agreement of MATLAB/Python does not independently validate junction physics or bed assumptions.\n'
    if len(table)==38:
        checks=pd.DataFrame(finest_checks)
        selected=checks[checks.length.eq(100)&checks.state.eq('S014_unit_10')].iloc[0]
        keys=[key+'_collapse_effect_difference_seconds' for key in ['peak','centroid','rising50','falling50']]
        finest=grids[grids.length.eq(10)&grids.state.eq('S014_unit_10')].iloc[0]
        lead=(f'FINDINGS: {len(table)} unique hydraulic configurations completed. All valid: {bool(table.valid.all())}.\n'
              f'At 10m the collapse peak/centroid shifts are {finest.peak_shift_min:.6f}/{finest.centroid_shift_min:.6f} minutes.\n'
              f'At 100m the largest difference from 10m across the four collapse timing metrics is {abs(selected[keys].astype(float)).max():.3f} seconds.\n'
              'A maximum 100m cell length with CFL0.35 is a practical working choice for this example/event and sub-minute collapse-timing comparisons. Keep 10m as a verification reference, not a requirement for every future run. Other networks still need their own stability/convergence checks.\n'
              'The normal-flow boundary can remain at the original outlet for this comparison: the 5/10/20km extension sensitivity is small in collapse effects and settles with increasing extension. Physical downstream uncertainty was accepted by the user.\n'
              'The remaining discrepancy from MC is much larger than the measured spatial/time-step/extension sensitivities under the common roughness used here. These tests do not isolate which hydraulic difference causes it.\n\n')
        report=lead+report
    (OUT/'REPORT.txt').write_text(report)
    sources=[Path(__file__),Path(__file__).with_name('extended_convergence56.py'),Path(__file__).with_name('inertial_verification_kernel.py'),Path(__file__).with_name('run_verification_matlab.m'),Path(__file__).with_name('verify_inertial56.py'),SAVED/'forcing.parquet']
    sources += [PREVIOUS/(state+'_mc_parameters.parquet') for state in ['S000_base','S014_unit_10']]
    (OUT/'manifest.json').write_text(json.dumps(dict(runs=len(table),all_runs_valid=bool(table.valid.all()),matlab_cases=parity,source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    print(child[['length','cells','peak_shift_min','centroid_shift_min','maximum_depth_m']].to_string(index=False));print(pd.DataFrame(parity).to_string(index=False))

if __name__=='__main__':main()

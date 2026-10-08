"""Read-only evidence for the Excel review. Never changes routing or peak metrics."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory, event_clock
from rapid_tools.analysis import _load_gauge_quality_sheet, _load_station_timeseries_for_observed_lag
from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT

OUT = ROOT / 'RAPID/validation/hydrograph_match_review_20261005'
EXAMPLES = ['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']


def peak_window(t, q, fraction):
    """Connected sampled block containing the first global maximum; no gap bridging."""
    t = pd.DatetimeIndex(t); q = np.asarray(q)
    i = int(np.argmax(q)); a = b = i
    gap = pd.Timedelta(seconds=1.5 * np.median(np.diff(t.asi8)) / 1e9)
    while a > 0 and q[a-1] >= fraction*q[i] and t[a]-t[a-1] <= gap:
        a -= 1
    while b+1 < len(q) and q[b+1] >= fraction*q[i] and t[b+1]-t[b] <= gap:
        b += 1
    return t[a], t[b]


def hydrographs():
    quality = _load_gauge_quality_sheet(); summary=[]; peaks=[]; starts=[]; events=[]
    old = pd.read_csv(ROOT/'RAPID/validation/peak_windows_20261005/summary.csv', dtype={'example':str}).set_index('example')
    for ex in EXAMPLES:
        qr = quality[pd.to_numeric(quality.Example, errors='coerce').eq(int(ex))].iloc[0]
        start, end = pd.to_datetime(qr.Period_start, utc=True), pd.to_datetime(qr.Period_end, utc=True)
        h = pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv'); t,_ = event_clock(ex,h)
        series = {'forcing':(pd.DatetimeIndex(t), h.q_inflow_cms.to_numpy()), 'model':(pd.DatetimeIndex(t),h.q_outlet_cms.to_numpy())}
        for name, station in [('upstream',qr.Up),('downstream',qr.Down)]:
            g = _load_station_timeseries_for_observed_lag(str(station)); g = g[g.time.between(start,end)]
            series[name] = (pd.DatetimeIndex(g.time),g.discharge_cms.to_numpy())
        rec = dict(example=ex, **old.loc[ex].to_dict())
        for name,(times,q) in series.items():
            eq=times[q==q.max()]; a,b=peak_window(times,q,.95)
            rec.update({f'{name}_peak_cms':float(q.max()),f'{name}_peak_first_utc':eq.min(),f'{name}_peak_last_utc':eq.max(),f'{name}_95_start_utc':a,f'{name}_95_end_utc':b,f'{name}_95_mid_utc':a+(b-a)/2,f'{name}_95_width_h':(b-a).total_seconds()/3600,f'{name}_median_step_min':np.median(np.diff(times.asi8))/6e10})
            for fraction in [1.,.995,.99,.95]:
                a,b=peak_window(times,q,fraction)
                peaks.append(dict(example=ex,series=name,fraction=fraction,start_utc=a,end_utc=b,midpoint_utc=a+(b-a)/2,width_h=(b-a).total_seconds()/3600,peak_cms=float(q.max()),definition='Connected sampled block containing FIRST global maximum; gaps >1.5 median steps excluded'))
            if ex == '48':
                # Prominence, not agreement with the model, selects independent events.
                step_s=np.median(np.diff(times.asi8))/1e9
                ix,props=find_peaks(q,prominence=.005*np.ptp(q),distance=max(1,int(np.ceil(18*3600/step_s))),plateau_size=True)
                order=sorted(sorted(range(len(ix)),key=lambda j:q[ix[j]],reverse=True)[:4],key=lambda j:ix[j])
                for rank,j in enumerate(order,1):
                    a,b=int(props['left_edges'][j]),int(props['right_edges'][j])
                    events.append(dict(example=ex,series=name,chronological_rank=rank,first_utc=times[a],last_utc=times[b],midpoint_utc=times[a]+(times[b]-times[a])/2,peak_cms=float(q[ix[j]]),prominence_cms=float(props['prominences'][j]),selection='Four highest local peaks, >=18h separation, prominence >=0.5% of series range; low prominence admits daily shoulders. Ranks are not proven matches.'))
        if ex=='48':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
            for name,color in [('upstream','#2563eb'),('downstream','#111827'),('model','#dc2626')]:
                times,q=series[name];mask=(times>=start)&(times<=end)
                axes[0].plot(times[mask],q[mask],label=name,color=color)
                axes[1].plot(times[mask],(q[mask]-q[mask].min())/np.ptp(q[mask]),label=name,color=color)
            axes[0].set(ylabel='Discharge (m³/s)',title='48: whole selected window; independent candidate peaks are listed in Excel')
            axes[1].set(ylabel='Within-series min–max scale',xlabel='UTC',title='Shape comparison only: amplitude differences removed, no time alignment')
            for ax in axes:ax.legend();ax.grid(alpha=.15);ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            fig.savefig(OUT/'48_full_event.png',dpi=120);plt.close(fig)
        rec['peak_ratio_downstream_to_upstream']=rec['downstream_peak_cms']/rec['upstream_peak_cms']
        rec['error_95_mid_h']=(rec['model_95_mid_utc']-rec['downstream_95_mid_utc']).total_seconds()/3600
        rec['model_peak_inside_downstream_95']=bool(rec['downstream_95_start_utc']<=rec['model_peak_first_utc']<=rec['downstream_95_end_utc'])
        summary.append(rec)
        m=json.loads((base_directory(ex)/'run/hydrograph_metrics.json').read_text())
        offset=t.iloc[0]-pd.to_timedelta(float(h.time_seconds.iloc[0]),unit='s')
        q=h.q_outlet_cms.to_numpy(); ts=h.time_seconds.to_numpy(); tol=.01*(q.max()-q[0]); ii=int(np.flatnonzero(abs(q-q[0])>tol)[0]); y=abs(q-q[0]); departure=ts[ii-1]+(ts[ii]-ts[ii-1])*(tol-y[ii-1])/(y[ii]-y[ii-1])
        starts.append(dict(example=ex,gauge_window_start_utc=start,gauge_window_end_utc=end,legacy_local_min_start_utc=offset+pd.to_timedelta(m['event_start_time_seconds'],unit='s'),legacy_start_source=m['event_start_source'],forcing_peak_utc=rec['forcing_peak_first_utc'],legacy_start_to_forcing_peak_min=(m['inflow_peak_time_seconds']-m['event_start_time_seconds'])/60,parent_1pct_departure_utc=offset+pd.to_timedelta(departure,unit='s')))
        print(ex,'hydrograph complete',flush=True)
    return pd.DataFrame(summary),pd.DataFrame(peaks),pd.DataFrame(starts),pd.DataFrame(events)


def tributaries():
    # Reuse the reviewed endpoint projection and corridor orientation, not a spatial buffer.
    nb=json.loads((ROOT/'RAPID/notebooks/sword_corridor_investigation.ipynb').read_text())
    ns={}; exec(''.join(nb['cells'][1]['source']),ns); exec(''.join(nb['cells'][2]['source']),ns)
    regional={}; rows=[]; summaries=[]
    for ex in EXAMPLES:
        directory=ROOT/f'network_variants/output/{ex}_indep__anchored'
        state=directory/'states/S000_base'; manifest=json.loads((state/'variant/summary/variant_manifest.json').read_text())['sword_matching']
        region=manifest['candidate_region'].upper()
        if region not in regional:
            path=Path('/Volumes/PhD/SWORD/v17c')/f'sword_{region}_v17c_reaches.parquet'
            f=gpd.read_parquet(path); f=f.drop_duplicates('reach_id').set_index('reach_id',drop=False)
            graph=nx.DiGraph(); graph.add_nodes_from(f.index.astype(int))
            for col in ['rch_id_dn_1','rch_id_dn_2','rch_id_dn_3','rch_id_dn_4']:
                graph.add_edges_from((int(u),int(v)) for u,v in zip(f.index,f[col]) if pd.notna(v) and int(v)>0 and int(v) in f.index)
            regional[region]=(f,graph)
        f,graph=regional[region]
        support=ns['_load_reference_section_support'](str(directory/'reference_section/reference_section_summary.json'))
        mf=json.loads((base_directory(ex)/'hydraulic_context/hydraulic_context_manifest.json').read_text())
        links=gpd.read_file(mf['input_paths']['directed_links']); nodes=gpd.read_file(mf['input_paths']['directed_nodes']).to_crs(support['centerline_crs'])
        sources=set(links.id_us_node)-set(links.id_ds_node); sinks=set(links.id_ds_node)-set(links.id_us_node)
        anchors=nodes[nodes.id_node.isin(sources|sinks)].copy(); anchors['anchor_role']=np.where(anchors.id_node.isin(sources),'network source','network sink')
        eligible,_=ns['load_eligible_sword_reaches'](manifest,support['centerline_crs'])
        projections=ns['project_network_endpoints'](anchors,eligible)
        core=ns['trace_main_path'](eligible,projections.loc['network source','sword_reach_id'],projections.loc['network sink','sword_reach_id'])
        reachable=nx.descendants(graph,core[0])|{core[0]}
        candidates=[]
        for receiver in core[1:]:  # Exclude uncertain confluences above the source projection.
            for u in graph.predecessors(receiver):
                if u in core: continue
                # Rejoining side arms originating above the clipped source are also not tributaries.
                upstream_main=nx.ancestors(graph,core[0])|{core[0]}
                external=(u not in reachable) and not bool((nx.ancestors(graph,u)|{u}) & upstream_main)
                a=float(f.loc[u,'facc']); b=float(f.loc[receiver,'facc'])
                entry=dict(example=ex,tributary_reach_id=str(u),receiving_reach_id=str(receiver),classification='External tributary candidate' if external else 'Rejoining connected branch',facc_km2=a if a>0 else None,receiver_facc_km2=b if b>0 else None,area_fraction=a/b if min(a,b)>0 else None,width_m=float(f.loc[u,'width']),river_name=str(f.loc[u,'river_name']),facc_quality=str(f.loc[u,'facc_quality']),receiver_order=core.index(receiver),source='SWORD v17c '+region+' directed reach topology')
                rows.append(entry)
                if external:candidates.append(entry)
        inlet=float(f.loc[core[0],'facc']); outlet=float(f.loc[core[-1],'facc'])
        summaries.append(dict(example=ex,region=region,corridor_reaches=len(core),external_tributary_candidates=len(candidates),largest_tributary_area_fraction=max([r['area_fraction'] or 0 for r in candidates],default=0),inlet_reach_facc_km2=inlet,outlet_reach_facc_km2=outlet,reach_area_growth_fraction=outlet/inlet-1 if inlet>0 else None,unmapped_tributary_flag_reaches=int(f.loc[core,'trib_flag'].eq(1).sum()),network_reference_length_km=float(support['line'].length)/1000,scope='Projected network endpoints; reach-level areas may include area outside clipped endpoint reaches. No tributary hydrographs.'))
        print(ex,'tributaries',len(candidates),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def hydraulics():
    rows=[]; summaries=[]
    for ex in EXAMPLES:
        inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv'); rec=inv[inv.state.eq('S000_base')].iloc[0]
        f=pd.read_parquet(rec.frame).copy(); graph=nx.DiGraph(); graph.add_edges_from(zip(f.id_us_node,f.id_ds_node))
        assert nx.is_directed_acyclic_graph(graph)
        mass={n:0. for n in graph}; roots=[n for n in graph if graph.in_degree(n)==0]; assert len(roots)==1; mass[roots[0]]=1.
        f['flow_fraction']=0.
        for n in nx.topological_sort(graph):
            edges=f[f.id_us_node.eq(n)]
            for i,r in edges.iterrows():
                share=mass[n]*r.wid_adj_routing/edges.wid_adj_routing.sum();f.loc[i,'flow_fraction']=share;mass[r.id_ds_node]+=share
        f['residence_contribution_h']=f.flow_fraction*f['count']*f.rapid_k/3600
        f['physical_contribution_h']=f.flow_fraction*f.link_length_m/f.rapid_celerity_mps/3600
        f['effective_path_length_m']=f.flow_fraction*f['count']*f.rapid_effective_length_m
        manifest=json.loads((base_directory(ex)/'hydraulic_context/hydraulic_context_manifest.json').read_text())
        nodes=pd.read_parquet(manifest['base_nodes']).set_index('id_node')
        f['reference_mid_dist_out_km']=(f.id_us_node.map(nodes.reference_section_dist_out_m)+f.id_ds_node.map(nodes.reference_section_dist_out_m))/2000
        f.insert(0,'example',ex);rows.append(f)
        summaries.append(dict(example=ex,links=len(f),minimum_slope=float(f.slope.min()),median_slope=float(f.slope.median()),flow_weighted_mean_residence_h=float(f.residence_contribution_h.sum()),physical_length_mean_residence_h=float(f.physical_contribution_h.sum()),length_floor_extra_h=float((f.residence_contribution_h-f.physical_contribution_h).sum()),flow_weighted_path_km=float((f.flow_fraction*f.link_length_m).sum()/1000),share_residence_slope_below_1e5=float(f.loc[f.slope.lt(1e-5),'residence_contribution_h'].sum()/f.residence_contribution_h.sum()),share_residence_slope_below_1e4=float(f.loc[f.slope.lt(1e-4),'residence_contribution_h'].sum()/f.residence_contribution_h.sum()),effective_mean_speed_mps=float(f.effective_path_length_m.sum()/(f.residence_contribution_h.sum()*3600))))
    return pd.concat(rows,ignore_index=True),pd.DataFrame(summaries)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in zip(['summary','peak_windows','event_starts','48_peak_events'],hydrographs()):frame.to_csv(OUT/f'{name}.csv',index=False)
    for name,frame in zip(['tributaries','corridors'],tributaries()):frame.to_csv(OUT/f'{name}.csv',index=False)
    for name,frame in zip(['routing_links','routing_summary'],hydraulics()):frame.to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(review_date='2026-10-05',production_root=str(DEFAULT_REFRESH_ROOT),routing_changed=False,peak_metrics_changed=False,threshold_note='95% OF MAXIMUM, not the 95th percentile. Connected sampled blocks; not confidence intervals.',residence_note='Sum of width-allocation visit probability times segment count times K: analytical mean residence diagnostic, not a peak-lag prediction or independent hydraulic validation.',sword_source='https://zenodo.org/records/15299138'),indent=2))

if __name__=='__main__':main()

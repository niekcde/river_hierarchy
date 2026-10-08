"""Analysis adapter for versioned hydraulic refreshes, including island states.

Numerical outputs come only from the requested refresh. Historical trees supply
selection/topology metadata and observed-gauge references, never old hydrographs
or resistance values. Comparison geometry follows the refreshed domain CSV.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import networkx as nx
from .plotting import (_classify_plot_category, _exposure_fraction,
    _interp_first_crossing_time, _plateau_center_time, _weighted_centroid_time,
    _peak_window_center_time, _peak_window_centroid_time, _peak_window_width)

ROOT = Path(__file__).resolve().parents[3]
POLICY = 'shared-head-anchored-depth-signed-inflow-20261003'
DEFAULT_REFRESH_ROOT = (ROOT/'network_variants/output').resolve().parent/'hydraulic_refresh'/POLICY


def plateau_event_duration(t, q, initial, final, rise_tolerance, fall_tolerance, min_settled_seconds=3600):
    """First departure to last sustained return, using fixed parent tolerances.

    Mirrors the accepted synthetic event_duration_verification.duration rule.
    A later secondary peak above tolerance postpones the final return.
    """
    t=np.asarray(t,float);q=np.asarray(q,float)
    if len(t)<3 or not np.isfinite(t).all() or not np.isfinite(q).all() or (np.diff(t)<=0).any():
        raise ValueError('Invalid hydrograph')
    if min(rise_tolerance,fall_tolerance)<=0:raise ValueError('Nonpositive event amplitude')
    departure=np.flatnonzero(abs(q-initial)>rise_tolerance);outside=np.flatnonzero(abs(q-final)>fall_tolerance)
    if not len(departure) or not len(outside):raise ValueError('No resolved event')
    i,j=int(departure[0]),int(outside[-1])
    if i==0 or j==len(t)-1:raise ValueError('Censored event boundary')
    def crossing(k,level,values):
        return float(t[k]+(t[k+1]-t[k])*(level-values[k])/(values[k+1]-values[k]))
    start=crossing(i-1,rise_tolerance,abs(q-initial));end=crossing(j,fall_tolerance,abs(q-final))
    if end<t[q.argmax()] or end<=start or t[-1]-end<min_settled_seconds:
        raise ValueError('Insufficient final settlement')
    return end-start


def interpolated_peak(t, q, index=None):
    """Three-point peak estimate; tied/boundary/nonconcave peaks stay unresolved."""
    t=np.asarray(t,float);q=np.asarray(q,float);i=int(np.argmax(q)) if index is None else int(index)
    if i==0 or i==len(q)-1:return np.nan,'boundary'
    if np.count_nonzero(q==q[i])>1:return np.nan,'tied_peak'
    x=t[i-1:i+2]-t[i];a,b,_=np.polyfit(x,q[i-1:i+2]-q[i],2)
    if a>=0:return np.nan,'nonconcave'
    delta=-b/(2*a)
    if not x[0]/2<=delta<=x[-1]/2:return np.nan,'vertex_outside_peak_bin'
    return float(t[i]+delta),'ok'


def _wave_metrics(run_dir, expected_forcing=None, thresholds=None):
    h=pd.read_csv(run_dir/'outlet_hydrograph.csv')
    t=h.time_seconds.to_numpy(float);q=h.q_outlet_cms.to_numpy(float);inflow=h.q_inflow_cms.to_numpy(float)
    if not np.isfinite(q).all() or (np.ptp(inflow)>0 and np.ptp(q)<=0):
        raise ValueError(f'Invalid or flat outlet response to varying forcing: {run_dir}')
    if expected_forcing is not None:
        np.testing.assert_allclose(t,expected_forcing[0],rtol=0,atol=1e-9)
        np.testing.assert_allclose(inflow,expected_forcing[1],rtol=0,atol=1e-9)
    saved=json.loads((run_dir/'hydrograph_metrics.json').read_text())
    m=dict(peak_time_seconds=saved['peak_time_seconds'],lag_seconds=saved['lag_to_inflow_peak_seconds'],
        peak_discharge=float(q.max()),peak_excess=float(q.max()-max(q[0],q[-1])),
        peak_reference_cms=float(max(q[0],q[-1])),
        rise95_time_seconds=_interp_first_crossing_time(t,q,.95*q.max()),
        plateau995_center_seconds=_plateau_center_time(t,q,frac=.995),
        centroid50_seconds=_weighted_centroid_time(t,q,frac=.50),
        centroid75_seconds=_weighted_centroid_time(t,q,frac=.75),
        peak_window_center95_seconds=_peak_window_center_time(t,q,frac=.95),
        peak_window_centroid95_seconds=_peak_window_centroid_time(t,q,frac=.95),
        peak_window_width95_seconds=_peak_window_width(t,q,frac=.95),
        rise_time_seconds=saved['time_to_peak_seconds'],fall_time_50_seconds=saved['fall_time_50_seconds'],
        fall_time_10_seconds=saved['fall_time_10_seconds'],e_folding_time_seconds=saved['e_folding_time_seconds'])
    tolerances=thresholds if thresholds is not None else (.01*(q.max()-q[0]),.01*(q.max()-q[-1]))
    m['interpolated_peak_seconds'],m['interpolated_peak_status']=interpolated_peak(t,q,int(np.argmin(abs(t-saved['peak_time_seconds']))))
    m['inflow_peak_seconds']=saved['inflow_peak_time_seconds']
    try:
        m['event_duration_seconds']=plateau_event_duration(t,q,float(q[0]),float(q[-1]),*tolerances)
        m['event_duration_status']='ok'
    except ValueError as error:
        m['event_duration_seconds']=np.nan;m['event_duration_status']=str(error)
    return m,(t,inflow),tolerances


def paired_responses(parent,child):
    out={}
    for key in ['peak_time_seconds','rise95_time_seconds','plateau995_center_seconds','centroid50_seconds',
                'centroid75_seconds','peak_window_center95_seconds','peak_window_centroid95_seconds','peak_window_width95_seconds']:
        out[key.replace('_seconds','_shift_hours')]=(child[key]-parent[key])/3600
    lag=parent['lag_seconds'];delta=child['lag_seconds']-lag
    r=delta/lag if lag>0 else np.nan
    out.update(lag_to_inflow_peak_pct_shift_vs_base=100*r,lag_to_inflow_peak_relative_to_base=r,
        lag_to_inflow_peak_hour_shift_vs_base=delta/3600,delta_lag_to_inflow_peak_hours=delta/3600,
        delta_peak_time_hours=out['peak_time_shift_hours'],
        peak_speed_pct_change_vs_base=100*(lag/child['lag_seconds']-1) if lag>0 and child['lag_seconds']>0 else np.nan,
        peak_magnitude_pct_change_vs_base=100*(child['peak_discharge']/parent['peak_discharge']-1),
        peak_excess_change_pct=100*((child['peak_discharge']-parent['peak_reference_cms'])/parent['peak_excess']-1) if parent['peak_excess']>0 else np.nan,
        event_duration_change_pct=100*(child['event_duration_seconds']/parent['event_duration_seconds']-1),
        parent_event_duration_hours=parent['event_duration_seconds']/3600,
        child_event_duration_hours=child['event_duration_seconds']/3600,
        event_duration_status=child['event_duration_status'] if parent['event_duration_status']=='ok' else 'parent: '+parent['event_duration_status'],
        parent_peak_lag_hours=lag/3600,child_peak_lag_hours=child['lag_seconds']/3600)
    dp=child['interpolated_peak_seconds']-parent['interpolated_peak_seconds']
    lp=parent['interpolated_peak_seconds']-parent['inflow_peak_seconds']
    out.update(interpolated_peak_shift_hours=dp/3600,
        interpolated_lag_shift_pct=100*dp/lp if lp>0 else np.nan,
        peak_interpolation_status=child['interpolated_peak_status'] if parent['interpolated_peak_status']=='ok' else 'parent: '+parent['interpolated_peak_status'])
    for key in ['rise_time','fall_time_50','fall_time_10','e_folding_time']:
        out['delta_'+key+'_hours']=(child[key+'_seconds']-parent[key+'_seconds'])/3600
    return out


def _ids(value):
    if isinstance(value,str):return [int(v) for v in re.findall(r'\d+',value)]
    return [] if pd.isna(value) else [int(value)]


def _base_flow_context(parent):
    graph=nx.DiGraph();graph.add_edges_from(zip(parent.id_us_node,parent.id_ds_node))
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Base graph is cyclic')
    roots=[n for n in graph if graph.in_degree(n)==0]
    if len(roots)!=1:raise ValueError('Base has multiple inlets')
    outgoing={int(n):list(g.itertuples()) for n,g in parent.groupby('id_us_node',sort=False)}
    totals={n:sum(e.width for e in edges) for n,edges in outgoing.items()}
    shares={roots[0]:1.};order=list(nx.topological_sort(graph))
    for node in order:
        for edge in outgoing.get(node,[]):
            shares[edge.id_ds_node]=shares.get(edge.id_ds_node,0.)+shares.get(node,0.)*edge.width/totals[node]
    distance={}
    for node in reversed(order):
        distance[node]=max((e.link_length_m+distance[e.id_ds_node] for e in outgoing.get(node,[])),default=0.)
    return shares,totals,distance


def domain_geometry(parent,child,parent_ids,child_ids,flow_context=None):
    """Lengths and base-width flow exposure on the actual paired domain."""
    p=parent.set_index('id_link',verify_integrity=True).loc[list(parent_ids)]
    c=child.set_index('id_link',verify_integrity=True).loc[list(child_ids)]
    sources=set(p.id_us_node)-set(p.id_ds_node);sinks=set(p.id_ds_node)-set(p.id_us_node)
    if len(sources)!=1 or len(sinks)!=1:raise ValueError('Comparison domain lacks unique boundaries')
    source=int(next(iter(sources)));sink=int(next(iter(sinks)))
    for full,domain in [(parent,p),(child,c)]:
        ds=set(domain.id_us_node)-set(domain.id_ds_node);de=set(domain.id_ds_node)-set(domain.id_us_node)
        if len(ds)!=1 or len(de)!=1:raise ValueError('Comparison domain lacks unique boundaries')
        source_node=next(iter(ds));sink_node=next(iter(de))
        inside=(set(domain.id_us_node)|set(domain.id_ds_node))-{source_node,sink_node}
        excluded=full[~full.id_link.isin(domain.index)]
        violation=(excluded.id_us_node.isin(inside)|excluded.id_ds_node.isin(inside)
                   |excluded.id_us_node.eq(source_node)|excluded.id_ds_node.eq(sink_node))
        if violation.any():raise ValueError('Comparison domain has external flow exchange')
    shares,totals,distance=flow_context or _base_flow_context(parent)
    entry=p[p.id_us_node.eq(source)].width.sum()/totals[source]*shares[source]
    if not -1e-10<=entry<=1+1e-10:raise ValueError('Invalid flow exposure')
    mid=np.array([(distance[e.id_us_node]+distance[e.id_ds_node])/2 for e in p.itertuples()])
    position=float(np.average(mid,weights=p.link_length_m))
    return dict(matched_base_len=float(p.link_length_m.sum()),child_len=float(c.link_length_m.sum()),
        matched_len_frac=float(p.link_length_m.sum()/parent.link_length_m.sum()),
        child_parent_len_frac=float(c.link_length_m.sum()/p.link_length_m.sum()),
        width_fraction_of_total_raw=float(entry),matched_base_upstream_node=source,matched_base_downstream_node=sink,
        parent_domain_n_links=len(p),child_domain_n_links=len(c),downstream_network_distance_m=position,
        along_network_frac=position/max(distance.values()),distance_method='physical_longest_downstream_path')


def _verify_completed(root,example,rec):
    directory=root/example/rec.fingerprint
    complete=json.loads((directory/'completed.json').read_text())
    if complete['status']!='routed' or not Path(complete['qout']).is_file():
        raise ValueError(f'Incomplete refreshed run: {example}/{rec.state}')
    policy=json.loads((directory/'policy_manifest.json').read_text())
    if policy['policy_version']!=POLICY:raise ValueError('Mixed hydraulic policy versions')
    return directory,complete


def build_refreshed_analysis_frame(example_ids, *, refresh_root=DEFAULT_REFRESH_ROOT,
                                   geometry_root=ROOT/'network_variants/output',length_mode='matched_parent_vs_network'):
    if length_mode not in {'matched_parent_vs_network','child_vs_matched_parent'}:raise ValueError('Unknown length mode')
    refresh_root=Path(refresh_root);geometry_root=Path(geometry_root);rows=[]
    for value in example_ids:
        example=str(value).zfill(2);dest=refresh_root/example
        inv=pd.read_csv(dest/'inventory.csv',dtype={'example':str});ready=inv[inv.status.eq('ready')]
        base=ready[(ready.family=='indep') & (ready.state=='S000_base')].iloc[0]
        base_dir,base_done=_verify_completed(refresh_root,example,base)
        parent=pd.read_parquet(base.frame);base_source=Path(base.manifest).parent.parent
        units=pd.read_csv(base_source/'hierarchy/unit_summary.csv').set_index('unit_id')
        membership=pd.read_csv(base_source/'variant/summary/unit_link_membership.csv')
        flow_context=_base_flow_context(parent)
        bm,forcing,tolerances=_wave_metrics(base_dir/'run');metrics={base.fingerprint:bm};frames={base.fingerprint:parent};done_cache={base.fingerprint:base_done}
        comparisons=pd.read_csv(dest/'selected_domain_resistance/comparisons.csv')
        selected=ready[~ready.state.eq('S000_base')]
        if comparisons.duplicated(['family','state']).any() or set(zip(comparisons.family,comparisons.state))!=set(zip(selected.family,selected.state)) or not comparisons.status.eq('recomputed').all():
            raise ValueError(f'Refreshed comparison coverage is incomplete for {example}')
        comparisons=comparisons.set_index(['family','state'])
        registries={};suffix='_direction130' if example=='15' else ''
        island_registry=(pd.read_csv(geometry_root/f'{example}_island{suffix}__anchored/island_analysis_manifest_strict.csv').set_index('island_id')
                         if selected.family.eq('island').any() else pd.DataFrame())
        for rec in selected.itertuples():
            if rec.fingerprint not in metrics:
                rd,complete=_verify_completed(refresh_root,example,rec)
                if complete['routing_dt_s']!=base_done['routing_dt_s']:raise ValueError('Parent/child clocks differ')
                metrics[rec.fingerprint]=_wave_metrics(rd/'run',forcing,tolerances)[0]
                frames[rec.fingerprint]=pd.read_parquet(rec.frame);done_cache[rec.fingerprint]=complete
            if rec.parent_state!='S000_base':raise ValueError('Non-base parent requires explicit pairing')
            child=frames[rec.fingerprint];cmp=comparisons.loc[(rec.family,rec.state)]
            canonical=re.sub(r'^\.(.*?)\.promotion-.*$',r'\1',rec.state)
            if rec.family=='island':
                selection=island_registry.loc[canonical];unit_ids=[int(selection.source_unit_id)];bubble=np.nan
            else:
                source=Path(rec.manifest).parent.parent;experiment=source.parents[1]
                if str(experiment) not in registries:registries[str(experiment)]=pd.read_csv(experiment/'state_registry.csv').set_index('state_id')
                selection=registries[str(experiment)].loc[canonical];unit_ids=_ids(selection.selected_unit_ids);bubble=selection.compound_bubble_id
            uid=unit_ids[0] if len(unit_ids)==1 else np.nan
            meta=units.loc[uid].to_dict() if pd.notna(uid) and uid in units.index else {}
            domain_path=dest/'selected_domain_resistance'/rec.family/rec.state/'domain_links.csv';domain=pd.read_csv(domain_path)
            pids=set(domain.loc[domain.domain_role.eq('parent'),'id_link'].astype(int));cids=set(domain.loc[domain.domain_role.eq('child'),'id_link'].astype(int))
            geometry=domain_geometry(parent,child,pids,cids,flow_context)
            selected_ids=set(membership[membership.unit_id.isin(unit_ids)].id_link)
            dtype=('closed_compound_unit' if rec.family=='compound_bubbles' or bool(meta.get('is_compound',False)) else 'closed_unit')
            if rec.family!='compound_bubbles' and pids!=selected_ids:dtype='containing_compound_context'
            explicit=domain.get('comparison_domain_type',pd.Series(dtype,index=domain.index)).dropna().unique()
            if len(explicit)==1:dtype=str(explicit[0])
            expanded=dtype=='enclosing_surviving_anchor_domain'
            mode={'indep':'indep','compound_bubbles':'compound','island':'island'}[rec.family]
            row=dict(example_id=example,state_id=rec.state,canonical_state_id=canonical,technical_alias=canonical!=rec.state,
                fingerprint=rec.fingerprint,mode=mode,family=rec.family,unit_id=uid,target_id=selection.island_component_id if rec.family=='island' else bubble if rec.family=='compound_bubbles' else uid,
                target_type='island' if mode=='island' else 'compound_bubble' if mode=='compound' else 'unit',
                compound_bubble_id=bubble if pd.notna(bubble) else meta.get('compound_bubble_id',np.nan),
                is_compound=meta.get('is_compound',mode=='compound'),in_compound_bubble=meta.get('in_compound_bubble',mode=='compound'),
                unit_class=meta.get('class',None),hydraulic_domain_type=dtype,domain_expanded=expanded,
                boundary_relation='enclosing_surviving_anchors' if expanded else 'matched_comparison_domain',
                matched_base_link_ids=','.join(map(str,sorted(pids))),child_link_ids=','.join(map(str,sorted(cids))),
                Lambda_matched=float(cmp.Lambda_K),Lambda_K=float(cmp.Lambda_K),K_eq_matched_base=float(cmp.parent_K),K_variant_realized=float(cmp.child_K),
                source_policy=POLICY,refreshed_run_dir=str(dest/rec.fingerprint/'run'),domain_membership_path=str(domain_path),
                routing_dt_s=done_cache[rec.fingerprint]['routing_dt_s'],**geometry,**paired_responses(bm,metrics[rec.fingerprint]))
            row['plot_category']='island collapse' if mode=='island' else _classify_plot_category(pd.Series(row))
            exposure,source=_exposure_fraction(pd.Series(row),raw_width_share=geometry['width_fraction_of_total_raw'])
            row.update(width_fraction=exposure,width_fraction_of_total=exposure,exposure_fraction_source=source,
                length_fraction=geometry['matched_len_frac'] if length_mode=='matched_parent_vs_network' else geometry['child_parent_len_frac'])
            log=-np.log10(row['Lambda_matched']);lf=row['length_fraction'];cf=row['child_parent_len_frac']
            row.update(neg_log10_lambda=log,length_weighted_log_lambda=lf*log,width_weighted_log_lambda=lf*exposure*log,
                len_fraction_width_weighted_log_lambda=lf*exposure*log,child_parent_weighted_log_lambda=cf*log,
                child_parent_width_weighted_log_lambda=cf*exposure*log)
            rows.append(row)
        print(f'{example}: loaded {len(selected)} refreshed comparisons ({len(metrics)} distinct hydrographs)',flush=True)
    frame=pd.DataFrame(rows);frame['compound_downstream_order']=np.nan
    compounds=frame[frame['mode'].eq('compound') & ~frame.technical_alias].sort_values(['example_id','downstream_network_distance_m']).drop_duplicates(['example_id','compound_bubble_id'])
    order={}
    for ex,group in compounds.groupby('example_id',sort=False):
        for rank,row in enumerate(group.itertuples(),1):order[ex,row.compound_bubble_id]=rank
    frame['compound_downstream_order']=[order.get((r.example_id,r.compound_bubble_id),np.nan) for r in frame.itertuples()]
    frame.attrs['source_policy']=POLICY;frame.attrs['refresh_root']=str(refresh_root)
    return frame

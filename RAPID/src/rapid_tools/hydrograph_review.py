"""Gauge and refreshed-model review on verified event clocks; no peak alignment."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .refreshed_analysis import ROOT,DEFAULT_REFRESH_ROOT
from .analysis import _load_gauge_quality_sheet,_load_station_timeseries_for_observed_lag

@lru_cache(maxsize=32)
def base_directory(example):
    ex=str(example).zfill(2);inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv')
    base=inv[(inv.state=='S000_base')&(inv.status=='ready')].iloc[0]
    return DEFAULT_REFRESH_ROOT/ex/base.fingerprint


def event_clock(example, hydrograph):
    """Recover UTC only from identical historical forcing; otherwise preserve native UTC."""
    ex=str(example).zfill(2);provenance=json.loads((DEFAULT_REFRESH_ROOT/ex/'forcing_provenance.json').read_text())
    if provenance['source']=='saved_actual_event':
        original=pd.read_csv(provenance['path'])
        np.testing.assert_allclose(hydrograph.time_seconds,original.time_seconds,rtol=0,atol=1e-8)
        np.testing.assert_allclose(hydrograph.q_inflow_cms,original.q_inflow_cms,rtol=1e-10,atol=1e-8)
        return pd.to_datetime(original.time_utc,utc=True),'Verified original forcing UTC'
    return pd.to_datetime(hydrograph.time_utc,utc=True),'Recorded gauge-event UTC'


def gauge_overlay(example, *, include_model=True, child_run_dir=None, normalize=False):
    ex=str(example).zfill(2);quality=_load_gauge_quality_sheet();r=quality.loc[pd.to_numeric(quality.Example,errors='coerce').eq(int(ex))].iloc[0]
    start,end=pd.to_datetime(r.Period_start,utc=True),pd.to_datetime(r.Period_end,utc=True)
    fig=go.Figure();records=[]
    def add(t,q,label,color,dash=None,full_peak=False):
        q=np.asarray(q,float);t=pd.DatetimeIndex(t);m=(t>=start)&(t<=end)
        if not m.any():raise ValueError(f'No samples in gauge event window: {ex}/{label}')
        peak_mask=np.ones(len(t),dtype=bool) if full_peak else m
        tq=t[peak_mask];z=q[peak_mask];values=(q-z.min())/np.ptp(z) if normalize and np.ptp(z)>0 else q
        fig.add_scatter(x=t,y=values,name=label,mode='lines',line=dict(color=color,dash=dash))
        equal=tq[z==z.max()];records.append(dict(series=label,peak_first_utc=equal.min(),peak_last_utc=equal.max(),peak_cms=z.max(),samples=len(z),peak_search_scope='Complete routed event' if full_peak else 'Selected gauge window',peak_inside_gauge_window=bool(start<=equal.min()<=end)))
    for side,station,color in [('upstream',r.Up,'#2563eb'),('downstream',r.Down,'#111827')]:
        g=_load_station_timeseries_for_observed_lag(str(station));g=g[g.time.between(start,end)];add(g.time,g.discharge_cms,f'Observed {side} ({station})',color)
    clock='Gauge UTC';display_end=end
    if include_model:
        h=pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv');t,clock=event_clock(ex,h)
        display_end=max(end,t.iloc[int(np.argmax(h.q_outlet_cms))]+(end-start)*.1)
        add(t,h.q_inflow_cms,'Applied upstream forcing','#93c5fd','dot',full_peak=True);add(t,h.q_outlet_cms,'Model parent outlet','#dc2626',full_peak=True)
        if child_run_dir is not None:
            c=pd.read_csv(child_run_dir/'outlet_hydrograph.csv');tc,_=event_clock(ex,c);add(tc,c.q_outlet_cms,'Model child outlet','#f59e0b',full_peak=True)
            display_end=max(display_end,tc.iloc[int(np.argmax(c.q_outlet_cms))]+(end-start)*.1)
    fig.update_layout(title=f'Example {ex}: gauges'+(' and model' if include_model else '')+f'<br><sup>{clock}; no peak alignment'+('; independent gauge-window / full-model-event normalization' if normalize else '')+'</sup>',template='plotly_white',height=420,xaxis_title='Time (UTC)',yaxis_title='Normalized discharge' if normalize else 'Discharge (m³/s)',legend=dict(orientation='h'),margin=dict(t=85,b=85))
    fig.update_xaxes(range=[start,display_end])
    if display_end>end:fig.add_vline(x=end.timestamp()*1000,line_dash='dot',line_color='gray')
    fig.update_layout(updatemenus=[dict(type='buttons',direction='right',x=0,y=1.2,buttons=[dict(label='Gauges only',method='update',args=[{'visible':[True,True]+[False]*(len(fig.data)-2)}]),dict(label='Gauges + model',method='update',args=[{'visible':[True]*len(fig.data)}])])])
    audit=pd.DataFrame(records);audit['example_id']=ex;audit['quality_reason']=str(r.get('Reason',''));return fig,audit


def select_network_pairs(analysis, example, *, units=None, modes=None, max_pairs=6):
    """Select explicit (mode, state_id) keys; None means all, capped visibly by caller."""
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
        raise ValueError('max_pairs must be a positive integer')
    ex = str(example).zfill(2)
    frame = analysis.loc[analysis.example_id.astype(str).str.zfill(2).eq(ex)].copy()
    if 'technical_alias' in frame:
        frame = frame.loc[~frame.technical_alias.fillna(False)]
    if modes is not None:
        frame = frame.loc[frame['mode'].isin(modes)]
    frame = frame.sort_values(['mode','state_id']).drop_duplicates(['mode','state_id'])
    if units is not None:
        units = list(units)
        if any(not isinstance(k, (tuple, list)) or len(k) != 2 for k in units):
            raise ValueError(
                'Use (mode, state_id) pairs from the available-unit table, '
                'e.g. units=[("indep", "S001_unit_31")]. '
                'State IDs alone are not valid, even when modes is set.'
            )
        requested = list(dict.fromkeys(tuple(k) for k in units))
        lookup = frame.set_index(['mode','state_id'], drop=False)
        missing = [k for k in requested if k not in lookup.index]
        if missing:
            raise ValueError(f'Unknown units for example {ex} and selected modes: {missing}')
        frame = lookup.loc[requested].reset_index(drop=True) if requested else frame.iloc[:0]
    return frame.iloc[:max_pairs].copy(), frame.iloc[max_pairs:].copy()


def parent_child_hydrograph_figure(record, *, show_forcing=True):
    """Compare saved whole-network outlet responses on their verified common clock."""
    from pathlib import Path
    parent = pd.read_csv(base_directory(record.example_id)/'run/outlet_hydrograph.csv')
    child = pd.read_csv(Path(record.refreshed_run_dir)/'outlet_hydrograph.csv')
    np.testing.assert_allclose(parent.time_seconds, child.time_seconds, rtol=0, atol=1e-8)
    np.testing.assert_allclose(parent.q_inflow_cms, child.q_inflow_cms, rtol=1e-10, atol=1e-8)
    tp, _ = event_clock(record.example_id, parent)
    tc, _ = event_clock(record.example_id, child)
    if not pd.DatetimeIndex(tp).equals(pd.DatetimeIndex(tc)):
        raise ValueError('Parent and child event UTC clocks differ')
    fig = go.Figure()
    if show_forcing:
        fig.add_scatter(x=tp, y=parent.q_inflow_cms, name='Shared upstream forcing',
                        mode='lines', line=dict(color='#94a3b8', dash='dot'))
    fig.add_scatter(x=tp, y=parent.q_outlet_cms, name='Parent outlet',
                    mode='lines', line=dict(color='#2563eb', width=2))
    fig.add_scatter(x=tc, y=child.q_outlet_cms, name='Child outlet',
                    mode='lines', line=dict(color='#dc2626', width=2, dash='dash'))
    fig.update_layout(
        title=f'{record.example_id} / {record["mode"]} / {record.state_id}'
              '<br><sup>Whole-network outlet responses; shared forcing; no peak alignment</sup>',
        template='plotly_white', height=420, hovermode='x unified',
        xaxis_title='Time (UTC)', yaxis_title='Discharge (m³/s)',
        legend=dict(orientation='h'), margin=dict(t=85, b=75),
    )
    return fig


def collapse_hydrograph_overlay(records, *, show_forcing=True):
    """Overlay selected child outlets with one shared parent and optional forcing.

    Use select_network_pairs first to apply the requested unit/mode/count limits.
    Each saved child must pass the same forcing and clock checks as a pair plot.
    """
    from plotly.colors import qualitative
    if records.empty:
        raise ValueError('Select at least one collapse state')
    examples = records.example_id.astype(str).str.zfill(2).unique()
    if len(examples) != 1:
        raise ValueError('Overlay requires states from one network example')
    fig = None
    for index, (_, record) in enumerate(records.iterrows()):
        pair = parent_child_hydrograph_figure(record, show_forcing=show_forcing)
        child = go.Scatter(pair.data[-1])
        child.name = f'{record["mode"]} / {record.state_id}'
        child.line.color = qualitative.Dark24[index % len(qualitative.Dark24)]
        child.line.dash = 'dash' if index < len(qualitative.Dark24) else 'dot'
        child.legendgroup = child.name
        if fig is None:
            fig = pair
            fig.data = fig.data[:-1]
        fig.add_trace(child)
    fig.update_layout(
        title=f'Example {examples[0]}: parent and {len(records)} selected collapse states'
              '<br><sup>Shared forcing and UTC; no peak alignment; click legend entries to toggle</sup>',
        height=520, margin=dict(t=90, b=110),
    )
    return fig


def parent_child_network_figure(record, *, zoom_to_domain=True):
    """Two actual network states, with matched resistance domains highlighted."""
    import geopandas as gpd
    from pathlib import Path
    from plotly.subplots import make_subplots
    def load(directory):
        manifest = json.loads((directory/'hydraulic_context/hydraulic_context_manifest.json').read_text())
        return gpd.read_file(manifest['input_paths']['directed_links'])
    def ids(value):
        if isinstance(value,str):
            return [int(v.strip()) for v in value.split(',') if v.strip()]
        return [int(v) for v in value]
    parent = load(base_directory(record.example_id))
    child = load(Path(record.refreshed_run_dir).parent).to_crs(parent.crs)
    # Project geographic data for an undistorted map and metre-scale axes.
    if parent.crs.is_geographic:
        crs = parent.estimate_utm_crs(); parent = parent.to_crs(crs); child = child.to_crs(crs)
    p = parent[parent.id_link.isin(ids(record.matched_base_link_ids))]
    c = child[child.id_link.isin(ids(record.child_link_ids))]
    if p.empty or c.empty:
        raise ValueError('Empty matched parent or child domain')
    fig = make_subplots(rows=1,cols=2,subplot_titles=['Parent: full base state','Child: selected collapse state'])
    for col,full,domain,color in [(1,parent,p,'#2563eb'),(2,child,c,'#dc2626')]:
        for frame,name,width,shade in [(full,'Network',1,'#cbd5e1'),(domain,'Compared domain',3,color)]:
            xs=[];ys=[];labels=[]
            for row in frame.itertuples():
                geom=row.geometry
                if geom is None or geom.is_empty: continue
                for line in list(geom.geoms) if geom.geom_type=='MultiLineString' else [geom]:
                    x,y=line.xy;xs.extend([*x,None]);ys.extend([*y,None]);labels.extend([f'Link {row.id_link}']*len(x)+[''])
            fig.add_trace(go.Scattergl(x=xs,y=ys,text=labels,hovertemplate='%{text}<extra></extra>',mode='lines',line=dict(color=shade,width=width),name=f'{"Parent" if col==1 else "Child"} {name}'),row=1,col=col)
    bounds=np.vstack([p.total_bounds,c.total_bounds]) if zoom_to_domain else np.vstack([parent.total_bounds,child.total_bounds])
    lo=bounds[:,:2].min(axis=0);hi=bounds[:,2:].max(axis=0);padding=max(float((hi-lo).max())*.08,50)
    for col in [1,2]:
        fig.update_xaxes(range=[lo[0]-padding,hi[0]+padding],title_text='Easting (m)',row=1,col=col)
        fig.update_yaxes(range=[lo[1]-padding,hi[1]+padding],title_text='Northing (m)',scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
    fig.update_layout(title=f'{record.example_id} / {record["mode"]} / {record.state_id} — {record.hydraulic_domain_type}',template='plotly_white',height=550,legend=dict(orientation='h'))
    return fig

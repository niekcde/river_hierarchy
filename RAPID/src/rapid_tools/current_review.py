"""Review active published Cunge registries, never the historical refresh tree."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .hydrograph_review import select_network_pairs

ROOT=Path(__file__).resolve().parents[3]
OUTPUT_ROOT=ROOT/'network_variants/output'


def build_current_analysis_frame(example_ids, *, output_root=OUTPUT_ROOT):
    rows=[]
    for example in example_ids:
        ex=str(example).zfill(2);parts=[]
        for family,mode in [('indep','indep'),('compound_bubbles','compound'),('island','island')]:
            path=Path(output_root)/f'{ex}_{family}__anchored/rapid_run_registry.csv'
            if not path.exists(): continue
            d=pd.read_csv(path);d['mode']=mode;d['registry_path']=str(path)
            required={'method','outlet_parquet','parameters_parquet','calibration_feature_utc','fingerprint'}
            if not required.issubset(d.columns):
                warnings.warn(f'{ex}/{family}: registry has historical schema; excluded from current review.')
                continue
            if not d.status.eq('ran').all() or not d.method.eq('cunge').all():
                raise ValueError(f'Not a completed Cunge registry: {path}')
            parts.append(d)
        if not parts:
            warnings.warn(f'{ex}: no current published registry; excluded (no historical fallback).')
            continue
        registry=pd.concat(parts,ignore_index=True)
        bases=registry[registry.state_id.eq('S000_base')]
        if bases.fingerprint.nunique()!=1: raise ValueError(f'{ex}: ambiguous base')
        base=bases.iloc[0]
        for fp,g in registry.groupby('fingerprint'):
            if g.outlet_parquet.nunique()!=1: raise ValueError(f'{ex}/{fp}: inconsistent aliases')
        forcing=pd.read_parquet(base.forcing_parquet)
        inlet_time=pd.Timestamp(forcing.time_utc.iloc[int(forcing.forcing_cms.argmax())])
        parent_peak=pd.Timestamp(base.peak_time_utc)
        lag=(parent_peak-inlet_time).total_seconds()
        seen=set()
        for r in registry.itertuples():
            if r.fingerprint==base.fingerprint: continue
            delta=(pd.Timestamp(r.peak_time_utc)-parent_peak).total_seconds()
            feature_delta=(pd.Timestamp(r.calibration_feature_utc)-pd.Timestamp(base.calibration_feature_utc)).total_seconds()
            rows.append(dict(example_id=ex,mode=r.mode,state_id=r.state_id,fingerprint=r.fingerprint,
                technical_alias=r.fingerprint in seen,target_id=r.state_id,hydraulic_domain_type='Current whole-network output',
                parent_domain_n_links=np.nan,child_domain_n_links=np.nan,
                parent_outlet_parquet=base.outlet_parquet,outlet_parquet=r.outlet_parquet,
                parent_parameters_parquet=base.parameters_parquet,parameters_parquet=r.parameters_parquet,
                forcing_parquet=r.forcing_parquet,parent_forcing_parquet=base.forcing_parquet,
                parent_feature_utc=base.calibration_feature_utc,child_feature_utc=r.calibration_feature_utc,
                parent_peak_utc=base.peak_time_utc,child_peak_utc=r.peak_time_utc,
                parent_peak_cms=base.peak_discharge_cms,child_peak_cms=r.peak_discharge_cms,
                peak_time_shift_hours=delta/3600,sampled_peak_shift_minutes=delta/60,
                reviewed_feature_shift_minutes=feature_delta/60,
                lag_to_inflow_peak_pct_shift_vs_base=100*delta/lag if lag>0 else np.nan,
                peak_total_change_pct=100*(r.peak_discharge_cms/base.peak_discharge_cms-1),
                forcing_dt_seconds=r.forcing_dt_seconds,registry_path=r.registry_path,
                source_policy='Active published Cunge registry',
                parent_feature_kind=base.calibration_feature_kind,child_feature_kind=r.calibration_feature_kind))
            seen.add(r.fingerprint)
    return pd.DataFrame(rows)


def add_centerline_comparisons(analysis, example_ids=None, *, output_root=OUTPUT_ROOT):
    """Add completed reference-centerline runs to an existing current analysis.

    Repeated calls replace the same centerline keys. Parent outputs and forcing
    must match the current comparison clock; no event-rerun fallback is used.
    """
    examples = analysis.example_id.unique() if example_ids is None else example_ids
    additions = []
    for example in examples:
        ex = str(example).zfill(2)
        matches = analysis[analysis.example_id.astype(str).str.zfill(2).eq(ex)]
        if matches.empty:
            raise ValueError(f'{ex}: no current parent comparison available')
        row = matches.iloc[0].copy()
        root = Path(output_root)/f'{ex}_indep__anchored/reference_section'
        run = root/'rapid/cunge/run'
        receipt = json.loads((run/'completion.json').read_text())
        result = json.loads((run/'result.json').read_text())
        if not receipt.get('complete') or result.get('method') != 'cunge':
            raise ValueError(f'{ex}: centerline is not a completed Cunge run')
        outlet = pd.read_parquet(run/'outlet.parquet')
        peak = outlet.iloc[int(outlet.q_outlet_cms.argmax())]
        forcing = pd.read_parquet(run/'forcing.parquet')
        delta = (pd.Timestamp(peak.time_utc)-pd.Timestamp(row.parent_peak_utc)).total_seconds()
        inlet = pd.Timestamp(forcing.time_utc.iloc[int(forcing.forcing_cms.argmax())])
        lag = (pd.Timestamp(row.parent_peak_utc)-inlet).total_seconds()
        updates = dict(
            mode='centerline', state_id='fully_collapsed', target_id='fully_collapsed',
            fingerprint=result['fingerprint'], technical_alias=False,
            hydraulic_domain_type='Whole-network centerline collapse',
            outlet_parquet=str(run/'outlet.parquet'),
            parameters_parquet=str(root/'rapid/cunge/prep/parameters.parquet'),
            forcing_parquet=str(run/'forcing.parquet'),
            registry_path=str(root/'rapid_run_registry.csv'),
            child_peak_utc=str(peak.time_utc), child_peak_cms=float(peak.q_outlet_cms),
            child_feature_utc=result['feature_utc'], child_feature_kind=result['feature_kind'],
            peak_time_shift_hours=delta/3600, sampled_peak_shift_minutes=delta/60,
            reviewed_feature_shift_minutes=(pd.Timestamp(result['feature_utc'])-pd.Timestamp(row.parent_feature_utc)).total_seconds()/60,
            peak_total_change_pct=100*(float(peak.q_outlet_cms)/row.parent_peak_cms-1),
            lag_to_inflow_peak_pct_shift_vs_base=100*delta/lag if lag>0 else np.nan,
            source_policy='Completed current reference-centerline run',
        )
        for key, value in updates.items():
            row[key] = value
        row['final_scientific_acceptance'] = receipt.get('final_scientific_acceptance', False)
        load_pair(row)
        additions.append(row)
    if not additions:
        return analysis.copy()
    added = pd.DataFrame(additions)
    replaced = (analysis['mode'].eq('centerline') & analysis.state_id.eq('fully_collapsed')
                & analysis.example_id.astype(str).str.zfill(2).isin(added.example_id.astype(str).str.zfill(2)))
    return pd.concat([analysis.loc[~replaced], added], ignore_index=True)


def load_pair(record):
    parent=pd.read_parquet(record.parent_outlet_parquet)
    child=pd.read_parquet(record.outlet_parquet)
    parent_clock = pd.DatetimeIndex(parent.time_utc)
    child_clock = pd.DatetimeIndex(child.time_utc)
    if record.get('mode') == 'centerline':
        # Completed runs may have different settling-tail lengths. Preserve both
        # full series, but require the same start and every shared sample time.
        shared = min(len(parent_clock), len(child_clock))
        same_clock = shared > 0 and parent_clock[:shared].equals(child_clock[:shared])
    else:
        same_clock = parent_clock.equals(child_clock)
    if not same_clock:
        raise ValueError('Current parent/child UTC clocks differ')
    forcing=pd.read_parquet(record.forcing_parquet)
    if Path(record.parent_forcing_parquet).resolve()!=Path(record.forcing_parquet).resolve():
        other=pd.read_parquet(record.parent_forcing_parquet)
        pd.testing.assert_frame_equal(forcing[['time_utc','forcing_cms']],other[['time_utc','forcing_cms']])
    for label,w in [('parent',parent),('child',child)]:
        i=int(w.q_outlet_cms.argmax())
        if pd.Timestamp(w.time_utc.iloc[i])!=pd.Timestamp(record[label+'_peak_utc']):
            raise ValueError(f'{label}: registry maximum timestamp disagrees with plotted data')
        np.testing.assert_allclose(w.q_outlet_cms.iloc[i],record[label+'_peak_cms'],rtol=1e-10)
    return parent,child,forcing


def parent_child_hydrograph_figure(record, *,
    show_forcing=True, show_downstream_gauge=False,
    show_parent=True, show_children=True, show_title = True,
    x_range=None, y_range=None,
    show_peaks=False):
    parent,child,forcing=load_pair(record);fig=go.Figure()
    if show_forcing:
        fig.add_scatter(x=forcing.time_utc.map(lambda t: pd.Timestamp(t).isoformat()),y=forcing.forcing_cms,
            name='Upstream gauge',line=dict(color='#94a3b8'))
    if show_downstream_gauge:
        q = forcing.downstream_cms.copy()

        # Preserve the prepared gauge's quality masks.
        if "downstream_gap" in forcing:
            q = q.mask(forcing.downstream_gap)
        if "downstream_supported" in forcing:
            q = q.where(forcing.downstream_supported)

        if q.notna().any() and q.max() > 0:
            fig.add_scatter(
                x=forcing.time_utc.map(lambda t: pd.Timestamp(t).isoformat()),
                y=q * forcing.forcing_cms.max() / q.max(),
                name="Downstream gauge",
                line=dict(color='#94a3b8', dash="5px,9px",),
                connectgaps=False,
            )
    hydros = []
    if (show_children and show_parent):
        hydros = [('Full complexity',parent,'#2563eb','solid'),('Simplified',child,'#dc2626','dash')]
    elif show_parent:
        hydros = [('Full complexity',parent,'#2563eb','solid')]
    elif show_children:
        hydros = [('Simplified',child,'#dc2626','dash')]
    if len(hydros) != 0:
        for name,w,color,dash in hydros:
            fig.add_scatter(x=w.time_utc.map(lambda t: pd.Timestamp(t).isoformat()),y=w.q_outlet_cms,
                name=name,line=dict(color=color,dash=dash))
            if show_peaks:
                peak = w.iloc[w.q_outlet_cms.argmax()]
                fig.add_scatter(
                    x=[pd.Timestamp(peak.time_utc).isoformat()],
                    y=[peak.q_outlet_cms],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=f"{name} peak",
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M} UTC"
                        "<br>%{y:.2f} m³/s"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )

    if show_title:
        fig.update_layout(title=f'{record.example_id} / {record["mode"]} / {record.state_id}<br><sup>Current published Cunge · UTC · no peak alignment · sampled shift {record.sampled_peak_shift_minutes:+g} min</sup>',
            template='plotly_white',height=440,xaxis_title='Time (UTC)',yaxis_title='Discharge (m³/s)',
            legend=dict(orientation='h'),margin=dict(t=85,b=75),
            meta=dict(parent_outlet=record.parent_outlet_parquet,child_outlet=record.outlet_parquet,registry=record.registry_path))
    else:
        fig.update_layout(
                    template='plotly_white',height=440,xaxis_title='Time (UTC)',yaxis_title='Discharge (m³/s)',
                    # legend=dict(orientation='h'),
                    legend=dict(
                        x=0.65,
                        y=0.35,
                        xanchor='right',
                        yanchor='top',
                        orientation='v',
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='lightgrey',
                        borderwidth=1),
                    margin=dict(t=85,b=75),

                    meta=dict(parent_outlet=record.parent_outlet_parquet,child_outlet=record.outlet_parquet,registry=record.registry_path))

    if x_range is not None:
        start, end = (pd.Timestamp(v) for v in x_range)
        fig.update_xaxes(range=[start.isoformat(), end.isoformat()])

    if y_range is not None:
        fig.update_yaxes(range=[y_range[0], y_range[1]])
    
    return fig


def collapse_hydrograph_overlay(records, *, show_forcing=True):
    from plotly.colors import qualitative
    if records.empty or records.example_id.nunique()!=1: raise ValueError('Select states from one network')
    fig=None
    for i,(_,record) in enumerate(records.iterrows()):
        pair=parent_child_hydrograph_figure(record,show_forcing=show_forcing)
        trace=go.Scatter(pair.data[-1]);trace.name=f'{record["mode"]} / {record.state_id}';trace.line.color=qualitative.Dark24[i%24]
        if fig is None: fig=pair;fig.data=fig.data[:-1]
        fig.add_trace(trace)
    fig.update_layout(title=f'Example {records.iloc[0].example_id}: current published Cunge parent and children<br><sup>Shared forcing and native UTC; no peak alignment</sup>')
    return fig


def gauge_overlay(example, *, analysis, include_model=True, normalize=False):
    r=analysis[analysis.example_id.eq(str(example).zfill(2))].iloc[0]
    parent,_,forcing=load_pair(r);fig=go.Figure();rows=[]
    series=[('Prepared upstream gauge',forcing,'upstream_cms'),('Prepared downstream gauge',forcing,'downstream_cms')]
    if include_model: series += [('Applied forcing',forcing,'forcing_cms'),('Current parent outlet',parent,'q_outlet_cms')]
    for label,w,col in series:
        q=w[col].copy()
        if col in ['upstream_cms','downstream_cms']:
            side=col.split('_')[0]
            if side+'_gap' in w: q=q.mask(w[side+'_gap'])
            if side+'_supported' in w: q=q.where(w[side+'_supported'])
        if not q.notna().any(): continue
        i=q.idxmax();rows.append(dict(example_id=r.example_id,series=label,peak_utc=w.time_utc.loc[i],peak_cms=q.loc[i]))
        if normalize and q.max()>q.min(): q=(q-q.min())/(q.max()-q.min())
        fig.add_scatter(x=w.time_utc.map(lambda t: pd.Timestamp(t).isoformat()),y=q,name=label,connectgaps=False)
    fig.update_layout(title=f'{r.example_id}: current published Cunge and prepared gauge series<br><sup>Published forcing clock; no peak alignment; gauge gaps retained</sup>',template='plotly_white',height=440,xaxis_title='Time (UTC)',yaxis_title='Normalized discharge' if normalize else 'Discharge (m³/s)',legend=dict(orientation='h'))
    return fig,pd.DataFrame(rows)


def geometry_manifest_path(record, key):
    """Resolve geometry through saved provenance, including island reruns."""
    state = Path(record[key]).parents[3]
    path = state/'hydraulic_context/hydraulic_context_manifest.json'
    if path.exists():
        return path
    catalog = Path(record.registry_path).parent/'island_analysis_manifest.csv'
    if key == 'parameters_parquet' and catalog.exists():
        entries = pd.read_csv(catalog)
        entries = entries[entries.island_id.eq(record.state_id)]
        if len(entries) == 1:
            path = Path(entries.iloc[0].hydraulic_frame_path).parent/'hydraulic_context_manifest.json'
            if path.exists():
                return path
    raise FileNotFoundError(f'No hydraulic geometry manifest for {record.state_id}: {key}')


# def parent_child_network_figure(record, *, zoom_to_domain=True):
    """Highlight and optionally zoom to a domain validated on current graphs."""
    import geopandas as gpd
    from plotly.subplots import make_subplots
    from .current_domains import load_domain_pair
    _,_,domain=load_domain_pair(record)
    frames=[]
    for key in ['parent_parameters_parquet','parameters_parquet']:
        if record.get('mode') == 'centerline' and key == 'parameters_parquet':
            root = Path(record.registry_path).parent
            summary = json.loads((root/'reference_section_summary.json').read_text())
            geometry_path = summary['paths']['directed_links_path']
        else:
            manifest=json.loads(geometry_manifest_path(record,key).read_text())
            geometry_path = manifest['input_paths']['directed_links']
        geo=gpd.read_file(geometry_path)
        params=pd.read_parquet(record[key])
        if set(geo.id_link)!=set(params.id_link): raise ValueError('Map reach IDs differ from routed physical graph')
        frames.append(geo)
    parent,child=frames;child=child.to_crs(parent.crs)
    if parent.crs.is_geographic:
        crs=parent.estimate_utm_crs();parent=parent.to_crs(crs);child=child.to_crs(crs)

    fig=make_subplots(rows=1,cols=2,subplot_titles=['Current parent network','Current child network'])
    p=parent[parent.id_link.isin(domain['parent_links'])]
    c=child[child.id_link.isin(domain['child_links'])]
    for col,frame,part,color in [(1,parent,p,'#2563eb'),(2,child,c,'#dc2626')]:
        for layer,label,shade,width in [(frame,'Network context','#cbd5e1',1),
                                        (part,'Compared collapse domain',color,1)]:
            x=[];y=[];labels=[]
            for r in layer.itertuples():
                geom=r.geometry
                if geom is None or geom.is_empty: continue
                for line in list(geom.geoms) if geom.geom_type=='MultiLineString' else [geom]:
                    xx,yy=line.xy;x.extend([*xx,None]);y.extend([*yy,None])
                    labels.extend([f'Link {r.id_link}']*len(xx)+[''])
            fig.add_trace(go.Scattergl(x=x,y=y,text=labels,
                                       hovertemplate='%{text}<extra></extra>',
                                       mode='lines',line=dict(color=shade,width=width),
                                       name=f'{"Parent" if col==1 else "Child"}: {label}'),row=1,col=col)
        fig.update_yaxes(scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
    bounds=np.vstack([p.total_bounds,c.total_bounds] if zoom_to_domain else [parent.total_bounds,child.total_bounds])
    low=bounds[:,:2].min(axis=0);high=bounds[:,2:].max(axis=0);pad=max((high-low).max()*.08,50.)
    # for col in [1,2]:
    #     fig.update_xaxes(range=[low[0]-pad,high[0]+pad],title_text='Easting (m)',row=1,col=col)
    #     fig.update_yaxes(range=[low[1]-pad,high[1]+pad],title_text='Northing (m)',row=1,col=col)
    # fig.update_layout(title=f'{record.example_id} / {record.state_id}<br><sup>Current {domain["domain_type"]}: node {domain["source"]} → {domain["sink"]}; {len(p)} parent / {len(c)} child reaches</sup>',template='plotly_white',height=550,meta=domain)

    for col in [1,2]:
        fig.update_xaxes(range=[low[0]-pad,high[0]+pad],visible=False,row=1,col=col)
        fig.update_yaxes(range=[low[1]-pad,high[1]+pad],visible=False,row=1,col=col)
    fig.update_layout(plot_bgcolor='white',paper_bgcolor='white')
    return fig


def parent_child_network_figure(record, *, zoom_to_domain=True, layout='panels',
                                show_children=True):
    """Highlight and optionally zoom to a domain validated on current graphs.

    layout : 'panels'  -> parent and child side by side
             'overlay' -> single map, shared context, child domain dashed on top
    """
    import geopandas as gpd
    from plotly.subplots import make_subplots
    from .current_domains import load_domain_pair

    if layout not in ('panels', 'overlay'):
        raise ValueError("layout must be 'panels' or 'overlay'")

    _,_,domain=load_domain_pair(record)
    frames=[]
    for key in ['parent_parameters_parquet','parameters_parquet']:
        if record.get('mode') == 'centerline' and key == 'parameters_parquet':
            root = Path(record.registry_path).parent
            summary = json.loads((root/'reference_section_summary.json').read_text())
            geometry_path = summary['paths']['directed_links_path']
        else:
            manifest=json.loads(geometry_manifest_path(record,key).read_text())
            geometry_path = manifest['input_paths']['directed_links']
        geo=gpd.read_file(geometry_path)
        params=pd.read_parquet(record[key])
        if set(geo.id_link)!=set(params.id_link): raise ValueError('Map reach IDs differ from routed physical graph')
        frames.append(geo)
    parent,child=frames;child=child.to_crs(parent.crs)
    if parent.crs.is_geographic:
        crs=parent.estimate_utm_crs();parent=parent.to_crs(crs);child=child.to_crs(crs)

    p=parent[parent.id_link.isin(domain['parent_links'])]
    c=child[child.id_link.isin(domain['child_links'])]

    def line_xy(layer):
        x=[];y=[];labels=[]
        for r in layer.itertuples():
            geom=r.geometry
            if geom is None or geom.is_empty: continue
            for line in list(geom.geoms) if geom.geom_type=='MultiLineString' else [geom]:
                xx,yy=line.xy;x.extend([*xx,None]);y.extend([*yy,None])
                labels.extend([f'Link {r.id_link}']*len(xx)+[''])
        return x,y,labels

    def trace(layer,name,color,width,dash='solid',gl=False,zorder=0):
        x,y,labels=line_xy(layer)
        kw=dict(x=x,y=y,text=labels,hovertemplate='%{text}<extra></extra>',
                mode='lines',line=dict(color=color,width=width,dash=dash),name=name)
        if gl: return go.Scattergl(**kw)          # Scattergl has no zorder
        return go.Scatter(**kw,zorder=zorder)

    def add_scale_bar(fig,xr,yr,xref='x',yref='y',frac=0.2,pos=(0.45,0.05)):
        """Scale bar of a 'nice' length (~frac of the x-range), anchored bottom-right."""
        span=xr[1]-xr[0]
        raw=span*frac
        mag=10**np.floor(np.log10(raw))
        length=max(n*mag for n in (1,2,5) if n*mag<=raw)
        x1=xr[0]+pos[0]*span; x0=x1-length
        y0=yr[0]+pos[1]*(yr[1]-yr[0])
        tick=(yr[1]-yr[0])*0.01
        line=dict(color='black',width=4)

        fig.add_shape(type='line',x0=x0,x1=x1,y0=y0,y1=y0,xref=xref,yref=yref,line=line
                      )
        for xe in (x0,x1):
            fig.add_shape(type='line',x0=xe,x1=xe,y0=y0-tick,y1=y0+tick,xref=xref,yref=yref,line=line)
        label=f'{length/1000:g} km'\
                if length>=1000 else f'{length:g} m'
        fig.add_annotation(x=(x0+x1)/2,y=y0+tick,text=label,showarrow=False,
                           yanchor='bottom',xref=xref,yref=yref,font=dict(size=30))

    if layout=='panels':
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Current parent network','Current child network'])
        for col,frame,part,color,prefix in [(1,parent,p,'#2563eb','Parent'),(2,child,c,'#dc2626','Child')]:
            fig.add_trace(trace(frame,f'{prefix}: Network context','#cbd5e1',1,gl=True),row=1,col=col)
            fig.add_trace(trace(part,f'{prefix}: Compared collapse domain',color,1,gl=True),row=1,col=col)
            fig.update_yaxes(scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
        cells=[(1,1),(1,2)]
    else:
        fig=go.Figure()
        fig.add_trace(trace(parent,'Network context','#cbd5e1',2,zorder=0))
        fig.add_trace(trace(p,'Full complexity','#2563eb',3,zorder=2))
        if show_children:
            fig.add_trace(trace(c,'Simplified','#dc2626',3,dash='dash',zorder=1))
        fig.update_yaxes(scaleanchor='x',scaleratio=1)
        cells=[(None,None)]

    bounds=np.vstack([p.total_bounds,c.total_bounds] if zoom_to_domain else [parent.total_bounds,child.total_bounds])
    low=bounds[:,:2].min(axis=0);high=bounds[:,2:].max(axis=0);pad=max((high-low).max()*.08,50.)
    xr=[low[0]-pad,high[0]+pad];yr=[low[1]-pad,high[1]+pad]


    for row,col in cells:
        fig.update_xaxes(range=xr,visible=False,constrain='domain',row=row,col=col)
        fig.update_yaxes(range=yr,visible=False,constrain='domain',row=row,col=col)

    # size the figure to the map's aspect ratio so no space is wasted
    aspect=(yr[1]-yr[0])/(xr[1]-xr[0])
    width=700
    ncols=2 if layout=='panels' else 1
    top=30 if layout=='panels' else 10          # room for subplot titles
    height=int(width/ncols*aspect)+top+10

    fig.update_layout(
        width=width,height=height,
        margin=dict(l=10,r=10,t=top,b=10),
        plot_bgcolor='white',paper_bgcolor='white',
        legend=dict(x=0.01,y=0.99,xanchor='left',yanchor='top',
                    bgcolor='rgba(255,255,255,0.8)',bordercolor='lightgrey',borderwidth=1),
    )
    unit=parent.crs.axis_info[0].unit_name
    if unit not in ('metre','meter'):
        raise ValueError(f'Scale bar expects metres, CRS uses {unit}')
    if layout=='panels':
        add_scale_bar(fig,xr,yr,'x','y')
        add_scale_bar(fig,xr,yr,'x2','y2')
    else:
        add_scale_bar(fig,xr,yr)

    return fig

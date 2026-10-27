"""Read-only notebook review of the agreed grid and saved Cunge pilot.

No routing, calibration, prediction or feature rebuilding is performed here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .design import build_original
from .networks import from_parameters, detect_units
from .collapse import drawable_graph, collapse_selected, nested_main_island_edges

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT/'synthetic_experiments/outputs/cunge_pilot_20261026_refined'
BRIDGE = ROOT/'synthetic_experiments/outputs/current_predictor_bridge_20261026'


def review_inputs():
    manifest=json.loads((PILOT/'manifest.json').read_text())
    complete=json.loads((PILOT/'completion.json').read_text())
    if not complete['numerical_screen_passed']:
        raise ValueError('Pilot did not pass its numerical screen')
    # Use settings saved with these waves, not potentially edited future configs.
    settings=manifest['settings']
    grid=json.loads((ROOT/'synthetic_experiments/configs/grid.json').read_text())['grid']
    return settings,grid


def grid_table(settings,grid):
    spec=[('reference_slope','Reference slope','m/m','Prescribed inlet–outlet head loss / extent'),
          ('length_multiplier','Selected branch length','×','Simple: A; nested: upper inner branch; crossover: cross edge'),
          ('total_length_m','Network extent','km','Scales unit, approach and exit together'),
          ('total_width_m','Total/reference width','m','Width scale; also changes the BASED depth anchor'),
          ('outer_a_fraction','Outer branch A fraction','fraction','Width distribution between main alternatives'),
          ('inner_fraction','Inner / crossover fraction','fraction of A','Inner allocation or transferred crossover width'),
          ('internal_location','Internal junction location','position','Upstream / central / downstream; outer test interval remains 25–75%')]
    rows=[]
    for key,label,unit,note in spec:
        values=grid[key];base=settings['base_geometry'][key]
        if key=='total_length_m':values=[x/1000 for x in values];base=base/1000
        fmt=lambda x:f'{x:g}' if isinstance(x,(int,float)) else str(x)
        rows.append({'Factor':label,'Core levels':', '.join(map(fmt,values)),'Unit':unit,'Reference':fmt(base),'Interpretation':note})
    return pd.DataFrame(rows)


def settings_table(s):
    return pd.DataFrame([
        ('Routing','Constant-reference Muskingum–Cunge; X calculated per link'),
        ('Manning n',f"{s['n_manning']:g} — current pilot reference"),
        ('Diffusion multiplier β',f"{s['beta']:g} — current pilot reference"),
        ('Depth',f"BASED at reference width/slope/Q; local width exponent {s['depth_alpha']:g}; anchor shared by parent/child"),
        ('Celerity',f"Lower bound {s['minimum_celerity_mps']:g} m/s; no active upper cap"),
        ('Forcing',f"{s['baseflow_cms']:g} → {s['peak_cms']:g} → {s['baseflow_cms']:g} m³/s; {s['rise_hours']:g} h rise, {s['fall_hours']:g} h fall"),
        ('Initial plateau',f"{s['spinup_hours']:g} h, steady-first-flow initialization"),
        ('Tail','Shared across states; settling plus full-tail volume-balance checks'),
        ('Numerics',f"Exact lengths; grid α={s['grid_alpha']:g}, then {s['subdivision_multiplier']}× subdivision; ≤{s['maximum_native_dt_seconds']:g} s native steps"),
        ('Reporting',f"{s['report_seconds']:g}-second interval means"),
        ('Centroid','Child minus parent; known baseline and shared 0.5% family window; 1%/2% sensitivity retained'),
        ('Peak','Child minus parent first sampled maximum in that same window; ties recorded'),
        ('Primary units','Minutes here; stored centroid responses in hours'),
    ],columns=['Setting','Reference treatment'])


def selected_pairs(cases,comparisons=None,max_pairs=6):
    if max_pairs<1:raise ValueError('max_pairs must be positive')
    data=pd.read_csv(PILOT/'responses.csv')
    data=data[np.isclose(data.threshold_fraction,.005)].copy()
    if len(set(cases))!=len(cases):raise ValueError('Duplicate case selection')
    if set(cases)-set(data.configuration):raise ValueError('Unknown case')
    if comparisons is not None and set(comparisons)-set(data.child_state):raise ValueError('Unknown comparison')
    parts=[]
    for case in cases:
        f=data[data.configuration.eq(case)]
        if comparisons is not None:
            f=f.set_index('child_state').loc[comparisons].reset_index()
        parts.append(f)
    result=pd.concat(parts,ignore_index=True)
    return result.head(max_pairs).copy(),len(result)-min(len(result),max_pairs)


def case_data(row):
    folder=PILOT/row['configuration']
    audit=json.loads((folder/'audit.json').read_text())
    return pd.read_parquet(folder/'hydrographs.parquet'),audit


def metrics(row):
    from rapid_tools.hydrograph_response import distribution
    wave,audit=case_data(row)
    t=wave.time_seconds.to_numpy();a,b=row['start_seconds'],row['end_seconds']
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    record={k:row[k] for k in ['configuration','child_state']}
    for side in ['parent','child']:
        q=wave[row[side+'_state']].to_numpy()
        e=np.interp(tt,t,q)-row['baseline_cms']
        if e.min()<0:raise ValueError('Unexpected negative excess in saved review window')
        m=distribution(tt,e)
        # Peak timing is a sampled-output diagnostic, not an interpolated peak.
        use=(t>=a)&(t<=b);times=t[use];qq=q[use]
        maxima=times[qq==qq.max()]
        record[side+'_centroid_seconds']=m['centroid_seconds']
        record[side+'_peak_seconds']=float(maxima[0])
        record[side+'_last_equal_peak_seconds']=float(maxima[-1])
        record[side+'_peak_cms']=float(qq.max())
    record['centroid_shift_min']=(record['child_centroid_seconds']-record['parent_centroid_seconds'])/60
    np.testing.assert_allclose(record['centroid_shift_min'],row['centroid_shift_hours']*60,rtol=1e-9,atol=1e-8)
    record['peak_shift_min']=(record['child_peak_seconds']-record['parent_peak_seconds'])/60
    return record


def selected_edges(graph,family,treatment):
    units=detect_units(graph)
    if treatment=='main_island_only':return nested_main_island_edges(graph)
    if treatment=='inner_only':return min((edges for _,edges in units),key=len)
    if treatment in ['upstream_only','downstream_only']:
        ordered=sorted((edges for _,edges in units),key=lambda ee:min(graph.nodes[u]['x'] for u,v,k in ee))
        return ordered[0 if treatment=='upstream_only' else 1]
    return set().union(*(edges for _,edges in units))


def draw(ax,graph,parameters,selected=None):
    extent=parameters['total_length_m']/1000
    ax.axvspan(0,.25*extent,color='#f1f5f9');ax.axvspan(.75*extent,extent,color='#f1f5f9')
    for u,v,k,d in graph.edges(keys=True,data=True):
        color=('#2563eb' if (u,v,k) in selected else '#cbd5e1') if selected is not None else d['color']
        ax.plot(np.asarray(d['plot_x'])/1000,d['plot_y'],color=color,
            lw=1+6*d['w']/parameters['total_width_m'],solid_capstyle='round')
        if d.get('length_factor',1)>1:
            x=np.asarray(d['plot_x']);y=np.asarray(d['plot_y']);j=len(x)//2
            ax.annotate(f"L={d['length_m']/1000:g} km",(x[j]/1000,y[j]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xlim=(-.025*extent,1.025*extent),ylim=(-4.3,4.3),yticks=[],xlabel='Downstream position (km)')
    ax.spines['left'].set_visible(False)


def review_figure(row,settings,full_tail=False):
    wave,audit=case_data(row);pars=audit['parameters'];m=metrics(row)
    family,treatment=row['child_state'].split('/')
    parent=drawable_graph(build_original(family,from_parameters(pars)))
    selection=selected_edges(parent,family,treatment);child=collapse_selected(parent,selection)
    fig,axes=plt.subplots(1,3,figsize=(16,3.7),gridspec_kw={'width_ratios':[1,1,1.55]},layout='constrained')
    draw(axes[0],parent,pars,selection);draw(axes[1],child,pars)
    axes[0].set_title('Parent · selected paths in blue',fontsize=10)
    axes[1].set_title('Child · replacement in purple',fontsize=10)
    ax=axes[2];onset=settings['spinup_hours']*3600
    hours=(wave.time_seconds.to_numpy()-onset)/3600
    for key,label,color in [(row['parent_state'],'Parent','#2563eb'),(row['child_state'],'Child','#d97706'),('inflow_cms','Input','#94a3b8')]:
        ax.plot(hours,wave[key],label=label,color=color,lw=1.6 if label!='Input' else 1,ls='--' if label=='Input' else '-')
    for side,color in [('parent','#2563eb'),('child','#d97706')]:
        ax.axvline((m[side+'_centroid_seconds']-onset)/3600,color=color,ls=':',lw=1.2)
        ax.scatter((m[side+'_peak_seconds']-onset)/3600,m[side+'_peak_cms'],color=color,s=23,zorder=5)
    a=(row['start_seconds']-onset)/3600;b=(row['end_seconds']-onset)/3600
    ax.axvspan(a,b,color='#64748b',alpha=.06)
    ax.set(xlabel='Hours since pulse onset',ylabel='Discharge (m³/s)',title=f"Δ centroid {m['centroid_shift_min']:+.2f} min · Δ peak {m['peak_shift_min']:+.1f} min")
    ax.set_xlim((hours[0],hours[-1]) if full_tail else (min(-2,a-2),b+max(2,.04*(b-a))))
    ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.12)
    status='core reference' if row['configuration']=='baseline' else 'existing stress pilot — outside core levels'
    fig.suptitle(f"{row['configuration']} · {row['child_state']} · {status}\nW={pars['total_width_m']:g} m; S={pars['reference_slope']:g}; A={pars['outer_a_fraction']:g}; inner={pars['inner_fraction']:g}; length ×{pars['length_multiplier']:g}; extent={pars['total_length_m']/1000:g} km",fontsize=10)
    return fig


def predictor_tables(selected):
    keys=selected[['configuration','child_state']]
    f=keys.merge(pd.read_parquet(BRIDGE/'features.parquet'),on=['configuration','child_state'],validate='one_to_one')
    rows=[]
    for r in f.to_dict('records'):
        for name,column,scale,unit in [('Mean route length','route_length_mean',.001,'km'),('Route length SD','route_length_sd',.001,'km'),('Mean width','width_mean',1,'m'),('Width CV','width_cv',1,'–'),('Route entropy','route_entropy',1,'nats')]:
            p,c=r['B_parent_'+column]*scale,r['B_child_'+column]*scale
            rows.append(dict(case=r['configuration'],comparison=r['child_state'],descriptor=name,unit=unit,parent=p,child=c,change=c-p))
        p,c=r['parent_G_inverse_depth'],r['child_G_inverse_depth']
        rows.append(dict(case=r['configuration'],comparison=r['child_state'],descriptor='G inverse depth',unit='–',parent=p,child=c,change=c-p))
    context=f[['configuration','child_state','domain_size','B_parent_dominance_mean','parent_slope_slow','length_delta_km','resistance_explicit_depth','delta_G_inverse_depth']]
    predictions=keys.merge(pd.read_csv(BRIDGE/'predictions.csv'),on=['configuration','child_state'],validate='one_to_many')
    for col in ['actual_hours','predicted_hours']:predictions[col.replace('_hours','_min')]=60*predictions[col]
    return pd.DataFrame(rows),context,predictions[['model','configuration','child_state','actual_min','predicted_min','error_minutes','status','undefined_features','outside_training_range']]

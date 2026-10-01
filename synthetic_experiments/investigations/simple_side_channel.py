"""Focused 20/80 simple-unit length probe, without changing the factorial grid.

Run from the repository root using the existing river-hierarchy-rivgraph Python.
Both parent variants and the common child use one timestep and forcing window.
"""
from pathlib import Path
from dataclasses import asdict, replace
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'synthetic_experiments/src', ROOT/'RAPID/src'):
    sys.path.insert(0, str(path))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synthetic_experiments.networks import make_states, canonical_network
from synthetic_experiments.routing import Settings, schedule, route, buffer_seconds
from synthetic_experiments.resistance import predictor
from synthetic_experiments.pipeline import write_json, source_fingerprint
from hierarchy_level_definition.metrics.equivalent_resistance import compute_K_chezy

OUT = ROOT/'synthetic_experiments/outputs/simple_side_channel_20_80'
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(total_width_m=500, reference_slope=1e-4, outer_a_fraction=.2,
              inner_fraction=.5, total_length_m=20000, internal_location='central')
settings = Settings()
variants = {}
for multiplier in (1., 5.):
    states, pairs = make_states(dict(PARAMS, length_multiplier=multiplier))
    pair = next(p for p in pairs if p['child_state']=='simple/full')
    variants[multiplier] = dict(parent=states['simple/original'],child=states['simple/full'],pair=pair)
child = canonical_network(variants[1.]['child'])
assert child == canonical_network(variants[5.]['child'])
networks = {f'parent_m{m:g}':canonical_network(v['parent']) for m,v in variants.items()}
networks['child'] = child
clock = schedule(networks.values(), settings)
write_json(OUT/'probe_manifest.json', dict(parameters=PARAMS,multipliers=[1,5],
    settings=asdict(settings),clock=clock,source=source_fingerprint(),networks=networks))
hydros={}; metrics={}; rows=[]
for name,net in networks.items():
    h,m,s = route(net,settings,clock)
    h.to_parquet(OUT/f'{name}_hydrograph.parquet',index=False)
    s.drop(columns='geometry').to_csv(OUT/f'{name}_segments.csv',index=False)
    write_json(OUT/f'{name}_metrics.json',m)
    hydros[name]=h;metrics[name]=m
for multiplier,variant in variants.items():
    parent=metrics[f'parent_m{multiplier:g}'];collapsed=metrics['child']
    resistance=predictor(variant['parent'],variant['child'],variant['pair'],settings)
    # Analytic parallel-resistance result and associated equilibrium allocation.
    ka,kb=compute_K_chezy(np.array([10000*multiplier,10000]),np.array([100.,400.]),settings.chezy_cz,settings.kb)
    analytic=1/(1/np.sqrt(ka)+1/np.sqrt(kb))**2
    assert np.isclose(resistance['K_before'],analytic,rtol=1e-9,atol=0)
    rows.append(dict(length_multiplier=multiplier,parent_lag_seconds=parent['peak_lag_seconds'],
        child_lag_seconds=collapsed['peak_lag_seconds'],
        lag_shift_pct=100*(collapsed['peak_lag_seconds']/parent['peak_lag_seconds']-1),
        parent_peak_cms=parent['peak_cms'],child_peak_cms=collapsed['peak_cms'],
        branch_A_resistance=ka,branch_B_resistance=kb,
        resistance_implied_A_fraction=(1/np.sqrt(ka))/(1/np.sqrt(ka)+1/np.sqrt(kb)),
        routing_A_fraction=.2,**resistance))
frame=pd.DataFrame(rows)
frame.to_csv(OUT/'comparison.csv',index=False)
# Linear decomposition of the long-parent outlet into A/B contributions.
# A single-path run retains original stem/branch widths and routing coefficients;
# only its prescribed inflow is scaled by the parent's fixed width allocation.
net=networks['parent_m5']
components={}
for branch,width,fraction in [('A',100.,.2),('B',400.,.8)]:
    path=dict(nodes=net['nodes'],edges=[e for e in net['edges'] if e[3] in (500.,width)])
    h,m,s=route(path,replace(settings,baseflow_cms=100*fraction,peak_cms=1000*fraction),clock)
    components[branch]=h
    h.to_parquet(OUT/f'branch_{branch}_outlet_contribution.parquet',index=False)
    write_json(OUT/f'branch_{branch}_metrics.json',m)
assert np.allclose(components['A'].outlet_cms+components['B'].outlet_cms,
                   hydros['parent_m5'].outlet_cms,rtol=1e-10,atol=1e-8)
# Use one colour per complete network in both views; plot event excess so
# area corresponds to event volume rather than an arbitrary baseflow window.
styles = [('parent_m5','Parent ×5: A = 50 km','#2563eb'),
          ('parent_m1','Parent ×1: A = 10 km','#ea580c'),
          ('child','Collapsed centerline','#16a34a')]
t=(hydros['child'].time_seconds-buffer_seconds(clock))/3600
fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
for ax in axes:
    for name,label,color in styles:
        ax.plot(t,hydros[name].outlet_cms-settings.baseflow_cms,label=label,color=color,lw=1.8)
    ax.set(xlabel='Time since pulse begins (hours)',ylabel='Discharge above baseflow (m³/s)',ylim=(-15,880))
    ax.grid(alpha=.15);ax.legend(fontsize=9)
axes[0].set(xlim=(0,350),title='Complete event response: equal total areas')
axes[1].set(xlim=(0,65),title='Zoom: the ×5 late pulse is outside this window')
axes[0].annotate('Delayed 20% of event volume',xy=(245,90),xytext=(125,260),
    arrowprops=dict(arrowstyle='->',color='#2563eb'),fontsize=10,color='#2563eb')
fig.suptitle('20/80 split · blue = ×5 in both panels · event volume = 58.32 million m³ per network')
fig.savefig(OUT/'hydrograph_comparison.png',dpi=180)
plt.close(fig)
# Keep branch decomposition in a separately labelled figure, avoiding reuse of
# the complete-network colours for different identities in an adjacent panel.
fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
ax.plot(t,hydros['parent_m5'].outlet_cms-settings.baseflow_cms,color='#2563eb',lw=3,label='Parent ×5: total')
for branch,fraction,color in [('A',.2,'#9333ea'),('B',.8,'#64748b')]:
    ax.plot(t,components[branch].outlet_cms-settings.baseflow_cms*fraction,
            color=color,ls='--',label=f'Branch {branch}: {fraction:.0%} of event volume')
ax.set(xlim=(0,350),xlabel='Time since pulse begins (hours)',ylabel='Discharge above baseflow (m³/s)',title='Outlet contributions to the ×5 parent')
ax.grid(alpha=.15);ax.legend()
fig.savefig(OUT/'branch_contributions.png',dpi=180)
plt.close(fig)
volume_rows=[]
for name,_,_ in styles:
    h=hydros[name];elapsed=h.time_seconds.to_numpy()-buffer_seconds(clock)
    excess=h.outlet_cms.to_numpy()-settings.baseflow_cms
    dt=clock['dt_seconds']
    total=float(excess[elapsed>=0].sum()*dt)
    # Interval-average output: weight a partial final interval at the zoom edge.
    overlap=np.maximum(0,np.minimum(elapsed+dt,65*3600)-np.maximum(elapsed,0))
    zoom=float(np.sum(excess*overlap))
    volume_rows.append(dict(network=name,total_excess_volume_m3=total,
                           first_65h_excess_volume_m3=zoom,after_65h_excess_volume_m3=total-zoom))
volume_audit=pd.DataFrame(volume_rows)
expected_volume=.5*(settings.peak_cms-settings.baseflow_cms)*(settings.rise_hours+settings.fall_hours)*3600
assert np.allclose(volume_audit.total_excess_volume_m3,expected_volume,rtol=1e-10,atol=1e-6)
volume_audit.to_csv(OUT/'volume_audit.csv',index=False)
print(frame[['length_multiplier','lag_shift_pct','K_before','K_after','Lambda_matched','length_fraction','width_fraction','resistance_predictor','resistance_implied_A_fraction']].to_string(index=False))
print('Common clock:',clock)
print('Results:',OUT)

# Peak timing is discrete: verify the direction of this modest contrast at 100 m.
fine_settings=replace(settings,target_length_m=100,effective_length_floor_m=100)
fine_clock=schedule(networks.values(),fine_settings)
fine_metrics={name:route(net,fine_settings,fine_clock)[1] for name,net in networks.items()}
resolution=[]
for label,collection in [('200m',metrics),('100m',fine_metrics)]:
    for multiplier in (1.,5.):
        parent=collection[f'parent_m{multiplier:g}'];collapsed=collection['child']
        resolution.append(dict(segmentation=label,length_multiplier=multiplier,
            dt_seconds=parent['dt_seconds'],lag_shift_pct=100*(collapsed['peak_lag_seconds']/parent['peak_lag_seconds']-1)))
resolution=pd.DataFrame(resolution)
resolution.to_csv(OUT/'resolution_check.csv',index=False)
write_json(OUT/'resolution_check.json',dict(clock_100m=fine_clock,metrics_100m=fine_metrics))
print(resolution.to_string(index=False))

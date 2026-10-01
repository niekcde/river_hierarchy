"""Focused width/length sweep and physical shortening, separate from the full grid.

Run: python synthetic_experiments/investigations/side_channel_width_length.py
Shortening compares the SAME split at A lengths 15 -> 12.5 -> 10 km.
The 10 km endpoint separation and inlet/outlet heads remain fixed. This is
branch shortening, not a topological collapse; the centerline is separate.
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
from scipy.signal import find_peaks
from synthetic_experiments.networks import make_states, canonical_network
from synthetic_experiments.routing import Settings, schedule, route, buffer_seconds
from synthetic_experiments.resistance import predictor
from synthetic_experiments.pipeline import write_json, source_fingerprint

OUT = ROOT/'synthetic_experiments/outputs/side_channel_width_length'
SHARES = (.5, .2, .1, .05)
MULTIPLIERS = (1., 1.25, 1.5, 2., 3.)
PARAMS = dict(total_width_m=500, reference_slope=1e-4, inner_fraction=.5,
              total_length_m=20000, internal_location='central')

def name(a, m):
    return f'a{a:g}_m{m:g}'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    networks, variants = {}, {}
    for a in SHARES:
        for m in MULTIPLIERS:
            states, pairs = make_states(dict(PARAMS, outer_a_fraction=a, length_multiplier=m))
            parent, child = states['simple/original'], states['simple/full']
            pair = next(p for p in pairs if p['child_state']=='simple/full')
            key = name(a,m)
            networks[key] = canonical_network(parent)
            if 'child' in networks:
                assert networks['child'] == canonical_network(child)
            networks['child'] = canonical_network(child)
            variants[key] = (parent, child, pair)
    clock = schedule(networks.values(), settings)
    write_json(OUT/'manifest.json',dict(parameters=PARAMS,shares=SHARES,multipliers=MULTIPLIERS,
        settings=asdict(settings),clock=clock,source=source_fingerprint(),networks=networks,
        shortening_reference_multiplier=1.5,shortening_shares=[.2,.05],
        note='Shortening preserves topology and width. Relative factors 1, 5/6, 2/3; physical factors 1.5, 1.25, 1.'))
    hydros, metrics, rows = {}, {}, []
    for key, net in networks.items():
        h, mt, segments = route(net,settings,clock)
        h.to_parquet(OUT/f'{key}_hydrograph.parquet',index=False)
        write_json(OUT/f'{key}_metrics.json',mt)
        segments.drop(columns='geometry').to_csv(OUT/f'{key}_segments.csv',index=False)
        hydros[key], metrics[key] = h, mt
        print(key, 'volume error',mt['volume_error_fraction'],flush=True)
    dt=clock['dt_seconds']
    t=(hydros['child'].time_seconds.to_numpy()-buffer_seconds(clock))/3600
    event=t>=0
    for a in SHARES:
        for m in MULTIPLIERS:
            key=name(a,m); mt=metrics[key]; h=hydros[key]
            excess=h.outlet_cms.to_numpy()[event]-settings.baseflow_cms
            times=t[event]; volume=excess.sum()*dt
            cumulative=np.cumsum(np.maximum(excess,0)); cumulative/=cumulative[-1]
            t10,t90=np.interp([.1,.9],cumulative,times)
            # Screening only: prominence relative to INPUT event amplitude.
            peaks,props=find_peaks(excess,prominence=.01*(settings.peak_cms-settings.baseflow_cms))
            resistance=predictor(*variants[key],settings)
            rows.append(dict(outer_a_fraction=a,length_multiplier=m,
                selected_branch_length_km=10*m,parent_peak_cms=mt['peak_cms'],
                parent_lag_seconds=mt['peak_lag_seconds'],
                lag_shift_pct=100*(metrics['child']['peak_lag_seconds']/mt['peak_lag_seconds']-1),
                collapse_peak_change_pct=100*(metrics['child']['peak_cms']/mt['peak_cms']-1),
                peak_excess_ratio=(mt['peak_cms']-settings.baseflow_cms)/(settings.peak_cms-settings.baseflow_cms),
                volume_t10_t90_hours=t90-t10,
                after_65h_volume_fraction=excess[times>=65].sum()*dt/volume,
                screened_peak_count=len(peaks),screened_peak_times_hours=times[peaks].tolist(),
                capped_segments=mt['n_celerity_capped'],volume_error_fraction=mt['volume_error_fraction'],
                **resistance))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'comparison.csv',index=False)
    short=[]
    for a in (.2,.05):
        ref=frame[(frame.outer_a_fraction==a)&(frame.length_multiplier==1.5)].iloc[0]
        for m in (1.5,1.25,1.):
            row=frame[(frame.outer_a_fraction==a)&(frame.length_multiplier==m)].iloc[0]
            short.append(dict(outer_a_fraction=a,reference_branch_length_km=15,
                branch_length_km=10*m,relative_length_factor=m/1.5,
                lag_change_pct=100*(row.parent_lag_seconds/ref.parent_lag_seconds-1),
                peak_excess_change_pct=100*(row.peak_excess_ratio/ref.peak_excess_ratio-1),
                unit_resistance_change_pct=100*(row.K_before/ref.K_before-1)))
    pd.DataFrame(short).to_csv(OUT/'shortening_comparison.csv',index=False)
    colors=plt.cm.viridis(np.linspace(.05,.9,len(MULTIPLIERS)))
    fig,axes=plt.subplots(4,1,figsize=(12,12),constrained_layout=True)
    # Display enough time to capture 99.99% of every parent event volume.
    end=0
    for h in hydros.values():
        q=np.maximum(h.outlet_cms.to_numpy()[event]-settings.baseflow_cms,0)
        end=max(end,float(t[event][np.searchsorted(np.cumsum(q)/q.sum(),.9999)]))
    for ax,a in zip(axes,SHARES):
        for color,m in zip(colors,MULTIPLIERS):
            ax.plot(t,hydros[name(a,m)].outlet_cms-settings.baseflow_cms,color=color,label=f'A length ×{m:g}')
        ax.plot(t,hydros['child'].outlet_cms-settings.baseflow_cms,'--',color='black',label='Collapsed centerline')
        ax.set(xlim=(0,end+5),ylabel='Excess discharge (m³/s)',title=f'A/B split {100*a:g}/{100*(1-a):g}')
        ax.grid(alpha=.15)
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Time since pulse begins (hours)')
    fig.savefig(OUT/'width_length_hydrographs.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for ax,a in zip(axes,(.2,.05)):
        for m,color in zip((1.5,1.25,1.),('#2563eb','#9333ea','#ea580c')):
            ax.plot(t,hydros[name(a,m)].outlet_cms-settings.baseflow_cms,color=color,
                label=f'A = {10*m:g} km ({m/1.5:.3g} × reference)')
        ax.set(xlim=(0,end+5),ylabel='Excess discharge (m³/s)',title=f'Same {100*a:g}/{100*(1-a):g} split: shorten A from 15 km toward 10 km')
        ax.legend();ax.grid(alpha=.15)
    axes[-1].set_xlabel('Time since pulse begins (hours)')
    fig.savefig(OUT/'shortening_hydrographs.png',dpi=160);plt.close(fig)
    print(pd.DataFrame(short).to_string(index=False))
    # Independent resolution check of the shortening comparisons.
    fine=replace(settings,target_length_m=100,effective_length_floor_m=100)
    keys=[name(a,m) for a in (.2,.05) for m in (1.,1.25,1.5)]
    fine_clock=schedule([networks[k] for k in keys],fine)
    fine_metrics={k:route(networks[k],fine,fine_clock)[1] for k in keys}
    checks=[]
    for label, collection in [('200m',metrics),('100m',fine_metrics)]:
        for a in (.2,.05):
            ref=collection[name(a,1.5)]
            for m in (1.,1.25,1.5):
                mt=collection[name(a,m)]
                checks.append(dict(segmentation=label,outer_a_fraction=a,length_multiplier=m,
                    lag_change_pct=100*(mt['peak_lag_seconds']/ref['peak_lag_seconds']-1),
                    peak_excess_change_pct=100*((mt['peak_cms']-settings.baseflow_cms)/
                        (ref['peak_cms']-settings.baseflow_cms)-1)))
    pd.DataFrame(checks).to_csv(OUT/'shortening_resolution_check.csv',index=False)
    write_json(OUT/'shortening_resolution_check.json',dict(clock=fine_clock,metrics=fine_metrics))
    print(pd.DataFrame(checks).to_string(index=False))
    print('Results:',OUT)

if __name__=='__main__':
    main()

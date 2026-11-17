"""Compare the two hydraulic roughness trials with the saved MC benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'LISFLOOD_FP/results/example56'


def main():
    trials={}
    records=[]
    for tag in ['dx100','dx100_mc_roughness']:
        p=OUT/tag
        m=json.loads((p/'manifest.json').read_text())
        if 'comparison' not in m: raise ValueError('Incomplete trial '+tag)
        f=pd.read_csv(p/'hydrographs.csv',parse_dates=['time_utc'])
        trials[tag]=(m,f)
        for state,r in m['states'].items():
            records.append(dict(model='custom_diffusive',trial=tag,state=state,n=m['settings']['n'],
                peak_cms=r['peak_cms'],peak_utc=r['peak_utc'],
                collapse_peak_shift_minutes=m['comparison']['peak_time_shift_minutes'],
                collapse_peak_change_cms=m['comparison']['peak_change_cms']))
    benchmark={}
    for state,saved in [('base','S000_base'),('collapse_S014_unit_10','S014_unit_10')]:
        p=ROOT/'RAPID/outputs/event_window_runs_20261013/56/tail_216h'/saved/'outlet.parquet'
        f=pd.read_parquet(p);benchmark[state]=f
        row=f.loc[f.q_outlet_cms.idxmax()]
        records.append(dict(model='saved_fixed_depth_MC',trial='saved_event_window_20261013',state=state,
            n=.020446875,peak_cms=float(row.q_outlet_cms),peak_utc=str(row.time_utc)))
    table=pd.DataFrame(records)
    table.to_csv(OUT/'roughness_comparison.csv',index=False)
    new,fine=trials['dx100_mc_roughness'];old,original=trials['dx100']
    # Verify the hydraulic trials differ only in n, not geometry/forcing/clock.
    assert old['source_hashes']==new['source_hashes']
    assert old['forcing']==new['forcing']
    for key in ['dx','max_step','rtol']:assert old['settings'][key]==new['settings'][key]
    raw=pd.read_parquet(ROOT/new['forcing']['path'])
    exact=np.trapz(raw.discharge_cms,(raw.time-raw.time.iloc[0]).dt.total_seconds())+96*3600*raw.discharge_cms.iloc[-1]
    audits={}
    for state,r in new['states'].items():
        error=(r['cumulative_inflow_m3']-exact)/exact
        assert abs(error)<1e-5 and r['maximum_mass_residual_fraction']<1e-6
        assert r['minimum_depth_m']>0 and r['final_relative_node_imbalance']<1e-6
        audits[state]=dict(independent_inflow_volume_relative_error=error,
            maximum_mass_residual_fraction=r['maximum_mass_residual_fraction'],
            maximum_froude=r['maximum_froude'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for ax,state,title in zip(axes,['base','collapse_S014_unit_10'],['Original network','Collapsed network · S014_unit_10']):
        ax.plot(original.time_utc,original[state+'_cms'],color='#86919a',lw=1.5,ls=':',label='Hydraulic · n = 0.035')
        ax.plot(fine.time_utc,fine[state+'_cms'],color='#146b9c',lw=2,label='Hydraulic · n = 0.020446875')
        mc=benchmark[state]
        ax.plot(mc.time_utc,mc.q_outlet_cms,color='#dc7628',lw=1.7,ls='--',label='Saved fixed-depth MC · n = 0.020446875')
        ax.set_title(title,loc='left');ax.set_ylabel('Discharge (m³/s)');ax.grid(alpha=.2)
        ax.set_xlim(pd.Timestamp('2025-06-25',tz='UTC'),pd.Timestamp('2025-07-01',tz='UTC'))
        ax.legend(frameon=False,fontsize=9,loc='lower right')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    axes[-1].set_xlabel('2025 · UTC')
    fig.suptitle('Example 56 · using the Muskingum–Cunge roughness in the hydraulic model',fontsize=13)
    fig.text(.08,.015,'Hydraulic trials use identical forcing, geometry and numerical settings.\nSaved MC uses a different forcing window/preprocessing; matching n does not isolate the equation differences.',fontsize=9,color='#58616a')
    fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(OUT/'roughness_comparison.png',dpi=180);plt.close(fig)
    a,b=new['states'].values()
    data=dict(roughness=new['settings']['n'],comparison=new['comparison'],audit=audits,
        same_geometry_and_forcing_between_hydraulic_trials=True,
        limitation='Saved MC benchmark has different forcing window/preprocessing and fixed reference depths; not an equations-only comparison',
        original_peak_shift_due_to_n_minutes=(a['peak_seconds']-old['states']['base']['peak_seconds'])/60)
    (OUT/'roughness_comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'ROUGHNESS_COMPARISON.md').write_text(f'''# Same-roughness trial

Both custom hydraulic states were rerun at the saved MC roughness **n = {new['settings']['n']}**.
Geometry, point-inflow series, 100 m maximum segment length, 300 s maximum internal
step, tolerances, normal-flow outlet and tail duration are unchanged from the
n = 0.035 hydraulic trial. Initialization is recomputed at the new roughness.

![Comparison](roughness_comparison.png)

| Quantity | Original | Collapse S014_unit_10 |
|---|---:|---:|
| Peak discharge, m³/s | {a['peak_cms']:.3f} | {b['peak_cms']:.3f} |
| Peak time, UTC | {a['peak_utc']} | {b['peak_utc']} |
| Lag behind raw inflow peak, hours | {a['peak_lag_from_inflow_hours']:.3f} | {b['peak_lag_from_inflow_hours']:.3f} |
| Minimum depth, m | {a['minimum_depth_m']:.3f} | {b['minimum_depth_m']:.3f} |
| Maximum depth, m | {a['maximum_depth_m']:.3f} | {b['maximum_depth_m']:.3f} |
| Maximum Froude diagnostic | {a['maximum_froude']:.3f} | {b['maximum_froude']:.3f} |

The collapse shifts the sampled peak by {new['comparison']['peak_time_shift_minutes']:.0f} minutes
and changes peak height by {new['comparison']['peak_change_cms']:+.3f} m³/s.
Changing roughness alone shifts the original hydraulic peak by
{data['original_peak_shift_due_to_n_minutes']:.0f} minutes relative to n = 0.035.
Negative shifts mean earlier. Peak sampling resolution is five minutes.

Both runs pass the mass-balance, independent input-volume and tail-settling checks
in `roughness_comparison.json`. The lower roughness trial has not received its own
additional spatial refinement study; the previous 100 m/200 m check was at n = 0.035.

The saved MC original peaks at 2025-06-27 14:39:30 UTC, and its collapse at
14:03:30 UTC (36 minutes earlier). The plot is a contextual comparison: MC uses a
different forcing window/preprocessing, fixed reference depths and fixed branch
fractions. Matching n does not make those assumptions equivalent, and the MC-calibrated
n is not automatically calibrated for the custom hydraulic model.

Full traces and provenance: `dx100_mc_roughness/hydrographs.csv` and `manifest.json`.
''')
    print(table.to_string(index=False));print(json.dumps(data,indent=2))


if __name__=='__main__':main()

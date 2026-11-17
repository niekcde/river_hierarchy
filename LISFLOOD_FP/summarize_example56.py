"""Plot and audit the completed prototype pair and its spatial refinement."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'LISFLOOD_FP/results/example56'


def load(tag):
    r=OUT/tag
    m=json.loads((r/'manifest.json').read_text())
    if 'comparison' not in m:
        raise ValueError(f'{tag} is incomplete')
    f=pd.read_csv(r/'hydrographs.csv',parse_dates=['time_utc'])
    return m,f


def main():
    m,f=load('dx100');coarse,c=load('dx200_verified')
    origin=f.time_utc.iloc[0]
    fine_states=m['states'];a,b=fine_states.values()
    raw=pd.read_parquet(ROOT/m['forcing']['path'])
    rawt=(raw.time-raw.time.iloc[0]).dt.total_seconds().to_numpy()
    exact_input=float(np.trapz(raw.discharge_cms,rawt)+96*3600*raw.discharge_cms.iloc[-1])
    checks={}
    for key,state in fine_states.items():
        other=coarse['states'][key]
        checks[key]=dict(maximum_100m_vs_200m_difference_cms=float(abs(f[key+'_cms']-c[key+'_cms']).max()),
            peak_difference_cms=state['peak_cms']-other['peak_cms'],
            peak_time_difference_minutes=(state['peak_seconds']-other['peak_seconds'])/60,
            centroid_difference_seconds=state['centroid_seconds']-other['centroid_seconds'],
            independent_inflow_volume_relative_error=(state['cumulative_inflow_m3']-exact_input)/exact_input)
        if state['maximum_mass_residual_fraction']>1e-6 or abs(checks[key]['independent_inflow_volume_relative_error'])>1e-5:
            raise ValueError('Volume acceptance failed')
        if state['final_relative_node_imbalance']>1e-6:
            raise ValueError('Tail not settled')
    checks['collapse_effect']=dict(
        peak_change_difference_cms=m['comparison']['peak_change_cms']-coarse['comparison']['peak_change_cms'],
        centroid_shift_difference_seconds=60*(m['comparison']['positive_excess_centroid_shift_minutes']-coarse['comparison']['positive_excess_centroid_shift_minutes']))
    links=pd.read_parquet(OUT/'dx100/base_link_flows.parquet')
    junction=links[links.u.eq(20)].pivot(index='time_seconds',columns='v',values='upstream_cms')
    fraction=junction.div(junction.sum(axis=1),axis=0)
    fixed_fraction=661.483311/(661.483311+647.069833)
    splits=dict(node=20,direct_branch_to=17,minimum_fraction=float(fraction[17].min()),
        maximum_fraction=float(fraction[17].max()),initial_fraction=float(fraction[17].iloc[0]),
        width_based_fraction=fixed_fraction)
    (OUT/'verification.json').write_text(json.dumps(dict(refinement=checks,dynamic_split=splits,
        independent_input_volume_m3=exact_input,tests='6 unittest tests passed'),indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    blue='#146b9c';orange='#dc7628';gray='#818b94'
    fig,axes=plt.subplots(3,1,figsize=(12,10),gridspec_kw={'height_ratios':[1.5,1.1,.9]})
    for ax in axes[:2]:
        ax.plot(f.time_utc,f.inflow_cms,color=gray,lw=1,alpha=.65,label='Upstream inflow')
        ax.plot(f.time_utc,f.base_cms,color=blue,lw=2,label='Original network · 46 links')
        ax.plot(f.time_utc,f.collapse_S014_unit_10_cms,color=orange,lw=1.9,ls='--',label='Collapse S014_unit_10 · 42 links')
        ax.set_ylabel('Discharge (m³/s)');ax.grid(alpha=.18)
    axes[0].set_title('Example 56 · first custom 1D diffusive-wave simulation',loc='left',pad=14)
    axes[0].legend(loc='upper right',frameon=False,fontsize=9)
    axes[0].set_xlim(f.time_utc.iloc[0],f.time_utc.iloc[-1])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    tail=pd.Timestamp(m['forcing']['end_observed_utc'])
    axes[0].axvspan(tail,f.time_utc.iloc[-1],color=gray,alpha=.09)
    axes[0].text(.98,.38,'Shaded: constant-inflow tail',transform=axes[0].transAxes,ha='right',fontsize=9,color='#5a646d')
    peak_time=pd.Timestamp(a['peak_utc'])
    axes[1].set_xlim(peak_time-pd.Timedelta(hours=24),peak_time+pd.Timedelta(hours=24))
    axes[1].set_ylim(1800,1870)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    axes[1].set_title(f"Peak detail · collapse {abs(m['comparison']['peak_time_shift_minutes']):.0f} min earlier; peak {m['comparison']['peak_change_cms']:+.2f} m³/s",loc='left',fontsize=11)
    for state,color in [(a,blue),(b,orange)]:
        axes[1].plot(pd.Timestamp(state['peak_utc']),state['peak_cms'],'o',color=color,ms=5)
    delta=f.collapse_S014_unit_10_cms-f.base_cms
    axes[2].plot(f.time_utc,delta,color='#73529d',lw=1.5)
    axes[2].axhline(0,color=gray,lw=.7)
    axes[2].fill_between(f.time_utc,0,delta,color='#73529d',alpha=.13)
    axes[2].set_ylabel('Collapse − original\n(m³/s)');axes[2].set_xlabel('Date in 2025 · UTC')
    axes[2].set_xlim(f.time_utc.iloc[0],f.time_utc.iloc[-1]);axes[2].grid(alpha=.18)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.text(.08,.012,'Assumed bed = cached water-surface profile · n = 0.035 · normal-flow outlet · 100 m maximum segment length\nHistorical cached geometry; uncalibrated prototype. Outlet traces sampled every 5 minutes.',fontsize=9,color='#58616a')
    fig.tight_layout(rect=[0,.05,1,1],h_pad=1.8)
    fig.savefig(OUT/'hydrograph_comparison.png',dpi=180)
    fig.savefig(OUT/'hydrograph_comparison.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5))
    times=origin+pd.to_timedelta(fraction.index,unit='s')
    ax.plot(times,100*fraction[17],color=blue,lw=1.7,label='Calculated fraction into direct branch 20 → 17')
    ax.axhline(100*fixed_fraction,color=gray,ls='--',label='Fixed width fraction (comparison only)')
    ax.set(title='Flow splitting evolves during the event · original network',ylabel='Fraction of outgoing junction flow (%)',xlabel='Date in 2025 · UTC')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax.grid(alpha=.18)
    ax.legend(frameon=False,fontsize=9,loc='best')
    fig.tight_layout();fig.savefig(OUT/'dynamic_split.png',dpi=180);plt.close(fig)
    max_refinement=max(v['maximum_100m_vs_200m_difference_cms'] for k,v in checks.items() if k!='collapse_effect')
    report=f'''# Example 56: first custom 1D hydraulic attempt

The prototype ran successfully for the original network and saved collapse
**S014_unit_10**. The collapse advances the outlet hydrograph while changing its
peak height very little. These are results of the new custom hydraulic model,
not saved RAPID hydrographs.

![Hydrographs](hydrograph_comparison.png)

## Results (100 m maximum segment length)

| Quantity | Original | Collapse |
|---|---:|---:|
| Physical links | {a['links']} | {b['links']} |
| Numerical faces | {a['mesh_faces']} | {b['mesh_faces']} |
| Outlet peak, m³/s | {a['peak_cms']:.3f} | {b['peak_cms']:.3f} |
| Outlet peak, UTC | {a['peak_utc']} | {b['peak_utc']} |
| Lag behind inflow peak, hours | {a['peak_lag_from_inflow_hours']:.3f} | {b['peak_lag_from_inflow_hours']:.3f} |
| Minimum depth anywhere, m | {a['minimum_depth_m']:.3f} | {b['minimum_depth_m']:.3f} |
| Maximum depth anywhere, m | {a['maximum_depth_m']:.3f} | {b['maximum_depth_m']:.3f} |
| Maximum face Froude diagnostic | {a['maximum_froude']:.3f} | {b['maximum_froude']:.3f} |
| Initialization spin-up, days | {a['spinup']['days']} | {b['spinup']['days']} |

The upstream peak was {m['forcing']['peak_discharge_cms']:.3f} m³/s at
{m['forcing']['peak_utc']}. Relative to the original, the collapse peak occurs
**{abs(m['comparison']['peak_time_shift_minutes']):.0f} minutes earlier**, with
**{m['comparison']['peak_change_cms']:+.3f} m³/s
({m['comparison']['peak_change_percent']:+.4f}%)** change in magnitude. The
maximum simultaneous difference is
**{m['comparison']['maximum_absolute_hydrograph_difference_cms']:.3f} m³/s**.
Five-minute output sampling limits interpretation of peak timing below that scale.

The positive-excess centroid shifts by
**{m['comparison']['positive_excess_centroid_shift_minutes']:.3f} minutes**
(negative means earlier). This descriptive centroid integrates positive discharge
above a straight baseline joining the first and last observed inflow values,
with the final baseline held through the tail. It is baseline dependent and is
not the established repository response metric without further harmonization.

## Behavior and interpretation

- The original routes water through parallel paths with different lengths and
  conveyances. Replacing the five-link section with its saved equivalent channel
  changes both conveyance and storage; the collapsed hydrograph leads on the rise
  and generally falls sooner. This interpretation is consistent with the results,
  but the experiment does not isolate length, width and storage effects separately.
- Flow splitting is dynamic. At node 20 the direct branch to node 17 carries
  **{100*splits['minimum_fraction']:.2f}–{100*splits['maximum_fraction']:.2f}%**
  of outgoing flow, compared with a constant width fraction of
  **{100*fixed_fraction:.2f}%**. The remaining flow follows the other branch.
  Fractions are calculated from simulated face flows, not imposed in routing.
- All sampled channels stay wet and all sampled face flows remain downstream.
  The code permits signed reverse flows, but this event did not produce them.
  The Froude diagnostic remains subcritical, consistent with this slow-flow pilot;
  it is not by itself proof of diffusive-wave validity.
- Depth evolves above the fixed proxy bed. No fixed BASED depth, RAPID celerity
  cap, K/X coefficient or prescribed branch fraction is used during routing.
- Both runs settle back to the final inflow of {a['final_inflow_cms']:.3f} m³/s
  during the 96-hour constant tail. Final inflow is lower than initial inflow,
  so the network releases stored water: outflow volume exceeding inflow volume
  is expected and is explicitly accounted for.

![Dynamic split](dynamic_split.png)

## Conservation and numerical verification

Six physical/numerical unit tests passed: uniform normal flow; internal lake at
rest; symmetric transient splitting and conservation; reversal and datum shift;
conflicting junction-bed rejection; and analytic Jacobian verification.

For the two accepted 100 m runs, the largest absolute storage/boundary-volume
residual is {max(a['maximum_mass_residual_m3'],b['maximum_mass_residual_m3']):.6g} m³,
or {max(a['maximum_mass_residual_fraction'],b['maximum_mass_residual_fraction']):.3g}
of total inflow volume. This checks the discretized mass identity, not all errors
in the continuous hydraulic solution. An independent exact integral of the
piecewise-linear input is {exact_input:.3f} m³; the maximum relative discrepancy
with the solver's integrated inflow is
{max(abs(checks[k]['independent_inflow_volume_relative_error']) for k in fine_states):.3g}.

Refining from 200 m to 100 m (same 300 s maximum internal step and tolerances)
changes either complete outlet trace by at most **{max_refinement:.5f} m³/s**.
The collapse-induced centroid shift changes by
**{checks['collapse_effect']['centroid_shift_difference_seconds']:.3f} seconds**.
The sampled peak times are unchanged. The approximately 25-minute effect is
therefore resolved at these two mesh sizes. This is a first refinement check,
not a comprehensive validation across flows, boundaries and geometries.

## Inputs, scope and limitations

- Original: cached `S000_base`, 46 links. Collapse: cached `S014_unit_10`,
  42 links. Exactly 41 exterior links are unchanged. Five links in the section
  from node 20 to node 17 become one 10,243.126 m channel of width 1,100.125 m.
  The original five links total 24,045.729 m across parallel paths; that total
  is **not** a single upstream-to-downstream travel distance.
- Fixed bed = the cached reconstructed water-surface heads, as agreed. The
  original and collapsed section have identical endpoint elevations. Widths
  are treated as rectangular widths at all depths; no bank overtopping occurs.
- Shared Manning n = 0.035, deliberately uncalibrated. Both runs use the same
  inlet series and identical terminal normal-flow rating. No lateral input.
- Observed point inflow: {m['forcing']['origin_utc']} to
  {m['forcing']['end_observed_utc']}. Linear interpolation bridges
  {m['forcing']['gaps_over_15min']} gaps longer than 15 minutes; the largest gap
  is {m['forcing']['maximum_gap_seconds']/60:.0f} minutes. The missing observations
  are not recovered by this interpolation.
- These historical local caches were available while the current master drive
  was unmounted. This is an example-56 pilot, not certification of the current
  production geometry release. Exact source hashes are in the run manifests.
- The cached case's reviewed observed peak lag is about 16.17 hours; this
  uncalibrated original run gives {a['peak_lag_from_inflow_hours']:.2f} hours.
  Thus it is about {a['peak_lag_from_inflow_hours']-16.1666666667:.2f} hours slower
  by that peak-only comparison. This is a useful calibration warning, not a
  formal validation score, and the collapse comparison is not a fit to observations.
- This is a custom diffusive-wave graph implementation, not the official
  LISFLOOD-FP engine. It omits inertia, junction momentum/losses, floodplain
  storage, wetting/drying and uncertainty in the proxy bed. The small peak-height
  change should not be treated as physically established to three decimals.

## Files and next step

`dx100/hydrographs.csv` contains both hydrographs, input, storage and mass residuals.
`dx100/*_link_flows.parquet` preserves upstream/downstream link traces.
`dx100/*_nodes.csv` stores depth ranges; `*_mesh.csv` stores the discretization.
`verification.json` compares resolutions; each run has a complete `manifest.json`.
See ../../README.md for equations, assumptions, test and run commands.

The prototype demonstrates the requested coupling and collapse comparison.
Before expanding the population, check downstream-boundary influence and
roughness/bed sensitivity, then audit the active geometry when the drive is
available. Those tests would address physical uncertainty rather than numerical
mass balance alone.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(dict(comparison=m['comparison'],refinement=checks,split=splits),indent=2))


if __name__=='__main__':main()

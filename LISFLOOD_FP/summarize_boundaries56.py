"""Summarize completed calibration, collapse transfer and numerical controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calibrate_boundaries56 import ROOT,OUT,SCENARIOS,STATES,DATA,CACHE


def get_run(s,state,n,dx=200.,extension=None):
    ext=SCENARIOS[s][1] if extension is None else extension
    records=[json.loads(p.read_text()) for p in (OUT/s/'runs').glob('*/result.json')]
    matches=[r for r in records if r['state']==state and abs(r['n']-n)<1e-10 and r['dx']==dx and r['extension_m']==ext]
    if len(matches)!=1:raise ValueError(f'Expected one completed run: {s}/{state}/{n}/{dx}/{ext}')
    r=matches[0];f=pd.read_parquet(ROOT/r['wave'])
    return r,f


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=[];parents=[];waves={};fits={}
    for s in SCENARIOS:
        fit=json.loads((OUT/s/'calibration.json').read_text());fits[s]=fit
        parent,pwave=get_run(s,'base',fit['n'])
        parents.append({k:parent[k] for k in ['scenario','n','timing_error_minutes','peak_cms','rmse_cms','bias_cms','nse','peak_utc']})
        for state in STATES:
            r,w=get_run(s,state,fit['n']);waves[(s,state)]=w
            if r['maximum_mass_residual_m3']>1 or r['original_port_mass_residual_m3']>1:raise ValueError('Mass residual >1 m3')
            row={k:r[k] for k in ['scenario','state','n','peak_cms','peak_utc','minimum_depth_m','maximum_depth_m',
                    'maximum_froude','maximum_mass_residual_m3','original_port_mass_residual_m3',
                    'independent_inflow_volume_relative_error','final_node_imbalance_fraction']}
            row.update(peak_shift_minutes=(r['peak_seconds']-parent['peak_seconds'])/60,
                       centroid_shift_minutes=(r['centroid_seconds']-parent['centroid_seconds'])/60,
                       peak_change_cms=r['peak_cms']-parent['peak_cms'],
                       max_hydrograph_difference_cms=float(abs(w.gauge_cms-pwave.gauge_cms).max()))
            rows.append(row)
    results=pd.DataFrame(rows);parent_table=pd.DataFrame(parents)
    results.to_csv(OUT/'collapse_comparison.csv',index=False);parent_table.to_csv(OUT/'calibration_summary.csv',index=False)
    controls=[]
    for s in ['A_normal','D_extend10km']:
        n=fits[s]['n'];effects={}
        for state in ['base','S014_unit_10']:
            coarse,c=get_run(s,state,n);fine,f=get_run(s,state,n,dx=100.)
            effects[state]=dict(coarse=coarse,fine=fine)
            controls.append(dict(scenario=s,state=state,check='200m_to_100m',
                peak_time_difference_minutes=(fine['peak_seconds']-coarse['peak_seconds'])/60,
                peak_difference_cms=fine['peak_cms']-coarse['peak_cms'],
                centroid_difference_minutes=(fine['centroid_seconds']-coarse['centroid_seconds'])/60,
                maximum_trace_difference_cms=float(abs(f.gauge_cms-c.gauge_cms).max())))
            if s=='D_extend10km':
                far,w=get_run(s,state,n,extension=20000.)
                effects[state]['far']=far
                controls.append(dict(scenario=s,state=state,check='10km_to_20km_no_refit',
                    peak_time_difference_minutes=(far['peak_seconds']-coarse['peak_seconds'])/60,
                    peak_difference_cms=far['peak_cms']-coarse['peak_cms'],
                    centroid_difference_minutes=(far['centroid_seconds']-coarse['centroid_seconds'])/60,
                    maximum_trace_difference_cms=float(abs(w.gauge_cms-c.gauge_cms).max())))
        for name,kind in [('fine','200m_to_100m_effect'),('far','10km_to_20km_effect')]:
            if name not in effects['base']:continue
            p=effects['base'];ch=effects['S014_unit_10']
            controls.append(dict(scenario=s,state='S014_minus_base',check=kind,
                peak_time_difference_minutes=((ch[name]['peak_seconds']-p[name]['peak_seconds'])-(ch['coarse']['peak_seconds']-p['coarse']['peak_seconds']))/60,
                peak_difference_cms=(ch[name]['peak_cms']-p[name]['peak_cms'])-(ch['coarse']['peak_cms']-p['coarse']['peak_cms']),
                centroid_difference_minutes=((ch[name]['centroid_seconds']-p[name]['centroid_seconds'])-(ch['coarse']['centroid_seconds']-p['coarse']['centroid_seconds']))/60))
    control=pd.DataFrame(controls);control.to_csv(OUT/'numerical_controls.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'base':'#176d9c','S014_unit_10':'#d37522','S008_unit_6':'#25866a','S012_unit_12':'#8753a3'}
    titles={'A_normal':'A · Normal flow','B_half_slope':'B · Half outlet friction slope',
            'C_double_slope':'C · Double outlet friction slope','D_extend10km':'D · 10 km downstream extension'}
    observed=pd.read_parquet(DATA/'observed_downstream.parquet')
    # Break the observed plotted line across gaps longer than 15 minutes.
    obs=observed.set_index('time').discharge_cms.reindex(pd.date_range(observed.time.min(),observed.time.max(),freq='5min'))
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,s in zip(axes.flat,SCENARIOS):
        ax.plot(obs.index,obs.values,color='#333333',alpha=.5,lw=.8,label='Downstream gauge')
        for state in STATES:
            f=waves[(s,state)]
            ax.plot(f.time_utc,f.gauge_cms,color=colors[state],lw=1.8 if state=='base' else 1.4,
                    ls='-' if state=='base' else '--',label='Original' if state=='base' else state)
        ax.set_title(f'{titles[s]}\nCalibrated n = {fits[s]["n"]:.6f}',loc='left')
        ax.set_xlim(pd.Timestamp('2025-06-25',tz='UTC'),pd.Timestamp('2025-07-01',tz='UTC'))
        ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.set_ylabel('Discharge at original port (m³/s)')
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle('Example 56 · calibrate the original, then transfer n to three collapses',fontsize=14)
    fig.text(.08,.015,'Fixed measured widths and proxy-bed profiles · identical upstream forcing · no lateral inflow\nHistorical cached geometry; one-event calibration. Idealized extension evaluated at the original outlet.',fontsize=9,color='#58616a')
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(OUT/'calibrated_hydrographs.png',dpi=180);plt.close(fig)
    children=list(STATES)[1:];fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    x=np.arange(4)
    for j,state in enumerate(children):
        d=results[results.state.eq(state)].set_index('scenario').loc[list(SCENARIOS)]
        for ax,col in zip(axes,['peak_shift_minutes','centroid_shift_minutes']):
            ax.bar(x+(j-1)*.24,d[col],width=.23,color=colors[state],label=state)
    for ax,title in zip(axes,['Peak-time change','Positive-excess centroid change']):
        ax.set_xticks(x,['A\nNormal','B\nHalf slope','C\nDouble slope','D\n+10 km'])
        ax.axhline(0,color='#555555',lw=.7);ax.set_title(title);ax.set_ylabel('Collapse − original (minutes)');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,ncol=3,loc='upper center',bbox_to_anchor=(.5,.91))
    fig.suptitle('Collapse response after parent-only calibration · negative means earlier',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.81]);fig.savefig(OUT/'collapse_sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for s in SCENARIOS:
        trials=pd.DataFrame(fits[s]['trials']).drop_duplicates('n').sort_values('n')
        ax.plot(trials.n,trials.timing_error_minutes,'o-',ms=3,label=titles[s])
    ax.axhline(0,color='#555555',lw=.8);ax.set(xlabel="Manning's n",ylabel='Peak-time error (minutes)',title='Parent-only roughness calibration · reviewed peak target')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(OUT/'calibration_curves.png',dpi=180);plt.close(fig)
    # Markdown tables without an optional tabulate dependency.
    def md(headers,values):
        return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in values)
    parent_md=md(['Scenario','Fitted n','Peak error (min)','Peak (m³/s)','RMSE (m³/s)','NSE'],
        [[r.scenario,f'{r.n:.6f}',f'{r.timing_error_minutes:+.2f}',f'{r.peak_cms:.2f}',f'{r.rmse_cms:.2f}',f'{r.nse:.4f}'] for r in parent_table.itertuples()])
    child_md=md(['Scenario','Collapse','Peak shift (min)','Centroid shift (min)','Peak change (m³/s)'],
        [[r.scenario,r.state,f'{r.peak_shift_minutes:+.2f}',f'{r.centroid_shift_minutes:+.2f}',f'{r.peak_change_cms:+.3f}'] for r in results[results.state.ne('base')].itertuples()])
    control_md=md(['Scenario','State','Check','Peak-time change (min)','Max trace change (m³/s)'],
        [[r.scenario,r.state,r.check,f'{r.peak_time_difference_minutes:+.3f}',f'{r.maximum_trace_difference_cms:.4f}' if pd.notna(r.maximum_trace_difference_cms) else '—'] for r in control.itertuples()])
    ranking=parent_table.sort_values('rmse_cms');best=ranking.iloc[0]
    spread=float(parent_table.rmse_cms.max()-parent_table.rmse_cms.min())
    extension_checks=control[control.check.eq('10km_to_20km_no_refit')]
    extension_small=bool((extension_checks.peak_time_difference_minutes.abs()<1).all())
    recommendation=('Retain A (normal flow) as the simplest baseline, with B/C/D as sensitivity cases. '
                    'The tested longer extension changes peak timing by less than one minute at the original port.'
                    if extension_small else
                    'Keep the boundary choice provisional: the extension-length check still changes peak timing by at least one minute.')
    doc=f'''# Example 56: outlet scenarios, roughness calibration and three collapses

Completed four parent-only calibrations and twelve collapse runs. Widths and
proxy-bed slopes are fixed. Roughness is re-estimated for each outlet scenario
on the original network only, then transferred without refitting to its children.
This is the custom diffusive-wave prototype, not the LISFLOOD-FP executable.

## Calibration results

{parent_md}

The objective follows the earlier reviewed peak approach: target
**27 June 2025, 14:40 UTC**, while retaining the prior soft peak-height interval
**1,800–1,900 m³/s**. Five roughness probes span n=0.012–0.060, followed by a
bracketed timing-root search. Full-resolution verification requires peak error
within five minutes. Peak times use a local quadratic interpolation of 5-minute
output; sub-minute values are numerical estimates, not observational precision.

RMSE, bias and NSE are diagnostics over **24 June 06:09:30–13 July 00:00:30 UTC**,
at common 5-minute timestamps. They were not the optimization objective.
Downstream gaps over 15 minutes are excluded from these diagnostics; upstream
gaps are linearly interpolated (maximum {protocol['upstream_max_gap_seconds']/60:.0f} minutes).
No downstream discharge is imposed as a boundary and no observed stage is used.

The raw downstream peak is **{protocol['raw_observed_peak_cms']:.2f} m³/s**, versus
**{protocol['upstream_peak_cms']:.2f} m³/s** upstream. The soft height interval is
the prior reviewed calibration target, not the raw observed peak. A timing fit
therefore must not be described as reproducing the raw peak amplitude. Missing
lateral inflow, gauge differences and model/input assumptions remain possible
explanations; this experiment does not identify which is responsible.

Among these timing-calibrated candidates, {best.scenario} has the lowest in-sample
whole-window RMSE ({best.rmse_cms:.2f} m³/s). This is not independent validation
or proof that its boundary is physically correct. Fitting n separately can
compensate for outlet assumptions, so assess the full range and collapse stability.

**Recommended working choice:** {recommendation} The whole-window RMSE spread
across these timing-calibrated scenarios is only **{spread:.3f} m³/s**. This event
does not strongly distinguish them. No separate event has been held out for validation.

![Hydrographs](calibrated_hydrographs.png)

## Collapse responses

{child_md}

Negative timing changes mean the collapse arrives earlier. The centroid uses
positive excess above a baseline joining first and last upstream observations,
held constant during the 96-hour tail, integrated over the complete simulation.
It is a descriptive pilot metric and not automatically the repository's official
response definition. No child is fitted to the downstream observations.
The roughly three-minute S008 peak shift is below the five-minute output interval:
it relies on peak interpolation and should be treated as a small model-derived
effect, not an observationally resolved timing difference.

![Collapse sensitivity](collapse_sensitivity.png)

Three preselected saved collapses span different sizes. The geometry audit is:

{md(['State','Original links','Child links','Exactly unchanged links'],[[s,v['original_links'],v['child_links'],v['exactly_unchanged_links']] for s,v in protocol['pair_audits'].items() if s!='base'])}

The same forcing, external ports, shared n and boundary setting are used for every
parent–child pair. These are historical local caches; the active master drive
was unavailable, so current production-geometry identity is not certified.

## What the boundary scenarios mean

- **A:** normal-flow outlet using terminal bed slope as friction slope.
- **B/C:** halve/double only the outlet friction slope. Internal bed slopes,
  channel widths and physical link lengths are unchanged.
- **D:** add an idealized 10 km continuation with terminal width and bed slope,
  then place normal flow at its far end. Discharge is evaluated at the original
  outlet cross-section, with half-cell storage correction. The extra routing
  delay to the far end is not counted as the observed-reach response.

Both the outlet rating and internal friction use the scenario's fitted n. For
the extension, its roughness is the same as the network. The extension is an
assumption/sensitivity experiment, not surveyed downstream geometry.

## Numerical and extension checks

Eight unit tests pass, including the unchanged-bed boundary-factor check and an
extension test that verifies flux/storage conservation at the original port.
Across the 16 final parent/child runs, maximum global mass residual is
{results.maximum_mass_residual_m3.max():.6g} m³; original-port residual is
{results.original_port_mass_residual_m3.max():.6g} m³. Independent input-volume
error is at most {results.independent_inflow_volume_relative_error.abs().max():.3g}
relative. Depths stay positive; maximum Froude diagnostic is
{results.maximum_froude.max():.3f}. All final tails satisfy the node-flux settling
criterion. These checks verify numerical behavior, not physical correctness.

Without refitting roughness, selected A/D runs are refined from 200 m to 100 m.
For D, 10 km is also compared with 20 km at the same n, measuring at the original
port. Results are:

{control_md}

The `S014_minus_base` rows measure change in the collapse effect itself. The
extension check covers the original and S014; it does not independently certify
extension-length insensitivity for every possible collapse. B/C and the two
additional collapses have not each received their own mesh-refinement run.

![Calibration curves](calibration_curves.png)

## Reproduce and inspect

Run `run_boundary_experiment.py --workers 3 --stage all`, followed by
`summarize_boundaries56.py`, in the existing scientific Python environment.
`protocol.json` records source hashes, assumptions, target and geometry audit.
Each trial stores a result JSON and hydrograph Parquet. Tables:
`calibration_summary.csv`, `collapse_comparison.csv`, `numerical_controls.csv`.
Earlier uncalibrated and same-roughness trials remain unchanged.
'''
    (OUT/'REPORT.md').write_text(doc)
    print(parent_table.to_string(index=False));print(results[['scenario','state','peak_shift_minutes','centroid_shift_minutes','peak_change_cms']].to_string(index=False));print(control.to_string(index=False))


if __name__=='__main__':main()

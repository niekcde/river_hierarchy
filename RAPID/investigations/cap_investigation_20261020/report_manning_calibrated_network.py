"""Report the completed isolated network-22 Manning calibration experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from manning_calibrated_network import DEST, digest, table, ROOT, OUT


def main():
    manifest=json.loads((DEST/'manifest.json').read_text());assert manifest['complete']
    cal=json.loads((DEST/'calibration.json').read_text());fit=cal['selected'];old=cal['old_selection']
    data=pd.read_csv(DEST/'comparison.csv',dtype={'example':str});d=data[data.window.eq('0.5%')].copy()
    summary=pd.read_csv(DEST/'summary.csv');audit=pd.read_csv(DEST/'routing_audit.csv')
    assert len(audit)==411 and audit.settled.all()
    assert audit.n.nunique()==audit.beta.nunique()==1
    assert np.isclose(audit.beta.iloc[0],.5)
    assert len(d)==410 and len(data)==1230
    primary=d[d.edit_type.ne('Single-centerline reference control')]
    regular=primary[primary.cohort.eq('original') | primary.cohort.eq('regular')]
    # Inventory labels regular records as original in some historical tables;
    # here use the explicit segmented category to avoid dependence on that label.
    regular=primary[primary.edit_type.ne('Segmented full network')]
    assert len(primary)==409 and len(regular)==408
    comparisons=[]
    for label,g in [('Regular collapses',regular),('Accepted full population',primary)]:
        comparisons.append(dict(population=label,cases=len(g),based_delays_gt1min=int((g.centroid_minutes_based>1).sum()),
          manning_delays_gt1min=int((g.centroid_minutes_manning>1).sum()),based_subminute_delays=int(((g.centroid_minutes_based>0)&(g.centroid_minutes_based<1)).sum()),
          manning_subminute_delays=int(((g.centroid_minutes_manning>0)&(g.centroid_minutes_manning<1)).sum()),
          changed_response_class=int((g.response_class_based!=g.response_class_manning).sum()),
          changed_sign=int((g.centroid_minutes_based*g.centroid_minutes_manning<0).sum())))
    totals=pd.DataFrame(comparisons);totals.to_csv(DEST/'population_summary.csv',index=False)
    cases=['S057_unit_5','island_22_018','S296_unit_30','CB001_bubble_9','all_closed_units_collapsed','fully_collapsed']
    selected=d.set_index('state').loc[cases].reset_index()
    selected['peak_minutes_based']=selected.peak_shift_hours_based*60
    selected['peak_minutes_manning']=selected.peak_shift_hours_manning*60
    selected.to_csv(DEST/'selected_comparisons.csv',index=False)
    # Numerically largest changes are diagnostic selections, not fitting targets.
    d['absolute_centroid_change_min']=abs(d.centroid_minutes_manning-d.centroid_minutes_based)
    d.nlargest(20,'absolute_centroid_change_min').to_csv(DEST/'largest_response_changes.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for kind,g in primary.groupby('edit_type'):
        axes[0].scatter(g.centroid_minutes_based,g.centroid_minutes_manning,label=kind,s=20,alpha=.65)
    extent=max(abs(primary.centroid_minutes_based).max(),abs(primary.centroid_minutes_manning).max())*1.1
    axes[0].plot([-extent,extent],[-extent,extent],color='grey',linestyle='--')
    axes[0].set(xscale='symlog',yscale='symlog',xlabel='BASED centroid shift (min)',ylabel='Manning centroid shift (min)',
                title='409 accepted comparisons; symmetric-log axes')
    axes[0].axhline(0,color='grey',lw=.5);axes[0].axvline(0,color='grey',lw=.5);axes[0].legend(fontsize=7)
    groups=['advance','near_zero','delay'];x=np.arange(3)
    for side,offset,color in [('based',-.18,'#666666'),('manning',.18,'#0072B2')]:
        counts=[int((regular['response_class_'+side]==g).sum()) for g in groups]
        bars=axes[1].bar(x+offset,counts,width=.36,label=side,color=color)
        axes[1].bar_label(bars,fontsize=9)
    axes[1].set(xticks=x,xticklabels=['Advance >1 min','Within ±1 min','Delay >1 min'],ylabel='Regular collapse cases',title='408 regular cases')
    axes[1].legend();axes[0].grid(alpha=.2);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(DEST/'overview.png',dpi=160);plt.close(fig)
    report=f'''# Network 22: calibrated Manning-depth model and every saved collapse

Completed 27 October 2026. Existing BASED results remain unchanged. The new
Manning formulation is calibrated only on the full-complexity reference state;
its selected n and active diffusion multiplier are then frozen across all
collapse states. This is a constant-reference-depth experiment, not variable
depth through time. No RF fitting, RF input replacement or production promotion.

## Calibration

| Quantity | Existing BASED | Manning-derived depth |
|---|---:|---:|
| Manning n | {cal['old_n']:.9f} | {fit['n_manning']:.9f} |
| Diffusion multiplier beta | {cal['old_beta']:.3f} | {fit['beta']:.3f} |
| Reviewed crest timing error, min | {old['timing_error_minutes']:.3f} | {fit['timing_error_minutes']:.3f} |
| Crest discharge, m³/s | {old['peak_cms']:.3f} | {fit['peak_cms']:.3f} |
| Shortfall from soft height range, m³/s | {old['height_distance_cms']:.3f} | {fit['height_distance_cms']:.3f} |

Target: {cal['timing_target']['target_utc']}; original roughness bounds 0.01–0.1.
Timing tolerance of one minute met: **{cal['timing_tolerance_met']}**.
Selected n is on a search boundary: **{cal['n_at_boundary']}**.
The 20,300–20,400 m³/s height interval is a soft modelling target, not a claimed
measurement uncertainty interval. It remains unmet. Timing is the fitted
objective; height is reported rather than independently optimized through beta.
The original observation-clock ambiguity remains unresolved. This calibration
is not independent validation and is not evidence that the new depths are true.

Depth is recalculated from allocated reference discharge for every trial n:
h=(n Q_ref / (B sqrt(S)))^(3/5). Then c=(5/3n)h^(2/3)sqrt(S), K=dx/c and
X=0.5−beta Q_ref/(2 B S c dx) are recalculated. The active beta=0.5 is retained
in every trial and state. The original geometry, cells, slopes and branch
fractions stay fixed within each state. No celerity clipping is applied.

The existing timing target, archived crest-extraction function, calibration
forcing, initial state and n bounds are reused. A coarse scan brackets the
target, followed by safeguarded secant refinement; all trials are saved.
The selected calibration is also checked at half the native timestep.

## Coverage

All **411 saved network states** were routed: one uncollapsed parent and
**410 collapse/control states**. The established analysis population consists
of 408 regular collapses plus one segmented full-network collapse (409
comparisons). The additional single-centerline reference control is reported
separately and is not silently added to that population. Aliases are retained,
not counted as additional independent states. Later combination-augmentation
experiments outside the saved main routing plan are not included.

{table(totals)}

## Response changes by case type

Delay counts use centroid shift >1 minute. Medians and shifts are child minus
the parent under the same formulation. All values below use the original shared
0.5% window and baseline; negative timing shifts are advances.

{table(summary)}

## Selected cases

Timing values are minutes. The reference single-centerline control
(`fully_collapsed`) differs from the segmented all-closed-units state.

{table(selected[['state','centroid_minutes_based','centroid_minutes_manning','peak_minutes_based','peak_minutes_manning']])}

![Centroid responses and regular-case counts](overview.png)

The full comparison also includes peak-excess magnitude, central 80% volume
duration, asymmetry and excess volume, at all three original windows (0.5%, 1%,
2%). Smaller absolute shifts or fewer delays are not by themselves evidence of
greater physical accuracy. This experiment changes the depth closure and then
recalibrates roughness; it therefore measures the representation effect under
two independently parameterized parent models, rather than a pure depth-only
intervention at identical n.

## Verification

All states have nonnegative discharge and settled tails. Tail extensions needed:
{int(audit.tail_extensions.sum())}. Parameters, outlets, metrics and hashed
completion receipts are retained for every state. Every state uses its original
native routing clock. Representative and extreme-response states are rerouted
at half that timestep; maximum centroid-contrast difference is
{manifest['maximum_centroid_clock_difference_s']:.6f} seconds. Peak-bin changes,
if any, remain explicit in clock_checks.csv. The parent calibration also has a
separate half-clock check. No claim of a new full spatial-convergence study is
made.

The router is the same constant-reference RAPID/Cunge DAG cascade used by the
accepted event-window workflow. Response calculations use the existing
piecewise-linear volume, centroid and quantile routines. The original shared
windows and baseline are retained for controlled comparison, not newly selected
after fitting. All input hashes are verified at completion.

## Correction to the earlier pilot

The earlier three-depth-option and four-collapse pilots inadvertently reset
beta to 1 for the Manning alternatives while the active BASED model used 0.5.
Their numerical outputs are retained as historical mixed sensitivity results,
and their reports now explicitly disclose that confounding. They should not
be cited as depth-only tests. This completed experiment preserves beta=0.5 and
adds the requested n recalibration, so its numbers supersede the pilot for the
present comparison. The existing BASED model and its outputs were not altered.

## Reproduction and files

Run `RAPID/investigations/cap_investigation_20261020/manning_calibrated_network.py`
with the river-hierarchy-rivgraph Python environment. Completed states resume
only after checking their n, source-parameter hash and output hashes. Run
`report_manning_calibrated_network.py` to rebuild this report.

- calibration.json, calibration_trials.csv and calibration_wave_clock*.parquet
- states/<state>/parameters.parquet, outlet.parquet, metrics.parquet, completion.json
- state_metrics.parquet and responses.csv (new Manning results)
- comparison.csv (all six response contrasts versus existing BASED)
- summary.csv, population_summary.csv, class_transitions.csv
- selected_comparisons.csv, largest_response_changes.csv, clock_checks.csv
- routing_audit.csv, design.json and manifest.json
'''
    (DEST/'REPORT.md').write_text(report)
    manifest['report_script_sha256']=digest(Path(__file__))
    manifest['summary_output_hashes']={str(p):digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(totals.to_string(index=False))
    print(selected[['state','centroid_minutes_based','centroid_minutes_manning']].to_string(index=False))


if __name__=='__main__':main()

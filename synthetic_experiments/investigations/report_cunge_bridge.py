"""Render frozen-model transfer diagnostics; no fits or routing."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'synthetic_experiments/outputs/current_predictor_bridge_20261026'
PREFLIGHT=ROOT/'synthetic_experiments/outputs/cunge_grid_preflight_20261026'

def main():
    d=pd.read_csv(OUT/'predictions.csv');valid=d[d.status.eq('predicted')]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colours={'narrow_long':'#0072B2','dominant_long':'#D55E00'}
    markers={'simple':'o','nested':'s','crossover':'^'}
    for col,(model,title) in enumerate([('geometry_9','Geometry: 9 inputs'),('depth_informed_10','Depth-informed: 10 inputs')]):
        group=valid[valid.model.eq(model)]
        for row in range(2):
            ax=axes[row,col];g=group if row==0 else group[group.configuration.eq('narrow_long')]
            for r in g.itertuples():
                ax.scatter(60*r.actual_hours,60*r.predicted_hours,c=colours[r.configuration],marker=markers[r.child_state.split('/')[0]],s=45)
            both=np.r_[60*g.actual_hours,60*g.predicted_hours];lo,hi=min(both.min(),0),max(both.max(),0)
            pad=max(1,(hi-lo)*.08);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='0.5',ls='--',lw=1)
            ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Routed centroid shift (minutes)',ylabel='Frozen predicted shift (minutes)')
            ax.set_title(title+(' · all 14 eligible pairs' if row==0 else ' · narrow-long cases only'))
            ax.grid(alpha=.15)
    fig.suptitle('Frozen real models on the synthetic pilot\nBlue: narrow-long; orange: dominant-long. Circle: simple; square: nested; triangle: crossover.',fontsize=12)
    fig.savefig(OUT/'frozen_transfer.png',dpi=170);plt.close(fig)
    c=json.load(open(OUT/'completion.json'));p=json.load(open(PREFLIGHT/'completion.json'))
    errors=pd.read_csv(OUT/'pilot_errors.csv');cost=pd.read_csv(PREFLIGHT/'conditional_cost.csv')
    lines=['# Frozen predictor transfer and factorial preflight — 26 October 2026','',
      'Reference: SYN-CUNGE-BRIDGE-2026-10-26. This follows the verified Cunge pilot. No real model was fitted or modified and no full factorial was routed.','',
      '## Feature parity and scope','',
      f"The shared descriptor/allocation functions plus explicit-depth resistance and inverse-depth G reproduced all 11 distinct inputs used across the two models in {c['real_parity_cases']} saved real cases spanning all 15 networks (including three containing-compound checks). Details: real_feature_parity.csv. Four additional semantic tests passed.",'',
      'Synthetic B domains are the closed outer 25–75% interval. For nested/crossover members this is the containing compound, not only the edited unit. Size sums parent B-domain allocation times length over selected unit membership. Main-island size uses its outer source-unit membership, matching the real source-unit convention; it is not the sum of removed edges. Geometry uses unsegmented physical edges; derived depths share the routing anchor.','',
      '## Frozen predictions','',
      'Both forests predict 14/21 pairs. All seven baseline pairs have equal path lengths and an undefined parent/child route-spread ratio (0/0). They remain in the routing results, but the frozen prediction wrapper requires finite predictors: no arbitrary zero/one substitution or imputation was introduced.','',
      '| Model | Eligible pairs | MAE (minutes) | Maximum absolute error (minutes) |','|---|---:|---:|---:|']
    for r in errors.itertuples():lines.append(f'| {r.model} | {r.n} | {r.mae_minutes:.2f} | {r.max_abs_error_minutes:.2f} |')
    lines+=['','These are errors on a small deliberately selected stress-test pilot, not representative external validation scores. The depth-informed model does not outperform the geometry model here. The dominant-long simple collapse has a route-length ratio outside the real training range; marginal range inclusion elsewhere does not establish joint support. See feature_coverage.csv and predictions.csv.','',
      'The baseline degeneracy is structural, not a routing failure. In the original grid, at least 5,103 pairs (all 729 length-multiplier=1 configurations) have undefined spread ratios. Another 486 simple/full pairs with equal branch widths and length multiplier >1 have undefined width-CV ratios. Thus at least 5,589/15,309 pairs cannot directly enter the unchanged frozen models. Retain these valid geometries for sensitivity; any revised ratio convention or descriptor requires a separately versioned model and real-data revalidation.','',
      '## Full factorial preflight','',
      f"All {p['passed_configurations']:,} configurations passed physical-input and shared-clock checks: {p['comparison_references']:,} comparison references, {p['state_references']:,} state references, and {p['provisional_unique_jobs']:,} provisional unique parameter/clock jobs. Every configuration permits dt=15 seconds under the current policy. Maximum path-sum K is {p['maximum_path_K_hours']:.2f} hours.",'',
      'This is input feasibility, not proof of full-grid positivity, settlement or spatial convergence. Job deduplication is provisional until exact shared windows and implementation provenance are included.','',
      '| Assumed common tail | In-memory single-core routing estimate | Uncompressed time+outlet storage |','|---|---:|---:|']
    for r in cost.itertuples():lines.append(f'| {r.tail_hours:g} h | {60*r.unique_routing_hours_fast:.1f}–{60*r.unique_routing_hours_slow:.1f} min | {r.uncompressed_time_and_outlet_GiB:.2f} GiB |')
    lines+=['','These throughput extrapolations exclude preparation, I/O, repeated tail extensions, descriptor calculation and refinement reruns; they are not end-to-end completion forecasts. Actual tail requirements remain per-configuration checks.','']
    stress=PREFLIGHT/'stress_runs/completion.json'
    if stress.exists():
        s=json.load(open(stress));checks=pd.read_csv(stress.parent/'numerical_checks.csv')
        lines += ['## Selected grid-extreme routing checks','',f"Routed {s['comparisons']} extra comparisons chosen by maximum path K, most negative X, highest section count and highest speed (duplicate selections merged). Numerical screen passed: {s['numerical_screen_passed']}. Largest refinement centroid difference: {checks.max_centroid_shift_difference_minutes.max():.4f} minutes. See ../cunge_grid_preflight_20261026/stress_runs/.",'']
    lines+=['## Next decision','',
      'The full grid can proceed as a controlled centroid-sensitivity experiment only with per-run settlement, volume and native-flow checks plus flagged refinement checks. Frozen transfer can proceed on the explicitly eligible subset. A universal frozen-predictor comparison needs a decision on the degenerate ratios; do not perturb baseline geometries merely to make model inputs finite. New training/augmentation is not performed by these scripts.','',
      'Reproduce: current_predictor_bridge.py; cunge_grid_preflight.py; report_cunge_bridge.py, under synthetic_experiments/investigations, using the project Python and single-thread BLAS. The selected extreme pilot config is saved at the preflight output root.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

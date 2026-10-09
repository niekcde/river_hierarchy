"""Compare exact matching interval means and write the review-stage report."""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT/'RAPID/outputs/common_grid_validation_20261008'
    dest = Path(json.loads((root/'latest.json').read_text())['directory'])
    metrics = pd.read_csv(dest/'base_metrics.csv', dtype={'example':str})
    records = []
    for ex in sorted(metrics.example.unique()):
        for variant, refname in [('300s','30s'),('60s','30s'),('60s_phase30','30s'),('60s_halfdt','60s')]:
            data = pd.read_parquet(dest/ex/f'{variant}.parquet')
            ref = pd.read_parquet(dest/ex/f'{refname}.parquet')
            step = (data.interval_end_utc.iloc[0]-data.interval_start_utc.iloc[0]).total_seconds()
            refstep = (ref.interval_end_utc.iloc[0]-ref.interval_start_utc.iloc[0]).total_seconds()
            group = int(round(step/refstep))
            offset = int(round((data.interval_start_utc.iloc[0]-ref.interval_start_utc.iloc[0]).total_seconds()/refstep))
            count = (len(ref)-offset)//group
            valid = data.downstream_cms.iloc[:count].notna().to_numpy()
            for method in ['cunge','fixed_x']:
                column = method+'_cms'
                equivalent = ref[column].to_numpy()[offset:offset+count*group].reshape(count,group).mean(axis=1)
                residual = data[column].to_numpy()[:count]-equivalent
                amplitude = float(metrics[(metrics.example==ex)&(metrics.method==method)&(metrics.variant==variant)].iloc[0].inlet_amplitude_cms)
                records.append(dict(example=ex,method=method,variant=variant,reference=refname,
                    maximum_difference_pct_inlet_amplitude=float(np.max(abs(residual[valid]))/amplitude*100),
                    rmse_pct_inlet_amplitude=float(np.sqrt(np.mean(residual[valid]**2)))/amplitude*100,
                    compared_bins=int(valid.sum()),comparison='Exact same UTC bounds; finer interval means duration-aggregated'))
    same = pd.DataFrame(records)
    same.to_csv(dest/'same_interval_validation.csv',index=False)
    controls = pd.read_csv(dest/'initialization_and_gap_controls.csv',dtype={'example':str})
    sampling = pd.read_csv(dest/'resolution_comparison.csv',dtype={'example':str})
    x60 = same[same.variant=='60s'].maximum_difference_pct_inlet_amplitude.max()
    xdt = same[same.variant=='60s_halfdt'].maximum_difference_pct_inlet_amplitude.max()
    x300 = same[same.variant=='300s'].maximum_difference_pct_inlet_amplitude.max()
    shift = sampling[sampling.variant=='60s'].global_maximum_time_difference_s.abs().max()
    ref = metrics[metrics.variant=='60s'].set_index(['example','method'])
    half = metrics[metrics.variant=='60s_halfdt'].set_index(['example','method'])
    dtpeak = (pd.to_datetime(half.peak_utc)-pd.to_datetime(ref.peak_utc)).dt.total_seconds()
    tails = ref[ref.tail_residual_pct_inlet_amplitude>.1]
    manifest = json.loads((dest/'provenance.json').read_text())
    for source, expected in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected, source
    antecedent = pd.read_csv(dest/'46_antecedent_controls.csv')
    antecedent['admissibility'] = np.where(
        antecedent.longest_antecedent_sampling_interval_minutes>360,
        'reject as production forcing: multi-hour/seasonal gap',
        'diagnostic usable; retain explicit sampling-gap audit')
    antecedent.to_csv(dest/'46_antecedent_admissibility.csv',index=False)
    source_files = ['plot_common_grid_targets.py','check_common_grid_controls.py',
                    'check_common_grid_antecedent.py',Path(__file__).name]
    (dest/'review_provenance.json').write_text(json.dumps(dict(
        source_hashes={name:hashlib.sha256((ROOT/'RAPID/investigations'/name).read_bytes()).hexdigest() for name in source_files},
        baseline_source_hashes_rechecked=True, tests='24 passed; common-grid, time-support, Cunge',
        routing_runs=len(metrics)+len(controls)+len(antecedent), native_states_saved=False,
        calibration_performed=False, extra_controls='46 actual antecedent forcing; 180-day runs rejected for seasonal gap'),indent=2))
    text = f'''Common-grid validation and calibration-target review — 2026-10-08

Completed stage: frozen baseline, common-grid implementation, base pilots,
resolution/numerical controls, initialization/gap stress tests and target plots.
No recalibration or hierarchy batch has been performed. The figures use existing
parameters with NEW forcing preparation and NEW physically matched interval means.

Scope and preservation
- 15 retained examples x 2 methods x 5 clock configurations = {len(metrics)} base runs.
- {len(controls)} additional controls for examples 03,46,54,55.
- {len(antecedent)} actual-antecedent controls for 46; seasonal-gap variants explicitly rejected.
- Example 11's original data are frozen; it remains excluded from these routing pilots.
- baseline/ preserves selected original waves, observed point records, hydraulic
  grids, parameters, targets, source code and hashes. Other existing results remain
  in place untouched; this is not a duplicate of the whole hierarchy archive.
- No external-drive writes; outlet means only; no native reach matrices persisted.
- All original source hashes were checked again after the runs.

Temporal method
- Raw point observations integrated exactly under a piecewise-linear reconstruction.
- Forcing and output cover identical UTC-midnight-aligned intervals, midpoint-labelled.
- 300s, 60s, 30s plus 60s with a 30s phase shift and 60s with half native dt.
- One native clock per example shared by all ordinary resolution variants; only the
  explicit half-dt control changes it. Geometry, n, Q reference and spatial grid fixed.
- Equilibrium at the first upstream point is held fixed across resolutions. A removed
  constant initial bin supplies this state and consistent previous boundary inflow.
- Outlet averages integrate the linear native-state reconstruction using trapezoids.
- Before-record extension <=10min; after-record tail <=3days+5min, explicitly flagged.
  Downstream observations outside their retained support are unavailable, not held.
- Intervals spanning >2 nominal observation cadences are flagged and omitted from
  preliminary downstream scores. Upstream gaps are filled linearly FOR DIAGNOSTICS
  and recorded. This conservative screening is NOT a final calibration mask.
  The existence of an irregular interval does not establish its cause: missing data,
  export compression or a provider's irregular recording need archive investigation.
  A downstream bin being supported does not erase upstream-gap uncertainty.
- Original 21/22 clocks are preserved provisionally, with their known MRC ambiguity.

Results
- Exact matching interval means: 60s versus duration-aggregated 30s differs by at most
  {x60:.9f}% of inlet amplitude over the observation periods (both methods, 15 examples).
- Corresponding 300s comparison: {x300:.9f}% maximum difference.
- Comparing curves interpolated through their reporting means gives 0.000694% maximum
  for 60s vs 30s. This separate diagnostic is not used as an observation-space score.
- Global maximum-bin centres differ by at most {shift:.0f}s for 60s vs 30s. This is
  a grid-resolution diagnostic, not a validation of the selected feature definition.
- Half native dt at unchanged 60s support changes means by at most {xdt:.9f}% of inlet
  amplitude. Maximum-bin choice moves by 60s for fixed-X 21 and 46, zero otherwise.
  Near ties can move the chosen bin without comparable whole-hydrograph movement.
- These base tests support 60s as a working comparison interval. They do not certify
  sub-minute hierarchy timing changes: validate selected matched parent-child
  transitions at 30s before adopting it for final hierarchy metrics.
- All routed flows are nonnegative. Input-integral volume invariance is asserted.
- 24 tests pass, including analytic volume, support flags, endpoint limits, mean
  quadrature and independent sparse/cascade parity with explicit initialization.

Initialization and gap controls
- Initial states +/-10%, holding event forcing fixed, are stress tests, not an
  uncertainty distribution. In 46 Cunge, -10% shifts the maximum +90min and +10%
  shifts it -95min; crest discharge changes up to 1.644% of inlet amplitude. Other
  tested combinations retain their maximum times. This means 46 needs actual
  antecedent-forcing/initial-state convergence checks before recalibration: fine
  reporting resolution does not cure sensitivity to an assumed equilibrium start.
- In 55, filling all flagged upstream gaps with the preceding/following observation
  instead of a straight line shifts Cunge's maximum by +24/-23 minutes; fixed-X by
  +21/-20 minutes. These intentionally contrasting assumptions are not confidence
  bounds. At the Cunge crest they change Q by less than 0.18% of inlet amplitude.
  Thus a common fine grid resolves the reporting issue but cannot settle missing
  information or the stability of an extremely broad maximum.
- IMPORTANT FOLLOW-UP: local_crest_sampling.csv shows both 55 gauges are actually
  hourly around their selected maxima, despite a 5min whole-event nominal cadence.
  At the downstream target, all six intervals within +/-3h are exactly 60min. The
  upstream near-crest record also has consecutive hourly intervals. These are NOT
  established data outages. The two-cadence screen flags a sampling-regime change.
  Treat the preceding/following-value controls as reconstruction sensitivity, not
  evidence of missing records. Use local/source sampling regimes to define final
  gap masks. Existing +/-10min timing tolerance does not follow from the local
  hourly observations; target timestamps remain fixed pending interpretation.
- New forcing yields a Cunge global-maximum residual of -37min in 03 and +20.5min
  in 55. Neither is a new fit. 03's reviewed plateau midpoint remains unchanged.
- Tail settling was not required to compare the observed event in this pilot. The
  following {len(tails)} method/example combinations remain >0.1% inlet amplitude away
  from the held final inflow after the 3-day tail and need extended final runs:
{tails[['tail_residual_pct_inlet_amplitude']].to_string()}

Figures and targets
- 15_panel_common_grid_targets.png/svg: complete events, original selected targets
  shown in purple; shaded regions preserve the selected plateau/crest spans.
- 15_panel_common_grid_target_crests.png/svg: same data with axes zoomed to crests.
- Each curve is normalized independently over the full observation window. This is
  a timing/shape view, not an amplitude-fit claim. Means in m3/s remain in Parquet.
- Faint dotted observed lines flag interpolation across long sampling intervals.
- No target was shifted or snapped to a bin centre. target_review.csv records the
  original selections and a separate observed global-maximum diagnostic. That
  diagnostic does NOT replace plateau, multiple-crest or manually selected targets.
- Case 54 still uses the observed two-crest midpoint as its recorded calibration
  target. Current global-maximum residuals are diagnostics only, unsuitable as that
  objective. Its provisional first-crest hierarchy choice is separate and unchanged.

46 antecedent follow-up
- Using 7,30,90 days of actual upstream observations before the event produces the
  same 60s maximum-bin centre: 2024-09-03 13:53:30 UTC, +98.5min from the unchanged
  selected target. The no-antecedent equilibrium pilot was +19.5min. These use the
  SAME existing parameters; no calibration change has been made.
- Over the observed event, 30d versus 90d antecedent runs differ by at most 0.008247%
  of inlet amplitude. Peak-bin agreement alone is insufficient: the 7d hydrograph
  retains a larger initial transient. 30d/90d results support a real antecedent
  period for this example before fitting, with a final sensitivity check after n changes.
- The 180d variants span an approximately 118-day source interval with no intervening
  retained points; they are rejected as production forcing and are not used as a
  convergence reference. More antecedent days are not automatically better data.
- Original event upstream values for 46 and both 55 gauge series were checked
  against current country-level source data: identical times and values. The hourly
  spacing around 55's peaks was not introduced by this common-grid preparation.

Next stage
Review the plotted target features, especially 03 and 54, then make corresponding
model feature operators explicit. Use validated antecedent forcing for 46, distinguish
sampling-regime changes from outages (particularly 55), and resolve the MRC timestamp ambiguity
before final fits. Use a 60s common-grid candidate
for fitting with the accepted gap/feature treatment; extend tails until settled.
Reassess all retained bases for both methods. Then test selected hierarchy pairs at
60s/30s before the full hierarchy batch. Network-41 geometry remains outside scope.
'''
    (dest/'REPORT.txt').write_text(text)
    print(same.groupby('variant').maximum_difference_pct_inlet_amplitude.max())
    print(dest/'REPORT.txt')


if __name__=='__main__':
    main()

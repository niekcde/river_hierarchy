"""Summarize the isolated cached-routing benchmark, without rerouting."""
from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parents[1]/'outputs/depth_reuse_benchmark_20261027'
f=pd.DataFrame(json.loads((OUT/'results.json').read_text()));c=json.loads((OUT/'weighted_census.json').read_text())
summary=f[f.variant!='smaller_tail'].groupby(['variant','method']).agg(seconds=('seconds','sum'),hits=('hits','sum'),misses=('misses','sum'))
print(summary.to_string());print('\nTail tests\n',f[f.variant=='smaller_tail'].to_string(index=False));print(c)
base=f[f.variant=='baseline'].seconds.sum();cached=f[f.variant=='disk_cache'].seconds.sum()
# Conditional projection: apply cell-weighted reuse to earlier measured cost scenarios.
prior=json.loads((OUT.parent/'depth_methods_preflight_20261027/benchmark_projection.json').read_text())
projection={case:sum(r['grid_hours_if_this_tail_and_cost_per_cell'] for r in prior if r['case']==case)*c['remaining_fraction'] for case in ['typical','long_low_slope']}
print('Conditional routing projection:',projection)
lines=['# Cached-routing and tail-growth benchmark — 27 October 2026','',
'Reference: SYN-DEPTH-REUSE-BENCHMARK-2026-10-27. Isolated investigation; production solver and pipeline files were not changed. No full factorial was launched.','',
'## Experiment','',
'Four core-grid geometries: typical 20-km and demanding 40-km low-slope scenarios, each with inner fraction 0.2 and 0.8. All three methods, all seven collapse comparisons. Twelve groups per variant. Baseline retains existing within-group reuse; the candidate additionally caches native state traces and audit values on disk across groups. Cache starts empty. Keys include prepared frame data, hydraulic settings including tail, forcing hash and source provenance. This is an experimental cache, not a production-ready implementation.','',
'The benchmark compares response tables to 1e-12 tolerance and saved hydrograph tables bit-for-bit between baseline and cached runs. All comparisons passed; existing conservation and convergence tests also passed. Each group is timed once; timings include preparation and usual output writing. CPU contention, startup and filesystem cache effects mean these are planning measurements, not precise performance guarantees.','',
'## Measured results','',
'| Method | Baseline seconds | Cached seconds | Saving |','|---|---:|---:|---:|']
for method in ['fixed_based','fixed_manning','variable_manning']:
 a=f[(f.variant=='baseline')&(f.method==method)].seconds.sum();b=f[(f.variant=='disk_cache')&(f.method==method)].seconds.sum()
 lines.append(f'| {method} | {a:.2f} | {b:.2f} | {100*(1-b/a):.1f}% |')
lines += [f'| All methods | {base:.2f} | {cached:.2f} | {100*(1-cached/base):.1f}% |','',
'Only inner-fraction duplication is sampled here. It is not the same reuse pattern as the full eight-factor grid. No repeated aggregate-file writing is included in either variant. The previously measured potential ~7.4-hour aggregation penalty is separately avoidable by final/periodic aggregation.','',
'## Smaller tail increments','',
'Both schedules rerun from the beginning. No state continuation or weaker tolerance is used. Compare doubling with multiplication by 1.5, starting at 96 hours. Only variable Manning tested for this option.','',
'| Geometry | Doubling seconds / tails (h) | Smaller increments seconds / tails (h) | Max centroid-shift difference (s) | Max peak-shift difference (s) |','|---|---|---|---:|---:|']
for _,r in f[f.variant=='smaller_tail'].iterrows():
 a=f[(f.variant=='baseline')&(f.method=='variable_manning')&(f['case']==r['case'])&(f.inner==.2)].iloc[0]
 lines.append(f"| {r['case']} | {a.seconds:.2f} / {a.tails} | {r.seconds:.2f} / {r.tails} | {r.max_centroid_shift_difference_seconds:.6g} | {r.max_peak_shift_difference_seconds:.6g} |")
lines += ['', '## Full-grid workload and prognosis','',
 f"Cell-weighted census, per method per tail attempt: {c['per_method_cells_without_global_cache']:,} cell traversals without global caching versus {c['per_method_cells_with_global_cache']:,} with it. Remaining fraction {c['remaining_fraction']:.3f}. This refines the earlier state-count reduction; it still does not know the convergence tails or nonlinear iterations of every state.",'',
 f"Applying this fraction to the earlier measured cost-per-cell scenarios gives {projection['typical']:.1f}–{projection['long_low_slope']:.1f} hours of conditional routing/preparation/output cost on one worker. This extrapolation assumes compatible reusable tails and broadly transferable per-cell costs; cache reads, per-group preparation/output and heterogeneous convergence add work. It is not a measured full-grid runtime or a guaranteed upper bound. The current prototype reuses only exactly matching tail settings; a full grid can require multiple tails per state.",'',
'No two-worker speedup was benchmarked. Do not divide by two and present that as a measured completion estimate. Shared-cache coordination and additional memory/I/O must first be implemented and checked.','',
'## Artifacts','',
'`results.json`: timed groups and tail checks. `routing_calls.csv`: timings for every newly calculated state/tail. `weighted_census.json`: exact geometry/cell enumeration. Individual group folders retain hydrographs, response tables, parameters and convergence audit. Reproduction: investigations/benchmark_depth_reuse.py (refuses to overwrite completed measured results); summarize with investigations/summarize_depth_reuse_benchmark.py.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');summary.to_csv(OUT/'timing_summary.csv')
with (OUT/'REPORT.md').open('a') as report:
 report.write('''

## Recommendation after the benchmark

Implement shared caching and periodic/final aggregation, retaining the current tail doubling. The small benchmark panel improved from 140.14 to 103.43 seconds (26.2%, including cache population); later variable-depth cases improved by 39–55%. Smaller tail increments saved about 14% for the typical case but cost about 18% more for the long case. Response differences were at most 0.0011 seconds in centroid shift and zero in peak shift, so runtime—not altered conclusions—is the issue here.

The cell-weighted full-grid extrapolation is approximately 10–24 hours before allowing for non-reusable preparation, output, cache overhead and incompatible tail histories. A provisional one-worker budget of roughly 12–30 hours is reasonable for planning, but is not a demonstrated bound or promised completion time. Two-worker runtime remains untested.

The experimental cache saves uncompressed native traces for audit and benchmark simplicity. Do not scale that disk layout blindly to the full grid with the currently limited free space. A production cache should consider compressed/shared traces or cached native-derived statistics plus reported hydrographs, without changing metric calculation. Storage remains a separate launch prerequisite.
''')

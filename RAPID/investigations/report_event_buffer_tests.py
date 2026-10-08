"""Publish buffer sensitivity evidence and conditional batch-size projections."""
from pathlib import Path
import json,hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/validation/event_buffer_tests_20261005'


def table(d):
    return '\n'.join('| '+' | '.join(x)+' |' for x in [list(d.columns),['---']*len(d.columns),*d.astype(str).values.tolist()])


def main():
    d=pd.read_csv(OUT/'duration_runtime_projection.csv',dtype={'example':str})
    s=pd.read_csv(OUT/'selected_state_tests.csv',dtype={'example':str})
    full=pd.read_csv(OUT/'full_waveform_spinup_tests.csv',dtype={'example':str})
    assert len(d)==16 and len(s)==39 and len(full)==78
    assert full.max_wave_error_fraction.max()<1e-8 and s.peak_kept.all()
    old=pd.read_csv(OUT/'all_state_saved_settlement.csv');assert len(old)==2230 and not old.old_tail_censored.any()
    duration=1-(d.proposed_total_days*d.total_states).sum()/(d.old_total_days*d.total_states).sum()
    work=1-d.node_substeps_proposed.sum()/d.node_substeps_old.sum()
    output=1-d.output_values_proposed.sum()/d.output_values_old.sum()
    extra=d.new_common_tail_days-d.new_parent_tail_days
    f=pd.read_parquet(ROOT/'RAPID/validation/all_pairs_lag_20261005/features.parquet')
    z=s.merge(f[['example_id','fingerprint','interpolated_peak_shift_hours']],left_on=['example','fingerprint'],right_on=['example_id','fingerprint'],how='inner')
    z['paired_timing_change_difference_h']=z.new_peak_change_vs_parent_hours-z.interpolated_peak_shift_hours
    z.to_csv(OUT/'cropped_event_response_sensitivity.csv',index=False)
    text=f'''# Event buffers: tests and conditional runtime projection

Reference **EVENT-BUFFERS-2026-10-05**. Requested after selecting shorter observed event windows and proposing a short steady spin-up plus settling-based spin-down. Production forcing, routing outputs and analysis metrics are unchanged. These are isolated tests at the existing hydraulic parameters, segmentation and native routing clocks.

## Tested scope

- Audited saved outlet settlement for **2,230 distinct routed states** (16 parents plus 2,214 children).
- Tested the newly cropped forcing on **39 states across all 16 examples**: every parent, largest cumulative travel-time candidate, latest-settling child under old forcing, and the three example-37 shape/peak cases where not already selected. `selection.csv` records the choices.
- All states use declared initial equilibrium (`steady_first_inflow`). Compare no initial buffer with 6 and 24 hours at the new event's first discharge. Buffers are rounded up to whole forcing intervals, keeping event samples and phase fixed.
- Full waveform comparison, including the long diagnostic tail, gives maximum difference **{full.max_wave_error_cms.max():.6g} m³/s**, or **{full.max_wave_error_fraction.max():.6g} of waveform range**. Maximum interpolated peak-time difference: **{full.peak_time_difference_seconds.abs().max():.6g} seconds**. This supports a 24-hour plateau under equilibrium initialization; it does not validate short warm-up from zero flow.
- The production engine bypasses exactly constant initial forcing while initialized at equilibrium. Thus retaining the old long initial plateau with the same NEW event/initial discharge is equivalent by that declared shortcut. The old event itself can have different antecedent conditions and is a separate sensitivity question.

## Reproduction and numerical checks

The efficient spatial recurrence calls the same linear Muskingum coefficients/held-forcing recurrence as the production sparse engine. Match the six-decimal internal bifurcation allocations and float32 connectivity representation used in preparation; keep inlet allocation unchanged. The optional serialized-allocation switch is in an investigation helper only, defaults off for earlier clock investigations, and changes no production code.

Each selected state is checked against 32 new-event forcing intervals evaluated with production sparse operations: maximum outlet difference **{s.sparse_parity_max_cms.max():.6g} m³/s**. A separate first-128-active-interval check against saved production output has maximum difference **{s.saved_parity_max_cms.max():.6g} m³/s** and passes a 1-ppm-of-forcing-range tolerance. The spin-up comparison then covers complete newly routed waveforms, not just these parity probes. No flow clipping or timestep changes are used.

## Final settling rule

Hold the input at the new event's final discharge. Use a tolerance of **0.1% of the new parent peak excess above max(initial, final discharge)**. From the last outlet exceedance after forcing-event end, require **24 continuous hours** within tolerance, followed by **another 24 hours**. The diagnostic runs continue substantially longer to check that this is not a temporary crossing. Use the latest required end among the selected parent/children, rounded up to a forcing interval.

All tested peaks remain within the proposed shortened output; maximum subsequent omitted-tail deviation is **{s.omitted_tail_max_parent_amplitude.max():.6g} of parent amplitude**, below the 0.001 threshold. The settlement helper was checked against a temporary plateau followed by a later excursion and a censored tail.

**{(extra>1e-5).sum()} examples** have a tested child settling later than the parent; maximum additional time is **{extra.max()*24:.2f} hours**. Therefore do not use the parent cutoff without a child check. These tests certify the 39 chosen states, not all newly forced children. A future full rerun must verify every state at the common cutoff and extend that example's shared tail if needed. All old saved states settled, but that alone does not certify the changed forcing.

## Event-cropping sensitivity

Compared with the existing saved responses, the maximum absolute change in the available selected parent–child interpolated timing differences is **{z.paired_timing_change_difference_h.abs().max()*3600:.6g} seconds** ({len(z)} child comparisons). This is encouraging, but is not a complete shape, volume or all-state invariance claim. Early/late excluded features and event-duration definitions can change. The source event windows are recorded separately, with July 5 confirmed for example 41.

## Estimated lengths and routing work

Proposed total = 24-hour initial plateau (rounded up) + new observed event + latest tested settlement tail including both 24-hour periods. Dates/end inclusivity are inherited from the user-reviewed proposals. Tiny duration differences reflect forcing-grid rounding.

'''+table(d[['example','total_states','old_total_days','event_days','new_common_tail_days','proposed_total_days','duration_reduction_pct','dynamic_work_reduction_pct']].round(2))+f'''

Weighted by the number of distinct states, total simulated state-days fall by **{duration*100:.1f}%**. Weighting active forcing steps by routing substeps and state segment counts gives **{work*100:.1f}% less routing work**, approximately **{1/(1-work):.2f}× fewer segment-substeps**. Full-network output-value count falls by **{output*100:.1f}%**; this is not a measured compressed-file-size reduction.

The routing-work projection excludes the already skipped initial plateau. It assumes the tested common tails are sufficient for the remaining states. It is not a full-batch wall-clock measurement or an ETA: preparation, equivalent resistance, sparse-solver sparsity, serialization, compression and I/O do not scale identically. Parent sparse seconds-per-interval are exported as local diagnostics, not used to promise an absolute completion time. No equivalent-resistance recalculation is needed merely to shorten forcing at unchanged geometry/hydraulics.

## Files and next implementation

Run `event_buffer_tests.py`, `full_event_spinup_check.py`, then `report_event_buffer_tests.py` under `RAPID/investigations`. Saved parquet trajectories are isolated test results. Exported selections, all-state old-settlement audit, new-event tests, full-waveform spin-up differences and per-example runtime projections accompany this report.

Recommended next implementation: retain equilibrium initialization; use a 24-hour initial plateau; rebuild forcing from observed event samples before padding once; use the tested tail as an initial plan and verify/extend for every child. Keep the existing long runs as references until the complete rerun's metrics are checked. This report documents the tests; it does not switch active production to them.
'''
    (OUT/'README.md').write_text(text)
    sources=['event_buffer_tests.py','full_event_spinup_check.py','check_fractional_routing_clocks.py']
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='EVENT-BUFFERS-2026-10-05',tested_states=len(s),saved_audited_states=len(old),production_changed=False,spinup_hours=[0,6,24],settling_tolerance_fraction=.001,settling_dwell_hours=24,extra_buffer_hours=24,script_sha256={n:hashlib.sha256((ROOT/'RAPID/investigations'/n).read_bytes()).hexdigest() for n in sources}),indent=2))
    print('duration reduction',duration,'work reduction',work,'output reduction',output)
    print('full spin max error',full.max_wave_error_cms.max(),'paired timing max seconds',z.paired_timing_change_difference_h.abs().max()*3600)


if __name__=='__main__':main()

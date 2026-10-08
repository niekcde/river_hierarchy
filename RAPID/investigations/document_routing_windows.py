"""Resolve absolute UTC timestamps for the tested event/buffer schedule."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory,event_clock
OUT=ROOT/'RAPID/validation/event_buffer_tests_20261005'


def main():
    windows=pd.read_csv(ROOT/'RAPID/reviews/proposed_event_windows_20261005.csv',dtype={'example':str}).set_index('example')
    projection=pd.read_csv(OUT/'duration_runtime_projection.csv',dtype={'example':str}).set_index('example')
    rows=[]
    for ex,w in windows.iterrows():
        d=projection.loc[ex];h=pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv');clock,_=event_clock(ex,h)
        start,end=pd.to_datetime(w.proposed_start_utc),pd.to_datetime(w.proposed_end_exclusive_utc)
        idx=np.flatnonzero(clock.between(start,end,inclusive='left').to_numpy())
        first,last=clock.iloc[idx[0]],clock.iloc[idx[-1]]
        dt=float(np.diff(h.time_seconds)[0]);nspin=int(np.ceil(86400/dt))
        simstart=first-pd.to_timedelta(nspin*dt,unit='s')
        total_steps=round(float(d.proposed_total_days)*86400/dt)
        simend=simstart+pd.to_timedelta(total_steps*dt,unit='s')
        settling=last+pd.to_timedelta(round((float(d.new_common_tail_days)-2)*86400,6),unit='s')
        confirmed=settling+pd.Timedelta(days=1)
        assert simend>=confirmed+pd.Timedelta(days=1)-pd.Timedelta(microseconds=10)
        assert abs((simend-simstart).total_seconds()/86400-d.proposed_total_days)<1e-8
        rows.append(dict(example=ex,requested_event_start_utc=start,requested_event_end_exclusive_utc=end,
                         first_retained_event_sample_utc=first,last_retained_event_sample_utc=last,
                         initial_plateau_discharge_cms=float(h.q_inflow_cms.iloc[idx[0]]),
                         final_plateau_discharge_cms=float(h.q_inflow_cms.iloc[idx[-1]]),
                         tested_simulation_start_utc=simstart,tested_settling_start_utc=settling,
                         tested_24h_settlement_confirmed_utc=confirmed,tested_simulation_end_utc=simend,
                         current_simulation_start_utc=clock.iloc[0],current_simulation_end_utc=clock.iloc[-1],
                         current_days=d.old_total_days,tested_days=d.proposed_total_days,
                         duration_reduction_fraction=d.duration_reduction_pct/100,
                         dynamic_work_reduction_fraction=d.dynamic_work_reduction_pct/100,
                         tested_states=int(d.tested_states),total_states=int(d.total_states),forcing_step_seconds=dt,
                         status='Tested proposal; not active production'))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'routing_schedule_utc.csv',index=False)
    data=json.loads(df.to_json(orient='records',date_format='iso'))
    (OUT/'routing_schedule_utc.json').write_text(json.dumps(data,indent=2))
    def table(frame):
        return '\n'.join('| '+' | '.join(row)+' |' for row in [list(frame.columns),['---']*len(frame.columns),*frame.astype(str).values.tolist()])
    text='''# Routing stages and exact UTC schedule

Reference **ROUTING-SCHEDULE-2026-10-05**. Recorded at the user's request after the spin-up and settling tests. These are the exact timestamps of the tested proposal, reconstructed on the existing forcing sample grid. They have **not** replaced production forcing. The current production start/end are shown separately.

## Where to find the values in Excel

Workbook: `RAPID/reviews/hydrograph_match_review.xlsx`, worksheet **Routing windows**. The root-level workbook with the same basename is an older empty workbook and is not the updated review file.

| Information | Cells |
| --- | --- |
| Example IDs in each table | A7:A22, A28:A43, A49:A64 |
| User-requested event start / exclusive end | B7:C22 |
| First / last retained event samples on the existing grid | D7:E22 |
| Initial / final plateau discharge (m³/s) | F7:G22 |
| Tested simulation start (initial plateau begins) | B28:B43 |
| Initial plateau ends / final plateau begins | C28:D43 |
| Outlet remains within the settling tolerance from | E28:E43 |
| 24-hour sustained settlement confirmed | F28:F43 |
| Tested simulation end, including extra 24-hour buffer | G28:G43 |
| Current production simulation start / end | B49:C64 |
| Current / tested duration in days and reduction | D49:F64 |
| Number of tested / total routed states | G49:G64 |
| Protocol, status and source explanation | A68:G82 |

Dates are UTC, displayed to the second. “End exclusive” means the new named end dates include the whole calendar day, ending at midnight the next day. Boundaries marked current keep their existing timestamp. The first/last retained samples can differ slightly from those requested bounds because the tested event preserves the existing forcing grid. Do not confuse the last retained event sample with the exclusive selection boundary.

## Routing stages

1. Initialize every link at equilibrium with the first retained event discharge (`steady_first_inflow`).
2. Hold that discharge for at least 24 hours, rounded up to complete forcing intervals. This is the initial plateau, from simulation start to the first event sample.
3. Apply the retained observed event samples at the unchanged forcing cadence and routing timestep.
4. Hold the last retained event discharge constant. Allow the parent and tested children to reach their final plateau. The tolerance is 0.1% of parent peak excess above max(initial, final discharge). Use the latest final exceedance among tested states, not a temporary early crossing.
5. Require 24 continuous hours within tolerance, then another 24-hour buffer. Round the final simulation endpoint up to a forcing sample. Parent and children use one shared endpoint.
6. When implementing the full batch, verify every child's final settlement and extend that example's shared tail if necessary. The schedule currently covers selected-state tests, not a certification of every newly forced child.

The source tests audited all 2,230 existing routed states and rerouted 39 selected states. Full 0/6/24-hour spin-up comparisons agree within numerical precision. Estimated batch reductions are 83% in simulated state-days and 77% in segment-routing substeps, not a measured whole-pipeline speedup. See [test report](../../validation/event_buffer_tests_20261005/README.md).

## Exact tested simulation boundaries

'''+table(df[['example','tested_simulation_start_utc','first_retained_event_sample_utc','last_retained_event_sample_utc','tested_simulation_end_utc']])+'''

## Reproducibility

Machine-readable full schedule: `RAPID/validation/event_buffer_tests_20261005/routing_schedule_utc.csv`. Requested windows: `RAPID/reviews/proposed_event_windows_20261005.csv`. Durations/settlement: the test report's `duration_runtime_projection.csv` and `selected_state_tests.csv`. Reconstruct with `RAPID/investigations/document_routing_windows.py`.

This report and the new worksheet preserve previous review values and notes. Changing workbook dates alone does not regenerate routing or recertify the tested tail. No active input files or model results were changed by documenting this schedule.
'''
    report_dir=ROOT/'RAPID/reviews/routing_windows'
    report_dir.mkdir(exist_ok=True)
    (report_dir/'README.md').write_text(text)
    print(df[['example','tested_simulation_start_utc','tested_simulation_end_utc']].to_string(index=False))


if __name__=='__main__':main()

"""Prepare isolated uncapped hierarchy runs; --run routes examples sequentially."""
import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from audit import ROOT, EXT, uncapped_grid, sha, save
from rapid_tools import cunge, event_window_batch as batch
from rapid_tools.common_grid_production import shared_clock

EXAMPLES = ['13', '15', '21', '22', '46', '48', '54']
OUT = ROOT / 'RAPID/outputs/uncapped_event_window_20261020'
CAL = ROOT / 'RAPID/outputs/uncapped_joint_calibration_20261020'


def prepare():
    plan_path = OUT / 'routing_plan.json'
    if plan_path.exists():
        doc = json.loads(plan_path.read_text())
        batch.verify(doc['source_sha256'])
        batch.verify(doc['code_sha256'])
        for case in doc['cases']:
            batch.verify(case['source_sha256'])
        return plan_path
    sources = {}

    def read(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    completion = read(CAL / 'completion.json')
    if not completion['complete'] or not completion['sources_unchanged']:
        raise ValueError('Joint calibration is not complete and verified.')
    original = read(ROOT / 'RAPID/outputs/event_window_plan_20261013/plan.json')
    extension = read(EXT / 'routing_plan.json')
    cases, rows = [], []
    for ex in EXAMPLES:
        case = copy.deepcopy(next(c for c in original['cases'] if c['example'] == ex))
        ec = next(c for c in extension['cases'] if c['example'] == ex)
        case['jobs'].append(copy.deepcopy(next(
            j for j in ec['jobs'] if j['state_id'] == 'all_closed_units_collapsed')))
        result = read(CAL / ex / 'result.json')
        receipts = [
            read(ROOT / f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json'),
            read(EXT / f'runs/{ex}/completion.json'),
        ]
        case['tail_hours'] = max(r['final_tail_hours'] for r in receipts)
        case['simulation_end_utc'] = (
            pd.Timestamp(case['window']['end_utc'])
            + pd.Timedelta(seconds=30, hours=case['tail_hours'])).isoformat()
        tracked = {case['forcing']: sha(case['forcing'])}
        for job in case['jobs']:
            source = job['parameters']
            tracked[source] = sha(source)
            f = pd.read_parquet(source)
            g = cunge.parameters(uncapped_grid(f), result['n_manning'], beta=result['beta'])
            if not (g.rapid_celerity_mps_raw < g.celerity_max_mps).all():
                raise ValueError(f'Upper cap still active: {ex}/{job["state_id"]}')
            # Retain or refine the old native clock, with the updated stability bound.
            dt = shared_clock([g], [30, 60], job['native_dt_seconds'])
            batch.equilibrium_check(g, dt)
            path = OUT / 'parameters' / ex / job['state_id'] / 'parameters.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            g.to_parquet(path, index=False)
            tracked[str(path)] = sha(path)
            job.update(parameters=str(path), native_dt_seconds=dt)
            rows.append(dict(example=ex, state=job['state_id'], links=len(g),
                             native_dt_seconds=dt, n=result['n_manning'], beta=result['beta']))
        case['source_sha256'] = tracked
        batch.forcing_with_buffers(case)
        cases.append(case)
        print(ex, len(case['jobs']), 'states prepared and stability checked', flush=True)
    batch.verify(sources)
    for case in cases:
        batch.verify(case['source_sha256'])
    code = batch.code_sources()
    code.update({str(p): sha(p) for p in [Path(__file__).resolve(),
                 Path(__file__).with_name('audit.py').resolve()]})
    save(plan_path, dict(
        schema='reviewed-event-window-batch-v2',
        status='isolated_uncapped_joint_calibration_rerun',
        policy='Original hierarchy plus segmented full collapse; no pilot pairs. '
               'Shared example calibration, unchanged geometry/forcing, lower celerity cap retained. '
               'Native clocks retain/refine original values; common one-minute reporting. '
               'Shared settling tails extend automatically if necessary.',
        source_sha256=sources, code_sha256=code, cases=cases))
    pd.DataFrame(rows).to_csv(OUT / 'prepared_states.csv', index=False)
    return plan_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    path = prepare()
    print('Verified plan:', path, flush=True)
    if args.run:
        batch.run(path, OUT / 'runs', EXAMPLES)
    else:
        print('Prepared only; use --run to route sequentially.')


if __name__ == '__main__':
    main()

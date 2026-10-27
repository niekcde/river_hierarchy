"""Compare completed uncapped reruns on windows shared with capped references."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_rerun import OUT, EXT, ROOT, batch, sha, save

sys.path.insert(0, str(ROOT / 'RAPID/investigations/closed_unit_extension_20261019'))
from responses import calculate
from analyze_pilot import response_window
from rapid_tools.hydrograph_response import response_rows

METRICS = ['centroid_shift_hours', 'peak_shift_hours', 'peak_excess_change_pct',
           'duration80_change_pct', 'volume_change_pct', 'asymmetry_change']


def main():
    planpath = OUT / 'routing_plan.json'
    plan = json.loads(planpath.read_text())
    batch.verify(plan['source_sha256'])
    batch.verify(plan['code_sha256'])
    oldpath = EXT / 'state_metrics.parquet'
    old = pd.read_parquet(oldpath)
    destination = OUT / 'review'
    destination.mkdir(exist_ok=True)
    states = {'capped': [], 'uncapped': []}
    windows, audits, sources = [], [], {str(oldpath): sha(oldpath), str(planpath): sha(planpath)}
    for case in plan['cases']:
        ex = case['example']
        receipt = OUT / 'runs' / ex / 'completion.json'
        done = json.loads(receipt.read_text())
        assert done['complete'] and done['states'] == len(case['jobs'])
        assert done['plan_sha256'] == sha(planpath)
        batch.verify(done['files'])
        batch.verify(case['source_sha256'])
        sources.update(done['files'])
        sources[str(receipt)] = sha(receipt)
        folder = Path(done['final_directory'])
        wave = pd.read_parquet(folder / 'forcing.parquet')
        t = wave.time_utc.astype('int64').to_numpy() / 1e9
        ii = np.flatnonzero(wave.phase.eq('input_event'))
        curves = {}
        reference = old[old.example.eq(ex)]
        base = min(float(reference.baseline_cms.min()), case['initial_cms'], case['final_cms'])
        for job in case['jobs']:
            state = 'fully_collapsed' if job['state_id'] == 'single_centerline' else job['state_id']
            path = folder / job['state_id'] / 'outlet.parquet'
            f = pd.read_parquet(path)
            assert f.time_utc.equals(wave.time_utc)
            q = f.q_outlet_cms.to_numpy()
            assert np.isfinite(q).all()
            base = min(base, float(q[ii[0]:].min()))
            curves[state] = (q, path, job['aliases'])
        assert set(curves) <= set(reference.state), 'Unmatched old state'
        for window in ['0.5%', '1%', '2%']:
            refs = reference[reference.window.eq(window) & reference.state.isin(curves)]
            assert len(refs) == len(curves) and not refs.state.duplicated().any()
            start, end = float(refs.rise_seconds.min()), float(refs.fall_seconds.max())
            tol = float(refs.tolerance_cms.iloc[0])
            for q, _, _ in curves.values():
                a, b = response_window(q, ii[0], ii[-1], case['initial_cms'], case['final_cms'], tol)
                start, end = min(start, t[a]), max(end, t[b])
            windows.append(dict(example=ex, window=window, baseline_cms=base,
                                start_seconds=start, end_seconds=end,
                                earlier_start_minutes=(refs.rise_seconds.min()-start)/60,
                                later_end_minutes=(end-refs.fall_seconds.max())/60))
            for row in refs.to_dict('records'):
                states['capped'].append(calculate(row, start, end, base, window))
                _, path, aliases = curves[row['state']]
                record = dict(example=ex, state=row['state'], aliases=json.dumps(aliases),
                              outlet=str(path), outlet_sha256=done['files'][str(path)],
                              tolerance_cms=tol)
                states['uncapped'].append(calculate(record, start, end, base, window))
        audits.append(dict(example=ex, states=len(curves), unmatched=0,
                           final_tail_hours=done['final_tail_hours'],
                           final_96h_error_cms=done['max_final_96h_error_cms'],
                           tolerance_cms=case['tolerance_cms']))
        print(ex, 'verified and compared', flush=True)
    tables = {}
    for version, records in states.items():
        st = pd.DataFrame(records)
        st.to_parquet(destination / f'{version}_state_metrics.parquet', index=False)
        tables[version] = response_rows(st)
        tables[version].to_parquet(destination / f'{version}_responses.parquet', index=False)
    comparison = tables['capped'].merge(tables['uncapped'], on=['example', 'state', 'window'],
                                      suffixes=('_old', '_new'), validate='one_to_one')
    for metric in METRICS:
        comparison[metric+'_difference'] = comparison[metric+'_new']-comparison[metric+'_old']
    comparison.to_parquet(destination / 'comparison.parquet', index=False)
    primary = comparison[comparison.window.eq('0.5%')]
    primary[primary.state.isin(['fully_collapsed', 'all_closed_units_collapsed'])].to_csv(
        destination / 'full_collapse_comparison.csv', index=False)
    primary.reindex(primary.centroid_shift_hours_difference.abs().sort_values(ascending=False).index).head(30).to_csv(
        destination / 'largest_changes.csv', index=False)
    pd.DataFrame(audits).to_csv(destination / 'completion_audit.csv', index=False)
    pd.DataFrame(windows).to_csv(destination / 'window_audit.csv', index=False)
    save(destination / 'manifest.json', dict(complete=True, sources=sources,
         code_sha256={str(Path(__file__).resolve()): sha(__file__)},
         convention='Shared capped/uncapped union windows and minimum baseline per example; '
         '0.5% primary, 1%/2% sensitivity; global maxima; child minus parent. '
         'All outputs isolated. No predictor retraining. Example 54 crest switching remains possible.',
         states=sum(a['states'] for a in audits), unmatched=0))
    print(primary[primary.state.eq('all_closed_units_collapsed')][
        ['example', 'centroid_shift_hours_old', 'centroid_shift_hours_new',
         'peak_shift_hours_old', 'peak_shift_hours_new']].to_string(index=False))


if __name__ == '__main__':
    main()

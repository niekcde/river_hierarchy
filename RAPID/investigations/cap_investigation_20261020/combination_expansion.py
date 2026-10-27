"""Geometry-selected 80 -> 240 closed-chain expansion, isolated from production.

Reuse the existing 80 routed states and construct/route only the additional 160.
The selection uses no observed targets or model errors.
"""
import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from chain_augmentation import (
    OUT, EXT, pilot, pair, COLUMNS, batch, shared_clock, sha, save,
    calculate, response_window, response_rows, METRICS,
)
from flow_resistance import solve_unit_flow

DEST = OUT / 'combination_expansion_20261026'
OLD = OUT / 'chain80_tuning'


def select(total=240):
    inventory = pilot.read(OUT/'physical_context_and_combinations/combination_inventory.csv')
    inventory = inventory[~inventory.duplicates_segmented].copy()
    physical = pilot.read(OUT/'physical_context_and_combinations/segmented_physical_context.csv').set_index('example')
    inventory['actual_fraction'] = [r.edited_parent_km/physical.loc[r.example, 'edited_parent_km'] for r in inventory.itertuples()]
    previous = pilot.read(OLD/'selection80.csv')
    chosen = previous.to_dict('records')
    used = {(r['example'], r['states']) for r in chosen}
    assert len(used) == 80 and total <= len(inventory)
    while len(chosen) < total:
        eligible = []
        for ex, g in inventory.groupby('example'):
            g = g[[(ex, s) not in used for s in g.states]]
            if len(g):
                eligible.append((sum(r['example'] == ex for r in chosen), ex, g))
        _, ex, g = min(eligible, key=lambda item: item[:2])
        prior = [r for r in chosen if r['example'] == ex]
        def novelty(row):
            members = set(row.states.split(' | '))
            return min(abs(row.actual_fraction-r['actual_fraction']) + .25*(1-len(members & set(r['states'].split(' | ')))/len(members | set(r['states'].split(' | ')))) for r in prior)
        g = g.copy()
        g['novelty'] = g.apply(novelty, axis=1)
        row = g.sort_values(['novelty', 'states'], ascending=[False, True]).iloc[0].to_dict()
        row['selection_reason'] = 'Balanced network count, then maximum minimum length-fraction / constituent-set distance; no targets.'
        chosen.append(row)
        used.add((ex, row['states']))
    result = pd.DataFrame(chosen)
    old_keys = {(r['example'], tuple(r['components'])): r['state'] for r in pilot.read(OLD/'variants.json')}
    result['previous80'] = [(r.example, tuple(r.states.split(' | '))) in old_keys for r in result.itertuples()]
    result['state'] = [old_keys.get((r.example, tuple(r.states.split(' | '))), 'expanded_'+hashlib.sha256(r.states.encode()).hexdigest()[:12]) for r in result.itertuples()]
    assert len(result) == total and result.previous80.sum() == 80
    assert not result.duplicated(['example', 'states']).any()
    assert not result.duplicated(['example', 'state']).any()
    return result


def resistance(frame):
    roots = set(frame.id_us_node)-set(frame.id_ds_node)
    sinks = set(frame.id_ds_node)-set(frame.id_us_node)
    assert len(roots) == len(sinks) == 1
    edges = pd.DataFrame(dict(u=frame.id_us_node, v=frame.id_ds_node,
                              K_e=frame.link_length_m/(frame.width**2*frame.depth_m**3)))
    result = solve_unit_flow(edges, next(iter(roots)), next(iter(sinks)), tolerance=1e-8, max_iterations=250)
    assert result['success'] and np.isfinite(result['K_eq']) and result['K_eq'] > 0
    return {k: float(result[k]) for k in ['K_eq', 'relative_gap', 'continuity']}


def depth_features(parent, child, lineage, kp):
    kc = resistance(child)
    values = []
    for side, frame in [('parent', parent), ('child', child)]:
        ids = set(lineage.loc[lineage.role.eq('collapsed_block'), side+'_link_id'])
        selected = frame[frame.id_link.isin(ids)]
        assert len(selected) and ids <= set(frame.id_link)
        w = selected.reference_flow_fraction*selected.link_length_m
        values.append(float((w/(selected.depth_m*np.sqrt(selected.slope))).sum()))
    return dict(resistance_explicit_depth=-np.log10(kc['K_eq']/kp['K_eq']),
                delta_G_inverse_depth=values[1]-values[0]), kc


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST/'routing_plan.json'
    if path.exists():
        manifest = json.loads((DEST/'preparation.json').read_text())
        assert manifest['script_sha256'] == sha(__file__)
        batch.verify(manifest['sources'])
        batch.verify(manifest['artifacts'])
        return path
    selection = select()
    selection.to_csv(DEST/'selection240.csv', index=False)
    blocks = pilot.read(pilot.OUT/'eligible_blocks.json')
    cases = {c['example']: c for plan in [pilot.read(EXT/'routing_plan.json'), pilot.read(OUT/'routing_plan.json')] for c in plan['cases']}
    old_plan = pilot.read(OLD/'routing_plan.json')
    old_jobs = {(c['example'], j['state_id']): j for c in old_plan['cases'] for j in c['jobs']}
    old_features = pilot.read(OLD/'combined_input.parquet').set_index(['example', 'state'])
    newcases, features, variants, solver = [], [], [], []
    for ex, group in selection.groupby('example'):
        case = copy.deepcopy(cases[ex])
        jobs = {j['state_id']: j for j in case['jobs']}
        parent = pilot.read(jobs['S000_base']['parameters'])
        segmented = pilot.read(jobs['all_closed_units_collapsed']['parameters'])
        kp = resistance(parent)
        lookup = {r['state']: dict(r, child_source=r['source'], child_sink=r['sink']) for r in blocks[ex]}
        sections = {name: segmented[segmented.block_state.eq(name)].copy() for name in lookup}
        assert all(len(f) == 1 for f in sections.values())
        case['jobs'] = [copy.deepcopy(jobs['S000_base'])]
        done = pilot.read(OLD/f'runs/{ex}/completion.json')
        case['tail_hours'] = done['final_tail_hours']
        case['simulation_end_utc'] = (pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30, hours=case['tail_hours'])).isoformat()
        tracked = {jobs['S000_base']['parameters']: sha(jobs['S000_base']['parameters']), case['forcing']: sha(case['forcing'])}
        for r in group.itertuples():
            names = r.states.split(' | ')
            if r.previous80:
                child = pilot.read(old_jobs[ex, r.state]['parameters'])
                lineage = pilot.read(OLD/f'parameters/{ex}/{r.state}/lineage.csv')
                m = old_features.loc[(ex, r.state)].to_dict()
            else:
                bs = [lookup[n] for n in names]
                child, lineage = pilot.compose(parent, bs, sections)
                reverse, _ = pilot.compose(parent, list(reversed(bs)), sections)
                pd.testing.assert_frame_equal(child, reverse)
                ids = {i for block in bs for i in block['parent_ids']}
                cids = set(lineage.loc[lineage.role.eq('collapsed_block'), 'child_link_id'])
                m = pair(parent, child, ids, cids)
                part = parent[parent.id_link.isin(ids)]
                m['parent_slope_slow'] = float(np.average(1/np.sqrt(part.slope), weights=part.reference_flow_fraction*part.link_length_m))
                m['length_delta_km'] = (m['B_child_route_length_mean']-m['B_parent_route_length_mean'])/1000
                np.testing.assert_allclose(m['domain_size'], r.edited_parent_km, rtol=1e-10)
                folder = DEST/f'parameters/{ex}/{r.state}'
                folder.mkdir(parents=True, exist_ok=True)
                target = folder/'parameters.parquet'
                child.to_parquet(target, index=False)
                lineage.to_csv(folder/'lineage.csv', index=False)
                dt = shared_clock([child], [30, 60], min(jobs['S000_base']['native_dt_seconds'], jobs['all_closed_units_collapsed']['native_dt_seconds']))
                batch.equilibrium_check(child, dt)
                case['jobs'].append(dict(state_id=r.state, parameters=str(target), native_dt_seconds=dt, aliases=[r.state]))
                tracked[str(target)] = sha(target)
            extra, kc = depth_features(parent, child, lineage, kp)
            m.update(extra, example=ex, state=r.state, cohort='combined', previous80=bool(r.previous80))
            assert np.isfinite([m[c] for c in COLUMNS+list(extra)]).all()
            features.append(m)
            solver.append(dict(example=ex, state=r.state, parent=kp, child=kc))
            variants.append(dict(example=ex, state=r.state, components=names, previous80=bool(r.previous80)))
        if len(case['jobs']) > 1:
            case['source_sha256'] = tracked
            newcases.append(case)
        print(ex, 'prepared', len(group), 'combined states', flush=True)
    pd.DataFrame(features).to_parquet(DEST/'predictors.parquet', index=False)
    save(DEST/'variants.json', variants)
    save(DEST/'resistance_audit.json', solver)
    save(path, dict(schema='reviewed-event-window-batch-v2', source_sha256=pilot.SOURCES,
                    code_sha256={str(Path(pilot.__file__)): sha(pilot.__file__), **batch.code_sources()}, cases=newcases))
    artifacts = {str(p): sha(p) for p in [path, DEST/'selection240.csv', DEST/'predictors.parquet', DEST/'variants.json', DEST/'resistance_audit.json']}
    save(DEST/'preparation.json', dict(script_sha256=sha(__file__), sources=pilot.SOURCES, artifacts=artifacts,
        total=240, reused=80, newly_constructed=160, production_modified=False,
        resistance_scope='Whole connected parent/child including unchanged connectors, matching full-segmentation convention.',
        inverse_depth_scope='Edited footprint only, excluding unchanged connectors.'))
    return path


def route(path, workers):
    doc = json.loads(path.read_text())
    output = DEST/'runs'
    output.mkdir(exist_ok=True)
    receipt = output/'batch.json'
    if not receipt.exists():
        save(receipt, dict(plan=str(path), plan_sha256=sha(path), selection=[c['example'] for c in doc['cases']]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(batch.run, path, output, [c['example']]) for c in doc['cases']]
        for f in as_completed(futures):
            f.result()


def responses():
    """Reconcile shared response windows across originals, previous80 and new160."""
    old = pd.read_parquet(OLD/'state_metrics.parquet')
    plan = json.loads((DEST/'routing_plan.json').read_text())
    new_cases = {c['example']: c for c in plan['cases']}
    rows, windows, reproduction = [], [], []
    for ex, refs in old.groupby('example'):
        if ex not in new_cases:
            rows.extend(refs.to_dict('records'))
            continue
        case = new_cases[ex]
        done = json.loads((DEST/f'runs/{ex}/completion.json').read_text())
        batch.verify(done['files'])
        folder = Path(done['final_directory'])
        wave = pd.read_parquet(folder/'forcing.parquet')
        t = wave.time_utc.astype('int64').to_numpy()/1e9
        ii = np.flatnonzero(wave.phase.eq('input_event'))
        curves = {j['state_id']: pd.read_parquet(folder/j['state_id']/'outlet.parquet') for j in case['jobs']}
        base = min(float(refs.baseline_cms.min()), min(float(f.q_outlet_cms.iloc[ii[0]:].min()) for f in curves.values()))
        reference = refs[(refs.state == 'S000_base') & (refs.window == '0.5%')].iloc[0]
        original = pd.read_parquet(reference.outlet)
        compare = original.merge(curves['S000_base'], on='time_utc', suffixes=('_old', '_new'), validate='one_to_one')
        assert len(compare) == len(original)
        np.testing.assert_allclose(compare.q_outlet_cms_old, compare.q_outlet_cms_new, rtol=1e-12, atol=1e-8)
        reproduction.append(dict(example=ex, max_difference_cms=float(abs(compare.q_outlet_cms_old-compare.q_outlet_cms_new).max())))
        for window in ['0.5%', '1%', '2%']:
            subset = refs[refs.window.eq(window)]
            ref = subset.iloc[0]
            start, end, tol = float(ref.rise_seconds), float(ref.fall_seconds), float(ref.tolerance_cms)
            for f in curves.values():
                assert f.time_utc.equals(wave.time_utc)
                a, z = response_window(f.q_outlet_cms.to_numpy(), ii[0], ii[-1], case['initial_cms'], case['final_cms'], tol)
                start, end = min(start, t[a]), max(end, t[z])
            changed = start != ref.rise_seconds or end != ref.fall_seconds or base != ref.baseline_cms
            windows.append(dict(example=ex, window=window, changed=changed, earlier_start_seconds=ref.rise_seconds-start,
                                later_end_seconds=end-ref.fall_seconds, baseline_change_cms=base-ref.baseline_cms))
            if changed:
                rows.extend(calculate(record, start, end, base, window) for record in subset.to_dict('records'))
            else:
                rows.extend(subset.to_dict('records'))
            for state in curves:
                if state != 'S000_base':
                    outlet = folder/state/'outlet.parquet'
                    rows.append(calculate(dict(example=ex, state=state, outlet=str(outlet), outlet_sha256=sha(outlet), tolerance_cms=tol), start, end, base, window))
    states = pd.DataFrame(rows)
    assert not states.duplicated(['example', 'state', 'window']).any()
    result = response_rows(states)
    primary = result[result.window.eq('0.5%')][['example', 'state', *METRICS]]
    new = pd.read_parquet(DEST/'predictors.parquet').drop(columns=METRICS, errors='ignore').merge(primary, on=['example', 'state'], how='left', validate='one_to_one')
    old_input = pd.read_parquet(OUT/'alternative_validation_20261023/input.parquet')
    existing = old_input.drop(columns=METRICS, errors='ignore').merge(primary, on=['example', 'state'], how='left', validate='one_to_one')
    assert len(new) == 240 and len(existing) == 2185
    assert np.isfinite(new[COLUMNS+METRICS]).all().all() and np.isfinite(existing[COLUMNS+METRICS]).all().all()
    # Refit every treatment, including control, on the harmonized targets.
    # Previously training-selected settings remain frozen across treatments.
    changes = existing[['example', 'state', 'centroid_shift_hours']].merge(
        old_input[['example', 'state', 'centroid_shift_hours']], on=['example', 'state'],
        validate='one_to_one', suffixes=('_new', '_old'))
    changes['change_seconds'] = 3600*(changes.centroid_shift_hours_new-changes.centroid_shift_hours_old)
    prior80 = pd.read_parquet(OLD/'combined_input.parquet')
    compare = new[new.previous80].merge(prior80, on=['example', 'state'], validate='one_to_one', suffixes=('_new', '_old'))
    compare['change_seconds'] = 3600*(compare.centroid_shift_hours_new-compare.centroid_shift_hours_old)
    states.to_parquet(DEST/'state_metrics.parquet', index=False)
    result.to_parquet(DEST/'responses.parquet', index=False)
    new.to_parquet(DEST/'combined_input.parquet', index=False)
    existing.to_parquet(DEST/'evaluation_input.parquet', index=False)
    pd.DataFrame(windows).to_csv(DEST/'window_audit.csv', index=False)
    pd.DataFrame(reproduction).to_csv(DEST/'parent_reproduction.csv', index=False)
    changes.to_csv(DEST/'existing_target_changes.csv', index=False)
    compare[['example', 'state', 'centroid_shift_hours_new', 'centroid_shift_hours_old', 'change_seconds']].to_csv(DEST/'previous80_target_changes.csv', index=False)
    save(DEST/'response_check.json', dict(complete=True, combined=240, existing=2185,
         unchanged_primary_targets=bool(changes.change_seconds.eq(0).all() and compare.change_seconds.eq(0).all()),
         changed_existing_targets=int(changes.change_seconds.ne(0).sum()),
         max_existing_target_change_seconds=float(changes.change_seconds.abs().max()),
         changed_previous80_targets=int(compare.change_seconds.ne(0).sum()),
         max_previous80_target_change_seconds=float(compare.change_seconds.abs().max()),
         changed_windows=int(pd.DataFrame(windows).changed.sum()), production_modified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    plan = prepare()
    if args.run:
        route(plan, args.workers)
        responses()

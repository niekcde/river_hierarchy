"""Missing-data, support, forest representation, and physical diagnostic audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from edited_size_check import b, OUT, EXT, save, sha

DEST = OUT/'remaining_error_audit'
PARTIAL = ['CB001_bubble_9', 'CB002_bubble_11', 'island_22_018',
           'S288_unit_17', 'island_22_219']


def main():
    DEST.mkdir(exist_ok=True)
    source = OUT/'edited_size_check/uncapped_edited_input.parquet'
    d = pd.read_parquet(source)
    params = json.loads((b.OUT/'final_model.json').read_text())['params']
    columns = list(b.COLUMNS)
    audit = []
    for c in columns:
        audit.append(dict(feature=c, missing=int(d[c].isna().sum()),
                          nonfinite=int((~np.isfinite(d[c])).sum()), zeros=int(d[c].eq(0).sum()),
                          unique=int(d[c].nunique()), minimum=float(d[c].min()), maximum=float(d[c].max())))
    pd.DataFrame(audit).to_csv(DEST/'input_audit.csv', index=False)
    assert np.isfinite(d[columns].to_numpy()).all()
    support, neighbors, results, leaves = [], [], [], []
    for ex in ['22', '41', '46', '55']:
        train = d[d.example.ne(ex)].copy()
        test = d[d.example.eq(ex) & (d.cohort.eq('segmented') | d.state.isin(PARTIAL))].copy()
        for variant in ['fixed', 'raw_ratios', 'no_pruning', 'fine_leaves', 'length_interaction']:
            tr, te = train.copy(), test.copy()
            settings = dict(params)
            b.COLUMNS = list(columns)
            if variant == 'raw_ratios':
                for c in columns:
                    if c.startswith('log1p__'):
                        tr[c], te[c] = np.expm1(tr[c]), np.expm1(te[c])
            if variant in ['no_pruning', 'fine_leaves']:
                settings['ccp_alpha'] = 0.
            if variant == 'fine_leaves':
                settings.update(min_samples_leaf=1, min_samples_split=2)
            if variant == 'length_interaction':
                # Mixed scopes for partial cases: a diagnostic feature, not asserted exact delta length.
                for f in [tr, te]:
                    f['size_times_length_fraction_change'] = f.domain_size*(np.expm1(f.log1p__B_ratio_route_length_mean)-1)
                b.COLUMNS.append('size_times_length_fraction_change')
            model = b.fit(tr, settings, jobs=4)
            pred = b.forest.predict(model, te)
            for r, value in zip(te.itertuples(), pred):
                results.append(dict(example=ex, state=r.state, variant=variant, actual=r.centroid_shift_hours,
                                    predicted=float(value), abs_error_hours=abs(value-r.centroid_shift_hours)))
            if variant != 'fixed':
                continue
            X = model['preprocessor'].transform(tr)
            Z = model['preprocessor'].transform(te)
            for i, row in enumerate(te.itertuples()):
                for c in columns:
                    support.append(dict(example=ex, state=row.state, feature=c, value=float(te.iloc[i][c]),
                         train_min=float(tr[c].min()), train_max=float(tr[c].max()),
                         outside_range=bool(te.iloc[i][c]<tr[c].min() or te.iloc[i][c]>tr[c].max()),
                         train_percentile=float((tr[c]<=te.iloc[i][c]).mean()),
                         train_target_min=float(tr.centroid_shift_hours.min()),
                         train_target_max=float(tr.centroid_shift_hours.max()),
                         actual=float(row.centroid_shift_hours)))
                near = tr.copy()
                near['distance'] = np.sqrt(((X-Z[i])**2).sum(axis=1))
                for cohort in ['all', 'segmented']:
                    pool = near if cohort == 'all' else near[near.cohort.eq('segmented')]
                    for r in pool.nsmallest(5, 'distance').itertuples():
                        neighbors.append(dict(held_example=ex, held_state=row.state, pool=cohort,
                            neighbor_example=r.example, neighbor_state=r.state, neighbor_cohort=r.cohort,
                            distance=r.distance, target=r.centroid_shift_hours, size=r.domain_size))
                encounter = np.zeros(len(tr))
                treepred, leafcounts = [], []
                for tree in model['estimator'].estimators_:
                    node = tree.apply(Z[i:i+1])[0]
                    mask = tree.apply(X) == node
                    encounter += mask
                    leafcounts.append(int(mask.sum()))
                    treepred.append(float(tree.predict(Z[i:i+1])[0]*model['target_sd']+model['target_mean']))
                for j in np.argsort(-encounter)[:10]:
                    r = tr.iloc[j]
                    leaves.append(dict(held_example=ex, held_state=row.state, neighbor_example=r.example,
                        neighbor_state=r.state, cohort=r.cohort, target=r.centroid_shift_hours,
                        encounter_fraction=float(encounter[j]/len(model['estimator'].estimators_)),
                        tree_prediction_min=min(treepred), tree_prediction_max=max(treepred),
                        tree_prediction_median=float(np.median(treepred)), median_leaf_rows=float(np.median(leafcounts))))
            print(ex, variant, 'done', flush=True)
        print(ex, 'five diagnostic variants done', flush=True)
    b.COLUMNS = columns
    for name, rows in [('support', support), ('neighbors', neighbors), ('diagnostic_models', results), ('leaf_neighbors', leaves)]:
        pd.DataFrame(rows).to_csv(DEST/f'{name}.csv', index=False)
    physics(d)
    save(DEST/'manifest.json', dict(complete=True, source_sha256={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)},
        notes=['20 targeted diagnostic fits, not a new model selection or full validation.',
               'Each examined network entirely held out. All original and segmented training rows retained.',
               'Length interaction combines edited size and outer ratios for partial cases; not an exact edited length delta.',
               'Routing times/celerities/depth are diagnostic model outputs, never predictive inputs.',
               'Co-leaf counts describe all training rows, not bootstrap-aware causal contributions.',
               'No production results or models modified.']))


def physics(d):
    root = OUT.parents[2]
    plans = [json.loads((EXT/'routing_plan.json').read_text()), json.loads((OUT/'routing_plan.json').read_text())]
    cases = {c['example']:c for plan in plans for c in plan['cases']}
    scopes = root/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
    meta = pd.read_parquet(scopes/'geometry_responses.parquet')
    graphs = pd.read_csv(scopes/'domain_graphs.csv', dtype={'example':str})
    rows = []
    for ex, case in cases.items():
        jobs = {j['state_id']:j for j in case['jobs']}
        parent = pd.read_parquet(jobs['S000_base']['parameters'])
        states = ['all_closed_units_collapsed'] + (PARTIAL if ex=='22' else [])
        for state in states:
            child = pd.read_parquet(jobs[state]['parameters'])
            if state == 'all_closed_units_collapsed':
                lineage = pd.read_csv(EXT/f'variants/{ex}/{state}/lineage.csv')
                line = lineage[lineage.role.eq('collapsed_block')]
                ids = [line.parent_link_id, line.child_link_id]
            else:
                m = meta[meta.example.eq(ex)&meta.state.eq(state)].iloc[0]
                g = graphs[graphs.example.eq(ex)&graphs.state.eq(state)&graphs.scope.eq(m.B_scope)].iloc[0]
                ids = [pd.read_parquet(scopes/'graphs'/f'{g[k]}.parquet').id_link
                       for k in ['parent_graph','child_graph']]
            entry = dict(example=ex, state=state, scope='edited' if state=='all_closed_units_collapsed' else 'outer')
            for label, full, selected in [('parent',parent,ids[0]), ('child',child,ids[1])]:
                f = full[full.id_link.isin(selected)]
                L = f.link_length_m.to_numpy()
                rL = L*f.reference_flow_fraction.to_numpy()
                entry.update({label+'_length_km':float(rL.sum()/1000),
                    label+'_width_m':float(np.average(f.width,weights=L)),
                    label+'_slope_mean':float(np.average(f.slope,weights=rL)),
                    label+'_slope_slow_proxy':float(np.average(1/np.sqrt(f.slope),weights=rL)),
                    label+'_depth_m':float(np.average(f.depth_m,weights=rL)),
                    label+'_time_h':float((rL/f.rapid_celerity_mps).sum()/3600),
                    label+'_effective_celerity':float(rL.sum()/(rL/f.rapid_celerity_mps).sum()),
                    label+'_full_time_h':float((full.reference_flow_fraction*full.link_length_m/full.rapid_celerity_mps).sum()/3600),
                    label+'_n':float(f.n_manning.iloc[0])})
            entry['full_route_time_change_h'] = entry['child_full_time_h']-entry['parent_full_time_h']
            row = d[d.example.eq(ex)&d.state.eq(state)].iloc[0]
            entry['centroid_shift_h'] = row.centroid_shift_hours
            entry['length_ratio'] = float(np.expm1(row.log1p__B_ratio_route_length_mean))
            rows.append(entry)
    pd.DataFrame(rows).to_csv(DEST/'physical_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()

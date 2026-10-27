"""Controlled edited-size substitution and held-out network 22 tree diagnostics."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from validate_forest import b, OUT, EXT, save, sha

PREVIOUS = OUT/'forest_validation'
DEST = OUT/'edited_size_check'


def main():
    DEST.mkdir(exist_ok=True)
    sizepath = EXT/'size_comparison/input.parquet'
    sizes = pd.read_parquet(sizepath)[['example', 'state', 'domain_size', 'domain_size_edited']]
    params = json.loads((b.OUT/'final_model.json').read_text())['params']
    files = [sizepath, PREVIOUS/'capped_input.parquet', PREVIOUS/'uncapped_input.parquet',
             PREVIOUS/'predictions.parquet', b.OUT/'final_model.json', Path(__file__),
             Path(b.__file__), Path(b.engine.__file__), Path(b.forest.__file__)]
    sources = {str(p.resolve()): sha(p) for p in files}
    design = dict(sources=sources, params=params, columns=b.COLUMNS, target='centroid_shift_hours',
                  validation='Fixed settings, leave all rows of one network out; 15 networks. No pairs.',
                  edited_size='sum(r_e * L_e) over selected parent unit membership / 1 km, '
                  'with r_e calculated in the original outer domain. Segmented sizes already exclude connectors and are unchanged.',
                  controls='All six other predictors, target definitions, weights, and seeds held fixed.',
                  diagnostic='Forest leaf co-occurrence and path frequencies for held-out 22; descriptive model behavior, not causal importance.',
                  scope='Research comparison, no production model changes.')
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    save(DEST/'design.json', design)
    oldpred = pd.read_parquet(PREVIOUS/'predictions.parquet')
    oldpred['size'] = 'current'
    rows = oldpred.to_dict('records')
    data = {}
    for version in ['capped', 'uncapped']:
        d = pd.read_parquet(PREVIOUS/f'{version}_input.parquet')
        d = d.merge(sizes, on=['example', 'state'], how='left', validate='one_to_one',
                    suffixes=('', '_reference'))
        np.testing.assert_allclose(d.domain_size, d.domain_size_reference, rtol=1e-12, atol=1e-12)
        assert d.domain_size_edited.notna().all()
        assert (d.domain_size_edited > 0).all()
        np.testing.assert_allclose(d.loc[d.cohort.eq('segmented'), 'domain_size'],
                                   d.loc[d.cohort.eq('segmented'), 'domain_size_edited'])
        data[version] = d.copy()
        d['domain_size'] = d.domain_size_edited
        d.to_parquet(DEST/f'{version}_edited_input.parquet', index=False)
        for variant in ['original_only', 'plus_segmented']:
            for ex in sorted(d.example.unique()):
                train, test = d[d.example.ne(ex)], d[d.example.eq(ex)]
                if variant == 'original_only':
                    train = train[train.cohort.eq('original')]
                assert train.example.nunique() == 14 and ex not in set(train.example)
                path = DEST/'folds'/version/variant/f'{ex}.json'
                record = b.forest.checked(path, fp)
                if record is None:
                    model = b.fit(train, params, jobs=4)
                    prediction = b.forest.predict(model, test)
                    record = dict(fingerprint=fp, rows=[
                        dict(example=r.example, state=r.state, cohort=r.cohort,
                             centroid_shift_hours=r.centroid_shift_hours, peak_shift_hours=r.peak_shift_hours,
                             predicted_hours=float(v)) for r, v in zip(test.itertuples(), prediction)])
                    save(path, record)
                rows.extend(dict(version=version, variant=variant, size='edited', **r) for r in record['rows'])
                print(version, variant, ex, 'edited complete', flush=True)
    predictions = pd.DataFrame(rows)
    assert len(predictions) == 17480 and not predictions.duplicated(['version', 'variant', 'size', 'example', 'state']).any()
    predictions.to_parquet(DEST/'predictions.parquet', index=False)
    scores = []
    for (version, variant, size, cohort), g in predictions.groupby(['version', 'variant', 'size', 'cohort']):
        for target in ['centroid_shift_hours', 'peak_shift_hours']:
            for subset, z in [('all', g), ('only22', g[g.example.eq('22')]),
                              ('excluding22', g[g.example.ne('22')]), ('excluding54', g[g.example.ne('54')])]:
                y, p, w = z[target].to_numpy(), z.predicted_hours.to_numpy(), b.engine.weights(z)
                scores.append(dict(version=version, variant=variant, size=size, cohort=cohort,
                    target=target, subset=subset, n=len(z),
                    relative_error_pct=b.forest.sweep.relative_score(y, p, w, 1),
                    **b.engine.score(y*60, p*60, w)))
    pd.DataFrame(scores).to_csv(DEST/'scores.csv', index=False)
    diagnose(data['uncapped'], predictions, params)
    for path, digest in sources.items():
        assert sha(path) == digest
    save(DEST/'completion.json', dict(complete=True, fingerprint=fp, validation_fits=60,
         diagnostic_refits=2, unmatched=0, production_modified=False))
    print(pd.DataFrame(scores).query("version=='uncapped' and target=='centroid_shift_hours' and subset=='all'").to_string(index=False))


def diagnose(d, predictions, params):
    selected = predictions[predictions.version.eq('uncapped') & predictions.example.eq('22')]
    wide = selected.pivot(index=['example', 'state', 'cohort', 'centroid_shift_hours'],
                          columns=['size', 'variant'], values='predicted_hours').reset_index()
    wide.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in wide.columns]
    for size in ['current', 'edited']:
        wide[size+'_original_abs_error_h'] = abs(wide[size+'_original_only']-wide.centroid_shift_hours)
        wide[size+'_augmented_abs_error_h'] = abs(wide[size+'_plus_segmented']-wide.centroid_shift_hours)
        wide[size+'_augmentation_penalty_h'] = wide[size+'_augmented_abs_error_h']-wide[size+'_original_abs_error_h']
    features = d[['example', 'state', *b.COLUMNS, 'domain_size_edited']].copy()
    for column in b.COLUMNS:
        if column.startswith('log1p__'):
            features[column.removeprefix('log1p__')] = np.expm1(features[column])
    wide = wide.merge(features, on=['example', 'state'], validate='one_to_one')
    wide = wide.sort_values('current_augmentation_penalty_h', ascending=False)
    wide.to_csv(DEST/'network22_errors_and_predictors.csv', index=False)
    segments = features.merge(d[['example', 'state', 'cohort', 'centroid_shift_hours']],
                              on=['example', 'state'], validate='one_to_one')
    segments[segments.cohort.eq('segmented')].to_csv(DEST/'segmented_predictors.csv', index=False)
    leafrows, pathrows = [], []
    for size in ['current', 'edited']:
        frame = d.copy()
        if size == 'edited':
            frame['domain_size'] = frame.domain_size_edited
        train = frame[frame.example.ne('22')]
        test = frame[frame.example.eq('22') & frame.cohort.eq('original')]
        segmented = train[train.cohort.eq('segmented')]
        model = b.fit(train, params, jobs=4)
        expected = selected[selected['size'].eq(size) & selected.variant.eq('plus_segmented')]
        expected = test[['example', 'state']].merge(expected, on=['example', 'state'], validate='one_to_one')
        np.testing.assert_allclose(b.forest.predict(model, test), expected.predicted_hours, atol=1e-12, rtol=1e-12)
        X = model['preprocessor'].transform(test)
        S = model['preprocessor'].transform(segmented)
        trees = model['estimator'].estimators_
        leaves = np.column_stack([tree.apply(X) for tree in trees])
        segleaves = np.column_stack([tree.apply(S) for tree in trees])
        kept = model['preprocessor'].kept
        topstates = set(wide[wide.cohort.eq('original')].head(10).state)
        for i, row in enumerate(test.itertuples()):
            matches = leaves[i][None, :] == segleaves
            best = int(matches.mean(axis=1).argmax())
            leafrows.append(dict(size=size, state=row.state,
                fraction_trees_sharing_any_segmented_leaf=float(matches.any(axis=0).mean()),
                strongest_segmented_example=segmented.iloc[best].example,
                strongest_segmented_leaf_fraction=float(matches[best].mean())))
            if row.state not in topstates:
                continue
            counts = {c: 0 for c in kept}
            thresholds = {c: [] for c in kept}
            for tree in trees:
                nodes = tree.decision_path(X[i:i+1]).indices
                seen = set()
                for node in nodes:
                    feature = tree.tree_.feature[node]
                    if feature >= 0:
                        name = kept[feature]
                        seen.add(name)
                        thresholds[name].append(float(tree.tree_.threshold[node]))
                for name in seen:
                    counts[name] += 1
            for name in kept:
                pathrows.append(dict(size=size, state=row.state, feature=name,
                    fraction_trees_path_uses_feature=counts[name]/len(trees)))
    pd.DataFrame(leafrows).to_csv(DEST/'network22_segmented_leaf_overlap.csv', index=False)
    pd.DataFrame(pathrows).to_csv(DEST/'network22_path_feature_usage.csv', index=False)


if __name__ == '__main__':
    main()

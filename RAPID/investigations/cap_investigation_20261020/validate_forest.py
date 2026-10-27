"""Paired capped/uncapped fixed-forest validation; no new hyperparameter selection."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_rerun import ROOT, OUT, EXT, save, sha
sys.path.insert(0, str(ROOT/'RAPID/investigations/closed_unit_extension_20261019'))
import augmentation as aug
b = aug.baseline
METRICS = ['centroid_shift_hours', 'peak_shift_hours', 'peak_excess_change_pct',
           'duration80_change_pct', 'volume_change_pct', 'asymmetry_change']
DEST = OUT/'forest_validation'


def main():
    DEST.mkdir(exist_ok=True)
    paths = [EXT/'augmentation/input.parquet', EXT/'responses.parquet',
             OUT/'review/capped_responses.parquet', OUT/'review/uncapped_responses.parquet',
             b.OUT/'final_model.json', Path(__file__), Path(b.__file__),
             Path(b.forest.__file__), Path(b.engine.__file__)]
    sources = {str(p.resolve()): sha(p) for p in paths}
    params = json.loads((b.OUT/'final_model.json').read_text())['params']
    base = pd.read_parquet(paths[0])
    base = base[base.cohort.isin(['original', 'segmented'])].copy()
    old = pd.read_parquet(paths[1])
    cohorts = {k: int(v) for k, v in base.cohort.value_counts().items()}
    assert cohorts == {'original': 2170, 'segmented': 15}
    datasets, join_audit = {}, []
    for version in ['capped', 'uncapped']:
        replacement = pd.read_parquet(OUT/f'review/{version}_responses.parquet')
        affected = set(replacement.example)
        responses = pd.concat([old[~old.example.isin(affected)], replacement], ignore_index=True)
        # Save all response columns and both full-collapse representations for later review.
        responses.to_parquet(DEST/f'{version}_responses_all_windows.parquet', index=False)
        primary = responses[responses.window.eq('0.5%')]
        d = base.drop(columns=METRICS, errors='ignore').merge(
            primary[['example', 'state', *METRICS]], on=['example', 'state'],
            how='left', validate='one_to_one', indicator=True)
        missing = d[d._merge.ne('both')][['example', 'state']]
        assert missing.empty
        d = d.drop(columns='_merge')
        assert len(d) == 2185 and d.example.nunique() == 15
        assert np.isfinite(d[b.COLUMNS+METRICS].to_numpy(float)).all()
        d.to_parquet(DEST/f'{version}_input.parquet', index=False)
        datasets[version] = d
        join_audit.append(dict(version=version, matched=len(d), unmatched=len(missing),
                               revised_rows=int(d.example.isin(affected).sum())))
    pd.testing.assert_frame_equal(datasets['capped'][b.COLUMNS], datasets['uncapped'][b.COLUMNS])
    design = dict(sources=sources, params=params, columns=b.COLUMNS, seed=b.engine.SEED,
                  target='centroid_shift_hours', networks=15, cohorts=cohorts,
                  validation='Leave one entire network out. Fixed settings, no tuning. '
                  'Train original-only or original plus segmented; evaluate both cohorts separately.',
                  weighting='Existing equal-total-network training/evaluation weights; training-only preprocessing and target standardization.',
                  comparison='Capped controls refitted on identical union windows/baselines. '
                  'Seven changed networks replaced; eight unchanged networks retained. '
                  'Pairs and legacy full collapses excluded from model training/evaluation.',
                  caveat='Exploratory reused networks, not an independent final test. Peak scores use centroid predictions, not a peak-trained model.')
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    save(DEST/'design.json', design)
    pd.DataFrame(join_audit).to_csv(DEST/'join_audit.csv', index=False)
    rows, folds = [], []
    for version, d in datasets.items():
        for variant in ['original_only', 'plus_segmented']:
            for ex in sorted(d.example.unique()):
                train = d[d.example.ne(ex)]
                if variant == 'original_only':
                    train = train[train.cohort.eq('original')]
                test = d[d.example.eq(ex)]
                assert set(train.example).isdisjoint(test.example)
                path = DEST/'folds'/version/variant/f'{ex}.json'
                saved = b.forest.checked(path, fp)
                if saved is None:
                    model = b.fit(train, params, jobs=4)
                    pred = b.forest.predict(model, test)
                    saved = dict(fingerprint=fp, training_rows=len(train), training_networks=int(train.example.nunique()),
                                 rows=[dict(example=r.example, state=r.state, cohort=r.cohort,
                                            centroid_shift_hours=r.centroid_shift_hours, peak_shift_hours=r.peak_shift_hours,
                                            predicted_hours=float(v)) for r, v in zip(test.itertuples(), pred)])
                    save(path, saved)
                rows.extend(dict(version=version, variant=variant, **r) for r in saved['rows'])
                folds.append(dict(version=version, variant=variant, held_example=ex,
                                  training_rows=saved['training_rows'], training_networks=saved['training_networks']))
                print(version, variant, ex, 'complete', flush=True)
    p = pd.DataFrame(rows)
    p.to_parquet(DEST/'predictions.parquet', index=False)
    pd.DataFrame(folds).to_csv(DEST/'fold_audit.csv', index=False)
    scores, networks = [], []
    for (version, variant, cohort), g in p.groupby(['version', 'variant', 'cohort']):
        for target in ['centroid_shift_hours', 'peak_shift_hours']:
            for label, z in [('all', g), ('excluding22', g[g.example.ne('22')]),
                             ('excluding54', g[g.example.ne('54')]),
                             ('excluding_extreme37', g[~(g.example.eq('37') & g.state.eq('CB001_bubble_3'))])]:
                y, pred = z[target].to_numpy(), z.predicted_hours.to_numpy()
                w = b.engine.weights(z)
                outside = abs(y)*60 > 1
                scores.append(dict(version=version, variant=variant, cohort=cohort, target=target,
                    subset=label, n=len(z), relative_error_pct=b.forest.sweep.relative_score(y, pred, w, 1),
                    mape_outside_1min_pct=float(np.average(abs((pred[outside]-y[outside])/y[outside]), weights=w[outside])*100),
                    **b.engine.score(y*60, pred*60, w)))
            for ex, z in g.groupby('example'):
                networks.append(dict(version=version, variant=variant, cohort=cohort, target=target, example=ex,
                                     mae_minutes=float(abs(z.predicted_hours-z[target]).mean()*60)))
    s = pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv', index=False)
    pd.DataFrame(networks).to_csv(DEST/'network_scores.csv', index=False)
    for path, digest in sources.items():
        assert sha(path) == digest
    save(DEST/'completion.json', dict(complete=True, fits=60, fingerprints=fp, unmatched=0,
                                    production_modified=False))
    print(s.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False))


if __name__ == '__main__':
    main()

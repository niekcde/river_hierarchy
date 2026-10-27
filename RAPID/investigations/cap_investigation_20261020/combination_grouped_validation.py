"""Matched compound-group validation: baseline, previous80, expanded240.

Original 435 family/reach groups and 15 full-collapse singleton folds are frozen.
Each combination is a separate singleton, including overlapping combinations.
Reuse training-only nested selections from the unaugmented outer training fold
to isolate adding cases; do not retune or use held-out errors for selection.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from alternative_validation import b, SEEDS, save, sha
from combination_expansion import DEST as EXPANSION, OUT
from ten_input_outlier_audit import score
from validation_group_inventory import response_bin

DEST = OUT/'combination_grouped_validation_20261026'
PRIOR = OUT/'alternative_validation_20261023'
VARIANTS = ['baseline', 'plus80', 'plus240']
KEYS = ['example', 'state']


def singleton_folds(data, seed, k=5):
    """Balanced count/response/network margins without tying cases by network.

    A separate call for the initial80 and extra160 makes the initial80 split
    invariant to the expanded treatment. Each combined entity has one row.
    """
    assert not data.duplicated(KEYS).any()
    rng = np.random.default_rng(seed)
    z = data.reset_index(drop=True)
    strata = z.response_bin.replace({'delay_over60min': 'delay_5to60min'})
    counts = np.zeros(k, int)
    by_response = {s: np.zeros(k, int) for s in strata.unique()}
    by_network = {s: np.zeros(k, int) for s in z.example.unique()}
    result = np.full(len(z), -1, int)
    for i in rng.permutation(len(z)):
        eligible = np.flatnonzero(counts == counts.min())
        costs = by_response[strata.iloc[i]][eligible] + .25*by_network[z.example.iloc[i]][eligible]
        best = eligible[np.isclose(costs, costs.min())]
        fold = int(rng.choice(best))
        result[i] = fold
        counts[fold] += 1
        by_response[strata.iloc[i]][fold] += 1
        by_network[z.example.iloc[i]][fold] += 1
    assert counts.max()-counts.min() <= 1 and (result >= 0).all()
    return result


def training_mask(data, folds, held, variant):
    allowed = data.cohort.ne('combined')
    if variant == 'plus80':
        allowed = allowed | data.previous80
    elif variant == 'plus240':
        allowed = np.ones(len(data), bool)
    elif variant != 'baseline':
        raise ValueError(variant)
    mask = np.asarray(allowed) & (folds != held)
    assert set(data.loc[mask, 'group_id']).isdisjoint(data.loc[folds == held, 'group_id'])
    return mask


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    assert json.loads((EXPANSION/'response_check.json').read_text())['complete']
    old = pd.read_parquet(EXPANSION/'evaluation_input.parquet')
    new = pd.read_parquet(EXPANSION/'combined_input.parquet')
    assert len(old) == 2185 and len(new) == 240 and new.previous80.sum() == 80
    old['previous80'] = False
    new['group_id'] = new.example+'_combination_'+new.state
    new['edit_type'] = 'Combined closed units'
    new['response_bin'] = new.centroid_shift_hours.map(response_bin)
    new['full_collapse_overlap_exception'] = True
    data = pd.concat([old, new], ignore_index=True)
    data['singleton_overlap_exception'] = data.cohort.isin(['segmented', 'combined'])
    assert not data.duplicated(KEYS).any() and data.group_id.nunique() == 690
    assert data.groupby('group_id').example.nunique().eq(1).all()
    sizes = data.groupby('group_id').size()
    assert sizes.loc[data[data.singleton_overlap_exception].group_id].eq(1).all()
    panels = json.loads((PRIOR/'design.json').read_text())['panels']
    assert np.isfinite(data[list(set(sum(panels.values(), [])))+['centroid_shift_hours']]).all().all()
    selections = [r for r in json.loads((PRIOR/'selections.json').read_text()) if r['strategy'] == 'compound_mixed']
    assignment = pd.read_csv(PRIOR/'outer_assignments.csv', dtype={'example': str})
    assignment = assignment[assignment.strategy.eq('compound_mixed')]
    paths = [Path(__file__), Path(b.__file__), Path(b.engine.__file__), Path(b.forest.__file__),
             Path(__file__).with_name('combination_expansion.py'),
             EXPANSION/'preparation.json', EXPANSION/'response_check.json', EXPANSION/'selection240.csv',
             EXPANSION/'existing_target_changes.csv', EXPANSION/'previous80_target_changes.csv',
             EXPANSION/'segmented_descriptor_reproduction.csv',
             EXPANSION/'evaluation_input.parquet', EXPANSION/'combined_input.parquet',
             PRIOR/'selections.json', PRIOR/'design.json', PRIOR/'predictions.parquet', PRIOR/'outer_assignments.csv']
    design = dict(sources={str(p): sha(p) for p in paths}, panels=panels, seeds=SEEDS,
        variants=VARIANTS, rows=len(data), groups=690, regular_groups=435, segmented_singletons=15,
        combined_singletons=240, fits=150,
        validation='Five repetitions x five compound-group outer folds. Frozen original folds; singleton combination folds balanced by counts, response and network margins. No grouping by example.',
        settings='Each panel uses its existing six-candidate nested selection from that outer training fold only. Identical settings across baseline/plus80/plus240. Augmented models are not retuned.',
        targets='All treatments including control are freshly fit on harmonized expanded-set response windows. Previous settings were selected on the earlier training targets; no retuning. Target changes are recorded separately.',
        weighting='Unchanged equal total weight per network and equal weight per case within network. Grouping ignores network identity; weighting retains the earlier scheme to isolate augmentation.',
        reporting='Same 2425 test cases in all three treatments; original, segmented, previous80, additional160, all combinations scored separately. Score each repetition first; report mean/min/max. Original+segmented equal-cohort score is the primary comparison.',
        overlap='User-requested singleton exception for full collapses and every combination. Constituent units, overlapping combinations and other states of the same network may be in training. Known-network related-state prediction, not unseen-network transfer.',
        production_modified=False)
    fp = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():
        assert json.loads((DEST/'design.json').read_text()) == design, 'Use a new output directory for changed design.'
    save(DEST/'design.json', design)
    tasks, records, audit = [], [], []
    variants = json.loads((EXPANSION/'variants.json').read_text())
    constituents = {(r['example'], r['state']): set(r['components']) for r in variants}
    for repeat, seed in enumerate(SEEDS):
        a = assignment[assignment['repeat'].eq(repeat)]
        existing = old[KEYS].merge(a[KEYS+['group_id', 'fold']], on=KEYS, validate='one_to_one')
        assert len(existing) == len(old)
        np.testing.assert_array_equal(existing.group_id, old.group_id)
        folds = np.full(len(data), -1, int)
        folds[:len(old)] = existing.fold
        for previous, offset in [(True, 700000), (False, 800000)]:
            mask = data.cohort.eq('combined') & data.previous80.eq(previous)
            folds[mask] = singleton_folds(data[mask], seed+offset)
        assert (folds >= 0).all()
        assert data.assign(fold=folds).groupby('group_id').fold.nunique().eq(1).all()
        records.extend(data[KEYS+['group_id', 'cohort', 'previous80', 'response_bin']].assign(repeat=repeat, fold=folds).to_dict('records'))
        for held in range(5):
            test = data[folds == held]
            assert len(test[test.cohort.eq('segmented')]) == 3
            assert len(test[test.cohort.eq('combined')]) == 48
            for variant in VARIANTS:
                train = data[training_mask(data, folds, held, variant)]
                train_keys = set(map(tuple, train[KEYS].to_numpy()))
                overlaps = [any((r.example, c) in train_keys for c in constituents[r.example, r.state])
                            for r in test[test.cohort.eq('combined')].itertuples()]
                audit.append(dict(repeat=repeat, fold=held, variant=variant, train_rows=len(train), test_rows=len(test),
                                  train_groups=train.group_id.nunique(), test_groups=test.group_id.nunique(),
                                  group_overlap=0, combined_test_with_component_in_training=sum(overlaps),
                                  same_network_test_rows=int(test.example.isin(train.example).sum())))
                for model, cols in panels.items():
                    choice = next(r for r in selections if r['repeat'] == repeat and r['fold'] == held and r['model'] == model)
                    tasks.append(dict(repeat=repeat, fold=held, variant=variant, model=model, columns=cols,
                                      params=choice['params'], selected_candidate=choice['name'],
                                      train=np.flatnonzero(training_mask(data, folds, held, variant)).tolist(),
                                      test=np.flatnonzero(folds == held).tolist()))
    assert len(tasks) == 150
    data.to_parquet(DEST/'input.parquet', index=False)
    pd.DataFrame(records).to_csv(DEST/'outer_assignments.csv', index=False)
    pd.DataFrame(audit).to_csv(DEST/'fold_audit.csv', index=False)
    save(DEST/'tasks.json', dict(fingerprint=fp, tasks=tasks))
    return design, fp, tasks


def fit_task(task, fp):
    key = f"{task['variant']}/r{task['repeat']}/f{task['fold']}/{task['model']}"
    path = DEST/'checkpoints'/key/'result.json'
    if path.exists():
        assert json.loads(path.read_text())['fingerprint'] == fp
        return key
    d = pd.read_parquet(DEST/'input.parquet')
    tr, te = d.iloc[task['train']], d.iloc[task['test']]
    assert set(tr.group_id).isdisjoint(te.group_id)
    b.COLUMNS = list(task['columns'])
    artifact = b.fit(tr, task['params'], jobs=1)
    predicted = b.forest.predict(artifact, te)
    assert np.isfinite(predicted).all()
    reproduced = False
    if task['variant'] == 'baseline':
        original = pd.read_parquet(PRIOR/'input.parquet')
        target_check = tr[KEYS+['centroid_shift_hours']].merge(original[KEYS+['centroid_shift_hours']], on=KEYS, validate='one_to_one', suffixes=('_new', '_old'))
        unchanged_training = np.array_equal(target_check.centroid_shift_hours_new, target_check.centroid_shift_hours_old)
        prior = pd.read_parquet(PRIOR/'predictions.parquet')
        prior = prior[(prior.strategy == 'compound_mixed') & (prior['repeat'] == task['repeat']) &
                      (prior.fold == task['fold']) & (prior.model == task['model'])]
        check = te[KEYS].assign(predicted=predicted).merge(prior, on=KEYS, validate='one_to_one')
        assert len(check) == len(prior)
        if unchanged_training:
            np.testing.assert_allclose(check.predicted, check.predicted_hours, rtol=1e-12, atol=1e-12)
            reproduced = True
    rows = te[KEYS+['cohort', 'previous80', 'group_id', 'centroid_shift_hours']].copy()
    rows['predicted_hours'] = predicted
    for field in ['variant', 'repeat', 'fold', 'model']:
        rows[field] = task[field]
    save(path, dict(fingerprint=fp, selected_candidate=task['selected_candidate'],
                   baseline_reproduced_if_training_targets_unchanged=reproduced, rows=rows.to_dict('records')))
    return key


def summarize(design, fp, tasks):
    rows = []
    reproduced = 0
    for t in tasks:
        path = DEST/f"checkpoints/{t['variant']}/r{t['repeat']}/f{t['fold']}/{t['model']}/result.json"
        result = json.loads(path.read_text())
        assert result['fingerprint'] == fp
        reproduced += result['baseline_reproduced_if_training_targets_unchanged']
        rows.extend(result['rows'])
    p = pd.DataFrame(rows)
    assert len(p) == 2425*5*2*3
    assert not p.duplicated(['variant', 'repeat', 'model', *KEYS]).any()
    p.to_parquet(DEST/'predictions.parquet', index=False)
    scores = []
    for (variant, repeat, model), g in p.groupby(['variant', 'repeat', 'model']):
        subsets = {
            'original': g.cohort.eq('original'), 'segmented': g.cohort.eq('segmented'),
            'previous80': g.cohort.eq('combined') & g.previous80,
            'additional160': g.cohort.eq('combined') & ~g.previous80,
            'combined240': g.cohort.eq('combined'), 'primary_existing': g.cohort.ne('combined'),
        }
        for cohort, mask in subsets.items():
            z = g[mask]
            scores.append(dict(variant=variant, repeat=repeat, model=model, cohort=cohort,
                               **score(z, 'equal_cohort_network' if cohort == 'primary_existing' else 'equal_network')))
    s = pd.DataFrame(scores)
    s.to_csv(DEST/'repeat_scores.csv', index=False)
    metrics = ['relative_error_pct', 'mae_minutes', 'r2']
    summary = s.groupby(['model', 'cohort', 'variant'])[metrics].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(DEST/'summary.csv', index=False)
    baseline = s[s.variant.eq('baseline')]
    paired = s[~s.variant.eq('baseline')].merge(baseline, on=['repeat', 'model', 'cohort'], suffixes=('', '_baseline'), validate='many_to_one')
    for metric in metrics:
        paired[metric+'_change'] = paired[metric]-paired[metric+'_baseline']
    paired.to_csv(DEST/'paired_changes.csv', index=False)
    for path, digest in design['sources'].items():
        assert sha(path) == digest, path
    save(DEST/'completion.json', dict(complete=True, fingerprint=fp, fits=len(tasks), test_rows_per_treatment=2425,
         repeated_predictions=len(p), regular_groups=435, segmented_singletons=15, combination_singletons=240,
         baseline_freshly_refit=True, unchanged_training_baselines_reproduced=reproduced, production_modified=False))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    design, fp, tasks = prepare()
    print('Prepared 150 fits, 690 groups, baseline / plus80 / plus240.', flush=True)
    if args.run:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_task, t, fp) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                print(i, '/150', f.result(), flush=True)
        summarize(design, fp, tasks)

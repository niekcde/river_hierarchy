"""Reproduce depth descriptors and audit the expanded geometry population."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combination_expansion as expansion
import flow_resistance
import edited_features
import responses as response_calculation
import analyze_pilot
from combination_expansion import DEST, OUT, EXT, depth_features, resistance, sha, save


def main():
    base = pd.read_parquet(OUT/'alternative_validation_20261023/input.parquet')
    plans = [EXT/'routing_plan.json', OUT/'routing_plan.json']
    cases = {c['example']: c for path in plans for c in json.loads(path.read_text())['cases']}
    rows = []
    for ex, case in cases.items():
        jobs = {j['state_id']: j for j in case['jobs']}
        parent = pd.read_parquet(jobs['S000_base']['parameters'])
        child = pd.read_parquet(jobs['all_closed_units_collapsed']['parameters'])
        lineage = pd.read_csv(EXT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv')
        features, _ = depth_features(parent, child, lineage, resistance(parent))
        ref = base[base.example.eq(ex) & base.cohort.eq('segmented')].iloc[0]
        for key, value in features.items():
            np.testing.assert_allclose(value, ref[key], rtol=1e-6, atol=1e-6)
            rows.append(dict(example=ex, feature=key, value=value, reference=ref[key], difference=value-ref[key]))
    path = DEST/'segmented_descriptor_reproduction.csv'
    reproduction = pd.DataFrame(rows)
    if path.exists():
        pd.testing.assert_frame_equal(reproduction, pd.read_csv(path, dtype={'example': str}), atol=1e-10, rtol=1e-10)
    else:
        reproduction.to_csv(path, index=False)

    selection = pd.read_csv(DEST/'selection240.csv', dtype={'example': str})
    original = pd.read_csv(expansion.OLD/'selection80.csv', dtype={'example': str})
    keys = lambda frame: {(r.example, frozenset(r.states.split(' | '))) for r in frame.itertuples()}
    assert len(keys(selection)) == 240 and keys(original) <= keys(selection)
    assert not selection.duplicates_segmented.any() and selection.previous80.sum() == 80
    all_new = pd.read_parquet(DEST/'predictors.parquet')
    old = pd.read_parquet(expansion.OLD/'combined_input.parquet')
    check = all_new[all_new.previous80].merge(old, on=['example', 'state'], validate='one_to_one', suffixes=('_new', '_old'))
    assert len(check) == 80
    for col in expansion.COLUMNS:
        np.testing.assert_array_equal(check[col+'_new'], check[col+'_old'])
    routing_plan = json.loads((DEST/'routing_plan.json').read_text())
    routed = 0
    for case in routing_plan['cases']:
        done = json.loads((DEST/f"runs/{case['example']}/completion.json").read_text())
        assert done['complete'] and done['plan_sha256'] == sha(DEST/'routing_plan.json')
        expansion.batch.verify(done['files'])
        routed += done['states']-1
    assert routed == 160
    sources = [Path(m.__file__) for m in [expansion, flow_resistance, edited_features,
               response_calculation, analyze_pilot, expansion.pilot, expansion.batch]]
    sources += [Path(__file__), DEST/'preparation.json', DEST/'response_check.json', path]
    save(DEST/'additional_audit.json', dict(complete=True, sources={str(p): sha(p) for p in sources},
        unique_constituent_sets=240, original80_preserved=True, unchanged_original80_predictors=True,
        new_states_routed=routed, segmented_depth_descriptor_checks=len(reproduction),
        max_resistance_reproduction_error=float(reproduction[reproduction.feature.eq('resistance_explicit_depth')].difference.abs().max()),
        production_modified=False))
    print('Verified: 240 unique combinations, 80 unchanged predictor rows, 160 settled new runs, 30 segmented descriptor checks.')

    # Target harmonization changes every experimental training fold. Verify the
    # fitting implementation separately on unchanged historical targets.
    from combination_grouped_validation import DEST as validation, PRIOR, b
    task_path = validation/'tasks.json'
    if task_path.exists():
        original = pd.read_parquet(PRIOR/'input.parquet')
        reference = pd.read_parquet(PRIOR/'predictions.parquet')
        tasks = json.loads(task_path.read_text())['tasks']
        checks = []
        for task in tasks:
            if task['variant'] != 'baseline' or task['repeat'] != 0 or task['fold'] != 0:
                continue
            b.COLUMNS = task['columns']
            train = original.iloc[task['train']]
            test = original.iloc[[i for i in task['test'] if i < len(original)]]
            model = b.fit(train, task['params'], jobs=1)
            predicted = b.forest.predict(model, test)
            prior = reference[(reference.strategy == 'compound_mixed') &
                (reference['repeat'] == 0) & (reference.fold == 0) & (reference.model == task['model'])]
            check = test[['example', 'state']].assign(predicted=predicted).merge(prior, on=['example', 'state'], validate='one_to_one')
            assert len(check) == len(test) == len(prior)
            np.testing.assert_allclose(check.predicted, check.predicted_hours, rtol=1e-12, atol=1e-12)
            checks.append(dict(model=task['model'], rows=len(check), max_prediction_difference=float(abs(check.predicted-check.predicted_hours).max())))
        assert len(checks) == 2
        save(DEST/'baseline_implementation_audit.json', dict(complete=True, fits=2, checks=checks,
             note='Separate implementation regression check using historical targets; experimental control uses harmonized targets.'))
        print('Both predictor panels reproduce historical baseline predictions within 1e-12.')


if __name__ == '__main__':
    main()

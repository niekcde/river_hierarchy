"""Report the paired, singleton-combination augmentation experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combination_grouped_validation import DEST, EXPANSION


def main():
    complete = json.loads((DEST/'completion.json').read_text())
    assert complete['complete'] and complete['baseline_freshly_refit']
    scores = pd.read_csv(DEST/'repeat_scores.csv')
    order = ['baseline', 'plus80', 'plus240']
    labels = ['No combinations', '+80 combinations', '+240 combinations']
    subsets = ['original', 'segmented', 'previous80', 'additional160']
    names = ['Regular unit / compound states', 'Fully segmented cases', 'Previous 80 combinations', 'Additional 160 combinations']
    models = ['geometry_9', 'depth_informed_10']
    titles = ['Geometry (9 predictors)', 'Depth informed (10 predictors)']
    colors = ['#536878', '#268caa', '#bc5b34']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    for i, model in enumerate(models):
        for j, subset in enumerate(subsets):
            ax = axes[i, j]
            for x, (variant, label, color) in enumerate(zip(order, labels, colors)):
                values = scores[(scores.model == model) & (scores.cohort == subset) & (scores.variant == variant)].relative_error_pct
                assert len(values) == 5
                mean = values.mean()
                ax.bar(x, mean, color=color, width=.65, label=label)
                ax.errorbar(x, mean, yerr=[[mean-values.min()], [values.max()-mean]], fmt='none', color='#26313b', capsize=4)
                ax.text(x, values.max()+max(values.max()*.03, .4), f'{mean:.1f}', ha='center', fontsize=9)
            ax.set_xticks(range(3), ['0', '80', '240'])
            ax.set_xlabel('Combinations added to training')
            ax.set_ylim(0, ax.get_ylim()[1]*1.15)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(names[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(titles[i]+'\nRelative error (%)')
    fig.suptitle('Combined cases as separate validation entities\nFive repeated compound-group splits; bars = mean, whiskers = range (not confidence intervals)', fontsize=13)
    fig.savefig(DEST/'comparison.png', dpi=170)
    plt.close(fig)

    summary = scores.groupby(['model', 'cohort', 'variant']).relative_error_pct.mean()
    table = ['| Predictor panel | Test population | No combinations | +80 | +240 |',
             '|---|---|---:|---:|---:|']
    for model, title in zip(models, titles):
        for subset, label in zip(['primary_existing', *subsets, 'combined240'],
                                  ['Existing cases, equal cohort weight', *names, 'All 240 combinations']):
            values = [summary.loc[(model, subset, v)] for v in order]
            table.append(f'| {title} | {label} | '+' | '.join(f'{v:.2f}%' for v in values)+' |')
    changes = pd.read_csv(DEST/'paired_changes.csv')
    improvement = []
    for model, title in zip(models, titles):
        for variant, label in zip(order[1:], labels[1:]):
            z = changes[(changes.model == model) & (changes.cohort == 'primary_existing') & (changes.variant == variant)]
            improvement.append(f'- {title}, {label}: {z.relative_error_pct_change.mean():+.2f} percentage points on the original+segmented objective; '
                               f'lower error in {int((z.relative_error_pct_change < 0).sum())}/5 repetitions.')
    routing = json.loads((EXPANSION/'response_check.json').read_text())
    audit = pd.read_csv(DEST/'fold_audit.csv')
    overlap = audit[audit.variant.eq('baseline')].combined_test_with_component_in_training.sum()/1200
    text = f"""# Combined-unit expansion with singleton validation

Completed both requested experiments: reuse the previous 80 routed combinations,
and expand to 240 combinations by constructing and routing 160 additional states.
All three treatments use the same test cases and original group assignments.

## Results

Mean one-minute-floored relative error across five complete repetitions (lower is better).
Networks have equal weight within each reporting cohort, matching the earlier validation.
The primary existing-case objective weights regular and fully segmented cohorts equally.

{chr(10).join(table)}

{chr(10).join(improvement)}

![Repeated validation comparison](comparison.png)

Full R², MAE, repetition ranges and paired changes are in `summary.csv`,
`repeat_scores.csv` and `paired_changes.csv`. The 80- and 240-combination treatments
are scored on exactly the same 240 held-out combined cases, including the additional
160 never used by the +80 training treatment.

## Population and validation

- Existing population: 2,170 regular states plus 15 fully segmented states.
- Training populations before holdout: 2,185 (control), 2,265 (+80), 2,425 (+240).
- Preserve all 435 regular family/reach groups; nested and compound members remain together.
- Each of the 15 fully segmented cases and each of the 240 combinations is a singleton entity.
- Total validation groups: 450, 530 and 690 for the respective training populations.
- Five repeated five-fold compound-group partitions; the original folds are unchanged.
- Every test fold contains three fully segmented cases, 16 previous combinations and 32 additional combinations.
- Both predictor panels reuse their training-only six-candidate nested choice for the corresponding original outer fold.
  Settings are identical between treatments; augmented fits are **not retuned**.
- Training-only preprocessing and target standardization; unchanged equal-network training weights.
  Network identity is retained for weights/reporting, **not for holding out cases**.
- Geometry-only expansion: balance counts across networks, then choose maximum minimum
  distance in edited-length fraction and constituent membership. Include all previous80;
  exclude duplicates of full segmentation and duplicate constituent sets. Some networks exhaust
  their available chains, so combination counts cannot be equal across all networks.
- Resistance uses whole connected networks including connectors, as for full segmentation.
  Geometry and inverse-depth delta use edited footprints excluding connectors.

## Checks and interpretation

All 160 additional states settled under the existing routing checks. Parent hydrographs
were reproduced. Harmonizing shared response windows changed {routing['changed_existing_targets']}
existing targets by at most {routing['max_existing_target_change_seconds']:.4f} seconds and
{routing['changed_previous80_targets']} previous-combination targets by at most
{routing['max_previous80_target_change_seconds']:.4f} seconds. Every treatment, including the control,
was freshly fit and scored on those same reconciled targets. Changed response windows
across the three threshold levels: {routing['changed_windows']}.
Both depth descriptors were independently reproduced on all 15 segmented cases.
Every experimental training fold contains at least one reconciled target, so exact historical
prediction reproduction is not expected for the new control. Two separate implementation
checks, one per predictor panel, refit on the unchanged historical targets and reproduce the
saved baseline predictions within 1e-12 tolerance. All 150 experimental model fits completed,
with one prediction per test case
per treatment/model/repetition (72,750 predictions). No validation group crosses a split.

This intentionally permits related states of the same network on opposite sides of a split.
In the control folds, {overlap:.1%} of combined test cases have at least one exact constituent
state in training. Overlapping combinations may also appear in training. These are additional
states of 15 existing networks, not additional independent rivers. Results measure known-network
related-state prediction. Repetition ranges are not confidence intervals, and prior exploratory
feature selection remains a limitation. Production models and previous experiment outputs are unchanged.

## Reproduce

Use the `river-hierarchy-rivgraph` environment and limit BLAS/OpenMP threads to one:

```sh
python RAPID/investigations/cap_investigation_20261020/combination_expansion.py --run --workers 4
python RAPID/investigations/cap_investigation_20261020/audit_combination_expansion.py
python RAPID/investigations/cap_investigation_20261020/combination_grouped_validation.py --run --workers 4
python RAPID/investigations/cap_investigation_20261020/audit_combination_expansion.py
python RAPID/investigations/cap_investigation_20261020/report_combination_expansion.py
python -m unittest discover -s RAPID/investigations/cap_investigation_20261020 -p test_combination_grouped_validation.py
```

Runs are resumable and isolated under `combination_expansion_20261026` and
`combination_grouped_validation_20261026`; design/source hashes are saved with the results.
"""
    (DEST/'REPORT.md').write_text(text)
    print('\n'.join(table))


if __name__ == '__main__':
    main()

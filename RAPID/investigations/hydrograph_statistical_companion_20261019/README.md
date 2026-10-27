# Hydrograph statistical companion

Start with [DESIGN.md](DESIGN.md), which preserves the research intent independently of implementation. Read [RESULTS.md](RESULTS.md) for current findings and [CHANGELOG.md](CHANGELOG.md) for amendments. The dedicated notebook is `RAPID/notebooks/hydrograph_statistical_companion_20261019.ipynb`.

## Reproduce

From repository root, use the existing project environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/hydrograph_statistical_companion_20261019/run.py --bootstrap 2000
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/hydrograph_statistical_companion_20261019/report.py
```

`run.py` writes only the companion output directory, records source hashes and verifies they do not change during the run. The same seed reproduces network draws. It rebuilds owned outputs; copy the output directory to a versioned location before deliberately changing the design if historical outputs are needed. `report.py` rebuilds owned figures, RESULTS.md, and the dedicated notebook. No shared notebook is a dependency for execution.

Independent validation uses the existing base Python's statsmodels. Export a bridge CSV with project Python, then validate:

```sh
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python - <<'PY'
import json
import pandas as pd
from pathlib import Path
root=Path('RAPID/outputs')
cols=json.loads((root/'hydrograph_no_exposure_20261019/final_model.json').read_text())['columns']
d=pd.read_parquet(root/'hydrograph_relative_sweep_20261019/input.parquet')
d[['example','state','centroid_shift_hours']+cols].to_csv('/tmp/companion_validation.csv',index=False)
PY
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python RAPID/investigations/hydrograph_statistical_companion_20261019/validate.py
```

## Output map

- `manifest.json`: input/source hashes, versions, seed and completion status.
- `transform_audit.csv`, `response_coverage.csv`, `dependence_audit.csv`, `*_correlation.csv`, `collinearity.csv`: cohort and design audits.
- `coefficients.csv`: full and LONO coefficients for all four training scenarios, in hours per transformed unit.
- `fit_diagnostics.csv`: REML variances, ICC, optimization objective and zero-variance boundaries. These are working-model estimates, not calibrated physical uncertainties.
- `predictions.parquet`, `scores.csv`, `per_network_scores.csv`: population LONO predictions and matched scoring. Read `training_scenario` and `evaluation` together.
- `bootstrap_coefficients.csv`, `bootstrap_failures.csv`, `bootstrap_diagnostics.csv`: all attempted network-resampling fits and their status.
- `effects_uncertainty.csv`: minute-scale coefficients, pointwise intervals, IQR/reference contrasts and LONO stability. No p-values.
- `contrast_support.csv`: descriptive nearest-geometry distances for conditional IQR endpoints, using equal-network standardized inputs; does not certify feasibility.
- `refit_coefficient_sensitivity.csv`: full-cohort versus explicit exclusion/refit coefficients in minutes.
- `residual_diagnostics.parquet`, `effect_contrasts.png`, `residual_patterns.png`: interpretation and adequacy evidence.
- `independent_validation.json`: independent statsmodels comparisons and prediction-invariance checks.

## Adjustment rules

Implementation fixes: document what changed, rerun affected outputs, retain the same research contract. Cohort, target, predictor, weighting, validation or inference changes: amend DESIGN.md explicitly and record rationale before treating results as comparable. A future GAM is an extension, not a silent substitution. Preserve previous run artifacts before a substantive revision. Further uncertainty runs must retain the failure ledger and distinguish attempted from successful draws.

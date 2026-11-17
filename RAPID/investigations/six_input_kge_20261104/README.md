# KGE diagnostics for the accepted six-input ridge

Reference SIX-INPUT-KGE-2026-11-04. Compute KGE-2009 and components from saved held-out predictions; no new fitting. Primary evaluation uses equal-river sample weights to match existing R²/MAE, with conventional equal-pair KGE retained as sensitivity. Signed responses are not transformed. Cohorts scored separately; per-repeat out-of-fold scores are averaged.

Run `python RAPID/investigations/six_input_kge_20261104/run.py` in the project environment. Outputs: `RAPID/outputs/six_input_kge_20261104/`. `report.md` describes interpretation; `repeat_scores.csv` and `summary.csv` preserve all components and baselines; `per_river_diagnostics.csv` checks signed-mean cancellation. Existing R²/MAE reproduced exactly. Perfect-prediction, zero-mean, unit-rescaling and joint-sign-reversal checks passed.

Decision and main interpretation: [six-input decision note](../../../synthetic_experiments/notes/2026-11-04_six_input_ridge_decision.md). KGE is supplementary, not a new training objective or gauge validation metric in this calculation.

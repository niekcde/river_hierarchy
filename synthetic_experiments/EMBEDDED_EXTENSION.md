# Fixed-unit embedding and width/length extension

This is a separate extension of the completed synthetic experiments. It changes
three controls while retaining all seven collapse comparisons, including main
island collapse. It does not alter or replace the original seven-factor grid.

| Control | Levels |
| --- | --- |
| Outer branch A fraction | 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95 |
| Selected branch length multiplier | 1, 1.25, 1.5, 2, 3 |
| Total network extent | 20, 100, 400, 800 km |
| Outer unit boundary separation | Fixed at 10 km |
| Total width | Fixed at 500 m |
| Reference slope | Fixed at 0.0001 |
| Inner/crossover fraction | Fixed at 0.5 |
| Internal junction positions | Fixed at the existing central positions within the unit |

The complete extension has **140 configurations and 980 parent–child
comparisons**. The selected hydraulic plan contains 981 distinct routing jobs
(all completed); the historical initial plan contained 955. Original
anchor levels remain so the extension connects to existing results.

## What varies and what stays fixed

For each combination of width allocation and branch multiplier, construct the
original 20 km network first. Then add equal straight serial lengths upstream
and downstream. All internal widths, lengths, local head drops, topology and
collapse operations are retained. A 10 km unit has approach/exit lengths of
5, 45, 195 or 395 km.

This differs from the original extent experiment, which scaled the unit and
surroundings together. Here the unchanged unit occupies a progressively smaller
fraction of its network. Hydraulic resistance ratio and exposure remain fixed
across the four embeddings; length fraction and omega decrease. Automated
checks enforce those properties for every comparison.

**Reference slope, rather than total inlet–outlet head drop, is held fixed.**
Total head drop therefore increases from 2 to 80 m. Keeping total head drop
fixed would change the unit slope and confound the intended embedding test.

Both 95/5 and 5/95 are included deliberately. For simple units the lengthened
branch is A: 95/5 lengthens the dominant branch; 5/95 lengthens the narrow one.
Nested cases lengthen the upper inner branch; crossover cases lengthen the
crossover link. These remain the original family-specific perturbations.
A multiplier of 1 means the reference geometric length. Multipliers below 1
are not used because they would violate the original endpoint-distance
constraint. Comparisons between longer and shorter permitted branches remain
possible without a topological collapse.

## Routing and metrics

The selected version uses 200 m segmentation, the existing length/celerity
safeguards, BASED-reference local kb with alpha=.65, n=.035, Muskingum X=.1
and the 100→1000→100 m³/s triangular forcing. These match the selected main grid. Temporal spin-up and
recession buffers increase with travel time; they are not fixed to the shortest
network's record length. Each configuration shares a stable clock across its
states. Identical networks/settings/clocks share routing jobs.

Evaluate lag change, 1% plateau-duration change and peak-excess change. Also
export 0.5% and 2% duration thresholds, celerity caps, effective-length floors,
volume errors and whether the lag difference is at most one saved timestep.
Small lag differences must be read with this resolution flag.

The matched-domain solver is applied before embedding, when the original
boundaries are known. The resulting local resistance comparison remains valid
because the internal geometry is unchanged. The extension then recalculates
network length fraction and shifts the stored boundaries. Calling the original
outer-boundary inference on the stretched network would incorrectly choose
25–75% of its new extent; this extension explicitly avoids that error.

## Current commands and status

The full selected extension is complete under `outputs/hydraulic_alpha065_20260930/embedded_width_length_extension`. Use `selected_hydraulic_workflow.py all --workers 4` for the selected main grid and extension; see [RUNBOOK.md](RUNBOOK.md). Current analyses resolve this version through `configs/active_analysis.json`. The unchanged geometry design is in `configs/embedded_extension.json`.

### Historical original-parameter runner

The commands below retain their original settings and output path. They do not run the selected alpha=.65 setup.

From the repository root, using the project Python environment:

```bash
python synthetic_experiments/investigations/embedded_width_length_extension.py --phase plan
python synthetic_experiments/investigations/embedded_width_length_extension.py --phase pilot
python synthetic_experiments/investigations/embedded_width_length_extension.py --phase full
```

The default phase is `pilot`. It covers both 5/95 and 95/5, multipliers 1 and 3,
and extents 20 and 800 km, plus the existing 80/20, multiplier-1, 20 km anchor:
**nine configurations, 63 comparisons, 61 unique jobs**. The full phase resumes
compatible completed jobs. Outputs and the manifest record what actually ran;
a completed pilot does not mean the full extension is complete.

Outputs: `outputs/embedded_width_length_extension/`. Configuration:
`configs/embedded_extension.json`. The existing [results synthesis](notebooks/synthetic_results_synthesis.ipynb), section 5, reads matching full results when available, otherwise it clearly labels the pilot.

The existing [predictor validation](notebooks/synthetic_predictor_validation.ipynb) contains a separate extension evaluation. Run `python synthetic_experiments/investigations/embedded_predictor_validation.py` after routing to regenerate it. Fixed model specifications are fitted only to original development rows; extension anchors are verified and excluded from scoring. Scores distinguish 20 km local extremes from longer embeddings. The selected notebook now reports full-extension transfer for all 917 novel comparisons. Original-grid interpolation scores should not be presented as evidence of extrapolation ability.

Do not mix this unbalanced union with the original seven-factor ANOVA or use
these new outputs to silently retune existing predictors. Its first purpose is
to test overlap with the real omega–lag range and separate local asymmetry from
how much of the network is affected. Remaining scope limitations include fixed
inner allocation, equal upstream/downstream extensions, fixed forcing and the
existing routing assumptions.

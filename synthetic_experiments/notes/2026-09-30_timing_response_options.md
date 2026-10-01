# Timing response alternatives after the selected hydraulic rerun

**Reference: SYN-2026-09-30-TIMING-OPTIONS**  
**Written: 30 September 2026.**  
**Why:** the user requested a comparison of timing responses before choosing one
for the main text and retaining alternatives in supplements. A general “lag
change per metre” is difficult to define. The agreed unit normalizer is the
**matched parent equivalent length**, not a new unit length and not the child
length. This follows the network-embedding and gauge-placement discussion.

## What was done

`investigations/timing_response_comparison.py` reads the selected alpha=.65,
BASED-reference, n=.035 results: 15,309 original-grid comparisons and all 980
extension comparisons. No routing, real-case analysis or primary metric
contract was changed. The existing synthesis (section 6) and predictor notebooks
contain executed figures and tables. Previous notebook copies are preserved in
`outputs/hydraulic_alpha065_20260930/notebooks_before_timing_comparison/`.

For model parent/child peak lags Tp/Tc, parent/child equivalent lengths Lp/Lc,
and the network single-centerline length Ln, the candidates are:

| Candidate | Formula | Units |
| --- | --- | --- |
| Current percentage lag | 100 (Tc/Tp − 1) | % |
| Absolute model timing | (Tc − Tp)/3600 | h |
| Model network speed change | Ln (1/Tc − 1/Tp) | m/s |
| Timing / parent Leq | (Tc − Tp)/Lp | s/m, displayed as min/km |
| Reference-scaled speed | (Ln/Tref) (Tp/Tc − 1) | m/s |
| Reference timing / parent Leq | (Tc/Tp − 1) Tref/Lp | s/m, displayed as min/km |
| State-specific equivalent-speed index | 100 [(Lc/Lp)(Tp/Tc) − 1] | % |

There are no observed gauge times in a synthetic experiment. The reference
variants use the saved single-centerline control lag Tref as a sensitivity
check, **not as an observation of the parent**. Actual gauge scaling still needs
the real-case reference. If Tref=Tp, these variants reduce exactly to their
direct model counterparts. The equivalent-speed index combines local lengths
with network travel times and can normalize away the shortening effect; it is
not local celerity. Its reciprocal slowness index is also exported.

## Controlled embedding result

All 245 local geometries have four extents: 20/100/400/800 km. Parent Leq stays
fixed within each geometry. The experiment adds **both approach and exit**
length, so it is not an isolated downstream-gauge-relocation test. Common
surrounding routing can also reshape the peaks.

Relative drift is the largest absolute departure across embeddings from the
20 km response, divided by that response's absolute value. For a conservative
summary, 172 geometries have peak-time differences exceeding three timesteps
at all four extents; all rows remain in the exported data and model evaluation.

| Response | Median relative drift | 90th percentile |
| --- | ---: | ---: |
| Percentage lag | 96.4% | 97.2% |
| Absolute timing | 23.6% | 150.0% |
| Timing / parent Leq | 23.6% | 150.0% |
| Network speed change | 96.1% | 97.1% |

Absolute timing and timing/parent-Leq have identical relative drift within
these groups because parent Leq is fixed. Leq normalization addresses comparison
across different local scales, not residual peak movement within an embedding.
This is not a claim of universal length independence: substantial residual
embedding dependence remains for some geometries.

For simple/full with A=.95 and multiplier 3, parent Leq is 29 km (not the 10 km
boundary span). The normalized timing effect changes from −23.12 to −22.66
min/km between 20 and 800 km, whereas lag changes from −68.04% to −5.07%.

The saved event-centroid differences provide an explanatory check: their maximum
departure across embeddings is less than 8.2e−8 seconds over all 245 geometries.
Mean arrival-time differences therefore remain constant to numerical precision
in this linear fixed-coefficient experiment while peak differences can change.
This diagnostic is not a new selected response or evidence of identical
behaviour in nonlinear real rivers.

## Fixed predictor comparison

All fits use the original development rows, original geometry-group holdout and
the same fixed shallow boosted estimator. No feature search or extension
training was performed. All 63 matching anchors reproduce the seven new metric
values and are excluded from extension scores, leaving 917 novel comparisons.

The previous six inputs are retained as a reference. A change block adds log
child/parent equivalent-length ratio, log child/parent elongation ratio, and
child-minus-parent path-length CV. Delta width dominance is a separate controlled
addition, not assumed useful because the earlier four-variable block helped.

Treatment-averaged R² for the six-plus-three-change set **without delta width
dominance**:

| Response | Original holdout | Novel 20 km extremes | Longer embeddings | Full novel extension |
| --- | ---: | ---: | ---: | ---: |
| Percentage lag | 0.937 | 0.592 | −27.708 | −1.990 |
| Absolute timing | 0.947 | 0.345 | 0.541 | 0.504 |
| Network speed change | 0.929 | 0.534 | −30.454 | −2.622 |
| Timing / parent Leq | 0.945 | 0.453 | 0.631 | 0.592 |
| Reference timing / parent Leq | 0.941 | 0.512 | 0.410 | 0.447 |
| State-specific equivalent-speed index | 0.893 | 0.350 | −16.605 | −2.086 |

R² values across different responses are not a sufficient basis for choosing
which physical question to ask. In particular, the normalized response contains
Leq algebraically, and Leq is also an input. MAEs are saved in each response's
own units and must not be directly ranked across units. Tiny-variance long-
embedding scopes can have very negative R², so read them alongside MAE.

Adding delta width dominance changes timing/parent-Leq R² by +.0017 on the
original holdout and +.0189 on the full extension (+.0742 for 20 km extremes).
This is mixed, limited evidence, not a reason to declare the variable necessary
or universally invalid. With a single-path child, delta Dw=1−Dw,parent; compound
path enumeration may share links and does not give independent flow shares.

## Provisional framing and next decision

**Timing change / parent Leq is the leading candidate for the local-impact
question**, supported by reduced embedding dilution and better transfer in this
test. It remains a width-weighted normalization of a network peak-time change,
not a measured local celerity. Real gauge anchoring and residual peak reshaping
remain unresolved. The current primary response is not silently replaced.

A possible paper structure is one main timing definition chosen for its meaning,
with the percentage, absolute-time, network-speed and reference-scaled alternatives
in supplements. Preserve the poorly transferring state-specific index as evidence
for why its interpretation differs. After the timing decision, allow a new compact
search to replace parent descriptors with defensible change descriptors rather
than continually adding to the original six. Do not use the exploratory extension
as an untouched confirmation set after making choices from it.

Reproduce from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  synthetic_experiments/investigations/timing_response_comparison.py
```

Outputs: the active run's `current_analysis/timing_response_comparison/`, including
metric definitions, source hashes/plan IDs, comparisons, per-treatment scores,
predictions, embedding diagnostics and figures. Algebra checks cover added-delay
invariance, units, reciprocal speed conversion, reference identity and pure
shortening. Earlier selected results and their validation partitions are preserved.

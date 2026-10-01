# Geometric predictor definitions

All current path descriptors are computed on the **matched parent domain**, not
necessarily just the edited links. Length fraction and exposure additionally
use network context. See the current predictor-validation notebook for their
standalone and joint predictive performance.

For each link e, let its physical length and width be ell_e and w_e. Enumerate
all directed paths p between the matched domain's upstream and downstream
boundaries. Define:

\[
L_p=\sum_{e\in p}\ell_e,\qquad
W_p=\frac{\sum_{e\in p}\ell_e}{\sum_{e\in p}\ell_e/w_e},\qquad
a_p=\frac{W_p}{\sum_jW_j}.
\]

W_j is the equivalent width of path j; j ranges over all paths. These are
geometric width weights, not necessarily routed discharge fractions.

## Current descriptors

\[
L_{eq}=\sum_pa_pL_p=\frac{\sum_pW_pL_p}{\sum_pW_p},\qquad
W_{eq}=\sum_pW_p
\]

\[
\mathrm{elongation}=\frac{L_{eq}}{W_{eq}}
=\frac{\sum_pW_pL_p}{(\sum_pW_p)^2}
\]

\[
D_w=\frac{\max_pW_p}{\sum_pW_p}
\]

\[
\bar L=\frac1n\sum_pL_p,\qquad
CV_L=\frac{\sqrt{\frac1n\sum_p(L_p-\bar L)^2}}{\bar L},\qquad
A_{LW}=\frac{L_{eq}}{\bar L}-1
\]

L_eq and W_eq have units of metres; elongation, D_w, CV_L and A_LW are
dimensionless. Positive A_LW indicates longer paths tend to be wider;
negative indicates longer paths tend to be narrower. Equal lengths or equal
path widths yield A_LW=0. D_w records width imbalance; CV_L records length
imbalance even when path widths are equal.

The three arrangement descriptors are complementary, but not independent.
For exactly two positive-width paths:

\[
|A_{LW}|=(2D_w-1)CV_L.
\]

The sign identifies whether the wider path is longer or shorter. At equal widths,
D_w=0.5 and A_LW=0 regardless of CV_L. This algebraic overlap is one reason
individual removal losses can be small; it does not make length diversity
irrelevant. The two-path identity does not generally apply to larger path sets.

## Network fraction, exposure and slope

\[
L_f=\frac{\sum_{e\in\mathcal D}\ell_e}{\sum_{e\in\mathcal N}\ell_e}.
\]

Each link is counted once in these sums, unlike path enumeration where shared
links can occur in several paths. Path-based W_eq is therefore not necessarily
an actual cross-sectional width in complex compounds.

For synthetic exposure, start with F(inlet)=1 and distribute each node's share
among outgoing links in proportion to link width:

\[
b_e=\frac{w_e}{\sum_{j\in out(u)}w_j},\qquad
F(v)=\sum_{e=(u,v)}F(u)b_e,\qquad E=F(u_{\mathcal D}).
\]

Outer enclosing-domain comparisons use E=1; nested inner collapse uses the
outer branch A fraction. Exposure uses upstream context.

\[
S_{ref}=\frac{H_{up}-H_{down}}{x_{down}-x_{up}}.
\]

This is the imposed reference slope in the synthetic grid, not a mean link
slope. The older real pilot used head drop / shortest domain path length; its
reference-length convention must remain explicit in any transfer analysis.

## Domain boundaries

| Treatment | Matched domain |
| --- | --- |
| Simple/full | Whole simple unit |
| Nested/inner-only | Inner unit |
| Nested/full | Whole outer unit |
| Nested/main-island-only | Whole outer unit, including retained inner branch |
| Crossover/full | Whole compound |
| Either partial crossover collapse | Enclosing closed compound |

## New, separately tested groups

**Parent-to-child change**, over the same matched boundaries:

\[
\log(L_{eq,c}/L_{eq,p}),\quad
\log(\mathcal E_c/\mathcal E_p),\quad
CV_{L,c}-CV_{L,p},\quad D_{w,c}-D_{w,p}.
\]

Natural logarithms are used. These describe the geometry retained by the child,
without using its routing response.

**Surrounding-network lengths**, measured in the parent:

\[
L_{up}=\min_{p:\ inlet\to u_{\mathcal D}}\sum_{e\in p}\ell_e,
\qquad
L_{down}=\min_{p:\ v_{\mathcal D}\to outlet}\sum_{e\in p}\ell_e.
\]

These are shortest directed physical path lengths in metres. They do not sum
all outside branches and do not describe bypass-channel width or length. They
are evaluated as an additional two-variable group, not assumed to be necessary.

## Reading the prediction analyses

- Correlation: direction/strength of a linear or monotonic association, within
  each treatment, computed on development data. It is not a prediction score.
- Standalone R²: prediction using only that variable, with one shared model
  across treatments. It is not a separately optimized fit within each treatment.
- Small combinations: joint prediction using separate input columns. These are
  not var-times-var scalar products.
- Removal loss: full-model R² minus refitted reduced-model R². Losses do not add
  to overall R² and do not measure independent physical contributions.

Omega is one constructed input axis but uses resistance, length fraction and
exposure. Counting it as one axis does not mean it needs only one raw measurement.
Compact selection compares input-axis counts and performance, not measurement cost.

## Selected hydraulic reference · 30 September 2026

Current synthetic reruns use a BASED reference section and alpha=0.65:

\[
D_i=D_{ref}(W_i/W_{ref})^{\alpha},\qquad
k_{b,i}=k_{b,ref}(W_i/W_{ref})^{1-\alpha}.
\]

The existing wide-channel Chezy equation is now evaluated with this local kb:

\[
K_i=\frac{\ell_i}{g C_z^2 W_i^2 D_i^3}
\propto \frac{\ell_i}{W_i^{2+3\alpha}}.
\]

The width exponent is therefore 3.95 at alpha=.65, rather than 5 under a common
width/depth ratio. The proportionality factor is common to parent and child
within each reference configuration; it cancels in the equivalent-resistance
ratio. Thus the ratio/omega remain geometric predictors under the fixed depth
law even though the dimensional resistance uses a BASED reference. Local kb is
applied before solving the matched domain, not as a correction to the old final
ratio. The Chezy predictor is not claimed to equal Manning resistance or to
reproduce the imposed width-based allocation.

Pure geometry descriptors and geometry-group validation splits are retained.
Responses, dimensional/relative resistance, omega and the fitted benchmarks
must be recomputed from the selected run. Real-case selection is deferred until
new runs with updated slopes. The old full-grid outputs are preserved and are
not relabeled as selected-run results.

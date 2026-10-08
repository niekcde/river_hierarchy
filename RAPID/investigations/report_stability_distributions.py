"""Save distribution interpretation and embed the figures in the existing notebook."""
from pathlib import Path
import base64,pandas as pd,nbformat
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/distributions';t=pd.read_csv(OUT/'tail_summary.csv',dtype={'example':str});q=pd.read_csv(OUT/'quantiles.csv',dtype={'example':str});raw=t[(t.variable=='celerity_raw_mps')&(t.weighting=='length')];conf=t[(t.variable=='numerical_conflict')&(t.weighting=='diagnostic_dt')]
text='''# Full-distribution stability review — 2026-10-02

Follow-up to ROBUST-SLOPES-2026-10-02: inspect whether numerical conflicts arise from the slow tail, fast tail or broad hydraulic heterogeneity before deciding which assumptions to revisit. All seven failing base networks are included, with every original link in each network. No hydraulic parameters or production outputs were changed.

## Main interpretation
| Case | Distribution finding | First issue to investigate |
|---|---|---|
| 05 | Predominantly a slow-tail problem. Fast link 87 is extreme but does not exceed the raw-celerity outlier fence under either weighting. | Narrow-branch depth/width extrapolation and local low slope. Do not lower the fast side just because it minimizes changed-link count. |
| 11 | Broad, mixed distribution; conclusions depend on weighting. Four slow links are flagged with equal-link weighting, none with channel-length weighting. | Compare the long low-gradient section and shorter steep sections separately. Respect potentially real spatial regimes; avoid a network-wide percentile correction. |
| 13 | Both low and high raw-celerity tails are flagged under both weightings. | Inspect both narrow/shallow branches and steep/wide fast links. No basis for blaming only the slow side. |
| 15 | Mainly slow-tail outliers. The fast tail remains within the raw-celerity fence. | Low-depth narrow branches combined with low allocated slopes. The numerical conflict involves multiple links but a small fraction of channel length. |
| 22 | A small slow tail; the fast side is a substantial part of the distribution, with 17.4% of channel length already above the upper cap in raw celerity. | Tiny narrow branch 882 and the depth law's extrapolation there, rather than compressing the broad fast population further. |
| 41 | Predominantly slow-tail outliers; no fast raw-celerity outlier. | Narrow branch 374 and the modest relative correction required by the timestep diagnostic. |
| 46 | Both raw-celerity tails are flagged under both weightings. | Both low-slope/shallow branches and steep, deep fast reaches; use a two-sided hydraulic review rather than a single-link fix. |

These identify priorities for checking hydraulic assumptions, not evidence that flagged links are physically erroneous. A statistical extreme can be a real channel. Do not use an outlier fence as a celerity cap or depth floor without separate justification.

## Definition and weighting
The distribution screen uses log10(raw celerity): lower fence Q25 − 1.5 IQR and upper fence Q75 + 1.5 IQR, transformed back to m/s. Log scale treats multiplicative deviations symmetrically. This is a transparent descriptive screen, not a confidence interval, formal error test or universal physical threshold.

Calculate every distribution twice: equal weight per original link and weight proportional to physical channel length. The latter prevents many short links from automatically dominating. It sums all branch lengths, not reference-centerline length, and it is not discharge, conveyance, volume or hydrograph-impact weighting. Original links can generate multiple routing segments; the K value shown is that of their effective routing segment.

Use **raw celerity** to assess modeled hydraulic extremes: caps otherwise hide high values and create artificial ties. Also inspect **used K** because actual timestep compatibility depends on effective segment length and capped celerity. Neither celerity nor K alone determines whether the depth or slope estimate is plausible.

## Raw-celerity tails, using channel-length-weighted fences
| Case | Slow-tail links | Slow-tail fraction of total channel length | Fast-tail links | Fast-tail fraction of total channel length |
|---|---:|---:|---:|---:|
'''
for ex,g in raw.groupby('example'):
 a=g[g.side=='lower'].iloc[0];b=g[g.side=='upper'].iloc[0];text+=f'| {ex} | {a["count"]} | {a.length_pct:.2f}% | {b["count"]} | {b.length_pct:.2f}% |\n'
text+='''
Case 11's zero flags do not imply homogeneous or well-constrained hydraulics: its wide interquartile range makes this screen permissive. Equal-link weighting flags four slow links covering 5.36% of channel length. This sensitivity is itself a reason to inspect spatial regimes rather than choose a correction from a single summary statistic.

## Size of the numerical conflicts
These are the slow/fast conflicts at the previously selected **diagnostic** integer timesteps that minimize the number of excluded intervals. These are not the same as statistical outliers, and these timesteps have not been selected for production routing.

| Case | Slow conflicts | Fraction of channel length | Fast conflicts | Fraction of channel length |
|---|---:|---:|---:|---:|
'''
for ex,g in conf.groupby('example'):
 a=g[g.side=='slow'].iloc[0];b=g[g.side=='fast'].iloc[0];text+=f'| {ex} | {a["count"]} | {a.length_pct:.3f}% | {b["count"]} | {b.length_pct:.3f}% |\n'
text+='''
For example, case 15's 12 conflicting links represent approximately **1.56% of total channel length**; case 46's 35 represent **6.07%**. Counts alone overstated how broadly case 15 is affected. These fractions do not predict changes in discharge or hydrograph shape.

Case 05 illustrates why optimizing changed-link count is not sufficient: the 180 s example selects one slow and one fast link for changes, but the full distribution points primarily to unusual slow values. The fast link is not automatically the preferable target merely because it is short.

## Reading the figures
- CDF x-axis: the variable value on a logarithmic scale, except effective segment length, which uses a linear scale. Y-axis: fraction of links or channel length at or below that value.
- For K, the left tail is fast and the right tail is slow. Grey weights links equally; green weights channel length; dashed orange shows the uncapped counterpart.
- Green shading shows the compatible K interval at a diagnostic timestep: dt/1.8 ≤ K ≤ dt/0.2. It is a numerical window, not an observational uncertainty band.
- Per-case six-panel figures cover width, estimated depth, slope, raw/used celerity, used K and effective segmentation length. Red/blue ticks identify the diagnostic slow/fast conflicts; these are not statistical-outlier labels.
- Purple lines mark BASED reference width/depth. All local depths are model estimates. With common n and alpha, raw celerity follows W^(2 alpha/3) S^(1/2) within a case; the width/depth and slope panels show the ingredients of its extremes.

## Practical implication
Keep the selected slope method. Investigate whether the reference-anchored width–depth extrapolation is too strong at the slow tails of 05,15,22,41, checking slope evidence as well. Review both tails in 13 and 46. For 11, assess the distinct spatial regimes and long-reach weighting before applying any broad cap. The distributions can guide **where** to investigate; they do not select a physically correct n, alpha, depth floor or cap by themselves.

Settings: selected robust base slopes, stored BASED depth anchor, alpha=.65, n=.035, common caps .05–2.524 m/s, X=.1 and 200 m target/effective floor. All values are conditional on these assumptions. Only the seven failing networks are covered; a proposed global adjustment must subsequently be checked on all 16.

Reproduction: `RAPID/investigations/analyze_stability_distributions.py` then `report_stability_distributions.py`. Outputs include `quantiles.csv`, `tail_summary.csv`, `links_with_distribution_flags.csv`, the combined K comparison and seven six-panel distribution figures.
'''
(OUT/'README.md').write_text(text)
p=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb';n=nbformat.read(p,as_version=4);tag='stability-distributions-20261002';n.cells=[c for c in n.cells if tag not in c.metadata.get('tags',[])];c=nbformat.v4.new_markdown_cell(text);c.metadata['tags']=[tag];n.cells.append(c)
for name in ['all_K_distributions.png']+[f'{ex}_distributions.png' for ex in sorted(q.example.unique())]:
 c=nbformat.v4.new_markdown_cell(f'![Full distributions: {name}](attachment:{name})');c['attachments']={name:{'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}};c.metadata['tags']=[tag];n.cells.append(c)
nbformat.validate(n);nbformat.write(n,p);print('Distribution report and notebook updated.')

"""Document the shared-cap decision and append revised distributions to the notebook."""
from pathlib import Path
import base64,json
import numpy as np,pandas as pd,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/common_caps';s=pd.read_csv(OUT/'summary.csv');c=pd.read_csv(OUT/'stability_conflict_counts.csv',dtype={'example':str});ex=pd.read_csv(OUT/'cap_exception_links.csv',dtype={'example':str});sel=s[(s.cap_min==.05)&(s.cap_max==5)].iloc[0]
text='''# Shared celerity caps and revised distributions — 2026-10-02

## Clarification and decision
The immediately preceding seven-network distribution analysis already used common **0.05–2.524 m/s** caps. The earlier hydraulic audit additionally reported saved Excel/per-case caps in a separate scenario. They were not used in that distribution analysis.

The new all-16 diagnostic uses **0.05–5.0 m/s for every base network**. The lower limit is retained because it currently clips no links; raising it to 0.1 would alter three additional links without improving the overall pass count. The upper limit is a broad, round diagnostic guardrail that affects only three links. These are explicitly listed for local review. Do not raise the shared limit simply to absorb an individual extreme, or tighten it to conceal remaining stability conflicts.

This is an operational diagnostic choice, not a universal physical maximum or a calibrated celerity model. If minimizing clipped-link count were the only objective, no clipping would trivially win: limits of 0.05–8 m/s already leave every present link unchanged. The 5 m/s threshold deliberately retains a small explicit review list. It is not claimed to be a uniquely optimal or literature-prescribed cutoff.

## Scientific interpretation of the guardrail
River wave celerity depends on the channel and event. [Allen et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018GL077914) discuss previously measured values spanning approximately 0.25–10 m/s. [HEC guidance](https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/applying-reach-routing-methods-within-hec-hms/applying-the-muskingum-cunge-routing-method) also treats appropriate celerity as channel- and event-dependent. Therefore exceeding 5 m/s is not proof that a link's geometry or estimated hydraulics are wrong. Local corrections need evidence; the cap itself is only a transparent sensitivity/guardrail assumption.

## Common-cap comparison
All comparisons retain n=.035, alpha=.65, stored BASED reference depth, the selected robust slopes, X=.1, and the 200 m segmentation target/effective floor. All 5,315 original links from 16 base networks are included. No saved Excel cap is applied.

| Common range (m/s) | Clipped links | Clipped fraction of total channel length | Networks passing integer-timestep preflight |
|---|---:|---:|---:|
'''
for lo,hi in [(.05,2.524),(.05,3),(.05,4),(.05,5),(.05,6),(.5,3)]:
 z=s[(s.cap_min==lo)&(s.cap_max==hi)].iloc[0];text+=f'| {lo:g}–{hi:g} | {int(z.clipped_links)} | {z.clipped_length_pct:.3f}% | {int(z.passing)}/16 |\n'
text+='''
A grid of **265 common cap pairs** was evaluated. The least-clipping all-pass pair *in this tested grid* is 0.5–3 m/s, modifying 611 links and 12.88% of channel length; it multiplies the slowest celerity by about 6.3. This is a much larger model intervention than the selected broad guardrail. It is not recommended merely to obtain 16/16 passes. This grid result is not a proof of the globally optimal continuous cap pair.

## The only three capped links
| Case | Link | Raw celerity (m/s) | Diagnostic value | Physical link length (m) |
|---|---:|---:|---:|---:|
'''
for z in ex.itertuples():text+=f'| {z.example} | {z.id_link} | {z.celerity_raw_mps:.3f} | 5.000 | {z.actual_length_m:.1f} |\n'
text+='''
Together these are **3/5,315 links (0.056%) and 0.078% of total channel length**. No link is raised by the 0.05 lower cap. Raw values are preserved alongside diagnostic values. There have been **no edits to their slopes, depths, widths, roughness or production files**. Those input changes would require a specific justification; they have not been invented simply to meet the cap.

## Stability conflicts exposed by the broader range
Seven networks pass: **03, 21, 37, 45, 48, 55, 56**. Nine have an empty common integer-timestep interval: **02, 05, 11, 13, 15, 22, 41, 46, 54**.

| Case | Minimum intervals still requiring change | Slow + fast at an illustrative minimum-change timestep | Diagnostic timestep (s) |
|---|---:|---:|---:|
'''
for z in c[~c.feasible].itertuples():text+=f'| {z.example} | {z.minimum_changed_intervals} | {z.slow_count_at_dt} + {z.fast_count_at_dt} | {z.representative_dt_s} |\n'
text+='''
These are mathematical lower bounds, not an instruction to adjust every listed link. They hold the current segmentation/X and other link intervals fixed, and assume excluded links could be adjusted independently. A shared n/alpha may change many links at once and need not realize these minima. No forcing-clock divisibility or full routing run is included.

Where several timesteps have the same minimum count, the representative now minimizes the sum of physical-length-weighted absolute log celerity changes needed to enter the timestep intervals. This makes the example less arbitrary than taking the first qualifying integer, without applying those changes.

Case **54** is an integer-clock edge case: its continuous interval is **86.530–86.616 s**. Its representative 87 s diagnostic flags fast link **152** (raw celerity 4.156 m/s). Reducing that link's diagnostic celerity to 360/87 = 4.138 m/s would remove this single interval conflict, a change of about 0.44%; that adjustment has **not** been applied or justified physically. This should not drive a reduction of the common upper cap for every river.

Case **22** changes from one minimum conflict under the old cap to 16 under the broader cap, demonstrating that the old 2.524 limit was materially shaping its stability. Cases **02** and **54** newly fail because their fast links are no longer clipped at the lower upper-limit. This is greater transparency about modeled heterogeneity, not evidence that widening the cap changed river geometry.

## Revised distribution interpretation
All 16 networks now have width, estimated depth, slope, raw/used celerity, used K and effective segment-length CDFs, both equally weighted by original links and weighted by physical channel length. The raw-celerity distribution is unchanged by choosing new caps; the used K distribution and timestep conflicts change.

The previous hydraulic-tail priorities remain: slow tails in 05,15,22,41; both tails in 13,46; mixed regimes and weighting sensitivity in 11. New failure 02 also has a statistically unusual slow tail under both weighting approaches. Case 54 has statistical extremes at both ends, but its present integer failure is tiny and should be treated as such. Log-IQR flags are descriptive screens, not physical-error labels.

The green K windows correspond to diagnostic timesteps, not production clock choices. Raw values are shown so cap-created ties cannot hide fast tails. Channel-length fractions are not discharge weights.

## Artifacts and scope
- `selected_diagnostic.json`: one cap range, shared by all 16; explicitly excludes Excel overrides.
- `summary.csv`, `scenarios.csv`: the 265-pair comparison.
- `all_base_links.csv`, `stability_link_details.csv`: raw and diagnostic hydraulics.
- `cap_exception_links.csv`: the three upper-cap review candidates.
- `stability_conflict_counts.csv`, `minimum_interval_change_links.csv`: remaining interval conflicts.
- `distributions/`: all-16 quantiles, outlier flags and figures.
- Scripts: `common_celerity_cap_audit.py`, `materialize_common_cap_candidate.py`, `analyze_stability_distributions.py --run-dir RAPID/assets/bounded_robust_slopes_20261002/common_caps --cap-max 5`, and `report_common_cap_audit.py` in `RAPID/investigations`.

This reruns the diagnostic analysis, not RAPID hydrographs or child-network preparation. Production/Excel files are unchanged. Re-evaluate clipping and stability when n or alpha is finally selected; the three-link count is conditional on n=.035 and alpha=.65.
'''
(OUT/'README.md').write_text(text)
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);scatter=ax.scatter(s.clipped_links,s.exceptions,c=s.passing,cmap='viridis',s=22,alpha=.65);fig.colorbar(scatter,ax=ax,label='Passing base networks');ax.set(xlabel='Number of original links clipped by common caps',ylabel='Minimum additional conflicting intervals',title='Common-cap trade-off · n=0.035, alpha=0.65');ax.grid(alpha=.15)
for lo,hi,offset in [(.05,2.524,(8,8)),(.05,4,(8,8)),(.05,5,(8,8)),(.5,3,(-80,15))]:
 z=s[(s.cap_min==lo)&(s.cap_max==hi)].iloc[0];ax.scatter(z.clipped_links,z.exceptions,s=60,facecolors='none',edgecolors='black');ax.annotate(f'{lo:g}–{hi:g} m/s',(z.clipped_links,z.exceptions),xytext=offset,textcoords='offset points',fontsize=8)
fig.savefig(OUT/'cap_tradeoff.png',dpi=140);plt.close(fig)
p=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb';n=nbformat.read(p,as_version=4);tag='common-cap-all16-20261002';n.cells=[c for c in n.cells if tag not in c.metadata.get('tags',[])];banner=nbformat.v4.new_markdown_cell('**Latest hydraulic diagnostic: common caps 0.05–5 m/s for all 16 base networks.** See the final section for the cap comparison, three clipped-link exceptions and revised distributions. Earlier cap scenarios remain historical comparisons. Production settings are unchanged.');banner.metadata['tags']=[tag];n.cells.insert(0,banner);cell=nbformat.v4.new_markdown_cell(text);cell.metadata['tags']=[tag];n.cells.append(cell)
for name in ['cap_tradeoff.png','distributions/all_K_distributions.png']+[f'distributions/{ex}_distributions.png' for ex in c.example]:
 path=OUT/name;key=path.name;cell=nbformat.v4.new_markdown_cell(f'![Shared caps: {key}](attachment:{key})');cell['attachments']={key:{'image/png':base64.b64encode(path.read_bytes()).decode()}};cell.metadata['tags']=[tag];n.cells.append(cell)
nbformat.validate(n);nbformat.write(n,p);print('Common-cap report and all-16 distribution notebook updated')

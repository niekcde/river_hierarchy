"""Dated audit report and notebook section; no production mutation."""
from pathlib import Path
import sys,json,base64
sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_signed_routing_validation import ROOT,BASE,OUT,EXAMPLES
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat


def markdown_table(frame):
    def fmt(value):
        return f'{value:.5g}' if isinstance(value, (float,np.floating)) else str(value)
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def main():
 inv=pd.concat([pd.read_csv(OUT/ex/'inventory.csv',dtype={'example':str}) for ex in EXAMPLES],ignore_index=True)
 inv.to_csv(OUT/'inventory.csv',index=False)
 settings=pd.concat([pd.read_csv(OUT/ex/'saved_settings.csv',dtype={'example':str}) for ex in EXAMPLES],ignore_index=True)
 settings.to_csv(OUT/'saved_settings.csv',index=False)
 comparisons=pd.concat([pd.read_csv(OUT/ex/'comparisons.csv',dtype={'example':str}) for ex in EXAMPLES],ignore_index=True)
 comparisons.to_csv(OUT/'comparisons.csv',index=False)
 summary=pd.concat([pd.read_csv(OUT/ex/'routing_summary.csv',dtype={'example':str}) for ex in EXAMPLES],ignore_index=True)
 summary.to_csv(OUT/'routing_summary.csv',index=False)
 errors=[]
 for ex in EXAMPLES:
  try:errors.append(pd.read_csv(OUT/ex/'routing_errors.csv',dtype={'example':str}))
  except pd.errors.EmptyDataError:pass
 errors=pd.concat(errors,ignore_index=True) if errors else pd.DataFrame(columns=['example','fingerprint','dt_s','error'])
 errors.to_csv(OUT/'routing_errors.csv',index=False)
 blocked=inv[inv.status!='ready'];blocked.to_csv(OUT/'blocked_inputs.csv',index=False)
 assert all((OUT/ex/'routing_method.json').exists() for ex in EXAMPLES)
 assert comparisons.groupby(['example','fingerprint']).size().eq(2).all()
 stats=[];coverage=[];mass=[]
 for ex in EXAMPLES:
  g=inv[inv.example==ex];c=comparisons[comparisons.example==ex]
  ranges=c.groupby('fingerprint')[['timing_change_minutes','duration_change_pct','peak_excess_change_pct']].agg(lambda a:a.max()-a.min())
  method=json.loads((OUT/ex/'routing_method.json').read_text());qa=json.loads((OUT/ex/'segment_validation.json').read_text());mass.extend(qa['mass_residual_relative'])
  coverage.append(dict(example=ex,entries=len(g),ready=int(g.status.eq('ready').sum()),blocked=int(g.status.ne('ready').sum()),unique_states=g[g.status=='ready'].fingerprint.nunique(),timesteps_s='/'.join(map(str,method['clocks'])),forcing=qa['forcing_kind'],base_segments=qa['n_segments'],max_timing_difference_s=60*ranges.timing_change_minutes.max(),max_duration_difference_pp=ranges.duration_change_pct.max(),max_peak_difference_pp=ranges.peak_excess_change_pct.max()))
  f=pd.read_csv(OUT/ex/'every_segment_audit.csv')
  for name,h in f.groupby('forcing'):
   stats.append(dict(example=ex,forcing=name,min_flow=h.minimum_cms.min(),min_branch_normalized_pct=100*h.minimum_divided_by_branch_fraction.min(),max_negative_volume_pct=100*h.negative_volume_fraction.max(),segments_negative=int((h.minimum_cms< -1e-8).sum())))
 coverage=pd.DataFrame(coverage);coverage.to_csv(OUT/'coverage_and_convergence.csv',index=False)
 stats=pd.DataFrame(stats);stats.to_csv(OUT/'segment_audit_summary.csv',index=False)
 cap=pd.concat([pd.read_csv(OUT/ex/'cap_sensitivity.csv',dtype={'example':str}) for ex in ['22','46']]);cap.to_csv(OUT/'cap_sensitivity.csv',index=False)
 cap_effect=[]
 for ex,g in cap.groupby('example'):
  a=g[g.policy=='cap5'].set_index('fingerprint');b=g[g.policy=='no_upper_cap'].set_index('fingerprint');delta=b[['peak_time_s','timing_change_minutes','duration_change_pct','peak_excess_change_pct']]-a[['peak_time_s','timing_change_minutes','duration_change_pct','peak_excess_change_pct']]
  cap_effect.append(dict(example=ex,base_peak_shift_s=delta.loc[a[a.base].index[0],'peak_time_s'],max_paired_timing_shift_s=60*delta.timing_change_minutes.abs().max(),max_duration_pp=delta.duration_change_pct.abs().max(),max_peak_excess_pp=delta.peak_excess_change_pct.abs().max()))
 cap_effect=pd.DataFrame(cap_effect);cap_effect.to_csv(OUT/'cap_effect_summary.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(13,4.4),constrained_layout=True)
 for ax,col,label in zip(axes,['max_timing_difference_s','max_duration_difference_pp','max_peak_difference_pp'],['Timing contrast difference (s)','Duration contrast difference (pp)','Peak-excess contrast difference (pp)']):
  ax.bar(coverage.example,coverage[col],color=['#f59e0b' if x in ['13','45','48'] else '#2563eb' for x in coverage.example]);ax.set_ylabel(label);ax.set_xlabel('Case');ax.tick_params(axis='x',rotation=90);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Maximum difference between routing clocks · fixed K · orange: standard pulse only')
 fig.savefig(OUT/'all_case_convergence.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,name,title in zip(axes,['triangle_12h_24h','rectangle_24h'],['Smooth zero-base pulse','Abrupt zero-base pulse']):
  h=stats[stats.forcing==name].set_index('example').reindex(EXAMPLES)
  ax.bar(h.index,(-h.min_branch_normalized_pct).clip(lower=0),color='#d97706');ax.set(title=title,ylabel='Worst artificial dip (% of branch-scaled amplitude)',xlabel='Case');ax.tick_params(axis='x',rotation=90)
 fig.savefig(OUT/'all_case_internal_dips.png',dpi=160);plt.close(fig)
 cfg=dict(reference='FULL-SIGNED-ROUTING-2026-10-02',n_manning=.035,depth_width_exponent=.65,depth_rule='D_i = D_ref (W_i/W_ref)^alpha; Kb_i = W_i/D_i',celerity_caps_mps=[.05,5.],X=.1,segmentation_target_m=200,effective_length_floor_m=200,stability_effective_length_correction=False,timestep_dependent_celerity_regularization=False,slopes='Bounded robust profile and original base unit allocation; shared heads. New child vertices solved locally with fixed boundaries.',coefficient_policy='Signed current inflow allowed for investigation; retain 0 <= previous-outflow coefficient < 1',status='Investigation configuration, NOT production-ready: unresolved child inputs and saved-method migration',production_changed=False)
 (OUT/'candidate_configuration.json').write_text(json.dumps(cfg,indent=2))
 native=pd.read_csv(OUT/'all_base_native_crosscheck.csv',dtype={'example':str});refined=pd.read_csv(OUT/'selected_pair_refinement.csv',dtype={'example':str})
 table=markdown_table(coverage[['example','entries','ready','blocked','unique_states','base_segments','timesteps_s','max_timing_difference_s']])
 actual=stats[(stats.forcing=='actual')&~stats.example.isin(['13','45','48'])]
 smooth=stats[stats.forcing=='triangle_12h_24h'];sharp=stats[stats.forcing=='rectangle_24h']
 report=f'''# Full fixed-hydraulic routing validation — 2026-10-02 to 2026-10-03

Reference: **FULL-SIGNED-ROUTING-2026-10-02**. Follow-up to SIGNED-MUSKINGUM-2026-10-02, requested to determine whether the proposed n/Kb/caps/slopes configuration is ready for production. Results remain conditional on forcing, allocation, segmentation and routing model.

## Decision

The tests support keeping the candidate hydraulic settings for further work. They do **not** establish an unconditional production-ready configuration. {len(blocked)} state entries still have unresolved hydraulic inputs; cases 13/45/48 have no saved actual-event base hydrograph; production RAPID and resistance inputs still use older settings. The processable child set also contains extreme candidate head allocations (up to 60.86 m/s raw celerity), so a successful numerical run is not hydraulic acceptance. No production inputs or results were overwritten.

## Fixed settings and input refresh

n=.035, alpha=.65, D_i=D_ref(W_i/W_ref)^alpha, Kb_i=W_i/D_i, X=.1, 200 m target and effective-length floor, common .05–5 m/s caps. Keep the bounded robust reference profile and original base-unit head allocation. No local 5 cm head edits, no stability-driven n or extra celerity correction, no stability-driven effective-length increase.

For a child, exact coordinate matches inherit base heads (not renumbered node IDs). New vertices receive a constrained equal-per-link relative-slope-error fit within their incident patch. Boundary heads remain fixed; its reference slope is boundary head range divided by longest directed patch length, with the existing 1% relative floor. Where a changed topology invalidates inherited internal unit heads, the affected unit interiors can be reallocated with unit boundary/direct-profile heads protected. This is an isolated candidate refresh, not a full production hydraulic-context rebuild. Infeasible or nonpositive-head cases are excluded explicitly, never assigned stale child slopes. Case 15 uses direction130 throughout.

## Coverage and temporal convergence

{table}

There are {len(inv)} conceptual entries, {sum(coverage.unique_states)} unique ready hydraulic states (including 16 bases), and {len(comparisons)//2} unique base–child comparisons tested at two clocks. Island aliases reuse identical geometry results. All registered parents are S000_base. {len(errors)} routing-screen errors were recorded. The largest two-clock paired timing difference is {coverage.max_timing_difference_s.max():.4g} s; duration and peak-excess differences reach {coverage.max_duration_difference_pp.max():.4g} and {coverage.max_peak_difference_pp.max():.4g} percentage points. These maxima are numerical differences, not physical accuracy claims.

Forcing is taken from each saved base event where available, cropped to one day before its active portion and initialized at exact equilibrium. End plateaus extend at least ten days or three longest-path K sums. Cases 13/45/48 use a labelled 12 h rise / 24 h fall standard triangle on unit baseflow: numerical checks only, not actual-event validation.

The exhaustive outlet screen uses the exact discrete transfer function evaluated with a band-limited representation of forcing on a 120 s reporting grid. This grid is distinct from the routing timestep. A padded smooth return outside the analyzed interval prevents circular FFT contamination; initial/final plateau errors are checked. Against native time-domain base routing, maximum waveform error fractions were {native[native.reporting_step_s==120].max_wave_error_fraction.max():.4g} at 120 s and {native[native.reporting_step_s==60].max_wave_error_fraction.max():.4g} at 60 s. Inspect `all_base_native_crosscheck.csv` for peak-time errors; small waveform errors can shift a flat maximum. Cases 21 and 56 receive additional 30 s reporting-grid / 5 s routing-clock checks (15→5 s changes the selected case-21 contrast by 0.193 s; 45→5 s changes the case-56 contrast by 1.173 s) in `selected_pair_refinement.csv`. The 5 s diagnostic need not divide saved forcing cadence because the same continuous interpolated forcing is evaluated directly.

## Every-base-segment native audit

All {coverage.base_segments.sum():,} base subreaches were routed with the native recurrence for their event/pulse and two zero-base stress inputs, without clipping negative values. Cases 11/46 reuse the earlier coarse native audit only after verifying identical K and segment counts; their previous fine audit remains available. Other cases use one common base-network clock below the upper bound. Internal child subreaches have **not** all received this separate native stress audit; all ready child outlets received the two-clock screen.

The minimum actual discharge across the 13 available real-event base runs is {actual.min_flow.min():.5g} m³/s. The largest smooth zero-base undershoot is {-smooth.min_branch_normalized_pct.min():.5g}% of branch-scaled input amplitude; for an abrupt rectangular pulse it is {-sharp.min_branch_normalized_pct.min():.5g}%. Maximum relative network mass-balance residual is {max(abs(x) for x in mass):.3g}. Signed coefficients are therefore not positivity-preserving, even where actual-event responses remain positive. Retaining 0<=C_previous_outflow<1 gives decaying homogeneous response; it is not a physical validation or a universal nonnegative-flow guarantee.

## Child hydraulic ranges: an additional blocker

The isolated inherited-head refresh is insufficient as a production rebuild. Case 37 / S156_unit_18 reaches a raw celerity of 34.36 m/s and island_37_161 reaches 60.86 m/s; the latter assigns about 59 m of head loss to an 86 m link (slope 0.686). Case 22 / S037_unit_220 reaches 12.79 m/s. These three states exceed twice their own base-network maximum raw celerity, a triage flag rather than a proposed physical threshold. `hydraulic_review_candidates.csv` and `candidate_hydraulic_ranges.csv` retain the evidence. The minimum child slope across processable states is 2.15e-7 m/m.

The base profile is not thereby shown to be wrong: the candidate child-head transfer must be replaced/verified by a full consistent child-unit allocation before those results receive physical interpretation. Do not fix these artifacts by expanding the cap. The table label **ready** means numerically processable in this investigation only.

## The three upper-capped base links

Base links 22/321, 46/292 and 46/307 have raw celerities about 5.21, 5.75 and 6.18 m/s. Only these three base links exceed 5 m/s (3/5315 original links). `base_upper_cap_links.csv` in each case records their widths, depths, slopes and lengths. Removing the upper cap was tested for every ready unique state in cases 22 and 46 with identical clocks and other parameters; any capped child links are also released.

{markdown_table(cap_effect)}

The largest paired effect is case 22 / CB001_bubble_9: a collapsed link is 81.8 km long with raw celerity 5.306 m/s. Removing the cap changes its timing contrast from -398.51 to -414.23 minutes (15.72 minutes, about 3.9% of the capped contrast). This is not the 12.79 m/s outlier. A small celerity correction applied over a long collapsed link can matter even when very few links are capped. Case 46 maximum paired cap effect is about 23.24 seconds.

These are **uncapped minus capped** results. They quantify the numerical effect of retaining 5 m/s; they do not independently prove the hydraulic plausibility of each raw celerity.

## Saved-method audit and remaining work

The 13 available RAPID prep manifests use n=.35, scalar BASED-reference Kb, case-dependent caps and stability effective-length correction. Cases 13/45/48 have no saved prep manifest in these roots. The saved base bubble-resistance edge tables use exactly one Kb per network in all 16 cases. Their equivalent-resistance manifests use **Chezy**, so changing Manning n does not directly change that resistance law; updating slopes and the depth/Kb geometry still matters. `saved_settings.csv` and `saved_resistance_kb_audit.csv` preserve the evidence.

Before a consistent production rerun: resolve missing contexts and failed child fits, and replace/verify the candidate allocation that creates extreme positive child slopes; rebuild and verify contexts under one frozen configuration; implement the width-dependent Kb consistently in RAPID and the chosen resistance geometry; prepare the missing actual-event runs; and make signed-coefficient routing an explicit audited policy rather than silently bypassing a preparation guard. Recompute affected equivalent-resistance results with fresh cache keys. This investigation does not justify mixing the new routed responses with old resistance tables.

The candidate settings themselves remain n=.035, alpha=.65, common .05–5 caps and the selected bounded robust slopes. The unresolved decision is how to promote and validate the numerical policy and all refreshed states, not another round of tuning n or slope to force coefficient positivity.
'''
 (OUT/'README.md').write_text(report)
 notebook=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb';nb=nbformat.read(notebook,as_version=4);marker='FULL-SIGNED-ROUTING-2026-10-02'
 nb.cells=[c for c in nb.cells if marker not in c.get('source','')]
 nb.cells.append(nbformat.v4.new_markdown_cell(report))
 for name in ['all_case_convergence.png','all_case_internal_dips.png']:
  cell=nbformat.v4.new_markdown_cell(f'<!-- {marker} -->\n![{name}](attachment:{name})')
  cell.attachments={name:{'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}};nb.cells.append(cell)
 nbformat.validate(nb);nbformat.write(nb,notebook)
 print(report)

if __name__=='__main__':main()

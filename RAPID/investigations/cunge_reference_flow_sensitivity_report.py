"""Report isolated reference-flow tests; does not alter hierarchy inputs."""
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/cunge_reference_flow_sensitivity_20261006'
f=pd.read_csv(OUT/'sensitivity_results.csv',dtype={'example':str})
alias=pd.read_csv(OUT/'state_aliases.csv',dtype={'example':str})
meta=json.loads((OUT/'provenance.json').read_text())
for item in meta['sources']:
 for side in ['parent','child']:
  assert hashlib.sha256(Path(item[f'{side}_frame']).read_bytes()).hexdigest()==item[f'{side}_sha256']
assert f.minimum_native_flow_cms.min()>=0 and f.tail_settled.all()
assert not f.grid_changed_for_comparison.any()
diag=f[~f.variant.isin(['actual','parent'])]
summary=diag.groupby('example').agg(max_wave_difference_pct_inlet_amplitude=('max_wave_difference_pct_inlet_amplitude','max'),max_abs_difference_cms=('max_abs_outlet_difference_cms','max'),max_saved_peak_shift_s=('peak_time_difference_s',lambda x:abs(x).max()),all_screening_pass=('screening_pass','all'))
summary.to_csv(OUT/'summary.csv')
relations=[]
for (ex,key),rows in f.groupby(['example','key']):
 r=rows.set_index('variant');p=r.loc['parent'];a=r.loc['actual']
 for name,b in r.iterrows():
  if name in ['actual','parent']:continue
  actual_delta=a.peak_cms-p.peak_cms;alternate_delta=b.peak_cms-p.peak_cms
  relations.append(dict(example=ex,state=a.state,variant=name,actual_child_minus_parent_peak_cms=actual_delta,alternate_child_minus_parent_peak_cms=alternate_delta,peak_magnitude_direction_reversed=bool(actual_delta*alternate_delta<0),actual_child_minus_parent_peak_time_s=(a.lag_h-p.lag_h)*3600,alternate_child_minus_parent_peak_time_s=(b.lag_h-p.lag_h)*3600))
relations=pd.DataFrame(relations);relations.to_csv(OUT/'parent_child_relationships.csv',index=False)
clock=pd.read_csv(OUT/'focused_clock_check.csv',dtype={'example':str})
text=f'''CUNGE REFERENCE-FLOW SENSITIVITY — 6 October 2026

Scope and conclusion
--------------------
Tested {len(alias)} flagged parent/child entries representing {len(meta['sources'])} distinct transitions across seven examples. These are targeted outlet sensitivity runs, not completion of all full-network RAPID runs. Existing calibrated Manning n, event forcing, physical inputs and alpha=2 were retained. No running-job inputs or production acceptance rules were edited.

The Q-change flag is not evidence of extra water or a numerical instability. It compares independently allocated STATIC reference discharge on a matched link. The earlier readiness test overreached by assuming that identical link geometry guarantees an identical hydraulic role. In 41, removing a parallel branch necessarily changes the surviving branch's flow. In 46, a changed competing branch width changes the estimated split.

Definition and provenance
-------------------------
Q_change = Q_ref_child(link) - Q_ref_parent(link).
Percentage change = 100 * Q_change / Q_ref_parent(link).
The scalar reference discharge is inherited from the stored BASED reference anchor, originally taken from the maximum forcing discharge (reference_section.py, compute_reference_section_kb). It is fixed across parent and child states: 7645.54857984 m3/s for 41; 2232.09821487415 m3/s for 46. The resampled event maximum for 46 is slightly different (2231.689848714555 m3/s); these tests retain the original stored anchor in every state.

At a bifurcation: Q_ref_i = Q_ref_arriving * B_i / sum(B_outgoing). At a confluence, incoming allocated flows are added. Thus per-link reference flows are model estimates, not individual branch gauge observations. In each state the split shares sum to one and the outlet allocation equals the inlet allocation. The same split fractions route the time-varying hydrograph.

Q enters X = 0.5 * [1 - Q_ref/(B*S*c*dx)]. For fixed geometry, c, and numerical length dx, higher Q makes X more negative. It does not directly change c or K. A width-derived redistribution can affect both these X coefficients and the fraction of water following different paths.

41: largest flagged change
--------------------------
State island_41_005, node 5 to node 7:
Parent narrow branch: width 72.451196 m, Q_ref 639.484901 m3/s.
Parent parallel branch: width 255.650007 m, Q_ref 2256.475099 m3/s.
Child retains the narrow branch alone: Q_ref 2895.960000 m3/s.
The increase is exactly the removed branch's 2256.475099 m3/s, or +352.858230% relative to the old narrow-branch flow. The node's total flow remains 2895.96 m3/s. It is not a 353% increase in network inflow.

For island_41_138, the corresponding increase 1416.849103 m3/s also equals flow formerly carried by the removed parallel branch (250.530842 m wide); the surviving 411.933035 m branch then carries 3746.490228 m3/s. In island_41_055 and island_41_075 the remaining physical links also cannot all retain their parent flows after removal: the node balance residuals are 819.992886 and 223.414801 m3/s, respectively, with no unknown/new edges to absorb them.

Conservation therefore supports recomputing Q and X in these changed hydraulic domains. It does not prove that keeping the surviving narrow channel's width/depth is a physically adequate representation of the removed parallel system. That is a separate structural approximation. These tests assess the consequence, not validate the collapsed geometry against observations.

46: why an unchanged branch gets less Q
--------------------------------------
State S075_unit_16 (same hydraulic transition as island_46_017), node 25:
The retained 25->23 branch stays 134.567537 m wide.
The competing 25->31 branch changes from 601.733436 to 620.512159 m wide.
Retained branch share: 18.2762% -> 17.8216%.
Retained branch Q_ref: 407.941821 -> 397.796367 m3/s, a change of -10.145454 m3/s (-2.486986%).
Competing branch Q_ref: 1824.156394 -> 1834.301848 m3/s.
Total: 2232.098215 m3/s in both cases. This is a width-allocation consequence, not a changed boundary event. The width change is an input difference; these tests do not establish which width is more accurate.

Experimental design
-------------------
1. Actual child: current child width-based split fractions and Q-derived X.
2. X_only: restore parent Q only in coefficients of exact shared links, keeping actual child routing shares and initial flow fractions. This isolates coefficient sensitivity. It is deliberately a counterfactual, not a proposed hydraulically consistent production configuration.
3. parent_bifurcation_splits: at bifurcations that still exist, hold the parent shares of surviving physical branches; distribute the remainder over other child branches in their child-width ratio. Recompute all downstream reference flows, X and steady initial flow fractions consistently. Near-identical branches are included only when endpoints, length and slope match exactly and width/depth differ by at most 1e-5 relative. This catches both slightly altered branches in example 22. It does not hold an old fractional share when only one outgoing branch remains.
4. conserved_parent_flow: solve conservation with every exact shared-link Q fixed to its parent value. Use only uniquely determined positive balanced solutions. This is available for 56. A failed uniqueness/positivity screen is NOT a proof that no possible conserved solution exists: many changed subgraphs are underdetermined. Only the explicit no-new-edge residual cases above prove that freezing all those Q values is impossible.
5. Parent: route the actual parent under the common clock to retain the parent/child comparison.

Each comparison uses a common grid for the child variants and a common timestep at most 15 s satisfying the signed-inflow policy's upper bound. None required a changed child section count. Common constant tails were extended until every variant met the existing 0.1% event-excess settlement criterion plus two days. All {len(f)} routed configurations remained nonnegative at every native section (overall minimum {f.minimum_native_flow_cms.min():.6f} m3/s). Frame hashes were rechecked after completion. The independent DAG recurrence was used; this investigation did not add a new full-engine equivalence test.

Sensitivity results
-------------------
All differences below compare an alternate treatment with the ACTUAL CHILD, not child with parent. Wave difference is max absolute outlet difference divided by upstream event amplitude, times 100; it is not the relative change in local reference Q. Maxima cover all tested variants for each example.

{summary.to_string()}

The declared screening limits were <=0.01% inlet-amplitude waveform difference, <=0.01% peak magnitude difference, zero change in saved-output peak time, and nonnegative native flow. These are a deliberately strict diagnostic screen, not a measured uncertainty bound or universal hydrological significance threshold.

Native timing and timestep refinement
-------------------------------------
{clock.to_string(index=False)}

For island_41_005, the parent-Q-in-X counterfactual has a reported peak one 1070 s output interval later, but its native peak is only about 223 s (3.72 min) later. Halving the native timestep retains that result. Thus output averaging/bin selection exaggerates the apparent timing difference. The actual parent and actual child have the same saved peak lag, 79.358333 h: the 17.83 min difference belongs to the X-only counterfactual and must not be presented as the actual collapse-induced peak shift.

Parent/child relationships
--------------------------
{relations[relations.peak_magnitude_direction_reversed].to_string(index=False)}

Small waveform sensitivity does not guarantee preservation of the sign of a near-zero parent/child metric. For example 46, the actual child peak is about 0.003576 m3/s below its parent; retaining parent bifurcation shares makes it about 0.012735 m3/s above. Both are tiny relative to its roughly 1700 m3/s outlet peak. Such pairs should be treated as near ties, not strong evidence of positive or negative attenuation relationships. See parent_child_relationships.csv for all comparisons. Predictor correlations across all hierarchy pairs were not rerun here.

Readiness interpretation
------------------------
13, 15, 22, 54 and 56: the flagged reference-allocation differences have negligible outlet effects in these targeted event tests and warrant a documented tolerance, rather than exact-Q rejection.
46: the allocation change has a clear explanation and tiny outlet sensitivity (maximum about 0.03331 m3/s, 0.002046% of inlet event amplitude). It is reasonable to proceed as an experimental run with the changed width/allocation documented; do not interpret signs of tiny pair effects as robust.
41: the large local Q changes are largely conservation-required responses to changed topology. Do not freeze Q on a surviving internal branch just because its physical row is unchanged. The largest X-only waveform difference is 1.233921 m3/s (0.042306% of inlet amplitude); this exceeds our strict screen, although it is small in absolute hydrograph terms. Treat 41 as a flagged exploratory run, not evidence of broken water balance and not a validated geometry representation.
The existing strict full-run gate remains unchanged and may still reject these states. A future gate correction should distinguish genuinely exterior unchanged hydraulic domains from branches inside the modified flow domain, while retaining geometry, conservation, coefficient, timestep and positivity checks. A blanket tolerance large enough to hide the 353% local Q redistribution would be inappropriate.
'''
(OUT/'REPORT.txt').write_text(text)
fig=make_subplots(rows=2,cols=2,subplot_titles=['41 / island_41_005 — outlet discharge','41 — alternate minus actual child','46 / S075_unit_16 — outlet discharge','46 — alternate minus actual child'],horizontal_spacing=.09,vertical_spacing=.15)
styles={'parent':('Actual parent','#222','dash'),'actual':('Actual child','#d62728','solid'),'X_only':('Parent Q in X only — diagnostic','#9467bd','dot'),'parent_bifurcation_splits':('Parent shares at surviving bifurcations','#0072B2','dash')}
seen=set()
for row,(ex,state) in enumerate([('41','island_41_005'),('46','S075_unit_16')],1):
 key=alias[(alias.example==ex)&(alias.state==state)].iloc[0]['key'];wave=pd.read_parquet(OUT/ex/key/'waveforms.parquet');item=next(x for x in meta['sources'] if x['key']==key);case=json.loads((Path(item['source_calibration'])/'case.json').read_text());end=(pd.Timestamp(case['last_sample'])-pd.Timestamp(case['first_sample'])).total_seconds()/86400
 wave=wave[wave.time_seconds<=end*86400];x=wave.time_seconds/86400
 for name,(label,color,dash) in styles.items():
  if name not in wave:continue
  fig.add_trace(go.Scatter(x=x,y=wave[name],name=label,legendgroup=name,showlegend=name not in seen,line=dict(color=color,dash=dash,width=2),hovertemplate='Day %{x:.4f}<br>%{y:.6f} m³/s<extra>'+label+'</extra>'),row=row,col=1);seen.add(name)
  if name not in ['actual','parent']:
   fig.add_trace(go.Scatter(x=x,y=wave[name]-wave.actual,name=label,legendgroup=name,showlegend=False,line=dict(color=color,dash=dash,width=2),hovertemplate='Day %{x:.4f}<br>Difference %{y:.6f} m³/s<extra>'+label+'</extra>'),row=row,col=2)
 for col in [1,2]:fig.update_xaxes(title_text='Days since event window start',row=row,col=col)
 fig.update_yaxes(title_text='Outlet discharge (m³/s)',row=row,col=1);fig.update_yaxes(title_text='Outlet difference (m³/s)',row=row,col=2)
fig.update_layout(title='Reference-flow sensitivity — fixed calibrated n; same event and clock',template='plotly_white',height=950,width=1400,margin=dict(t=150,b=70,l=80,r=35),legend=dict(orientation='h',x=0,y=1.07,yanchor='bottom'),hovermode='x unified')
fig.write_html(OUT/'targeted_hydrographs.html',include_plotlyjs=True)
print(summary.to_string());print('Report and interactive figure written; source hashes unchanged.')

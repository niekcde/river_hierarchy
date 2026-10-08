"""Read-only diagnosis of completed full-hierarchy preflight checks.

Does not edit routing code, source hydraulic frames or active-run directories.
"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
OUT=ROOT/'RAPID/outputs/cunge_readiness_audit_20261006'
ASSETS=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'
cal=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str});cal=cal[cal.method.eq('cunge')].set_index('example')
log=Path('/private/tmp/cunge_readiness.log').read_text();(OUT/'preflight.log').write_text(log)
passed={}
for line in log.splitlines():
 if line.startswith('{'):
  record=json.loads(line)
  if record.get('status')=='preflight_passed':passed[record['example']]=record
summary=[];details=[];splits=[];inputs={}
for ex in cal.index:
 inv=pd.read_csv(ASSETS/ex/'inventory.csv');audit=pd.read_csv(OUT/ex/'exterior_invariance.csv');failed=audit[~audit.unchanged_parameters]
 m=json.loads((Path(cal.loc[ex,'output_dir'])/'case.json').read_text());qref=m['reference_discharge_cms'];cache={}
 def get(fp):
  if fp not in cache:
   path=Path(inv[inv.fingerprint.eq(fp)].iloc[0].frame);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   cache[fp]=cunge.freeze_grid(pd.read_parquet(path),qref)
  return cache[fp]
 for row in failed.itertuples():
  ch=inv[inv.family.eq(row.family)&inv.state.eq(row.state)].iloc[0]
  par=inv[inv.family.eq(row.family)&inv.state.eq(row.parent_state)]
  if par.empty:par=inv[inv.state.eq(row.parent_state)].drop_duplicates('fingerprint')
  assert len(par)==1
  a=get(par.iloc[0].fingerprint);b=get(ch.fingerprint)
  keys=['u','v','link_length_m','width','slope','depth_m'];z=a.merge(b,on=keys,suffixes=('_p','_c'))
  z=z[~np.isclose(z.reference_discharge_cms_p,z.reference_discharge_cms_c,rtol=1e-10,atol=1e-8)]
  for edge in z.itertuples():
   details.append(dict(example=ex,family=row.family,state=row.state,parent_state=row.parent_state,u=edge.u,v=edge.v,parent_q_cms=edge.reference_discharge_cms_p,child_q_cms=edge.reference_discharge_cms_c,delta_q_cms=edge.reference_discharge_cms_c-edge.reference_discharge_cms_p,relative_change_pct=100*(edge.reference_discharge_cms_c/edge.reference_discharge_cms_p-1),width_m=edge.width))
  # Find altered outgoing branch sets at common nodes; constant vertical
  # profile offsets cannot explain these changes in width-based allocation.
  for u in sorted(set(a.u)&set(b.u)):
   aa=a[a.u.eq(u)];bb=b[b.u.eq(u)]
   if len(aa)<=1 and len(bb)<=1:continue
   if np.isclose(aa.width.sum(),bb.width.sum(),rtol=1e-12,atol=1e-10) and sorted(aa.v)==sorted(bb.v):continue
   for side,edges in [('parent',aa),('child',bb)]:
    for e in edges.itertuples():splits.append(dict(example=ex,family=row.family,state=row.state,parent_state=row.parent_state,node=u,side=side,id_link=e.id_link,v=e.v,width_m=e.width,split_fraction=e.split_fraction,reference_discharge_cms=e.reference_discharge_cms))
 local=[d for d in details if d['example']==ex]
 status='preflight_passed' if not len(failed) else 'blocked_shared_link_flow_changes'
 if ex=='11' and not len(failed):status='numerically_ready_calibration_unresolved'
 summary.append(dict(example=ex,status=status,ready_state_entries=int(inv.status.eq('ready').sum()),excluded_states=int(inv.status.eq('excluded_topology_review').sum()),unique_states=int(inv[inv.status.eq('ready')].fingerprint.nunique()),affected_transitions=len(failed),max_abs_delta_Q_cms=max([abs(d['delta_q_cms']) for d in local],default=0),max_abs_relative_Q_change_pct=max([abs(d['relative_change_pct']) for d in local],default=0),max_abs_delta_X=float(failed.rapid_x_max_abs_change.max()) if len(failed) else 0,parent_timing_fit=bool(cal.loc[ex,'within_sampling']),estimated_Qout_MiB=passed.get(ex,{}).get('estimated_Qout_MiB'),routing_dt_s=passed.get(ex,{}).get('routing_dt_s')))
s=pd.DataFrame(summary);s.to_csv(OUT/'readiness_summary.csv',index=False);pd.DataFrame(details).to_csv(OUT/'affected_shared_links.csv',index=False);pd.DataFrame(splits).to_csv(OUT/'changed_branch_allocations.csv',index=False)
(OUT/'inspected_input_hashes.json').write_text(json.dumps(inputs,indent=2))
report='''CUNGE FULL-HIERARCHY READINESS REVIEW — 6 October 2026

Scope: all 16 examples. Input-only preflight in an isolated audit directory.
No full hierarchy runs were started. The active 02/05 output directories, routing implementation and source hydraulic inputs were not edited by this review.
Passing input preflight is not a completed routing/settlement/positivity test; the runner still performs those checks before publishing results.

'''+s.to_string(index=False)+'''

CAUSE
The current Cunge preparation recalculates reference-flow fractions in each state by allocating flow in proportion to outgoing branch width. A width change in a replaced or reevaluated branch therefore redistributes flow into an otherwise unchanged branch. Its Qref changes, and consequently its Cunge X changes, despite unchanged length, depth, width and slope on that surviving branch. Celerity, K and cell counts remain unchanged on the matched exterior links in these failures.
The inherited basin reference discharge and calibrated Manning n have not changed. This is a conflict between per-state width-proportional allocation and the strict requirement to preserve every unchanged exterior link's hydraulic parameters. It is not, by itself, evidence of nonconservation, routing instability or a failed calibration.

13 DETAIL
indep/S002_unit_17, parent S000_base: at node 31, the affected branch changes from width 443.283722 m (toward node 32) to 442.759015 m (toward node 118). The unchanged neighbour toward node 36 retains width 588.196791 m. The smaller competing branch changes the neighbour's reference flow by +0.250495 m³/s (about +0.050895%). This is a real recalculation consequence, not floating-point roundoff. The source width product labels the replacement changed_evaluated.

15 DETAIL
Six island states fail: island_15_023, _035, _044, _111, _188 and _194. Maximum absolute Qref shift is 0.000476 m³/s. In inspected island_15_111, routing-corrected widths at multiple splits differ from the base by about 0.00007–0.00017 m while their underlying wet widths are unchanged. Those small recalculations propagate downstream. This is much smaller than the 13 change, but not machine-epsilon arithmetic noise. One pre-existing topology-review exclusion remains excluded; it is separate from these six transitions.

WHY NOT RELAX THE CHECK GLOBALLY
The same check finds material redistribution in 41 (up to about 2,256.48 m³/s) and 46 (10.15 m³/s). A blanket relaxation would mask changes relevant to parent-child comparison. Smaller failures also occur in 22, 54 and 56; see exact transition and edge records. All reported maxima are over the checked affected transitions, not whole-network totals.

NEXT METHOD DECISION
To maintain the agreed strict exterior-invariance experiment, inherit parent branch-flow shares at surviving collapse boundaries and assign only genuinely new interior flows by conservation. Reference Q, actual routing split fractions and steady initial flows must then be derived consistently; simply overwriting X or Q on selected links is insufficient. This would be a distinct preparation policy, not a relaxed numerical tolerance. It has not been implemented or validated by this audit.
Alternatively, per-state width-based redistribution can be retained and explicitly treated as part of the network transformation, but that relaxes the earlier requirement that untouched exterior links keep identical parameters. The currently running experiments retain the original strict check.
Small routing-width recalculations in island products should also be checked against the existing unchanged-link width-inheritance contract before choosing a numerical tolerance. No source widths were changed here.

READY TO SCHEDULE
02, 03, 05, 21, 37, 45, 48, 55 pass current input checks; 03 was already fully routed previously. The user's 02/05 jobs should be left running.
11 passes numerical-input checks but retains its documented Cunge timing failure, so it is an explicitly qualified sensitivity run rather than a satisfactory calibrated example.
13, 15, 22, 41, 46, 54, 56 remain blocked by the current shared-link policy.
37 includes one previously excluded topology state; passing refers to all ready, nonexcluded states.

STORAGE
Estimates cover uncompressed Qout only at the archived base horizon. Prep inflow arrays are of similar size, and common-tail extensions can increase both. Reserve extra space for temporary arrays, metadata and logs. Network 37's Qout alone is about 4.09 GiB; 55 about 1.42 GiB; 48 about 0.78 GiB. Use the external drive for a larger batch rather than assuming local space suffices for every hierarchy. Existing runner free-space checks remain active.

FILES
readiness_summary.csv — per-example readiness and estimates.
affected_shared_links.csv — exact Qref changes on matched exterior links.
changed_branch_allocations.csv — parent/child widths and shares at changed splits.
<example>/exterior_invariance.csv — every audited transition and parameter check.
<example>/excluded_states.csv — preserved, pre-existing topology exclusions.
preflight.log — original check output, including aggregate failure status.
inspected_input_hashes.json — hashes for hydraulic frames used in detailed diagnosis.
'''
(OUT/'REPORT.txt').write_text(report)
print(s[['example','status','affected_transitions','max_abs_delta_Q_cms','max_abs_relative_Q_change_pct']].to_string(index=False))

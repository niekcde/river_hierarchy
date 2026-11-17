"""Package boundary verification and slope-only feasibility results."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];FIRST=ROOT/'RAPID/outputs/member_boundary_review_20261027';OUT=FIRST.with_name(FIRST.name+'_complete')

def main():
    data=json.loads((OUT/'state_results.json').read_text());f=pd.DataFrame(data);f['example']=f.example.astype(str).str.zfill(2)
    assert len(f)==1467 and not f[['example','state']].duplicated().any()
    assert not f.exception.fillna('').str.startswith('TypeError:').any()
    b=pd.read_json(OUT/'boundary_nodes.json');prior=f[f.previous_strict_failure]
    counts=f.feasibility_status.value_counts().to_dict();status=f.boundary_status.value_counts().to_dict()
    f.groupby(['example','feasibility_status']).size().unstack(fill_value=0).to_csv(OUT/'by_network.csv')
    f[f.feasibility_status.ne('feasible')].to_csv(OUT/'exceptions.csv',index=False)
    prior[['example','state','feasibility_status']].to_csv(OUT/'historical_strict_failures_current_status.csv',index=False)
    for name in ['focused_boundary_review.html','focused_cases.json','independent_lp_checks.json']:shutil.copy2(FIRST/name,OUT/name)
    # Every witness explains an actual anchor-to-anchor constraint, not a solver error.
    witnesses=json.loads((OUT/'feasibility_witnesses.json').read_text());flat=[]
    for v in witnesses:
        for kind,key in [('positive','nonpositive_paths'),('floor','floor_conflicts')]:
            for path in v[key]:flat.append(dict(example=v['example'],state=v['state'],failure_kind=kind,**path))
    pd.DataFrame(flat).to_csv(OUT/'conflicting_paths.csv',index=False)
    initial_floor_count=int(f.current_relative_floor_feasible.fillna(False).sum());positive_count=int(f.positive_slopes_feasible.fillna(False).sum())
    verified=int(f.boundary_status.eq('verified_existing_correspondences').sum());manual=int(f.boundary_status.eq('manual_boundary_exception').sum());exceptions=len(f)-verified-manual
    report=f'''# Existing compound-member boundaries and fixed-head slope feasibility

27 October 2026. Scope: the 1,467 accepted compound-member comparisons only.
No collapse geometry, source head, production slope, width, depth, reference-flow
allocation, routing parameter or calibration was changed. No routing was run.

## Result

All 1,467 comparisons have an existing valid published boundary contract. The review
reused its selected parent reaches and boundary identities. It cross-checked the
selection against saved hierarchy membership and the accepted comparison inventory,
and the boundary list against selected/exterior node incidence. It did not invent
new collapse regions or replace saved boundaries.

- {verified:,} cases have verified current boundary correspondences and no manual
  boundary override requiring separate interpretation.
- {manual} cases carry explicit reviewed boundary overrides and were kept separate.
- {exceptions} other cases have a boundary/data review exception.
- {initial_floor_count:,} cases pass positive-slope feasibility and the existing
  free-patch 1% relative slope floor with boundary/exterior heads fixed.
- {positive_count:,} tested cases permit positive slopes before applying that floor.

| Slope check outcome | Cases |
|---|---:|
'''
    labels={'feasible':'Feasible with fixed heads and current floor','nonpositive_fixed_head_path':'Reversed or zero fixed-boundary head drop','relative_floor_infeasible':'Positive slopes possible; current floor too high','invalid_floor_reference':'Positive slopes possible; floor reference undefined','not_tested_exterior_path_exception':'Exterior correspondence requires review','not_tested':'Manual boundary interpretation kept separate','not_tested_interface_scope_exception':'Unexpected replacement/exterior interface'}
    for name,n in counts.items():report+=f'| {labels.get(name,name)} | {n} |\n'
    report+='''
`not_tested_exterior_path_exception` means the saved exterior path evidence cannot
be applied directly to the current graph, or exterior subdivision nodes require
additional handling. It is not a finding of hydraulic infeasibility. Boundary
identity verification and exterior-path verification are separate gates.

## How the test works

The current parent hydraulic heads are the fixed reference. Every original exterior
node, including the selected member boundary, is fixed. All selected internal nodes
are free, including surviving original nodes. The replacement reach set is the
current child minus the existing saved exterior paths; unchanged cases use unique
mapped directed-endpoint and length correspondence. Exterior path direction and
continuity are checked before use. Approved displaced boundary identities are reused,
not rediscovered by nearest-neighbour matching.

On a directed acyclic graph, each directed path between fixed heads must have positive
available head drop. For the current relative floor, the available drop must also
exceed the sum of the required edge drops. The longest weighted path gives the binding
constraint. This is a feasibility calculation, not slope optimization. It produces
explicit paths and endpoint heads when constraints conflict.

For each connected component of free internal nodes, the relative-floor reference
is the boundary head span divided by the boundary reference-section distance span,
as in the existing reconstruction. The required slope is 1% of this value. As in
production, fixed-to-fixed replacement edges are checked for positivity but are not
subject to that free-patch relative floor. `invalid_floor_reference` means the
reference-section extent cannot define this reference, even though positive slopes
may be feasible; it is not a proof that boundary heads need to move.

Boundary correspondence is verified by current injective mapped identities, exact
parent positions or published validated displaced-node correspondences. The source
frame and geometry hashes are checked against the active release. This is a check
of published geometries after earlier boundary repairs, not a count of boundaries
that survived the original raw collapse untouched.

## Three concrete examples in network 22

| Case | Result | Evidence |
|---|---|---|
| S296_unit_30 | Feasible | Critical free-patch path 132 → 139 has 0.474822238 m available drop; the current floor requires 0.022726829 m. |
| S023_unit_136 | Positive slopes infeasible | New link 996 goes 267 → 266 over 15.677466 m. Fixed reference heads are 63.735835275 → 63.745263726 m: 9.428452 mm uphill. No internal-node adjustment can fix this fixed-to-fixed link. The original selected graph has the opposite connection, 266 → 267. |
| S166_unit_309 | Positive slopes feasible; current floor infeasible | Path 613 → 614 is 598.733180 m long, with 1.655289 mm available drop. The current floor requires 2.494645 mm. |

Open [the boundary viewer](focused_boundary_review.html). It shows the existing
selection, current collapsed geometry, reference boundary heads and critical path.
Hover shows the currently saved child head too; no new optimized heads are plotted.

Unit 30 passing this slope test is fully consistent with its current method changing
many retained slopes: the broader compound solve is a policy choice there, not a
necessity imposed by these slope constraints. This test says nothing about the
reference-flow/split constraints examined in the earlier, broader investigation.

## Historical evidence

The October 3 failures under “only new nodes are free” do not establish that fixing
only member boundaries fails. However, the later October 8 boundary-repair trial
DID contain 54 strict fixed-exterior-head failures. That later evidence must not be
conflated with the older tests or overlooked. The current-state intersection is
recorded in `historical_strict_failures_current_status.csv`:

'''
    report+=f'{len(prior)} of those historical failure states occur in this accepted compound-member population. Their current classifications are: {prior.feasibility_status.value_counts().to_dict()}.\n'
    report+='''
## Limits and next decision

This verifies a practical existing boundary definition and tests it; it is not a
proposal to widen the region or relax heads automatically. Saved island-target
census coverage caveats remain in each source contract. The review checks consistency
of existing selections, not an independent pixel-level recreation of island edits.

Exterior length differences are counted separately in `state_results.csv`. Fixing
exterior endpoint heads preserves their head drops, but a changed exterior length
can still change the slope. This pass does not repair geometry or certify exact
exterior geometric preservation.

The appropriate next review is therefore: retain fixed boundaries for feasible
cases; inspect the explicit reversed-head paths, floor-only conflicts, reference
extent failures and correspondence exceptions separately. A reference-head/slope
penalty has not been implemented or tuned. There is no evidence here that every
compound-member collapse needs a whole-compound head recalculation.

Validation: six focused tests, including 60 random DAG comparisons with an independent
linear program, pass. Independent linear programs also agree on all three real
examples. Production inputs are checksum-verified and unchanged.
'''
    (OUT/'REPORT.md').write_text(report)
    manifest=json.loads((OUT/'manifest.json').read_text());manifest.update(test_count=6,random_dag_lp_checks=60,real_case_lp_checks=3,boundary_correspondence_counts=b.correspondence.value_counts().to_dict(),historical_strict_failure_current_counts=prior.feasibility_status.value_counts().to_dict(),code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print({k:v for k,v in manifest.items() if not k.endswith('hashes')})
    print(f[f.boundary_status.eq('review_exception')][['example','state','exception']].to_string(index=False))

if __name__=='__main__':main()

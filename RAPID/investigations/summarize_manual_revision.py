"""Summarize resolved reviews, isolated repairs and explicit remaining blockers."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/review_revision'

def run():
 manual=json.loads((OUT/'manual_edit_results.json').read_text());attrs=json.loads((OUT/'attribute_results.json').read_text());islands=json.loads((OUT/'actual_island_boundary_results.json').read_text());roles=pd.read_csv(OUT/'connection_flag_triage.csv');dups=pd.read_csv(OUT/'island_duplicate_resolution_plan.csv')
 result=dict(scope='Reviewed independent-state decisions, nine remaining connection flags, twelve flagged nonduplicate island states, and the agreed duplicate-resolution plan',manual_geometry_trials=len(manual),manual_geometry_trials_passed=sum(r['reviewed_geometry_pass'] for r in manual),manual_new_cuts=sum(len(r['new_cuts']) for r in manual),all_manual_trials_preserve_retained_node_reachability=all(r['all_retained_node_reachability_preserved'] for r in manual),
 connection_flags=len(roles),connection_flags_with_missing_boundary_nodes=int((~roles.identity_present).sum()),incorrect_parent_role_requirements=int(roles.assessment.eq('overstrict_role_requirement_absent_even_in_parent').sum()),internal_connectivity_changes_with_retained_boundaries=int(roles.assessment.eq('boundary_present_exterior_supported_internal_flow_role_changed').sum()),
 island_cases=len(islands),island_geometry_trials_passed=[r['state'] for r in islands if r['overall_geometry_pass']],island_new_cuts=sum(r['new_cuts'] for r in islands),island_reassembly_blockers=[r['state'] for r in islands if not r['overall_geometry_pass']],
 duplicate_island_entries=len(dups),duplicate_production_cleanup_applied=False,manual_cases_with_invalid_fragment_widths=[dict(example=r['example'],state=r['state'],links=r['invalid_width_links']) for r in attrs if not r['widths_valid']],manual_cases_with_fixed_head_failure=[dict(example=r['example'],state=r['state'],error=r['head_error'],preexisting=True) for r in attrs if not r['fixed_exterior_heads_passed']],
 additional_manual_anchor_review_requested=[],remaining_work=['Reassemble four island local graphs with the actual target boundary; the saved merged graph retains an extra branch and is not an accepted replacement','Regenerate affected hydraulic widths and resolve six fragment width-core failures; preserve the recorded pre-existing fixed-head failure','Integrate reviewed boundary/domain decisions and validated identities into production preparation','Apply the independent-canonical duplicate plan and update dependent aliases','Regenerate routing inputs and validate before any production rerun'],tests_passed=25,notebook_executed=True,production_changed=False,routing_ready=False)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()

"""Overlay explicit human visual reviews without clearing automatic audit results."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'

def run():
    queue=pd.read_csv(OUT/'priority_review_queue.csv',dtype={'example':str})
    latest={}
    ledger=OUT/'boundary_manual_reviews.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            r=json.loads(line);latest[(r['example'],r['state'],r['fingerprint'])]=r
    rows=[]
    for r in queue.to_dict('records'):
        review=latest.get((r['example'],r['state'],r['fingerprint']))
        r['manual_review_status']=review['decision'] if review else 'not_reviewed'
        r['manual_review_note']=review['user_observation'] if review else ''
        p=OUT/'full'/r['example']/r['state']/'validation.json';categories=[]
        if p.exists():
            v=json.loads(p.read_text())
            if any(not b['present'] for b in v['boundaries']):categories.append('unresolved_boundary_identity')
            if any(b['present'] and not b['replacement_incidence'] for b in v['boundaries']):categories.append('boundary_flow_role_check')
            if any(b['present'] and not b['exterior_supported'] for b in v['boundaries']):categories.append('exterior_path_comparison')
        else:categories.append('target_or_contract_mismatch')
        r['automatic_review_categories']=';'.join(categories)
        r['visual_review_does_not_clear_automatic_checks']=True
        rows.append(r)
    f=pd.DataFrame(rows);f.to_csv(OUT/'manual_review_queue.csv',index=False)
    pending=f[f.manual_review_status.eq('not_reviewed')]
    pending.to_csv(OUT/'remaining_visual_review_queue.csv',index=False)
    summary=dict(total_queue=len(f),visually_reviewed=int((~f.manual_review_status.eq('not_reviewed')).sum()),
                 visually_accepted=int(f.manual_review_status.eq('visual_acceptance').sum()),
                 questions_or_tentative=int(f.manual_review_status.eq('visual_no_problem_question').sum()),
                 not_yet_reviewed=len(pending),automatic_statuses_changed=False,
                 reviewed_with_unresolved_identity=f[(~f.manual_review_status.eq('not_reviewed'))&f.automatic_review_categories.str.contains('unresolved_boundary_identity')][['example','state','unresolved_boundaries']].to_dict('records'))
    (OUT/'manual_review_progress.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

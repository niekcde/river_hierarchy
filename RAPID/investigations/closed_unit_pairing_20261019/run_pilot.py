"""Route the prepared closed-unit pilot with the established event-window solver."""
import argparse,json
from pathlib import Path
import pandas as pd
from pilot import OUT,ROOT,PLAN,sha,save
from rapid_tools import event_window_batch as batch

def make_plan():
    prepared=json.loads((OUT/'preparation_manifest.json').read_text())
    batch.verify(prepared['sources']);batch.verify(prepared['code_sha256'])
    old=json.loads(PLAN.read_text());jobs=json.loads((OUT/'prepared_jobs.json').read_text());cases=[]
    for ex in sorted({j['example'] for j in jobs}):
        case=dict(next(c for c in old['cases'] if c['example']==ex));oldjobs={j['state_id']:j for j in case['jobs']}
        oldreceipt=ROOT/f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json';done=json.loads(oldreceipt.read_text())
        case['tail_hours']=done['final_tail_hours'];case['simulation_end_utc']=(pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30,hours=case['tail_hours'])).isoformat()
        case['jobs']=[oldjobs['S000_base'],oldjobs['single_centerline']]
        for j in jobs:
            if j['example']==ex:
                case['jobs'].append(dict(state_id=j['state'],parameters=j['parameters'],native_dt_seconds=j['native_dt_seconds'],aliases=['closed_unit_pilot/'+j['state']]))
        case['source_sha256']={p:sha(p) for p in [case['forcing'],str(oldreceipt),*[j['parameters'] for j in case['jobs']]]}
        case['pilots']=[];cases.append(case)
    doc=dict(schema='reviewed-event-window-batch-v2',status='isolated_closed_unit_pilot',code_sha256=batch.code_sources(),source_sha256={str(OUT/'preparation_manifest.json'):sha(OUT/'preparation_manifest.json'),str(OUT/'prepared_jobs.json'):sha(OUT/'prepared_jobs.json'),str(Path(__file__)):sha(__file__)},cases=cases)
    path=OUT/'routing_plan.json'
    if path.exists() and json.loads(path.read_text())!=doc:raise ValueError('Plan differs; use a new run version')
    save(path,doc);return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args();path=make_plan();print(path,flush=True)
    if args.run:batch.run(path,OUT/'runs')

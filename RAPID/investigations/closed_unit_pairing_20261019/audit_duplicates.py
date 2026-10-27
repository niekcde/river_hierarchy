"""Check exact normalized hydraulic graphs against existing states and pilot peers."""
import json,hashlib
import pandas as pd
from pilot import OUT,PLAN,sha,save

def signature(frame):
    cols=['u','v','link_length_m','width','depth_m','slope','head_us','head_ds']
    table=frame[cols].astype(float).sort_values(cols).reset_index(drop=True)
    return hashlib.sha256(table.to_json(orient='values',double_precision=12).encode()).hexdigest()

def main():
    plan=json.loads(PLAN.read_text());jobs=json.loads((OUT/'prepared_jobs.json').read_text());rows=[];sources={};known={}
    for ex in sorted({j['example'] for j in jobs}):
        case=next(c for c in plan['cases'] if c['example']==ex)
        for j in case['jobs']:
            sources[j['parameters']]=sha(j['parameters']);assert sources[j['parameters']]==case['source_sha256'][j['parameters']]
            key=(ex,signature(pd.read_parquet(j['parameters'])));known.setdefault(key,[]).append(j['state_id'])
    for j in jobs:
        key=(j['example'],signature(pd.read_parquet(j['parameters'])));matches=known.get(key,[])
        if j['kind']!='single_control' and matches:raise ValueError('New intervention duplicates existing graph: '+str(matches))
        rows.append(dict(example=j['example'],state=j['state'],kind=j['kind'],hydraulic_graph_sha256=key[1],matches=' | '.join(matches)))
        known.setdefault(key,[]).append('pilot/'+j['state'])
    pd.DataFrame(rows).to_csv(OUT/'duplicate_audit.csv',index=False)
    save(OUT/'duplicate_audit_manifest.json',dict(sources=sources,code_sha256=sha(__file__),convention='Stable routed endpoint IDs and physical section properties, sorted independent of reach IDs; JSON 12-digit precision. Controls may reproduce originals; new pair/all-block variants must not.'))
    print('Six pairs and three segmented variants are distinct from original graphs and each other; controls audited separately.')
if __name__=='__main__':main()

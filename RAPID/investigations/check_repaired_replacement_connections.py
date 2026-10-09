"""Final reachability and exterior-multiplicity audit, including cached trials."""
from pathlib import Path
import sys,json
from functools import lru_cache
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from network_variants.boundary_contract_repair import replacement_connections
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/full'
@lru_cache(maxsize=20)
def base(path):return pd.read_parquet(path,columns=['id_link','id_us_node','id_ds_node'])

def main():
    reg=pd.read_csv(ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv',dtype={'example':str}).set_index(['example','state'])
    rows=[]
    for path in sorted(OUT.glob('*/*/validation.json')):
        ex,state=path.parent.parent.name,path.parent.name;row=reg.loc[(ex,state)];v=json.loads(path.read_text());contract=json.loads((path.parent/'contract.json').read_text())
        parent=base(str(Path(row.frame_path).with_name(row.parent_fingerprint+'.parquet')))
        trial=path.parent/'trial_links.parquet'
        frame=pd.read_parquet(trial if trial.exists() else row.frame_path,columns=['id_link','id_us_node','id_ds_node','u','v'])
        pairs=pd.concat([frame[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'p'}),frame[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'p'})]).drop_duplicates()
        mapping=dict(zip(pairs.raw,pairs.p));used={}
        for r in v['exterior']:
            if r['path_unique']:
                for c in r['child_links']:used.setdefault(c,[]).append(r['parent_link'])
        collisions={int(c):ps for c,ps in used.items() if len(set(ps))>1}
        connections=replacement_connections(parent,frame,mapping,contract,set(used))
        okay=not collisions and all(r['expected']==r['actual'] for r in connections)
        v['exterior_path_reuse']=collisions;v['replacement_connections']=connections
        v['replacement_connections_preserved']=okay
        v['anchor_pass']=not collisions and all(b['present'] and b['exterior_supported'] and b['replacement_incidence'] for b in v['boundaries'])
        path.write_text(json.dumps(v,indent=2))
        result_path=path.parent/'result.json';result=json.loads(result_path.read_text())
        result['replacement_connections_pass']=okay
        result['replacement_connectivity_review']=not okay
        result['anchor_pass']=v['anchor_pass']
        changed=result['new_cuts']+result['restored_identities']>0
        result['status']=('repaired_anchor_pass' if changed else 'unchanged_anchor_pass') if v['anchor_pass'] else ('partial_repair_unresolved' if changed else 'unresolved')
        result['reason']='exterior_multiplicity_failure' if collisions else ''
        result_path.write_text(json.dumps(result,indent=2))
        rows.append(dict(example=ex,state=state,pass_check=okay,exterior_reuse=bool(collisions),missing_pairs=sum(r['expected'] and not r['actual'] for r in connections),unexpected_pairs=sum(r['actual'] and not r['expected'] for r in connections)))
    pd.DataFrame(rows).to_csv(OUT.parent/'replacement_connection_checks.csv',index=False)
    f=pd.DataFrame(rows);print(len(f),'states checked; parent/child reachability differences',int((~f.pass_check).sum()));print(f[~f.pass_check].groupby('example').size().to_string())
if __name__=='__main__':main()

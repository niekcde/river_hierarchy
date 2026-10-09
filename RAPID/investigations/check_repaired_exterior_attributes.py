"""Check exact-geometry exterior paths against saved parent hydraulics."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/full'
@lru_cache(maxsize=20)
def parent(path):return pd.read_parquet(path).set_index('id_link')

def main():
    reg=pd.read_csv(ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv',dtype={'example':str}).set_index(['example','state']);rows=[]
    for path in OUT.glob('*/*/trial_fragment_attributes.parquet'):
        ex,state=path.parent.parent.name,path.parent.name;r=reg.loc[(ex,state)]
        p=parent(str(Path(r.frame_path).with_name(r.parent_fingerprint+'.parquet')));f=pd.read_parquet(path).set_index('id_link')
        v=json.loads((path.parent/'validation.json').read_text());links=[];endpoints=[]
        for a in v['exterior']:
            if not a['geometry_exact']:continue
            links.extend((lid,a['parent_link']) for lid in a['child_links'])
            endpoints.append((a['child_links'][0],a['child_links'][-1],a['parent_link']))
        if not links:continue
        child_ids,parent_ids=zip(*links);c=f.loc[list(child_ids)];q=p.loc[list(parent_ids)];failures={}
        for field in ['width','depth_m','slope']:
            failures[field]=int((~np.isclose(c[field].to_numpy(),q[field].to_numpy(),rtol=1e-8,atol=1e-9)).sum())
        first,last,ps=zip(*endpoints)
        failures['upstream_head']=int((~np.isclose(f.loc[list(first),'head_us'].to_numpy(),p.loc[list(ps),'head_us'].to_numpy(),rtol=0,atol=1e-8)).sum())
        failures['downstream_head']=int((~np.isclose(f.loc[list(last),'head_ds'].to_numpy(),p.loc[list(ps),'head_ds'].to_numpy(),rtol=0,atol=1e-8)).sum())
        result=dict(example=ex,state=state,checked_exterior_paths=len(endpoints),all_preserved=not any(failures.values()),**failures)
        (path.parent/'exterior_attributes_check.json').write_text(json.dumps(result,indent=2));rows.append(result)
    f=pd.DataFrame(rows);f.to_csv(OUT.parent/'exterior_attribute_checks.csv',index=False)
    print(len(f),'derived states checked;',int((~f.all_preserved).sum()),'with exterior attribute differences')
    print(f[~f.all_preserved].head(12).to_string(index=False))
if __name__=='__main__':main()

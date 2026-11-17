"""Check that replacement interfaces introduce no undeclared exterior anchors."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from network_variants.reviewed_geometry import ReviewedGeometry
OUT=ROOT/'RAPID/outputs/member_boundary_review_20261027_complete'

def main():
    rows=json.loads((OUT/'state_results.json').read_text());r=ReviewedGeometry();cache={};checks=[]
    for row in rows:
        if row['feasibility_status'].startswith('not_tested'):continue
        ex=str(row['example']).zfill(2);st=row['state'];record=r.lookup(ex,'indep',st)
        if ex not in cache:cache[ex]=pd.read_parquet(r.lookup(ex,'indep','S000_base')['original_frame'])
        p=cache[ex];c=json.loads(Path(record['contract_path']).read_text());f=pd.read_parquet(record['original_frame']);ext=p[~p.id_link.isin(c['selected_parent_links'])];extnodes=set(ext.u)|set(ext.v)
        pathfile=Path(record.get('source_bundle') or '')/'exterior_paths.json'
        if record.get('source_bundle') and pathfile.exists():outside={lid for path in json.loads(pathfile.read_text()) for lid in path['child_links']}
        else:
            bypair={}
            for z in f.itertuples():bypair.setdefault((z.u,z.v),[]).append((z.id_link,z.link_length_m))
            outside=set()
            for z in ext.itertuples():
                hits=[lid for lid,length in bypair.get((z.u,z.v),[]) if np.isclose(length,z.link_length_m,rtol=1e-8,atol=1e-5)]
                outside.add(z.id_link if z.id_link in hits else hits[0])
        patch=f[~f.id_link.isin(outside)];interfaces=(set(patch.u)|set(patch.v))&extnodes;declared=set(c['boundary_nodes'])
        extra=interfaces-declared;missing=declared-(set(patch.u)|set(patch.v))
        checks.append(dict(example=ex,state=st,unexpected_exterior_interface_nodes=sorted(map(int,extra)),missing_declared_boundary_nodes=sorted(map(int,missing))))
        if extra or missing:
            row['ungated_feasibility_status']=row['feasibility_status'];row['feasibility_status']='not_tested_interface_scope_exception'
            row['unexpected_exterior_interface_nodes']=sorted(map(int,extra));row['positive_slopes_feasible']=None;row['current_relative_floor_feasible']=None
    (OUT/'replacement_interface_checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'state_results.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(OUT/'state_results.csv',index=False)
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['feasibility_status']=pd.Series([z['feasibility_status'] for z in rows]).value_counts().to_dict();manifest['replacement_interface_checks']=len(checks);manifest['unexpected_interface_cases']=sum(bool(z['unexpected_exterior_interface_nodes'] or z['missing_declared_boundary_nodes']) for z in checks);(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Checked',len(checks),'unexpected',manifest['unexpected_interface_cases'])

if __name__=='__main__':main()

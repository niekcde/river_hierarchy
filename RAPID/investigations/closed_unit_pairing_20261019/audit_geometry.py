"""Persist geometric anchor/length checks for selected and excluded pilot networks."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pilot import OUT,PLAN,sha,save

def main():
    plan=json.loads(PLAN.read_text());blocks=json.loads((OUT/'eligible_blocks.json').read_text());rows=[];sources={str(PLAN):sha(PLAN)}
    for ex in ['03','13','22','55']:
        case=next(c for c in plan['cases'] if c['example']==ex);jobs={j['state_id']:j for j in case['jobs']}
        pp=Path(jobs['S000_base']['parameters']);pgpath=pp.parents[3]/'variant/widths/links_with_width_families.gpkg';pa=pd.read_parquet(pp);pg=gpd.read_file(pgpath).set_index('id_link');sources[str(pp)]=sha(pp);sources[str(pgpath)]=sha(pgpath);points={}
        for r in pa.itertuples():
            line=pg.loc[r.id_link].geometry;np.testing.assert_allclose(line.length,r.link_length_m,atol=1e-6,rtol=1e-9)
            points[int(r.u)]=Point(line.coords[0]);points[int(r.v)]=Point(line.coords[-1])
        for b in blocks[ex]:
            cp=Path(jobs[b['state']]['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg';cg=gpd.read_file(cp).set_index('id_link');sources[str(cp)]=sha(cp);line=cg.loc[b['child_id']].geometry
            start=Point(line.coords[0]).distance(points[b['source']]);end=Point(line.coords[-1]).distance(points[b['sink']]);matches=max(start,end)<=1e-5
            rows.append(dict(example=ex,state=b['state'],inlet_offset_m=start,outlet_offset_m=end,anchors_match=matches,network_used=ex!='13'))
            if ex!='13' and not matches:raise ValueError('Selected geometry anchors mismatch')
    results=pd.DataFrame(rows);results.to_csv(OUT/'geometric_anchor_audit.csv',index=False)
    lengths=[]
    for j in json.loads((OUT/'prepared_jobs.json').read_text()):
        dest=Path(j['parameters']).parent;f=pd.read_parquet(dest/'parameters.parquet');g=gpd.read_file(dest/'links.gpkg').set_index('id_link')
        mismatch=np.abs(np.array([g.loc[i].geometry.length for i in f.id_link])-f.link_length_m.to_numpy())
        np.testing.assert_allclose([g.loc[i].geometry.length for i in f.id_link],f.link_length_m,rtol=1e-9,atol=1e-6)
        lengths.append(dict(example=j['example'],state=j['state'],max_length_difference_m=float(mismatch.max()),links=len(f)))
    pd.DataFrame(lengths).to_csv(OUT/'geometry_length_audit.csv',index=False)
    save(OUT/'geometry_audit_manifest.json',dict(sources=sources,code_sha256=sha(__file__),rejected_network='13',reason='Saved section endpoint coordinates do not all match original parent boundary anchors. No snapping performed.'))
    print(results.groupby('example').agg(blocks=('state','size'),matching=('anchors_match','sum')).to_string())
if __name__=='__main__':main()

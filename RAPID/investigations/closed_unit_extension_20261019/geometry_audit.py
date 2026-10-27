"""Read-only audit of all existing outer-block sections before composition."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/investigations/closed_unit_pairing_20261019')]
import pilot
OUT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=pilot.read(pilot.PLAN);blocks=pilot.read(pilot.OUT/'eligible_blocks.json');rows=[]
    for case in plan['cases']:
        ex=case['example'];jobs={j['state_id']:j for j in case['jobs']};p=pilot.read(jobs['S000_base']['parameters']);heads=pilot.head_map(p)
        pg=pilot.geometry_frame(Path(jobs['S000_base']['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg').set_index('id_link');coords={}
        for r in p.itertuples():
            line=pg.loc[r.id_link].geometry
            for node,pt in [(r.u,Point(line.coords[0])),(r.v,Point(line.coords[-1]))]:
                if node in coords:assert coords[node].distance(pt)<1e-5
                coords[node]=pt
        for b in blocks[ex]:
            state=b['state'];f=pilot.read(jobs[state]['parameters']);r=f[f.id_link==b['child_id']].iloc[0];g=pilot.geometry_frame(Path(jobs[state]['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg').set_index('id_link');line=g.loc[b['child_id']].geometry
            rows.append(dict(example=ex,state=state,length_m=r.link_length_m,inlet_offset_m=Point(line.coords[0]).distance(coords[b['source']]),outlet_offset_m=Point(line.coords[-1]).distance(coords[b['sink']]),head_us_difference=r.head_us-heads[b['source']],head_ds_difference=r.head_ds-heads[b['sink']],celerity=r.rapid_celerity_mps,count=int(r['count'])))
        print(ex,'audited',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'source_anchor_audit.csv',index=False);pilot.save(OUT/'audit_sources.json',pilot.SOURCES)
    print(d.groupby('example').agg(max_inlet=('inlet_offset_m','max'),max_outlet=('outlet_offset_m','max'),max_head_us=('head_us_difference',lambda x:abs(x).max()),max_head_ds=('head_ds_difference',lambda x:abs(x).max())).to_string())
if __name__=='__main__':main()

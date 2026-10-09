"""Read-only applicability test of exact-on-line cuts across the full unit audit.

Mechanical applicability is not proof of intended anchor correspondence.
No trial geometry or production file is saved/replaced by this scan.
"""
from pathlib import Path
import sys,json
import pandas as pd
from shapely.geometry import Point
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools import collapse_creation_review as r
from rapid_tools.anchor_repair_trial import repair,node_mapping,topology
OUT=r.ROOT/'RAPID/outputs/anchor_repair_trial_20261008/full_scan'
REVIEWED={('41','S016_unit_4',5),('41','S016_unit_4',7),('02','S023_unit_12',27),('02','S026_unit_16',32),('13','S039_unit_33',64),('15','S135_unit_144',281)}

def run():
    registry,b,q=r.load_review_data()
    b=b[b.family.isin(['indep','compound_bubbles'])]
    candidate=b[b.nearest_child_line_distance_m.le(1e-5)&b.nearest_child_node_distance_m.gt(1e-5)]
    details=[];states=[]
    for (ex,state,fp),group in candidate.groupby(['example','state','fingerprint']):
        row=registry[(registry.example==ex)&(registry.state==state)].iloc[0]
        try:
            frame=pd.read_parquet(row.frame_path);mapping=node_mapping(frame)
            links=r.vectors(row.child_links_path,'links')[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node']],on='id_link',validate='one_to_one')
            nodes=r.vectors(row.child_nodes_path,'nodes');pn=r.vectors(row.base_nodes_path,'nodes')
            pp=pn.set_index('id_node').geometry;valid=[];entries=[]
            for a in group.itertuples():
                point=pp.loc[a.parent_node_id]
                hits=[int(z.id_link) for z in links.itertuples() if z.geometry.distance(point)<=1e-5 and 1e-5<z.geometry.project(point)<z.geometry.length-1e-5]
                status='unique_interior_cut' if len(hits)==1 else 'ambiguous_or_no_interior_cut'
                if a.parent_node_id in mapping.values():status='identity_already_present_elsewhere'
                if status=='unique_interior_cut':valid.append(int(a.parent_node_id))
                entries.append(dict(example=ex,state=state,fingerprint=fp,parent_node=int(a.parent_node_id),status=status,candidate_links=','.join(map(str,hits)),nearest_existing_node_m=a.nearest_child_node_distance_m,reviewed=(ex,state,int(a.parent_node_id)) in REVIEWED))
            if valid:
                after,an,actions=repair(links,nodes,pn,mapping,r.int_ids(row.required_boundary_nodes),cuts=valid)
                before_top,after_top=topology(links),topology(after)
                for k in ['dag','components','cycle_rank','inlets','outlets']:assert before_top[k]==after_top[k],k
                assert abs(links.length.sum()-after.length.sum())<1e-5
                for lid,g in after.groupby('source_link'):
                    old=links[links.id_link==lid].geometry.iloc[0]
                    assert abs(old.length-g.length.sum())<1e-5
                    assert old.hausdorff_distance(g.geometry.union_all())<1e-5
                remaining=set(r.int_ids(row.required_boundary_nodes))-set(an.parent_identity)
            else:remaining=set(r.int_ids(row.required_boundary_nodes))-set(mapping.values())
            details.extend(entries)
            states.append(dict(example=ex,state=state,fingerprint=fp,mechanical_test_pass=True,n_insertable_cuts=len(valid),remaining_boundary_identities=','.join(map(str,sorted(remaining))),error=''))
        except Exception as error:
            details.extend([dict(example=ex,state=state,fingerprint=fp,parent_node=int(a.parent_node_id),status='test_failed',error=str(error),reviewed=(ex,state,int(a.parent_node_id)) in REVIEWED) for a in group.itertuples()])
            states.append(dict(example=ex,state=state,fingerprint=fp,mechanical_test_pass=False,n_insertable_cuts=0,error=str(error)))
        if len(states)%20==0:print('Scanned',len(states),'state entries',flush=True)
    OUT.mkdir(exist_ok=True,parents=True)
    d=pd.DataFrame(details);s=pd.DataFrame(states)
    d.to_csv(OUT/'boundary_applicability.csv',index=False);s.to_csv(OUT/'state_applicability.csv',index=False)
    unique=d.drop_duplicates(['example','fingerprint','parent_node']);physical=s.drop_duplicates(['example','fingerprint'])
    summary=dict(scope='Independent and compound outputs; mechanical dry-run, not validated production repairs',all_outputs=len(q),classification=q.review_class.value_counts().to_dict(),tested_logical_states=len(s),tested_distinct_outputs=len(physical),tested_boundaries=len(unique),boundary_status=unique.status.value_counts().to_dict(),failed_state_tests=int((~s.mechanical_test_pass).sum()),insertable_nearest_node_under_30m=int(((unique.status=='unique_interior_cut')&unique.nearest_existing_node_m.lt(30)).sum()),production_mutations=0)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

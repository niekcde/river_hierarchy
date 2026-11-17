"""Deduplicate repeated reach geometries; audit full junction context."""
import json,hashlib
import numpy as np
import pandas as pd
from review import SOURCE,DEST,RUNS

def key(ex,r):
 # Exclude depth, flow, slope and numerical subdivisions: these can vary across
 # states for the same geometrical reach. IDs alone are reused for replacements.
 return (ex,int(r.u),int(r.v),round(float(r.link_length_m),6),round(float(r.width),6))
def main():
 unique={};junctions={};parents=set()
 flags=pd.read_parquet(DEST/'flagged_depth_connections.parquet')
 flags=flags[(flags.abs_depth_jump_m>5)&(flags.depth_ratio>2)]
 candidates={(str(ex),state):g for (ex,state),g in flags.groupby(['example','state'])}
 for exdir in sorted((SOURCE/RUNS[2]/'states').iterdir()):
  ex=exdir.name
  for folder in sorted(exdir.iterdir()):
   if folder.name=='single_centerline':continue
   f=pd.read_parquet(folder/'parameters.parquet');lookup={}
   geometry_keys=[key(ex,r) for r in f.itertuples()]
   assert len(set(geometry_keys))==len(f), 'Simultaneous identical parallel geometries need separate identities'
   for r in f.itertuples():
    k=key(ex,r);lookup[int(r.id_link)]=k
    if folder.name=='S000_base':parents.add(k)
    if k not in unique:
     unique[k]=dict(example=ex,reach_id=int(r.id_link),u=int(r.u),v=int(r.v),length_m=r.link_length_m,width_m=r.width,
       depth_min_m=r.depth_m,depth_max_m=r.depth_m,depth_max_state=folder.name,reference_depth_m=np.nan,appearances=0,reach_id_aliases=set())
    a=unique[k];a['reach_id_aliases'].add(int(r.id_link));a['appearances']+=1;a['depth_min_m']=min(a['depth_min_m'],r.depth_m)
    if r.depth_m>a['depth_max_m']:a['depth_max_m']=r.depth_m;a['depth_max_state']=folder.name
    if folder.name=='S000_base':a['reference_depth_m']=r.depth_m;a['reach_id']=int(r.id_link)
   incoming={n:g for n,g in f.groupby('v')};outgoing={n:g for n,g in f.groupby('u')}
   candidate=candidates.get((ex,folder.name))
   if candidate is None:continue
   for node in candidate.node.unique():
    ins=incoming[node];outs=outgoing[node]
    context_in=[dict(reach_id=int(r.id_link),depth_m=float(r.depth_m),width_m=float(r.width),q_cms=float(r.reference_discharge_cms)) for r in ins.itertuples()]
    context_out=[dict(reach_id=int(r.id_link),depth_m=float(r.depth_m),width_m=float(r.width),q_cms=float(r.reference_discharge_cms)) for r in outs.itertuples()]
    for a in ins.itertuples():
     for b in outs.itertuples():
      jump=abs(b.depth_m-a.depth_m);ratio=max(a.depth_m,b.depth_m)/min(a.depth_m,b.depth_m)
      if jump<=5 or ratio<=2:continue
      pair=(lookup[int(a.id_link)],lookup[int(b.id_link)])
      match_out=outs[(outs.depth_m/a.depth_m).between(.8,1.25)]
      # Discharge share of outgoing branches whose depths match the incoming one.
      share=float(match_out.reference_discharge_cms.sum()/outs.reference_discharge_cms.sum())
      min_gap=float(abs(outs.depth_m-a.depth_m).min())
      total_in=float(ins.reference_discharge_cms.sum());total_out=float(outs.reference_discharge_cms.sum())
      if pair not in junctions or jump>junctions[pair]['depth_jump_m']:
       junctions[pair]=dict(example=ex,node=int(node),up_reach=int(a.id_link),down_reach=int(b.id_link),
        up_depth_m=a.depth_m,down_depth_m=b.depth_m,depth_jump_m=jump,depth_ratio=ratio,
        up_width_m=a.width,down_width_m=b.width,width_change_m=b.width-a.width,
        up_q_cms=a.reference_discharge_cms,down_q_cms=b.reference_discharge_cms,
        matching_outgoing_flow_share=share,closest_outgoing_depth_gap_m=min_gap,
        incoming_count=len(ins),outgoing_count=len(outs),up_flow_share=float(a.reference_discharge_cms/total_in),down_flow_share=float(b.reference_discharge_cms/total_out),
        example_state=folder.name,all_incoming=json.dumps(context_in),all_outgoing=json.dumps(context_out),
        up_key=repr(pair[0]),down_key=repr(pair[1]))
  print('Unique reach audit',ex,flush=True)
 rows=[]
 for k,a in unique.items():
  a['reach_id_aliases']='|'.join(map(str,sorted(a['reach_id_aliases'])));a['in_reference']=k in parents;a['reach_key']=repr(k);h=a['depth_max_m'];w=a['width_m']
  a['hydraulic_radius_at_max_depth_m']=w*h/(w+2*h)
  a['radius_overestimate_pct']=100*(h/a['hydraulic_radius_at_max_depth_m']-1)
  a['discharge_overestimate_pct']=100*((h/a['hydraulic_radius_at_max_depth_m'])**(2/3)-1)
  rows.append(a)
 d=pd.DataFrame(rows);d.to_csv(DEST/'distinct_reaches_depth_radius.csv',index=False)
 counts=[]
 for threshold in [5,10,20]:
  counts.append(dict(depth_above_m=threshold,reference_reaches_at_reference_depth=int((d.reference_depth_m>threshold).sum()),
   distinct_reaches_across_all_configurations=int((d.depth_max_m>threshold).sum()),
   reference_geometries_at_any_config_depth=int(((d.depth_max_m>threshold)&d.in_reference).sum()),
   additional_collapse_geometries=int(((d.depth_max_m>threshold)&~d.in_reference).sum())))
 pd.DataFrame(counts).to_csv(DEST/'distinct_reach_depth_counts.csv',index=False)
 d[d.discharge_overestimate_pct>10].sort_values('discharge_overestimate_pct',ascending=False).to_csv(DEST/'distinct_reach_radius_flags.csv',index=False)
 j=pd.DataFrame(junctions.values()).sort_values('depth_jump_m',ascending=False);j.to_csv(DEST/'distinct_connection_jumps.csv',index=False)
 print(pd.DataFrame(counts).to_string(index=False),flush=True)
 print('unique geometries',len(d),'reference geometries',d.in_reference.sum(),'Qerror>10%',d.discharge_overestimate_pct.gt(10).sum(),'Qerror>20%',d.discharge_overestimate_pct.gt(20).sum(),flush=True)
 print(j[['example','up_reach','down_reach','up_depth_m','down_depth_m','up_width_m','down_width_m','incoming_count','outgoing_count','matching_outgoing_flow_share']].head(12).to_string(index=False),flush=True)
 (DEST/'reach_deduplication.json').write_text(json.dumps(dict(key='example, endpoints, length and width rounded to 1 micrometre. Reach IDs are aliases and are not part of identity; simultaneous duplicate geometries within each state are rejected. Depth, discharge, slope, state and computational cell count excluded.',
 meaning='Distinct reach geometries, not unique physical river locations. Alternative collapsed geometries remain distinct; repeated appearances are counted once. Maximum depth across configurations used for any-configuration flags.',
 radius_flag='Illustrative >10% discharge overestimate from wide-rectangular approximation at the same depth, n, slope; full rectangular R=B*h/(B+2h). Not proof depth is physically impossible.',
 jump_flag='Adjacent difference >5 m AND depth ratio >2. Repeated geometry pairs merged, keeping maximum jump and full junction context from that configuration.',
 similar_depth='Outgoing depth within incoming depth /1.25 to incoming depth *1.25. Flow share is a contextual screen; does not prove water-surface continuity.',
 script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()

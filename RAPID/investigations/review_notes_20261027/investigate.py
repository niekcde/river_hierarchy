from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/frozen_floor_review_20261027')]
from check import release,frame,js,outside_paths
import pyogrio,numpy as np
from shapely.ops import unary_union
OUT=ROOT/'RAPID/outputs/review_notes_20261027';OUT.mkdir(exist_ok=False)
reviews=js('/Users/6256481/Downloads/boundary-review-decisions.json');evidence=js(ROOT/'RAPID/outputs/frozen_floor_review_20261027/exception_review_evidence.json');ev={v['key']:v for v in evidence};results=[];cache={}
focus={'02':[31,32,55],'15/S113_unit_189':[384,385,386],'15/S185_unit_30':[420],'22/S152_unit_12':[21,22],'22/S244_unit_262':[517,522],'22/S135_unit_211':[418,419],'22/S262_unit_287':[573,575,574,589],'22/S299_unit_14':[662],'37/S252_unit_201':[395,396],'37/S051_unit_256':[509,510,651,519],'37/S289_unit_143':[296],'41':[274],'55':[6,11]}
for key,review in reviews.items():
 ex,state=key.split('/');r=release.lookup(ex,'indep',state);c=js(r['contract_path']);f=frame(r['original_frame']);cg=pyogrio.read_dataframe(r['links']).set_index('id_link');cn=pyogrio.read_dataframe(r['nodes']).set_index('id_node')
 if ex not in cache:
  br=release.lookup(ex,'indep','S000_base');cache[ex]=(frame(br['original_frame']),pyogrio.read_dataframe(br['links']).set_index('id_link'),pyogrio.read_dataframe(br['nodes']).set_index('id_node'))
 p,pg,pn=cache[ex];pi=p.set_index('id_link');fi=f.set_index('id_link');selected=set(c['selected_parent_links']);paths=outside_paths(r,p,f,selected)
 def mapping(fr):
  return dict(zip(fr.id_us_node,fr.u))|dict(zip(fr.id_ds_node,fr.v))
 mp=mapping(f);pm=mapping(p);inverse={v:k for k,v in mp.items()};pinv={v:k for k,v in pm.items()}
 nodes=focus.get(key,focus.get(ex,[]));bad={int(x) for x in (ev[key]['details'].get('path_records') or []) for x in [x['parent_link']]}
 if not bad:bad={int(q.id_link) for q in p.itertuples() if q.id_link not in selected and (q.u in nodes or q.v in nodes)}
 details=[]
 for lid in sorted(bad):
  g=pg.loc[lid].geometry;pr=pi.loc[lid];dist=cg.geometry.map(g.hausdorff_distance);nearest=[]
  for cid in dist.nsmallest(3).index:
   q=fi.loc[cid];gg=cg.loc[cid].geometry
   nearest.append(dict(link=int(cid),u=int(q.u),v=int(q.v),length=float(q.link_length_m),hausdorff=float(dist.loc[cid]),length_difference=float(gg.length-g.length),simple=bool(gg.is_simple),source_link=str(cg.loc[cid].get('source_link',''))))
  details.append(dict(parent_link=lid,u=int(pr.u),v=int(pr.v),length=float(pr.link_length_m),simple=bool(g.is_simple),nearest=nearest))
 nodedata=[]
 for node in nodes:
  rr=dict(node=node,parent_present=node in pinv,child_present=node in inverse,saved_boundary=node in c['boundary_nodes'])
  for label,fr,ng,inv in [('parent',p,pn,pinv),('child',f,cn,inverse)]:
   rr[label+'_incident']=fr.loc[fr.u.eq(node)|fr.v.eq(node),['id_link','u','v','link_length_m']].to_dict('records')
   if node in inv:
    pt=ng.loc[inv[node]].geometry;rr[label+'_xy']=[pt.x,pt.y];rr[label+'_raw']=int(inv[node])
  if node in inverse:
   pt=cn.loc[inverse[node]].geometry;rr['coincident_child_nodes']=[int(mp[i]) for i,g in cn.geometry.items() if i!=inverse[node] and g.distance(pt)<1e-5]
  nodedata.append(rr)
 history={}
 for folder in [r.get('source_bundle'),r.get('review_folder')]:
  if not folder:continue
  for name in ['repair_proposal.json','result.json','fragment_result.json']:
   path=Path(folder)/name
   if path.exists():history[str(path)]=js(path)
 results.append(dict(key=key,review=review,selected=sorted(selected),boundaries=c['boundary_nodes'],manual={k:v for k,v in c.items() if k.startswith('reviewed_')},exterior=details,nodes=nodedata,history=history,record=r))
 print(key,'selected',sorted(selected),'boundaries',c['boundary_nodes'],flush=True)
(OUT/'evidence.json').write_text(json.dumps(results,indent=2,default=str));(OUT/'user_review.json').write_text(json.dumps(reviews,indent=2))

from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from network_variants.reviewed_geometry import active_geometry
release=active_geometry()
cases=[('02','S023_unit_12',[27],[],{}),('02','S026_unit_16',[32],[],{}),('13','S039_unit_33',[64,66],[],{64:66}),('15','S135_unit_144',[281],[],{}),('02','S004_unit_5',[7],[],{58:7}),('02','S005_unit_25',[50],[],{56:50}),('22','S262_unit_287',[573,575],[574,589],{}),('37','S067_unit_303',[598],[],{653:598}),('37','S289_unit_143',[277,278],[296],{}),('41','S061_unit_57',[110,114,127,128],[112,115],{})]
results=[]
for ex,state,present,absent,mapping in cases:
 links,nodes,contract,_,_=release.load(ex,'indep',state)
 mp=dict(zip(nodes.id_node,nodes.parent_identity))
 assert set(present)<=set(mp.values()),(ex,state,'missing')
 assert not set(absent)&set(mp.values()),(ex,state,'not removed')
 assert all(mp[k]==v for k,v in mapping.items()),(ex,state,'mapping')
 results.append(dict(example=ex,state=state,present=present,absent=absent,mapping=mapping,passed=True))
p=release.manifest.parent/'completeness_audit/manual_decision_checks.json';p.write_text(json.dumps(results,indent=2)+'\n');print('All 10 explicit cut/mapping/domain decision checks pass.')

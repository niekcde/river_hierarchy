"""Three size definitions, fixed forest, whole-network holdouts; no final refit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import augmentation as aug
from geometry_audit import OUT,pilot
b=aug.baseline
DEST=OUT/'size_comparison'

def prepare():
 d=pd.read_parquet(OUT/'augmentation/input.parquet');meta=pd.read_parquet(pilot.SCOPES/'geometry_responses.parquet');graphs=pd.read_csv(pilot.SCOPES/'domain_graphs.csv',dtype={'example':str});plan=pilot.read(pilot.PLAN);audit=[];cache={}
 for case in plan['cases']:
  ex=case['example'];job=next(j for j in case['jobs'] if j['state_id']=='S000_base');membership=pilot.read(Path(job['parameters']).parents[3]/'variant/summary/unit_link_membership.csv')
  for row in d[(d.example==ex)&(d.cohort=='original')].itertuples():
   m=meta[(meta.example==ex)&(meta.state==row.state)];assert len(m)==1;m=m.iloc[0]
   selected=[int(x) for x in str(m.selected_unit_ids).split(',')];ids=set(membership[membership.unit_id.isin(selected)].id_link.astype(int));assert ids
   g=graphs[(graphs.example==ex)&(graphs.state==row.state)&(graphs.scope==m.B_scope)];assert len(g)==1;key=g.iloc[0].parent_graph
   if key not in cache:
    frame=pilot.read(pilot.SCOPES/'graphs'/f'{key}.parquet');cache[key]=pilot.allocation_tables(frame)[1]
   edges=cache[key];assert ids<=set(edges.id_link),(ex,row.state,'membership outside B')
   total=edges.mean_length_contribution_m.sum()/1000;edited=edges.loc[edges.id_link.isin(ids),'mean_length_contribution_m'].sum()/1000
   assert np.isclose(total,row.domain_size,rtol=1e-10,atol=1e-10),(ex,row.state,total,row.domain_size)
   assert 0<edited<=total+1e-10
   audit.append(dict(example=ex,state=row.state,domain_size_current=total,domain_size_edited=edited,edited_fraction=edited/total,selected_links=len(ids),domain_links=len(edges),scope=m.B_scope))
 a=pd.DataFrame(audit);d=d.merge(a[['example','state','domain_size_edited']],how='left',on=['example','state'],validate='one_to_one');mask=d.cohort.ne('original');d.loc[mask,'domain_size_edited']=d.loc[mask,'domain_size'];assert d.domain_size_edited.notna().all();a.to_csv(DEST/'footprint_audit.csv',index=False);d.to_parquet(DEST/'input.parquet',index=False)
 pilot.save(DEST/'footprint_sources.json',pilot.SOURCES)
 return d

def main():
 DEST.mkdir(exist_ok=True);d=prepare();columns=list(b.COLUMNS);params=json.loads((b.OUT/'final_model.json').read_text())['params']
 design=dict(method='Hold out entire networks; fixed settings; original-only and all additions training; original ratios unchanged',size_options=['current','none','edited'],edited_definition='sum(parent outer-domain edge visit probability * length) over selected unit membership; new composite edited sizes retained',params=params,columns=columns,sources={str(p):b.engine.sha(p) for p in [Path(__file__),Path(aug.__file__),Path(b.__file__),DEST/'input.parquet',DEST/'footprint_sources.json']},limitations='Exploratory after inspecting network22; fixed settings selected previously. No final model promotion.')
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();pilot.save(DEST/'design.json',design)
 old=pd.read_parquet(OUT/'augmentation/predictions.parquet');results=[]
 for training,oldvariant in [('original_only','original_only'),('augmented','plus_both')]:
  for size in ['current','none','edited']:
   if size=='current':
    z=old[old.variant==oldvariant].copy();z['training']=training;z['size']=size;results.extend(z.to_dict('records'));continue
   data=d.copy();b.COLUMNS=[c for c in columns if size!='none' or c!='domain_size']
   if size=='edited':data['domain_size']=data.domain_size_edited
   for ex in sorted(data.example.unique()):
    tr=data[(data.example!=ex)&((data.cohort=='original') if training=='original_only' else True)];te=data[data.example==ex];assert ex not in set(tr.example)
    path=DEST/'checkpoints'/training/size/f'{ex}.json';saved=b.forest.checked(path,fp)
    if saved is None:
     artifact=b.fit(tr,params);p=b.forest.predict(artifact,te)
     saved=dict(fingerprint=fp,training=training,size=size,held=ex,rows=[dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(v)) for r,v in zip(te.itertuples(),p)])
     pilot.save(path,saved);print(training,size,ex,'saved',flush=True)
    results.extend(dict(training=training,size=size,**r) for r in saved['rows'])
 b.COLUMNS=columns;p=pd.DataFrame(results);p.to_parquet(DEST/'predictions.parquet',index=False);scores=[];net=[]
 for (training,size,cohort),g in p.groupby(['training','size','cohort']):
  for target in ['centroid_shift_hours','peak_shift_hours']:
   for subset,z in [('all',g),('excluding22',g[g.example!='22']),('excluding54',g[g.example!='54'])]:
    w=b.engine.weights(z);y=z[target].to_numpy();pr=z.predicted_hours.to_numpy();scores.append(dict(training=training,size=size,cohort=cohort,target=target,subset=subset,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
   for ex,z in g.groupby('example'):
    net.append(dict(training=training,size=size,cohort=cohort,target=target,example=ex,n=len(z),mae_minutes=float(abs(z.predicted_hours-z[target]).mean()*60)))
 pd.DataFrame(scores).to_csv(DEST/'scores.csv',index=False);pd.DataFrame(net).to_csv(DEST/'network_scores.csv',index=False)
 print(pd.DataFrame(scores).query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False))
if __name__=='__main__':main()

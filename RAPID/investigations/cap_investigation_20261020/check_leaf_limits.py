import sys
sys.path.insert(0,'RAPID/investigations/cap_investigation_20261020')
from investigate_remaining import *
d=pd.read_parquet(OUT/'edited_size_check/uncapped_edited_input.parquet');params=json.loads((b.OUT/'final_model.json').read_text())['params'];rows=[];nodes=[]
for ex in ['22','41','46','55']:
 tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)&(d.cohort.eq('segmented')|d.state.isin(PARTIAL))]
 for variant in ['fixed','split2_only']:
  model=b.fit(tr,dict(params,**({'min_samples_split':2} if variant=='split2_only' else {})),4);X=model['preprocessor'].transform(te)
  pred=b.forest.predict(model,te)
  for i,r in enumerate(te.itertuples()):
   rows.append(dict(example=ex,state=r.state,variant=variant,actual=r.centroid_shift_hours,predicted=float(pred[i])))
   counts=[]
   for tree in model['estimator'].estimators_:
    node=tree.apply(X[i:i+1])[0];counts.append(tree.tree_.n_node_samples[node])
   nodes.append(dict(example=ex,state=r.state,variant=variant,median_unique_inbag_samples=float(np.median(counts)),fraction_leaves_below_20=float((np.array(counts)<20).mean())))
 print(ex,'done',flush=True)
pd.DataFrame(rows).to_csv(DEST/'split_limit_check.csv',index=False);pd.DataFrame(nodes).to_csv(DEST/'leaf_sample_counts.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(nodes).to_string(index=False))

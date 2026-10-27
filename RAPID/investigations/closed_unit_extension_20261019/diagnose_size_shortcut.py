import sys,json
sys.path.insert(0,'RAPID/investigations/closed_unit_extension_20261019')
import augmentation as a
import pandas as pd,numpy as np
b=a.baseline;out=a.OUT/'augmentation/diagnostic22';d=pd.read_parquet(a.OUT/'augmentation/input.parquet');tr=d[(d.example!='22')&d.cohort.isin(['original','segmented'])];te=d[(d.example=='22')&(d.cohort=='original')];params=json.loads((b.OUT/'final_model.json').read_text())['params']
rows=[]
for variant in ['full','without_size']:
 if variant=='without_size':b.COLUMNS=[c for c in b.COLUMNS if c!='domain_size']
 model=b.fit(tr,params);pred=b.forest.predict(model,te);rows.append(dict(variant=variant,mae_minutes=float(abs(pred-te.centroid_shift_hours).mean()*60)))
 if variant=='full':
  cols=model['preprocessor'].kept;roots=pd.Series([cols[t.tree_.feature[0]] for t in model['estimator'].estimators_]).value_counts();print('root splits',roots.to_dict());print('importance',dict(zip(cols,model['estimator'].feature_importances_)))
 print(rows[-1])
pd.DataFrame(rows).to_csv(out/'size_ablation.csv',index=False)

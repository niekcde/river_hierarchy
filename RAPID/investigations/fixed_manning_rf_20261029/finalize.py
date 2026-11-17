"""Save all-data research forests and diagnostic summaries after held-out refits."""
import json
import joblib
import numpy as np
import pandas as pd
import run as task

def main():
 out=task.DEST;completion=json.loads((out/'completion.json').read_text())
 for path,h in completion['files'].items():assert task.sha(path)==h
 d=pd.read_parquet(out/'input.parquet');p=pd.read_parquet(out/'predictions.parquet')
 settings=json.loads((task.BASE/'current_model_comparison_20261023/settings.json').read_text())
 for model,config in settings.items():
  task.tuning.b.COLUMNS=list(config['columns'])
  artifact=task.tuning.b.fit(d,config['params'],jobs=2)
  path=out/(model+'.joblib');joblib.dump(artifact,path,compress=3)
  task.save(out/(model+'.json'),dict(artifact=str(path),sha256=task.sha(path),columns=config['columns'],params=config['params'],
   training_rows=len(d),training_networks=sorted(d.example.unique()),target='fixed Manning centroid shift hours, 0.5% window',
   evaluation='Use predictions.parquet (held-network fits); all-data forest training scores are not test results.',
   settings='Retained prior all-data selected hyperparameters, no new tuning.',input_sha256=task.sha(out/'input.parquet')))
  print('Saved all-data forest',model,flush=True)
 rows=[]
 for model,g in p.groupby('model'):
  for ex,z in g.groupby('example'):rows.append(dict(model=model,example=ex,**task.score(z[z.cohort=='original'])))
 pd.DataFrame(rows).to_csv(out/'regular_network_scores.csv',index=False)
 p['observed_minutes']=60*p.centroid_shift_hours;p['predicted_minutes']=60*p.predicted_hours
 p['absolute_error_minutes']=abs(p.predicted_minutes-p.observed_minutes)
 p[(p.observed_minutes>1+1e-6)&(p.predicted_minutes<=1+1e-6)].sort_values('observed_minutes',ascending=False).to_csv(out/'missed_delays.csv',index=False)
 p[(p.observed_minutes<=1+1e-6)&(p.predicted_minutes>1+1e-6)].sort_values('predicted_minutes',ascending=False).to_csv(out/'false_delays.csv',index=False)
 paired=p.pivot(index=['example','state','cohort','edit_type','observed_minutes'],columns='model',values=['predicted_minutes','absolute_error_minutes'])
 paired.columns=['_'.join(c) for c in paired.columns];paired=paired.reset_index()
 paired['depth_model_error_reduction_min']=paired.absolute_error_minutes_geometry_9-paired.absolute_error_minutes_depth_informed_10
 paired.to_csv(out/'paired_model_errors.csv',index=False)
 dependency_paths=[task.Path(__file__),task.Path(task.__file__),task.Path(task.tuning.b.__file__),task.Path(task.tuning.b.forest.__file__),task.Path(task.tuning.b.engine.__file__),
  task.ROOT/'RAPID/investigations/closed_unit_extension_20261019/edited_features.py',task.ROOT/'RAPID/investigations/hydrograph_predictors_20261014/flow_resistance.py',
  task.ROOT/'RAPID/src/rapid_tools/current_domains.py']
 task.save(out/'research_artifacts.json',dict(complete=True,all_data_fits=2,held_out_fits=30,production_modified=False,
  dependencies={str(x):task.sha(x) for x in dependency_paths},
  files={str(x):task.sha(x) for x in out.glob('*.joblib')}))

if __name__=='__main__':main()

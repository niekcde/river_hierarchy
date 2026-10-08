"""Frozen omega additions on the same synthetic holdout; exploratory, no tuning."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from deduplicated_response_benchmark import OUT,EXTENDED,RESPONSES,scores,estimator
s=pd.read_parquet(OUT/'synthetic_matched_features.parquet');train=s[s.split.eq('development')];test=s[~s.split.eq('development')];rows=[]
for name in ['omega','omega_plus_size','omega_plus_dominance','omega_plus_association','omega_plus_slope','omega_plus_context','components','selected_six']:
 for response in RESPONSES:
  cols=EXTENDED[name];fit=estimator('boosted').fit(train[cols],train[response]);prediction=fit.predict(test[cols]);baseline=np.full(len(test),train[response].mean())
  row=next(scores(test[response].to_numpy(),prediction,baseline,test,feature_set=name,response=response,validation='synthetic_geometry_holdout',model='boosted'));rows.append(row)
 print(name,flush=True)
pd.DataFrame(rows).to_csv(OUT/'synthetic_predictor_additions.csv',index=False)
print(pd.DataFrame(rows).pivot(index='feature_set',columns='response',values='r2').round(3).to_string())

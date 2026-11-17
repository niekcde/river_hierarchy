import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
from control_coverage import coverage_table

def test_conditional_coverage_keeps_missing_separate():
    rows=[]
    for x in [0,1,2]:
        for z in [0,1,2]:
            rows.append(dict(method='m',child_state='s',x=x,z=z,centroid_shift_minutes=x*z,p=x if z!=2 else np.nan,q=z))
    r=coverage_table(pd.DataFrame(rows),dict(x=[0,1,2],z=[0,1,2]),['p','q'])
    a=r[(r.factor=='x')&(r.predictor=='p')].iloc[0]
    assert a.response_changing_contexts==2 and a.valid_predictor_contexts==1
    assert a.predictor_available_pct==50 and a.predictor_changes_pct==100
    b=r[(r.factor=='x')&(r.predictor=='q')].iloc[0]
    assert b.predictor_changes_pct==0 and b.median_predictor_span_sd==0

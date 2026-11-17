import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
from predictor_association_matrices import correlations

def test_correlation_direction_scale_and_constant():
    d=pd.DataFrame([dict(method='m',child_state=s,x=x,z=z,p=sign*x,q=1000*sign*x,c=2.) for s,sign in [('a',1),('b',-1)] for x in [1,2,3] for z in [1,2,3]])
    r=correlations(d,dict(x=[1,2,3],z=[1,2,3]),['p','q','c'],'m')
    a=r[r.factor.eq('x')].pivot(index='child_state',columns='predictor',values='correlation')
    np.testing.assert_allclose(a.p,[1,-1]);np.testing.assert_allclose(a.q,a.p)
    assert a.c.isna().all()

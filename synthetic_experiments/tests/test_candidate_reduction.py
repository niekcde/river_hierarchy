"""Numerical safeguards for the descriptive screen, independent of external data."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parents[1]/'investigations/candidate_reduction_20261030/audit.py'
s=importlib.util.spec_from_file_location('candidate_reduction_audit',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def test_null_stratum_roundoff_uses_full_column_scale():
    values=np.r_[1e6,2e6,-3e-9,2e-9]
    cleaned=a.clean(values)
    np.testing.assert_array_equal(cleaned[-2:],0.)
    # The meaningful large values remain unchanged; no source values overwritten.
    np.testing.assert_allclose(cleaned[:2],values[:2])
    assert values[-1]!=0

def test_constant_columns_are_not_correlations_and_nan_is_not_zero():
    f=pd.DataFrame({'constant':[1e12]*100,'varying':np.arange(100.)})
    c,valid=a.matrix(f,list(f))
    assert not valid[0] and valid[1]
    assert np.isnan(c[0]).all()
    assert np.isnan(a.corr(np.array([np.nan,np.nan]),np.array([1.,2.])))

def test_monotonicity_does_not_imply_linearity():
    x=np.arange(1.,101.);y=x**4
    np.testing.assert_allclose(a.corr(x,y,True),1.)
    assert a.corr(x,y)<.9
    np.testing.assert_allclose(a.corr(x,-y,True),-1.)

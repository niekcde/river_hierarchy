"""Fractional-clock diagnostic must reproduce the production sparse recurrence."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / 'investigations/check_fractional_routing_clocks.py'
spec = importlib.util.spec_from_file_location('fractional_clock_investigation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize('forcing_dt,nsub', [(755., 8), (1085., 16)])
def test_fractional_clock_split_rejoin_with_steady_initial_flow(forcing_dt, nsub):
    frame = pd.DataFrame(dict(id_link=[1,2,3,4,5],u=[0,1,1,2,3],v=[1,2,3,4,4],
                              width=[10.,3.,7.,3.,7.],count=[2,3,1,1,2],
                              rapid_k=[100.,120.,90.,130.,80.]))
    flow = np.r_[np.full(4,10.),np.linspace(10.,20.,12),np.linspace(20.,12.,16),np.full(30,12.)]
    dt = forcing_dt / nsub
    native, _, _ = module.sparse_benchmark(frame, flow, nsub, dt, 10.)
    candidate, audit = module.cascade(frame, flow, nsub, dt, 10., True)
    substeps, _ = module.cascade(frame, flow, nsub, dt, 10., return_native=True)
    np.testing.assert_allclose(substeps.reshape(-1,nsub).mean(axis=1),candidate,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(candidate,native,rtol=1e-12,atol=1e-12)
    assert len(audit) == frame['count'].sum()
    assert all(a['minimum'] >= 0 for a in audit)

import numpy as np
import pytest
from solver import quadratic_projection


def test_equal_drop_penalty_splits_adjustment():
    x,qa=quadratic_projection(-np.eye(2),[0,0],[1,1],[[-1,1]],[1.])
    np.testing.assert_allclose(x,[-.5,.5]);assert qa['stationarity']<1e-12


def test_slope_penalty_moves_long_reach_endpoint_more():
    x,_=quadratic_projection(-np.eye(2),[0,0],[1/100,1/1000],[[-1,1]],[1.])
    np.testing.assert_allclose(x,[-1/101,100/101])


def test_external_floor_can_bind():
    x,_=quadratic_projection(-np.eye(2),[0,0],[1,1],[[-1,1],[1,0]],[1.,-.1])
    np.testing.assert_allclose(x,[-.1,.9])


def test_geometry_offset_is_in_objective():
    x,_=quadratic_projection(-np.eye(2),[0,-.2],[1,1],[[-1,1]],[1.])
    np.testing.assert_allclose(x,[-.4,.6])


def test_infeasible_constraints_rejected():
    with pytest.raises(ValueError,match='No certified'):
        quadratic_projection(np.eye(1),[0],[1],[[1],[-1]],[1,1])


def test_random_projections_match_independent_slsqp():
    from scipy.optimize import minimize,LinearConstraint
    rng=np.random.default_rng(42)
    for _ in range(40):
        C=rng.normal(size=(5,3));target=rng.normal(size=5);w=rng.uniform(.1,2,5);A=rng.normal(size=(7,3));seed=rng.normal(size=3);b=A@seed-rng.uniform(.01,1,7)
        x,qa=quadratic_projection(C,target,w,A,b)
        opt=minimize(lambda z:np.sum((w*(C@z-target))**2),seed,jac=lambda z:2*C.T@(w*w*(C@z-target)),constraints=[LinearConstraint(A,b,np.inf)],method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
        assert opt.success
        np.testing.assert_allclose(x,opt.x,atol=2e-5)

"""Known factorial interactions, missing ratios, and feature homogeneity."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'synthetic_experiments/investigations'))
import control_predictor_screen as s

def test_known_interaction():
    x,y=np.meshgrid([-1.,0.,1.],[-1.,0.,1.],indexing='ij')
    a=x+2*y+3*x*y;r=s.tensor_stats(a)
    np.testing.assert_allclose(r['main'],[1/11,4/11])
    np.testing.assert_allclose(r['pairs'],[6/11])
    np.testing.assert_allclose(r['total'],[7/11,10/11])

def test_matched_direction_and_missing():
    a=np.array([[1.,2.,3.],[3.,2.,1.],[2.,2.,2.],[1.,np.nan,3.]])
    r=s.matched_stats(a,1)
    assert r['valid_contexts']==3 and r['varying_contexts']==2
    assert r['rho_median']==0 and r['constant_contexts']==1
    assert s.tensor_stats(a) is None

def test_scaling_shared_feature_implementation():
    cfg=dict(s.bridge.pilot.CONFIG and __import__('json').loads(s.bridge.pilot.CONFIG.read_text()))
    pars=dict(cfg['base_geometry'],length_multiplier=1.5,outer_a_fraction=.8)
    _,_,tables,edgeids=s.bridge.synthetic_frames(pars,cfg)
    p=tables['nested/original'];c=tables['nested/full']
    a,b=.25*pars['total_length_m'],.75*pars['total_length_m']
    p=p[p.x0.ge(a)&p.x1.le(b)].copy();c=c[c.x0.ge(a)&c.x1.le(b)].copy()
    base=s.bridge.features(p,c,set(p.id_link))
    L,W,S,H=2.,3.,5.,1.7
    def scale(f):
        f=f.copy();f.link_length_m*=L;f.width*=W;f.slope*=S;f.depth_m*=H
        return f
    changed=s.bridge.features(scale(p),scale(c),set(p.id_link))
    for col in s.COLUMNS[:-1]:
        multiplier=L if col in ['domain_size','length_delta_km'] else S**-.5 if col=='parent_slope_slow' else L/H/S**.5 if col=='delta_G_inverse_depth' else 1.
        np.testing.assert_allclose(changed[col],base[col]*multiplier,rtol=1e-7,atol=1e-8,equal_nan=True)

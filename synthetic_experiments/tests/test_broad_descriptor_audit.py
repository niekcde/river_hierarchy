import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
import broad_descriptor_audit as b

def test_descriptor_scaling_against_shared_implementation():
    settings=json.loads(b.cp.bridge.pilot.CONFIG.read_text())
    pars=dict(settings['base_geometry'],length_multiplier=1.5,outer_a_fraction=.8)
    _,_,tables,_=b.cp.bridge.synthetic_frames(pars,settings)
    f=tables['nested/original'];lo=.25*pars['total_length_m'];hi=.75*pars['total_length_m'];f=f[f.x0.ge(lo)&f.x1.le(hi)].copy()
    original=b.cp.bridge.descriptors(f)
    scaled=f.copy();scaled.link_length_m*=2;scaled.width*=3
    changed=b.cp.bridge.descriptors(scaled)
    for metric in b.BASE_METRICS:
        np.testing.assert_allclose(changed[metric],original[metric]*b.scale_metric(metric,2,3),atol=1e-8,rtol=1e-9,equal_nan=True,err_msg=metric)

def test_path_geometry_scales_independently():
    key=(.8,.5,1.5,'central',1.1)
    reference=b.extended_reference(key)['nested/original']
    pars=dict(zip(b.cp.SHAPE,key),total_width_m=1500.,total_length_m=40000.,reference_slope=5e-4)
    states,_=b.cp.make_states(pars);actual=b.legacy.domain_metrics(states['nested/original'],10000,30000)
    for metric in actual:
        scale=2 if metric=='equivalent_length' else 3 if metric=='equivalent_width' else 2/3 if metric=='elongation' else .5 if metric=='branching_density_by_length' else 1
        np.testing.assert_allclose(actual[metric],reference[metric]*scale,atol=1e-8,rtol=1e-9)

"""Semantic checks for the frozen-real-model bridge, not forest retraining."""
import sys
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'synthetic_experiments/investigations'))
import current_predictor_bridge as b

class FeatureSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings=json.loads(b.pilot.CONFIG.read_text())
        cls.parameters=cls.settings['base_geometry']
        _,_,cls.frames,_=b.synthetic_frames(cls.parameters,cls.settings)

    def domain(self,state):
        f=self.frames[state]
        return f[f.x0.ge(5000)&f.x1.le(15000)].copy()

    def test_equal_routes_have_undefined_spread_ratio(self):
        p,c=self.domain('simple/original'),self.domain('simple/full')
        r=b.features(p,c,set(p.id_link))
        self.assertTrue(np.isnan(r['log1p__B_ratio_route_length_sd']))
        self.assertAlmostEqual(r['length_delta_km'],0.)
        self.assertAlmostEqual(r['domain_size'],10.)

    def test_nested_size_is_selected_footprint_in_outer_allocation(self):
        p,c=self.domain('nested/original'),self.domain('nested/inner_only')
        ids=set(p[p.x0.ge(8000)&p.x1.le(12000)].id_link)
        r=b.features(p,c,ids)
        self.assertAlmostEqual(r['B_parent_route_length_mean'],10000.)
        self.assertAlmostEqual(r['domain_size'],3.2)

    def test_depth_resistance_single_channel_series(self):
        f=self.domain('simple/full')
        expected=float((f.link_length_m/(f.width**2*f.depth_m**3)).sum())
        self.assertAlmostEqual(b.resistance(f)/expected,1.,places=10)

    def test_G_depth_scale_and_resistance_ratio(self):
        p,c=self.domain('nested/original'),self.domain('nested/full')
        r=b.features(p,c,set(p.id_link))
        rr=b.features(p.assign(depth_m=p.depth_m*2),c.assign(depth_m=c.depth_m*2),set(p.id_link))
        self.assertAlmostEqual(rr['delta_G_inverse_depth'],r['delta_G_inverse_depth']/2,places=9)
        self.assertAlmostEqual(rr['resistance_explicit_depth'],r['resistance_explicit_depth'],places=9)

if __name__=='__main__':unittest.main()

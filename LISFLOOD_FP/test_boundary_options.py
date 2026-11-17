import unittest
import numpy as np
import pandas as pd
from boundary_options import build_boundary,gauge_discharge
from diffusive_network import normal_depth


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([(0,1,2000.,100.,2.,0.)],columns=['u','v','link_length_m','width','head_us','head_ds'])

    def test_boundary_factor_does_not_modify_bed(self):
        a,_=build_boundary(self.frame,.035,100)
        b,_=build_boundary(self.frame,.035,100,slope_factor=.5)
        np.testing.assert_array_equal(a.bed,b.bed)
        h=np.full(len(a.bed),2.)
        np.testing.assert_array_equal(a.fluxes(h)[0],b.fluxes(h)[0])
        self.assertAlmostEqual(b.fluxes(h)[1]/a.fluxes(h)[1],np.sqrt(.5))

    def test_extension_gauge_steady_and_storage_identity(self):
        m,info=build_boundary(self.frame,.035,100,extension_m=2000.)
        h=np.full(len(m.bed),normal_depth(100,100,.001,.035))
        self.assertAlmostEqual(gauge_discharge(m,info,h)[0],100.,places=8)
        sol=m.integrate(h,np.linspace(0,12000,101),lambda t:100+20*np.exp(-((t-3000)/1000)**2))
        upstream_area=m.storage_area-info['downstream_area']
        for t,y in zip(sol.t,sol.y.T):
            rhs=m.rhs(t,y,lambda t:100+20*np.exp(-((t-3000)/1000)**2))
            qg,_=gauge_discharge(m,info,y[:-2])
            self.assertAlmostEqual(float(upstream_area@rhs[:-2]),rhs[-2]-qg,places=8)
        self.assertEqual(info['original_outlet_node'],1)


if __name__=='__main__':unittest.main()

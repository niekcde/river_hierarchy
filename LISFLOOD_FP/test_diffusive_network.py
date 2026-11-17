import unittest
import numpy as np
import pandas as pd
from diffusive_network import Network, normal_depth


def frame(rows):
    return pd.DataFrame(rows,columns=['u','v','link_length_m','width','head_us','head_ds'])


class PhysicsTests(unittest.TestCase):
    def test_uniform_normal_flow(self):
        m=Network.from_frame(frame([(0,1,2000.,100.,2.,0.)]),dx=100)
        h=np.full(len(m.bed),normal_depth(100.,100.,.001,.035))
        q,out=m.fluxes(h)
        np.testing.assert_allclose(q,100.,rtol=1e-12)
        self.assertAlmostEqual(out,100.)
        sol=m.integrate(h,np.array([0.,3600.]),lambda _:100.)
        np.testing.assert_allclose(sol.y[:-2,-1],h,atol=1e-9)

    def test_lake_at_rest_internal_flux(self):
        m=Network.from_frame(frame([(0,1,1000.,100.,1.,0.)]))
        q,_=m.fluxes(5.-m.bed)
        np.testing.assert_allclose(q,0.,atol=1e-12)

    def test_junction_symmetry_and_conservation(self):
        m=Network.from_frame(frame([(0,1,1000.,100.,3.,2.),
            (1,2,1000.,50.,2.,1.),(1,2,1000.,50.,2.,1.),(2,3,1000.,100.,1.,0.)]))
        h,_=m.spinup(100.)
        sol=m.integrate(h,np.linspace(0,20000,101),lambda t:100+50*np.exp(-((t-5000)/1500)**2))
        q=np.array([m.fluxes(x)[0] for x in sol.y[:-2].T])
        np.testing.assert_allclose(q[:,m.faces.link.eq(1)],q[:,m.faces.link.eq(2)],rtol=1e-9,atol=1e-8)
        storage=m.storage_area@(sol.y[:-2]-h[:,None])
        residual=storage-(sol.y[-2]-sol.y[-1])
        self.assertLess(np.max(abs(residual)),1e-5)

    def test_reverse_flow_and_bed_datum_invariance(self):
        f=frame([(0,1,1000.,100.,1.,0.)]);m=Network.from_frame(f)
        h=2.+np.linspace(0,3,len(m.bed)) # explicitly impose opposite stage slope below
        h=5.-2*m.bed
        self.assertTrue((m.fluxes(h)[0]<0).all())
        shifted=f.copy();shifted[['head_us','head_ds']]+=100
        n=Network.from_frame(shifted)
        np.testing.assert_allclose(m.fluxes(h)[0],n.fluxes(h)[0],atol=1e-9)

    def test_reject_conflicting_junction_bed(self):
        with self.assertRaises(ValueError):
            Network.from_frame(frame([(0,1,100.,50.,2.,1.),(1,2,100.,50.,1.1,0.)]))

    def test_analytic_jacobian(self):
        m=Network.from_frame(frame([(0,1,1000.,100.,1.,0.)]))
        y=np.r_[np.full(len(m.bed),2.),0.,0.]
        analytic=m.jacobian(0.,y).toarray()
        numerical=np.empty_like(analytic)
        for j in range(len(y)):
            delta=np.zeros(len(y));delta[j]=1e-6
            numerical[:,j]=(m.rhs(0,y+delta,lambda _:100)-m.rhs(0,y-delta,lambda _:100))/(2e-6)
        np.testing.assert_allclose(analytic,numerical,atol=1e-6,rtol=1e-6)


if __name__=='__main__': unittest.main()

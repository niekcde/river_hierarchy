import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'boundary_objectives_20261027'))
import numpy as np,pandas as pd
from solve import fit_primary,fit_internal
class Tests(unittest.TestCase):
 def test_unpenalised_boundary_tie(self):
  x,qa=fit_primary(np.array([[0.,1.]]),np.array([1.]),np.array([[1.,-1.]]),np.array([2.]))
  np.testing.assert_allclose(x,[2.,0.],atol=1e-8);self.assertTrue(qa['rank_deficient'])
 def test_joint_relative_drops(self):
  x,qa=fit_primary(np.eye(2),np.array([2.,1.]),np.array([[1.,-1.]]),np.array([1.]))
  np.testing.assert_allclose(x,[.2,-.8],atol=1e-8)
 def test_internal_shared_head_fit(self):
  f=pd.DataFrame({'id_link':[1,2],'u':[0,1],'v':[1,2],'link_length_m':[100.,200.]})
  h,qa=fit_internal(f,{0:1.,2:0.},.003,np.array([.01,.02]));self.assertAlmostEqual(h[1],.68,places=7)
 def test_active_internal_floor(self):
  f=pd.DataFrame({'id_link':[1,2],'u':[0,1],'v':[1,2],'link_length_m':[100.,200.]})
  h,qa=fit_internal(f,{0:1.,2:0.},.003,np.array([.8,.02]));self.assertAlmostEqual(h[1],.2,places=7)
 def test_zero_width_feasible_face(self):
  f=pd.DataFrame({'id_link':[1,2],'u':[0,1],'v':[1,2],'link_length_m':[100.,200.]})
  h,qa=fit_internal(f,{0:1.,2:0.},.003,np.array([.4,.6]));self.assertAlmostEqual(h[1],.6,places=7)
 def test_infeasible_not_softened(self):
  f=pd.DataFrame({'id_link':[1,2],'u':[0,1],'v':[1,2],'link_length_m':[100.,200.]})
  with self.assertRaises(ValueError):fit_internal(f,{0:1.,2:0.},.003,np.array([.8,.8]))
if __name__=='__main__':unittest.main()

import unittest
import pandas as pd
from check import path_constraints,lp_check
class FrozenFloorTests(unittest.TestCase):
 def test_fixed_fixed_floor(self):
  f=pd.DataFrame({'id_link':[1],'u':[0],'v':[1],'link_length_m':[100.]});h=pd.Series({0:1.,1:.9})
  for required,expected in [(.05,True),(.2,False)]:
   p=path_constraints(f,h,[required])[0]
   self.assertEqual(p['available_drop_m']>=required,expected)
   self.assertEqual(lp_check(f,h,{0,1},[required]),expected)
 def test_path_sum_internal_node(self):
  f=pd.DataFrame({'id_link':[1,2],'u':[0,1],'v':[1,2]});h=pd.Series({0:1.,2:.9})
  for weights,expected in [([.04,.04],True),([.06,.06],False)]:
   p=path_constraints(f,h,weights)[0]
   self.assertEqual(p['available_drop_m']>=p['path_weight'],expected)
   self.assertEqual(lp_check(f,h,{0,2},weights),expected)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import pandas as pd
from alternative_validation import balanced_groups


class BalancedSplitTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for ex in ['01','02','03','04']:
            for i in range(4):
                rows.append(dict(example=ex,state=str(i),group_id=ex+'_family'+str(i//2),
                    cohort='original',edit_type='Compound member',response_bin='near_zero_<=1min' if i<2 else 'advance_5to60min'))
            rows.append(dict(example=ex,state='full',group_id=ex+'_full',cohort='segmented',
                edit_type='Segmented full network',response_bin='advance_over60min'))
        return pd.DataFrame(rows)

    def test_whole_network_and_reproducibility(self):
        d=self.frame();fold,_,sig=balanced_groups(d,'example',2,100)
        repeat,_,sig2=balanced_groups(d,'example',2,100)
        np.testing.assert_array_equal(fold,repeat);self.assertEqual(sig,sig2)
        for k in range(2):
            self.assertEqual(d[fold==k].example.nunique(),2)
            self.assertTrue(set(d[fold==k].example).isdisjoint(d[fold!=k].example))

    def test_family_integrity_and_full_collapse_balance(self):
        d=self.frame();fold,_,_=balanced_groups(d,'group_id',2,200)
        check=d.assign(fold=fold)
        self.assertTrue(check.groupby('group_id').fold.nunique().eq(1).all())
        self.assertTrue(check[check.cohort.eq('segmented')].groupby('fold').size().eq(2).all())

    def test_duplicate_partition_exclusion(self):
        d=self.frame();_,_,sig=balanced_groups(d,'example',2,100)
        _,_,other=balanced_groups(d,'example',2,101,banned=[sig])
        self.assertNotEqual(sig,other)


if __name__=='__main__':unittest.main()

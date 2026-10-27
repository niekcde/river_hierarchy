"""Check transitivity and conservative reach overlap before fold assignment."""
import unittest
from validation_group_inventory import grouped, response_bin


def record(state, families=(), reaches=(), context=()):
    return dict(state=state,family_tokens=list(families),parent_footprint_links=list(reaches),descriptor_links=list(context))


class GroupInventoryTests(unittest.TestCase):
    def test_family_and_reach_transitivity(self):
        rows=[record('compound',[1],[10]),record('member',[1],[20]),record('overlap',[2],[20]),record('other',[3],[30])]
        families,_=grouped(rows)
        self.assertNotEqual(families['compound'],families['overlap'])
        groups,_=grouped(rows,use_links=True)
        self.assertEqual(groups['compound'],groups['overlap'])
        self.assertNotEqual(groups['compound'],groups['other'])

    def test_segmented_bridge_and_context_sensitivity(self):
        rows=[record('a',[1],[10],[10,11]),record('b',[2],[20],[20,21])]
        groups,_=grouped(rows,use_links=True)
        self.assertEqual(len(set(groups.values())),2)
        groups,_=grouped(rows+[record('segmented',(),[10,20])],use_links=True)
        self.assertEqual(len(set(groups.values())),1)
        contexts,_=grouped([record('a',[1],[10],[10,20]),record('b',[2],[20],[20])],use_links=True,use_context=True)
        self.assertEqual(contexts['a'],contexts['b'])

    def test_order_invariance(self):
        rows=[record('a',[1],[10]),record('b',[1],[20]),record('c',[2],[20])]
        self.assertEqual(grouped(rows,True)[0],grouped(list(reversed(rows)),True)[0])

    def test_signed_bins(self):
        self.assertEqual(response_bin(1/60),'near_zero_<=1min')
        self.assertEqual(response_bin(-5/60),'advance_1to5min')
        self.assertEqual(response_bin(1),'delay_5to60min')
        self.assertEqual(response_bin(-2),'advance_over60min')


if __name__=='__main__':unittest.main()

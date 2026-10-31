"""Check paired physical-frame bootstrap direction and repeatability."""
import unittest
from compare import bootstrap_difference
from evaluate import aggregate


class PairedBootstrapTest(unittest.TestCase):
    def test_constant_difference_and_reproducibility(self):
        a=[3.0]*176;b=[2.0]*176
        result=bootstrap_difference(a,b)
        self.assertEqual(result['mean'],1.0)
        self.assertEqual(result['ci95'],[1.0,1.0])
        self.assertEqual(result,bootstrap_difference(a,b))
        self.assertEqual(bootstrap_difference(b,a)['mean'],-1.0)
        with self.assertRaises(ValueError): bootstrap_difference(a[:-1],b)

    def test_overlaps_are_averaged_by_physical_frame(self):
        rows=[dict(start=s,horizon=h,frame=s+h,metrics={'correct':{'mae':float(s)}},
                   delta={'correct':{'delta_mae':float(h)}}) for s in range(168) for h in range(1,10)]
        result=aggregate(rows,['correct'])['correct']
        self.assertEqual(result['by_physical_frame']['1']['mae'],0.)
        self.assertEqual(result['by_physical_frame']['9']['mae'],4.)
        self.assertEqual(result['by_physical_frame']['176']['mae'],167.)
        self.assertEqual(result['delta_by_physical_frame']['9']['delta_mae'],5.)
        with self.assertRaises(ValueError): aggregate(rows[:-1],['correct'])


if __name__=='__main__': unittest.main()

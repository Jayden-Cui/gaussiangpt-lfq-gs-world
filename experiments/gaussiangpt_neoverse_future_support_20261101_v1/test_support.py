import unittest
import torch
from support import transfer_initial_features


class InitialFeatureTransfer(unittest.TestCase):
    def test_exact_cells_new_cells_and_destination_order(self):
        source=torch.tensor([[2,0,0],[0,0,0],[0,3,0]],dtype=torch.int32)
        feature=torch.tensor([[20.,21.],[0.,1.],[30.,31.]])
        target=torch.tensor([[0,3,0],[1,0,0],[2,0,0],[0,2,0]],dtype=torch.int32)
        mapped,index,report=transfer_initial_features(source,feature,target)
        torch.testing.assert_close(mapped,feature[torch.tensor([2,1,0,2])])
        self.assertEqual(index.tolist(),[2,1,0,2])
        self.assertEqual(report['exact_overlap_nodes'],2)
        self.assertEqual(report['nearest_filled_nodes'],2)
        self.assertEqual(report['nearest_distance_cells']['max'],1.)

    def test_tie_break_is_source_order_invariant_and_inputs_are_immutable(self):
        source=torch.tensor([[1,0,0],[-1,0,0],[0,2,0]],dtype=torch.int64)
        feature=torch.tensor([[8.,9.],[2.,3.],[5.,6.]],requires_grad=True)
        target=torch.tensor([[0,0,0],[0,2,0]],dtype=torch.int64)
        copies=[x.detach().clone() for x in (source,feature,target)]
        first,_,_=transfer_initial_features(source,feature,target)
        shuffled,_,_=transfer_initial_features(source.flip(0),feature.flip(0),target)
        torch.testing.assert_close(first,shuffled)
        torch.testing.assert_close(first[0],feature[1])
        first.add_(100)
        for current,before in zip((source,feature,target),copies): torch.testing.assert_close(current,before)
        self.assertFalse(first.requires_grad)

    def test_identity_transfer_and_duplicate_rejection(self):
        coords=torch.tensor([[3,-2,1],[0,0,0]],dtype=torch.int32)
        features=torch.tensor([[10.,20.],[30.,40.]])
        mapped,_,report=transfer_initial_features(coords,features,coords)
        torch.testing.assert_close(mapped,features)
        self.assertEqual(report['nearest_filled_nodes'],0)
        self.assertFalse(report['future_feature_values_used'])
        with self.assertRaises(ValueError):
            transfer_initial_features(coords,features,coords[torch.tensor([0,0])])


if __name__=='__main__': unittest.main()

import copy,unittest
from unittest.mock import patch
import torch
from io_utils import ROOT,read
from posterior import build_pair,make_memories


def sample_window(nodes):
    generator=torch.Generator().manual_seed(100+nodes)
    coordinates=torch.stack((torch.arange(nodes),torch.arange(nodes)%3,torch.arange(nodes)%5),dim=1).int()
    return [dict(coordinates=coordinates.clone(),features=torch.randn(nodes,64,generator=generator),pts=float(h)/30)
            for h in range(10)]


class PosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.teacher,cls.posterior=build_pair(read(ROOT/'protocol.json'),'cpu')
        cls.posterior.eval()
        cls.window=sample_window(12)

    def test_future_features_and_geometry_cannot_change_earlier_codes(self):
        a=make_memories(self.teacher.base,self.window)
        changed=copy.deepcopy(self.window)
        changed[9]['features']+=3
        changed[9]['coordinates']+=100
        b=make_memories(self.teacher.base,changed)
        with torch.no_grad():
            x=self.posterior([a])['codes'];y=self.posterior([b])['codes']
        self.assertTrue(torch.equal(x[:,:8],y[:,:8]))
        self.assertGreater(float((x[:,8]-y[:,8]).abs().max()),1e-6)
        self.assertEqual(tuple(x.shape),(1,9,16))

    def test_initial_only_ignores_all_future_values(self):
        changed=copy.deepcopy(self.window)
        for state in changed[1:]:
            state['features']*=9;state['coordinates']+=200
        a=make_memories(self.teacher.base,self.window,True)
        b=make_memories(self.teacher.base,changed,True)
        with torch.no_grad():
            x=self.posterior([a])['codes'];y=self.posterior([b])['codes']
        self.assertTrue(torch.equal(x,y))

    def test_variable_length_batch_mask_gradients_and_no_quantization(self):
        a=make_memories(self.teacher.base,self.window)
        b=make_memories(self.teacher.base,sample_window(17))
        self.posterior.zero_grad(set_to_none=True)
        with patch.object(self.teacher.base.quantizer,'forward',side_effect=AssertionError('Quantizer executed')):
            batch=self.posterior([a,b])
            individual=torch.cat([self.posterior([m])['codes'] for m in [a,b]])
            torch.testing.assert_close(batch['codes'],individual,rtol=2e-5,atol=2e-6)
            batch['codes'].square().mean().backward()
        self.assertEqual(tuple(batch['vectors'].shape),(2,9,64))
        self.assertTrue(bool((self.posterior.queries.weight.grad.norm(dim=-1)>0).all()))
        self.assertGreater(float(self.posterior.project_in.weight.grad.norm()),0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.teacher.parameters()))


if __name__=='__main__':unittest.main()

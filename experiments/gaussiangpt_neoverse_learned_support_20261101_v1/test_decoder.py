import types,unittest
import torch
from io_utils import ROOT,read
from runtime import build_model
from support_decoder import candidate_coordinates,make_condition,target_labels,select_state,state_loss


class CandidateDecoderTests(unittest.TestCase):
    def test_candidates_depend_only_on_initial_coordinates_and_preserve_all(self):
        c=torch.tensor([[0,0,0],[2,0,0]],dtype=torch.int32)
        a=candidate_coordinates(c,1,100)
        b=candidate_coordinates(c.flip(0),1,100)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(tuple(v) in set(map(tuple,a.tolist())) for v in c.tolist()))
        with self.assertRaises(ValueError):candidate_coordinates(c,1,1)

    def test_missing_target_stops_and_prediction_has_no_target_count(self):
        c=torch.tensor([[0,0,0],[1,0,0]],dtype=torch.int32);f=torch.zeros(2,64)
        condition=types.SimpleNamespace(coordinates=c,features=f)
        with self.assertRaises(ValueError):target_labels(condition,[{'coordinates':torch.tensor([[3,0,0]]),'features':torch.zeros(1,64)}])
        result=select_state(c,f,torch.tensor([-1.,2.]),10)
        self.assertTrue(torch.equal(result['coordinates'],c[1:]))
        with self.assertRaises(ValueError):select_state(c,f,torch.tensor([-1.,-1.]),10)

    def test_supervised_geometry_and_features_have_separate_gradients(self):
        logits=torch.zeros(1,3,requires_grad=True);features=torch.zeros(1,3,64,requires_grad=True)
        target={'occupancy':torch.tensor([[1.,0.,0.]]),'indices':[torch.tensor([0])],'features':[torch.ones(1,64)]}
        loss,_=state_loss({'features':features,'logits':logits},target,torch.ones(64));loss.backward()
        self.assertLess(float(logits.grad[0,0]),0);self.assertGreater(float(logits.grad[0,1]),0)
        self.assertGreater(float(features.grad[0,0].norm()),0);self.assertEqual(float(features.grad[0,1:].norm()),0)

    def test_actual_decoder_causal_codes_and_native_width(self):
        torch.set_num_threads(2);torch.manual_seed(81)
        model=build_model(read(ROOT/'protocol.json'),'free16','cpu').eval()
        initial={'coordinates':torch.tensor([[0,0,0],[1,0,0],[0,1,0]],dtype=torch.int32),
                 'features':torch.randn(3,64),'pts':0.}
        condition=make_condition(model.base,initial,1,200)
        with torch.no_grad():
            model.base.features.weight.normal_(std=.05);model.occupancy.weight.normal_(std=.05)
        codes=torch.randn(9,16);changed=codes.clone();changed[-1]+=2.5
        with torch.no_grad():
            a=model(condition,codes);b=model(condition,changed)
        self.assertEqual(tuple(a['features'].shape),(9,len(condition.coordinates),64))
        torch.testing.assert_close(a['features'][:-1],b['features'][:-1],atol=2e-5,rtol=1e-6)
        torch.testing.assert_close(a['logits'][:-1],b['logits'][:-1],atol=2e-5,rtol=1e-6)
        self.assertGreater(float((a['features'][-1]-b['features'][-1]).abs().max()),1e-4)
        codes.requires_grad_();model(condition,codes)['features'][-1].square().mean().backward()
        self.assertGreater(float(codes.grad.norm()),0)


if __name__=='__main__':unittest.main()

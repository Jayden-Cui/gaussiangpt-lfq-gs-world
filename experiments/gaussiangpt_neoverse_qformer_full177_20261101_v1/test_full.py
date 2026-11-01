import hashlib,json,unittest
from collections import Counter,defaultdict
import torch
from io_utils import ROOT,read
from full_model import build_models,frozen_digest,parameters,schedule,aggregate
from runtime import digest
from posterior import make_memories
from support_decoder import make_condition,target_labels,state_loss


class FullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.p=read(ROOT/'protocol.json')
        cls.decoder,cls.posterior=build_models(cls.p,'cpu')
        cls.initial_decoder_digest=digest(cls.decoder);cls.initial_posterior_digest=digest(cls.posterior)
        cls.items=schedule(cls.p)
        cls.schedule_sha256=hashlib.sha256(json.dumps(cls.items,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def test_full_sampler_no_lost_tail_and_all_render_horizons(self):
        items=self.items
        self.assertEqual(items,schedule(self.p));self.assertEqual(len(items),2000)
        visits=Counter();horizons=defaultdict(Counter)
        for item in items:
            self.assertEqual(len({v['start'] for v in item['batch']}),4)
            for v in item['batch']:
                visits[v['start']]+=1
                if item['phase']=='render':horizons[v['start']][v['horizon']]+=1
        self.assertEqual(set(visits),set(range(168)))
        self.assertEqual(set(f for s in visits for f in range(s,s+10)),set(range(177)))
        self.assertEqual(set(visits.values()),{47,48})
        self.assertTrue(all(set(v)==set(range(1,10)) and min(v.values())>=3 for v in horizons.values()))

    def test_joint_update_freezes_state_encoder_and_unused_free_codes(self):
        torch.manual_seed(7);d,q=self.decoder,self.posterior
        coordinates=torch.stack((torch.arange(8),torch.arange(8)%2,torch.arange(8)%3),dim=1).int()
        base=torch.randn(8,64)
        states=[dict(coordinates=coordinates.clone(),features=base+.03*torch.randn(8,64),pts=h/30) for h in range(10)]
        condition=make_condition(d.base,states[0],1,1000)
        memories=make_memories(d.base,states);target=target_labels(condition,states[1:])
        frozen=frozen_digest(d);head=d.base.features.weight.detach().clone();projection=q.project_in.weight.detach().clone()
        opt=torch.optim.Adam(parameters(d,q),lr=1e-5);opt.zero_grad(set_to_none=True)
        code=q([memories])['codes'][0]
        prediction=d(condition,code);loss,_=state_loss(prediction,target,d.base.output_std)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(q.project_in.weight.grad.norm()),0)
        self.assertGreater(float(d.base.features.weight.grad.norm()),0)
        self.assertIsNone(d.codes.grad);self.assertFalse(d.codes.requires_grad)
        opt.step()
        self.assertEqual(frozen_digest(d),frozen)
        self.assertFalse(torch.equal(head,d.base.features.weight));self.assertFalse(torch.equal(projection,q.project_in.weight))

    def test_physical_frame_equal_aggregation_is_not_context_weighted(self):
        rows=[dict(frame=f,metrics={'correct':dict(metrics={'mae':v})}) for f,v in [(1,0.),(1,0.),(2,1.)]]
        result=aggregate(rows,['correct'])
        self.assertEqual(result['unique_frame_equal']['correct']['mae'],.5)
        self.assertAlmostEqual(result['window_prediction_equal']['correct']['mae'],1/3)
        self.assertEqual(result['unique_frames'],2);self.assertEqual(result['window_predictions'],3)


if __name__=='__main__':unittest.main()

"""CPU numerical/coverage tests, never initialize native GS or CUDA."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
import worker as w
import pool_index as pool

def ref(clip,frame):
    return SimpleNamespace(clip=clip,frame=frame,current_role='TRAIN',source_group=pool.EXPECTED[clip])

class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__();self.ae=nn.Module();self.ae.vq=nn.Identity()
        self.ae.embeddings=nn.Linear(4,4);self.set_encoder=nn.Linear(4,4)
        self.ae.encoder=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.SiLU())
        self.ae.decoder=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.SiLU())
        self.ae.feature_heads=nn.Linear(4,2);self.ae.gaussian_multiplier=nn.Linear(2,2)
    def forward(self,payload,force):
        e=self.ae.encoder(self.ae.embeddings(payload)+self.set_encoder(payload))
        self.ae.decoder.train(force)
        return self.ae.gaussian_multiplier(self.ae.feature_heads(self.ae.decoder(e)))

def tiny_loader(r):
    return torch.arange(32,dtype=torch.float32).reshape(8,4)/31+r.frame*.1

def tiny_terms(model,payload,vgg,force):
    loss=.01*(model(payload,force)-.7).square().mean()
    return {name:loss for name in w.LOSS_WEIGHTS},dict(latent_count=len(payload),output_count=len(payload))

class WorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(w.SEED)
        self.clips=sorted(pool.EXPECTED)

    def test_two_epochs_cover_all_frames_and_keep_two_separate_tail_batches(self):
        index=tuple(ref(c,f) for c in self.clips for f in range(177))
        plans=w.make_epochs(index)
        self.assertEqual([len(p) for p in plans],[2213,2213])
        self.assertEqual([len(p[-1]) for p in plans],[2,2])
        self.assertEqual(sum(len(b) for p in plans for b in p),17700)
        self.assertEqual(plans,w.make_epochs(index))
        self.assertNotEqual([(r.clip,r.frame) for b in plans[0] for r in b],
                            [(r.clip,r.frame) for b in plans[1] for r in b])
        self.assertTrue(all(pool.assert_coverage(index,p) for p in plans))

    def test_forcing_changes_only_after_one_complete_epoch(self):
        self.assertEqual([w.forcing_for_step(s) for s in range(1,4427)],[True]*2213+[False]*2213)
        for s in (0,4427,1.5,True):
            with self.assertRaises(RuntimeError):w.forcing_for_step(s)

    def test_pressure_batches_allow_cross_batch_repetition_but_not_internal_duplicates(self):
        batch=[dict(clip=c,frame=f) for c,f in (
            ('buildingblock_tower_17_176',117),('buildingblock_tower_18_352',151),
            ('buildingblock_tower_7_0',135),('buildingblock_tower_0_176',166))]
        w.validate_preflight_batches([batch,deepcopy(batch)])
        invalid=deepcopy(batch);invalid[3]=deepcopy(invalid[0])
        with self.assertRaises(RuntimeError):w.validate_preflight_batches([batch,invalid])
        invalid=deepcopy(batch);invalid[3]['clip']='buildingblock_tower_14_0'
        with self.assertRaises(RuntimeError):w.validate_preflight_batches([batch,invalid])

    def _arithmetic(self,n):
        model=nn.Linear(1,1,bias=False);model.weight.data.zero_()
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        batch=tuple(ref(self.clips[i],i) for i in range(n));seen=[]
        targets=[.1,.3,.2,.4][:n]
        def loader(r):seen.append(r.frame);return torch.tensor([[targets[r.frame]]])
        def terms(m,target,vgg,force):
            loss=(m(torch.ones(1,1))-target).square().mean()
            return {k:loss if k=='real_l1' else loss*0 for k in w.LOSS_WEIGHTS},dict(latent_count=1,output_count=1)
        result=w.accumulated_update(model,optimizer,batch,loader,terms,None,False)
        self.assertEqual(seen,list(range(n)))
        self.assertAlmostEqual(result['loss'],sum(t*t for t in targets)/n,places=6)
        self.assertAlmostEqual(result['gradient_norm'],2*sum(targets)/n,places=6)
        self.assertAlmostEqual(float(model.weight.detach()),.2*sum(targets)/n,places=6)
        self.assertEqual(result['actual_batch_size'],n)

    def test_full_b4_gradient_is_true_frame_mean(self):self._arithmetic(4)
    def test_tail_b2_gradient_is_divided_by_two_not_four(self):self._arithmetic(2)

    def test_decoder_only_keeps_encoder_parameters_buffers_and_no_grad(self):
        model=TinyCodec();w.legacy.configure_arm(model,'decoder_only')
        frozen=w.legacy.snapshot(model,encoder_only=True);before=w.legacy.snapshot(model,parameters_only=True)
        optimizer=w.legacy.fresh_optimizer(model)
        batch=tuple(ref(self.clips[i],i) for i in range(4))
        for force in (True,False):
            w.accumulated_update(model,optimizer,batch,tiny_loader,tiny_terms,None,force)
            w.legacy.assert_frozen(model,frozen);w.legacy.assert_training_modes(model,'decoder_only')
        changed={n for n,v in w.legacy.snapshot(model,parameters_only=True).items() if v!=before[n]}
        self.assertTrue(changed);self.assertTrue(all(w.legacy.in_decoder(n) for n in changed))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not w.legacy.in_decoder(n)))

    def test_full_codec_updates_all_original_groups_and_fresh_optimizer(self):
        model=TinyCodec();w.legacy.configure_arm(model,'full_codec')
        before=w.legacy.snapshot(model,parameters_only=True);optimizer=w.legacy.fresh_optimizer(model)
        self.assertFalse(optimizer.state)
        self.assertEqual((optimizer.param_groups[0]['lr'],optimizer.param_groups[0]['betas']),(.0001,(.9,.95)))
        row=w.accumulated_update(model,optimizer,tuple(ref(self.clips[i],i) for i in range(4)),tiny_loader,tiny_terms,None,False)
        groups={w.legacy.group_name(n) for n,v in w.legacy.snapshot(model,parameters_only=True).items() if v!=before[n]}
        self.assertEqual(groups,set(w.legacy.TRAINABLE_DECODER)|{'ae.encoder','ae.embeddings','set_encoder'})
        self.assertTrue(all(v['nonzero_grad']>0 for v in row['gradient_groups'].values()))

    def test_eval_duplicate_and_empty_batches_rejected_before_load(self):
        a=ref(self.clips[0],0);bad=deepcopy(a);bad.current_role='EVAL-only'
        model=TinyCodec();optimizer=w.legacy.fresh_optimizer(model)
        for batch in ((),(bad,),(a,a),(a,)*5):
            with self.assertRaises(RuntimeError):
                w.accumulated_update(model,optimizer,batch,lambda _:self.fail('loaded invalid batch'),tiny_terms,None,True)

    def test_parameterized_state_quantizer_and_encoder_buffer_drift_refused(self):
        model=TinyCodec();frozen=w.legacy.snapshot(model,encoder_only=True)
        model.ae.encoder[1].running_mean.add_(1)
        with self.assertRaises(RuntimeError):w.legacy.assert_frozen(model,frozen)
        model.ae.vq=nn.Linear(4,4)
        with self.assertRaises(RuntimeError):w.legacy.configure_arm(model,'full_codec')

    def test_loss_term_drift_and_nonfinite_loss_fail_before_step(self):
        model=nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        before=deepcopy(model.state_dict());batch=(ref(self.clips[0],0),)
        for invalid in ('terms','nan'):
            def terms(*_):
                loss=model(torch.ones(1,1)).sum()*float('nan')
                return ({'unexpected':loss} if invalid=='terms' else {k:loss for k in w.LOSS_WEIGHTS}),{}
            with self.assertRaises(RuntimeError):w.accumulated_update(model,optimizer,batch,lambda _:None,terms,None,False)
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))

if __name__=='__main__':unittest.main()

"""CPU integration checks against actual pinned QFormer and LFQ dependencies."""
import copy
import json
import unittest
import torch
from aligned_ldm import AlignedLDM,make_condition
from model_core import MaskedGroupNorm64
from latent_objective import latent_reconstruction_loss

def window(seed=71,requires_grad=False):
    generator=torch.Generator().manual_seed(seed)
    coords=torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=torch.long)
    return [dict(coordinates=coords.clone(),features=(torch.randn(4,64,generator=generator)+h*.03).requires_grad_(requires_grad),
        pts=h/29.97,frame_index=h) for h in range(10)]

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(20261030)
        cls.model=AlignedLDM().eval()
        cls.windows=[window(71),window(81)]
        cls.conditions=[make_condition(w[0],radius=1) for w in cls.windows]

    def test_shapes_and_exact_id_inverse(self):
        with torch.no_grad():
            z=self.model.encode_windows(self.windows)
            self.assertEqual(tuple(z['z_prequant'].shape),(2,9,64))
            self.assertEqual(tuple(z['indices'].shape),(2,9))
            self.assertEqual(len({id(m) for m in self.model.posteriors}),9)
            inverse=self.model.quantizer.indices_to_codes(z['indices'],project_out=True)
            torch.testing.assert_close(inverse,z['z_quantized'],atol=1e-6,rtol=1e-6)

    def test_shared_initial_context_is_identical_with_or_without_future(self):
        with torch.no_grad():
            for w in self.windows:
                full=self.model.prepare(w)['initial']
                alone=self.model.encode_initial(w[0])
                self.assertEqual(tuple(full.shape),(4,4,64))
                torch.testing.assert_close(full,alone,atol=1e-5,rtol=1e-5)
            changed=copy.deepcopy(self.windows[0])
            for h in range(1,10):
                changed[h]['coordinates'][0]=torch.tensor([100+h,2,3])
                changed[h]['features']+=100
            actual=self.model.prepare(changed)['initial']
            torch.testing.assert_close(actual,self.model.encode_initial(changed[0]),atol=1e-5,rtol=1e-5)

    def test_unquantized_future_cannot_change_prefix_codes(self):
        with torch.no_grad():
            base=self.model.encode_windows([self.windows[0]])
            changed=copy.deepcopy(self.windows[0])
            for h in range(5,10):
                changed[h]['coordinates'][0]=torch.tensor([300+h,8,9])
                changed[h]['features']=changed[h]['features']*20+50
            altered=self.model.encode_windows([changed])
            torch.testing.assert_close(base['z_prequant'][:,:4],altered['z_prequant'][:,:4],atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.equal(base['indices'][:,:4],altered['indices'][:,:4]))

    def test_prefix_gradient_excludes_future_states(self):
        w=window(91,True)
        z=self.model.encode_windows([w])['z_prequant'][0,3]
        loss=(z*torch.linspace(-1,1,64)).sum()
        grads=torch.autograd.grad(loss,[s['features'] for s in w],allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads[:5] if g is not None),0)
        self.assertEqual(sum(float(g.abs().sum()) for g in grads[5:] if g is not None),0)

    def test_ten_learned_outputs_and_initial_only_replay(self):
        with torch.no_grad():
            z=self.model.encode_windows(self.windows)
            pred=self.model.decode_latents(z['z_quantized'],self.conditions,initial_contexts=z['initial_contexts'])
            replay=self.model.decode_ids(z['indices'],self.conditions)
            self.assertEqual(len(pred),2)
            for b in range(2):
                self.assertEqual(len(pred[b]),10)
                for h in range(10):
                    torch.testing.assert_close(pred[b][h]['features'],replay[b][h]['features'],atol=1e-5,rtol=1e-5)
                    torch.testing.assert_close(pred[b][h]['occupancy_logits'],replay[b][h]['occupancy_logits'],atol=1e-5,rtol=1e-5)

    def test_all_ten_loss_positions_and_motion_gradients(self):
        self.model.train();self.model.zero_grad(set_to_none=True)
        try:
            z=self.model.encode_windows(self.windows)
            pred=self.model.decode_latents(z['z_quantized'],self.conditions,initial_contexts=z['initial_contexts'])
            for p in pred[0]:p['features'].retain_grad()
            losses=latent_reconstruction_loss(pred,self.windows)
            self.assertEqual(tuple(losses['per_frame'].shape),(2,10))
            losses['loss'].backward() # Deliberately exclude entropy auxiliary loss.
            for p in pred[0]:
                self.assertIsNotNone(p['features'].grad)
                self.assertGreater(float(p['features'].grad.abs().sum()),0)
            query_norms=self.model.queries.weight.grad.norm(dim=-1)
            self.assertTrue(bool(torch.isfinite(query_norms).all()))
            self.assertTrue(bool((query_norms>0).all()),query_norms)
            for module in [self.model.encoder_space,self.model.encoder_time,self.model.decoder_space,self.model.decoder_time]:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
        finally:self.model.eval();self.model.zero_grad(set_to_none=True)

    def test_window_decoder_can_depend_on_future_quantized_codes(self):
        codes=torch.randn(1,9,64,requires_grad=True)
        pred=self.model.decode_latents(codes,self.conditions[:1])
        loss=(pred[0][1]['features']*torch.linspace(-1,1,64)).sum()
        grad=torch.autograd.grad(loss,codes)[0]
        self.assertGreater(float(grad[:,8].abs().sum()),0)

    def test_no_initial_feature_output_residual(self):
        # Zero output heads must produce zero outputs, even for nonzero initial state.
        model=copy.deepcopy(self.model)
        with torch.no_grad():
            model.features.weight.zero_();model.features.bias.zero_()
            model.occupancy.weight.zero_();model.occupancy.bias.zero_()
            pred=model.decode_latents(torch.randn(1,9,64),self.conditions[:1])
            for p in pred[0]:
                self.assertEqual(float(p['features'].abs().max()),0)
                self.assertEqual(float(p['occupancy_logits'].abs().max()),0)

    def test_wrong_initial_context_rejected(self):
        with torch.no_grad():
            z=self.model.encode_windows(self.windows)
            with self.assertRaisesRegex(ValueError,'different or modified'):
                self.model.decode_latents(z['z_quantized'],list(reversed(self.conditions)),initial_contexts=z['initial_contexts'])

    def test_masked_group_norm_matches_dense_group_norm(self):
        x=torch.randn(4,5,64)
        active=torch.ones(4,5,dtype=torch.bool)
        dense=torch.nn.GroupNorm(4,64,eps=1e-6)
        frame=MaskedGroupNorm64(False)
        global_norm=MaskedGroupNorm64(True)
        expected=dense(x.permute(0,2,1)).permute(0,2,1)
        torch.testing.assert_close(frame(x,active),expected)
        expected_global=dense(x.permute(2,0,1)[None])[0].permute(1,2,0)
        torch.testing.assert_close(global_norm(x,active),expected_global)
        extended=torch.cat([x,torch.randn(4,3,64)*1e4],1)
        mask=torch.cat([active,torch.zeros(4,3,dtype=torch.bool)],1)
        torch.testing.assert_close(frame(extended,mask)[:,:5],frame(x,active),atol=1e-5,rtol=1e-5)
        extended[:,5:]=float('nan')
        torch.testing.assert_close(frame(extended,mask)[:,:5],frame(x,active),atol=1e-5,rtol=1e-5)
        self.assertTrue(bool(torch.isfinite(frame(extended,mask)).all()))

if __name__=='__main__':
    unittest.main(verbosity=2)


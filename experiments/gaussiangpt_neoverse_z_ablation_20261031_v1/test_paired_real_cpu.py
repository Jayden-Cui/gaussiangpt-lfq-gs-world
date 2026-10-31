"""Real Q-Former/LFQ + tiny synthetic renderer, never a native-GS experiment."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from model_adapter import load_pinned, attach_adapters, clone_pair
from paired_step import step_native


def bound_module(name,path,digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError('External source SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    exec(compile(content,str(path),'exec'),module.__dict__)
    return module


def window(seed):
    gen=torch.Generator().manual_seed(seed)
    coords=torch.tensor([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[4,0,0],[8,0,0]])
    return [dict(coordinates=coords.clone(),features=torch.randn(6,64,generator=gen)+.03*h,
                 pts=h/29.97,frame_index=h) for h in range(10)]


class TinyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec=nn.Module()
        self.codec.ae=nn.Module()
        self.codec.ae.decoder=nn.Module()
        self.codec.ae.decoder.out=nn.Linear(64,3)
        self.codec.ae.feature_heads=nn.Linear(3,3)
        self.codec.ae.gaussian_multiplier=nn.Linear(3,3)
        self.last_count=6

    def render_views(self,prediction,record):
        hidden=self.codec.ae.decoder.out(prediction['features']).mean(0)
        rgb=torch.sigmoid(self.codec.ae.feature_heads(hidden)+self.codec.ae.gaussian_multiplier(hidden))
        return torch.stack([rgb.expand(3,3,3), (rgb*.9).expand(3,3,3)])


class TinyData:
    def __init__(self):
        gen=torch.Generator().manual_seed(7)
        self.records=[{'real':torch.rand(3,3,3,generator=gen)} for _ in range(13)]
        self.teacher_targets=[torch.rand(2,3,3,3,generator=gen) for _ in range(13)]
        self.teacher_masks=[torch.ones(3,3,dtype=torch.bool) for _ in range(13)]
        self.roi=torch.ones(3,3,dtype=torch.bool)
        self.windows=[window(100+s) for s in range(4)]

    def window(self,start): return self.windows[start]

    def loss(self,image,start,h):
        error=(image-self.records[start+h]['real']).abs().mean(-1)
        full=error.mean()
        return full/10+full/10+(full/9 if h else 0), dict(full_l1=float(full.detach()),
            roi_l1=float(full.detach()),motion_l1=float(full.detach()) if h else 0.)


def direct_gradients(model,bridge,data,api,teacher):
    starts=[0,1,2,3]
    windows=[data.window(s) for s in starts]
    conditions=[api.make_condition(w[0]) for w in windows]
    encoded=model.encode_windows(windows)
    predictions=model.decode_latents(encoded['z_quantized'],conditions,initial_contexts=encoded['initial_contexts'])
    total=encoded['aux_loss']*.1
    for b,start in enumerate(starts):
        images=[bridge.render_views(predictions[b][h],data.records[start+h]) for h in range(10)]
        for h in range(10):
            base,_=data.loss(images[h][0],start,h)
            other,_=teacher.teacher_loss(images[h],data.teacher_targets[start+h],(0.,.25),data.teacher_masks[start+h])
            total=total+(base+other)/4
            if h:
                actual=data.records[start+h]['real']-data.records[start]['real']
                error=((images[h][0]-images[0][0])-actual).abs().mean(-1)
                total=total+(error.mean()+error[data.roi].mean())/(4*9)
    total.backward()
    params=list(model.parameters())+list(bridge.parameters())
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)


class RealAdapterStepTest(unittest.TestCase):
    def test_both_action_modes_execute_one_synthetic_cpu_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        parent=Path(os.environ['Z_ADAPTER_SOURCE_PARENT'])
        api=load_pinned(parent)
        model=api.AnchoredLDM(blocks=4,checkpoint_blocks=False).eval()
        model.initialize_output_units(window(100)[0]['features'])
        api.attach_spatial_action(model,True)
        with torch.no_grad():
            model.features.weight.copy_(torch.eye(64)*.2)
            model.features.bias.fill_(.015)
            model.decoder_backbone.Wo.weight.copy_(torch.eye(64)*.07)
        attach_adapters(model,s_ref=1.)
        pair=clone_pair(model)
        temporal=bound_module('_paired_test_temporal',parent/'gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py',
                              '6a1de737b3e87d88c6678352236258c031f9e8d66f17d79a04e2b944560562ef')
        teacher=bound_module('_paired_test_teacher',parent/'gaussiangpt_neoverse_gs_supervision_20261031_v10b/loss_step.py',
                             'ce304e2d861e303f1780c1267cf05146e7d5d4a13f8b30f54b28368608ea16ad')
        torch.manual_seed(19)
        bridge=TinyBridge()
        data=TinyData()
        for mode in ('continuous16','lfq16'):
            current=pair[mode].train()
            actual_bridge=copy.deepcopy(bridge).train()
            reference=copy.deepcopy(current)
            reference_bridge=copy.deepcopy(actual_bridge)
            # A zero-learning-rate test keeps parameters fixed for a direct gradient comparison.
            optimizer=torch.optim.AdamW(list(current.parameters())+list(actual_bridge.parameters()),lr=0.,weight_decay=0.)
            torch.manual_seed(345)
            stats,codes,ids=step_native(current,actual_bridge,optimizer,data,[0,1,2,3],
                make_condition=api.make_condition,temporal=temporal,teacher_loss=teacher.teacher_loss)
            torch.manual_seed(345)
            direct_gradients(reference,reference_bridge,data,api,teacher)
            self.assertEqual((stats['action_mode'],list(codes.shape)),(mode,[4,9,16]))
            self.assertGreater(stats['preclip_grad_norm'],0)
            self.assertTrue(all(all(x>0 for x in row) for row in stats['reconstruction_latent_grad_per_sample_h']))
            for left,right in ((current,reference),(actual_bridge,reference_bridge)):
                for (name,a),(other,b) in zip(left.named_parameters(),right.named_parameters()):
                    self.assertEqual(name,other)
                    if a.grad is None or b.grad is None:
                        self.assertTrue(a.grad is None and b.grad is None,name)
                    else:
                        torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=3e-4,msg=lambda m:name+' '+m)
            if mode=='continuous16': self.assertIsNone(ids)
            else: self.assertEqual(list(ids.shape),[4,9])


if __name__ == '__main__': unittest.main()

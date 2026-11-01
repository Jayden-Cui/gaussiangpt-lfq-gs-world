"""Conditional stage B: shared motion decoder plus one free code per window.

This is preparation, not an automatic training entry point. The caller must
first decide stage B from completed stage-A dynamic evidence.
"""
from pathlib import Path
import importlib.util
import math
import sys
import torch
from torch import nn


STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')


def load_bound_sources(device='cpu'):
    spec=importlib.util.spec_from_file_location('_stage_b_stage_a_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    fm,data,runtime=helper.model_modules(orig)
    if binding.sha(p['teacher_checkpoint'])!=p['teacher_checkpoint_sha256']:
        raise RuntimeError('Two-window decoder checkpoint changed')
    if binding.sha(p['posterior_checkpoint'])!=p['posterior_checkpoint_sha256']:
        raise RuntimeError('Two-window posterior checkpoint changed')
    parent=binding.read(Path(p['parent_root'])/'protocol.json')
    if parent['starts']!=[35,123] or parent['horizon']!=9:
        raise RuntimeError('Two-window code ordering changed')
    decoder,posterior=fm.build_models(p,'cpu',train_decoder=True)
    if runtime.digest(decoder)!=p['teacher_tensor_digest']:
        raise RuntimeError('Wrong initial two-window decoder')
    if runtime.digest(posterior)!=p['posterior_tensor_digest']:
        raise RuntimeError('Wrong initial two-window Q-Former')
    if any(param.requires_grad!=name.startswith(fm.TRAIN_PREFIXES)
           for name,param in decoder.named_parameters()):
        raise RuntimeError('Trainable decoder parameter scope differs')
    posterior.requires_grad_(False).eval()
    store=data.WindowStore(p)
    frozen=fm.frozen_digest(decoder)
    if frozen!=store.manifest['frozen_model_digest']:
        raise RuntimeError('Initial state encoder/context differs from the frozen cache')
    return dict(binding=binding,c=c,helper=helper,p=p,paths=paths,receipt=receipt,
                decoder=decoder.to(device),posterior=posterior.to(device),
                store=store,fm=fm,runtime=runtime,frozen_digest=frozen)


@torch.no_grad()
def initialize_codes(posterior,decoder,store,starts,device):
    """Use the matching two-window Q-Former gauge, then preserve both anchor codes."""
    if len(set(starts))!=len(starts) or any(not 0<=s<168 for s in starts):
        raise ValueError('Invalid training windows')
    if decoder.mode!='free16' or tuple(decoder.codes.shape)!=(2,9,16):
        raise RuntimeError('Wrong learned-support decoder')
    posterior.eval();values=[]
    for start in starts:
        window=store.get(start,device=device)
        codes=posterior([window['memories']])['codes'][0]
        if tuple(codes.shape)!=(9,16) or not bool(torch.isfinite(codes).all()):
            raise RuntimeError('Invalid warm-start posterior code')
        if start in (35,123):
            # These are the free codes paired with the exact two-window decoder.
            codes=decoder.codes[(35,123).index(start)]
        values.append(codes.detach().cpu().clone())
        del window,codes
    return torch.stack(values)


class FreeCodes(nn.Module):
    def __init__(self,starts,initial,device):
        super().__init__()
        if len(set(starts))!=len(starts) or tuple(initial.shape)!=(len(starts),9,16):
            raise ValueError('Free-code initialization does not match window order')
        self.starts=list(starts);self.index={s:i for i,s in enumerate(starts)}
        self.values=nn.ParameterList([nn.Parameter(initial[i].detach().clone().to(device)) for i in range(len(starts))])

    def code(self,start):return self.values[self.index[start]]

    def stack(self):return torch.stack([x.detach().cpu() for x in self.values])


def optimizer_for(decoder,bank,p,decoder_lr,code_lr):
    trainable=[x for x in decoder.parameters() if x.requires_grad]
    if not trainable or not list(bank.parameters()):raise RuntimeError('Missing trainable decoder or code parameters')
    return torch.optim.Adam([dict(params=trainable,lr=decoder_lr),dict(params=list(bank.parameters()),lr=code_lr)],
                            betas=tuple(p['betas']),eps=p['eps'])


def train_step(decoder,bank,bridge,store,std,optimizer,batch,p,*,probe_rgb=False):
    from core import image_losses
    from support_decoder import state_loss,select_state
    from runtime import render
    if len(batch)!=4:raise ValueError('Expected four accumulated window/horizon examples')
    decoder_trainable=[x for x in decoder.parameters() if x.requires_grad]
    expected=decoder_trainable+list(bank.parameters())
    actual=[x for group in optimizer.param_groups for x in group['params']]
    if len(actual)!=len(expected) or {id(x) for x in actual}!={id(x) for x in expected}:
        raise RuntimeError('Optimizer must contain exactly shared-decoder and free-code parameters')
    if any(x.requires_grad for x in bridge.parameters()):raise RuntimeError('Native GS codec is not frozen')
    optimizer.zero_grad(set_to_none=True);rows=[]
    for member in batch:
        start,h=member['start'],member['horizon']
        if start not in bank.index or not 1<=h<=9:raise ValueError('Unknown training example')
        window=store.get(start);code=bank.code(start)
        prediction=decoder(window['condition'],code)
        latent,terms=state_loss(prediction,window['targets'],std)
        state=select_state(window['condition'].coordinates,prediction['features'][h-1],
                           prediction['logits'][h-1],p['max_predicted_latent_nodes'])
        sample=store.sample(start,h)
        images,gs,sh=render(bridge,state,sample['record'])
        rgb,rgb_terms=image_losses(images[0],sample['record']['real'],window['roi'],sample['motion'])
        if probe_rgb:
            gradients=torch.autograd.grad(rgb,(code,decoder.base.features.weight),retain_graph=True)
            rgb_grads=[float(x.norm()) for x in gradients]
            if not all(math.isfinite(x) and x>0 for x in rgb_grads):
                raise RuntimeError('RGB gradients did not reach free code and decoder features')
        loss=rgb+.1*latent
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
        row=dict(start=start,horizon=h,loss=float(loss.detach()),rgb_loss=float(rgb.detach()),
                 latent_loss=float(latent.detach()),occupancy_loss=float(terms['occupancy'].detach()),
                 feature_loss=float(terms['feature'].detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
        if probe_rgb:row['rgb_to_code_and_decoder_gradients']=rgb_grads
        (loss/len(batch)).backward();rows.append(row)
        del window,code,prediction,latent,terms,state,sample,images,gs,sh,rgb,rgb_terms,loss
    selected={x['start'] for x in batch}
    for start in bank.starts:
        if start not in selected and bank.code(start).grad is not None:
            raise RuntimeError('Unselected window unexpectedly acquired a gradient')
    if any(x.grad is not None and not bool(torch.isfinite(x.grad).all()) for x in expected):
        raise RuntimeError('Nonfinite shared-decoder or free-code gradient')
    if any(x.grad is not None for x in decoder.parameters() if not x.requires_grad):
        raise RuntimeError('Frozen initial encoder or unused anchor codes acquired gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(expected,p['grad_clip']));optimizer.step()
    if any(not bool(torch.isfinite(x).all()) for x in expected):raise RuntimeError('Nonfinite updated parameter')
    return dict(windows=rows,preclip_gradient_norm=norm,clip_triggered=norm>p['grad_clip'])

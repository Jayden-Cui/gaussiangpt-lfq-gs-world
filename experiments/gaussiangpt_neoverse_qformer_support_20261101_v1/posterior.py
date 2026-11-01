"""Native-64 prefix Q-Former; decoder receives only S0 and continuous codes."""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PrefixMemory:
    keys: torch.Tensor
    coordinates: torch.Tensor


@torch.no_grad()
def make_memories(base, window, initial_only=False):
    if len(window)!=10:raise ValueError('Expected ten real time slots')
    states=window
    if initial_only:
        states=[dict(window[0],pts=state['pts']) for state in window]
    memories=[]
    for h in range(1,10):
        # Building the union before slicing would allow later geometry to enter.
        prepared=base.prepare(states[:h+1])
        present=prepared['present']
        keys=prepared['hidden'][present].detach()
        coordinates=prepared['coordinates'][None].expand(h+1,-1,-1)[present].detach()
        if keys.shape!=(len(coordinates),64) or not len(keys):raise ValueError('Invalid prefix memory')
        memories.append(PrefixMemory(keys,coordinates))
    return memories


class PrefixPosterior(nn.Module):
    def __init__(self,base,projection_std=.01):
        super().__init__()
        self.posteriors=copy.deepcopy(base.posteriors)
        self.queries=copy.deepcopy(base.queries)
        self.position=copy.deepcopy(base.position)
        self.project_in=copy.deepcopy(base.quantizer.project_in)
        if self.project_in.weight.shape!=(16,64) or self.queries.weight.shape!=(9,64):
            raise ValueError('Native bottleneck dimension changed')
        nn.init.normal_(self.project_in.weight,std=projection_std)
        if self.project_in.bias is not None:nn.init.zeros_(self.project_in.bias)
        self.requires_grad_(True)

    def forward(self,windows):
        from model_core import qformer_sequence
        if not windows or any(len(w)!=9 for w in windows):raise ValueError('Nine prefixes required')
        vectors=[]
        for h in range(9):
            keys=[w[h].keys for w in windows]
            positions=[self.position(w[h].coordinates) for w in windows]
            k=pad_sequence(keys,batch_first=True)
            pos=pad_sequence(positions,batch_first=True)
            lengths=torch.tensor([len(v) for v in keys],device=k.device)
            valid=torch.arange(k.shape[1],device=k.device)[None]<lengths[:,None]
            query=self.queries.weight[h:h+1][None].expand(len(windows),-1,-1)
            vectors.append(qformer_sequence(self.posteriors[h],k,pos,query,valid))
        vectors=torch.cat(vectors,dim=1)
        codes=self.project_in(vectors.float())
        if codes.shape!=(len(windows),9,16) or not bool(torch.isfinite(codes).all()):
            raise RuntimeError('Invalid continuous posterior code')
        return dict(vectors=vectors,codes=codes)


def build_pair(p,device):
    from runtime import build_model,digest
    torch.manual_seed(p['seed'])
    teacher=build_model(p,'free16','cpu')
    checkpoint=torch.load(p['teacher_checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint['step']!=1200 or checkpoint['mode']!='free16':raise RuntimeError('Wrong teacher')
    teacher.load_state_dict(checkpoint['model'],strict=True)
    if digest(teacher)!=p['teacher_tensor_digest']:raise RuntimeError('Teacher tensor binding mismatch')
    teacher.requires_grad_(False).eval()
    posterior=PrefixPosterior(teacher.base,p['code_projection_init_std'])
    return teacher.to(device),posterior.to(device)

"""Native-64 building blocks, independent of the GS renderer.

Spatial masked normalization follows explicit axes; encoder never normalizes
across time. Decoder may use full quantized-window statistics, as in the
released VideoWorld decoder. This does not grant access to unquantized futures.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MaskedGroupNorm64(nn.Module):
    def __init__(self, across_time=False, eps=1e-6):
        super().__init__()
        self.across_time=across_time; self.eps=eps
        self.weight=nn.Parameter(torch.ones(64)); self.bias=nn.Parameter(torch.zeros(64))
    def forward(self,x,active):
        assert x.ndim==3 and x.shape[-1]==64 and active.shape==x.shape[:2]
        clean=torch.where(active[...,None],x,torch.zeros_like(x))
        g=clean.reshape(*x.shape[:2],4,16)
        m=active[...,None,None].to(x.dtype)
        dims=(0,1,3) if self.across_time else (1,3)
        count=(m.sum(dims,keepdim=True)*16).clamp_min(1)
        mean=(g*m).sum(dims,keepdim=True)/count
        var=((g-mean).square()*m).sum(dims,keepdim=True)/count
        y=((g-mean)*torch.rsqrt(var+self.eps)).reshape_as(x)
        return (y*self.weight+self.bias)*active[...,None]

class TemporalResidual64(nn.Module):
    def __init__(self,dilation=1,*,across_time=False):
        super().__init__()
        self.dilation=dilation
        self.norm1=MaskedGroupNorm64(across_time)
        self.norm2=MaskedGroupNorm64(across_time)
        self.conv1=nn.Conv1d(64,64,3,dilation=dilation)
        self.conv2=nn.Conv1d(64,64,3,dilation=dilation)
    def convolve(self,x,active,norm,conv):
        x=F.silu(norm(x,active)).permute(1,2,0)
        return conv(F.pad(x,(2*self.dilation,0))).permute(2,0,1)*active[...,None]
    def forward(self,x,active):
        y=self.convolve(x,active,self.norm1,self.conv1)
        y=self.convolve(y,active,self.norm2,self.conv2)
        return (x+y)*active[...,None]

class AdaptiveWindowNorm64(nn.Module):
    """Full-window code conditioning; intentionally not prefix-causal."""
    def __init__(self):
        super().__init__()
        self.norm=MaskedGroupNorm64(across_time=True)
        self.affine=nn.Linear(64,128)
    def forward(self,x,condition,active):
        count=active.sum().clamp_min(1)
        clean=torch.where(active[...,None],condition,torch.zeros_like(condition))
        pooled=clean.sum((0,1))/count
        scale,shift=self.affine(pooled).chunk(2,-1)
        return (self.norm(x,active)*(1+scale)+shift)*active[...,None]

class Position3D(nn.Module):
    def __init__(self,coordinate_scale=.032):
        super().__init__()
        self.coordinate_scale=float(coordinate_scale)
        self.register_buffer('frequencies',2**torch.arange(6,dtype=torch.float32))
        self.project=nn.Linear(39,64,bias=False)
    def forward(self,coordinates):
        xyz=coordinates.float()*self.coordinate_scale
        phase=xyz[...,None]*self.frequencies*torch.pi
        return self.project(torch.cat([xyz,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1))

def masked_attention(module,q,k,v,valid):
    q=module._separate_heads(module.q_proj(q),module.num_heads)
    k=module._separate_heads(module.k_proj(k),module.num_heads)
    v=module._separate_heads(module.v_proj(v),module.num_heads)
    logits=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1])
    if valid is not None:
        if not bool(valid.any(1).all()): raise ValueError('Empty attention prefix')
        logits=logits.masked_fill(~valid[:,None,None,:],float('-inf'))
    return module.out_proj(module._recombine_heads(logits.softmax(-1)@v))

def qformer_sequence(module,keys,positions,query,valid):
    for block in module.layers:
        query=block.norm2(query+masked_attention(block.cross_attn_token_to_image,query,keys+positions,keys,valid))
        query=block.norm3(query+block.mlp(query))
    return module.norm_final_attn(query+masked_attention(
        module.final_attn_token_to_image,query+query,keys+positions,keys,valid))


"""Function-preserving width/code expansion of the bound fifth-round decoder."""
import copy,math
import torch
from torch import nn


def repeated_linear(source,out_repeat=1,in_repeat=1,*,affine=False,break_symmetry=False):
    old=source.weight.detach();o,i=old.shape
    result=nn.Linear(i*in_repeat,o*out_repeat,bias=source.bias is not None)
    # Conditional affine outputs concatenate scale and shift; preserve that ordering.
    rows=torch.arange(o).reshape(2,-1).repeat_interleave(out_repeat,0).reshape(-1) if affine else torch.arange(o).repeat(out_repeat)
    weight=old[rows].repeat(1,in_repeat)/in_repeat
    if break_symmetry and in_repeat==2:
        noise=torch.randn(weight.shape[0],i)*(.01*float(old.std().clamp_min(1e-6)))
        weight=weight+torch.cat((noise,-noise),1)
    with torch.no_grad():
        result.weight.copy_(weight)
        if source.bias is not None:result.bias.copy_(source.bias.detach()[rows])
    return result


def repeated_conv(source,r):
    result=nn.Conv1d(source.in_channels*r,source.out_channels*r,source.kernel_size,
        dilation=source.dilation,bias=source.bias is not None)
    w=source.weight.detach().repeat(r,r,1)/r
    if r==2:
        noise=torch.randn(w.shape[0],source.in_channels,w.shape[2])*(.01*float(source.weight.detach().std().clamp_min(1e-6)))
        w=w+torch.cat((noise,-noise),1)
    with torch.no_grad():
        result.weight.copy_(w)
        if source.bias is not None:result.bias.copy_(source.bias.detach().repeat(r))
    return result


def repeated_ln(source,r):
    result=nn.LayerNorm(source.normalized_shape[0]*r,eps=source.eps)
    with torch.no_grad():result.weight.copy_(source.weight.detach().repeat(r));result.bias.copy_(source.bias.detach().repeat(r))
    return result


class InitialReferenceNorm(nn.Module):
    def __init__(self,source,r):
        super().__init__();self.eps=source.eps;self.width=64*r
        self.weight=nn.Parameter(source.weight.detach().repeat(r));self.bias=nn.Parameter(source.bias.detach().repeat(r))
    def forward(self,x,active):
        if x.shape[0]<4 or x.shape[-1]!=self.width or not active[:4].any():raise ValueError('Invalid initial reference')
        clean=torch.where(active[...,None],x,torch.zeros_like(x));g=clean.reshape(*x.shape[:2],self.width//16,16)
        ref=g[:4];mask=active[:4,:,None,None].to(x.dtype);count=mask.sum((0,1,3),keepdim=True)*16
        mean=(ref*mask).sum((0,1,3),keepdim=True)/count
        var=((ref-mean).square()*mask).sum((0,1,3),keepdim=True)/count
        y=((g-mean)*torch.rsqrt(var+self.eps)).reshape_as(x)*self.weight+self.bias
        return torch.where(active[...,None],y,torch.zeros_like(y))


class InitialConditionNorm(nn.Module):
    def __init__(self,source,r):
        super().__init__();self.norm=InitialReferenceNorm(source.norm,r)
        self.affine=repeated_linear(source.affine,r,r,affine=True,break_symmetry=True)
    def forward(self,x,condition,active):
        valid=active[:4,:,None];clean=torch.where(valid,condition[:4],torch.zeros_like(condition[:4]))
        pooled=clean.sum((0,1))/active[:4].sum();scale,shift=self.affine(pooled).chunk(2,-1)
        y=self.norm(x,active)*(1+scale)+shift
        return torch.where(active[...,None],y,torch.zeros_like(y))


class SpatialAction(nn.Module):
    def __init__(self,source,r):
        super().__init__();self.width=64*r;self.base=copy.deepcopy(source.base);self.base.width=self.width
        for block in self.base.blocks:
            block.spatial.width=self.width
            old=block.spatial.message
            block.spatial.message=nn.Sequential(repeated_ln(old[0],r),repeated_linear(old[1],r,r,break_symmetry=True),
                nn.SiLU(),repeated_linear(old[3],r,r,break_symmetry=True))
            block.temporal.norm1=InitialReferenceNorm(block.temporal.norm1,r)
            block.temporal.norm2=InitialReferenceNorm(block.temporal.norm2,r)
            block.temporal.conv1=repeated_conv(block.temporal.conv1,r);block.temporal.conv2=repeated_conv(block.temporal.conv2,r)
            block.condition_norm=InitialConditionNorm(block.condition_norm,r)
        self.initial_norm=repeated_ln(source.initial_norm,r);self.action_norm=repeated_ln(source.action_norm,r)
        self.Wf=repeated_linear(source.Wf,r,r,break_symmetry=True);self.Wz=repeated_linear(source.Wz,r,r,break_symmetry=True)
        self.Wo=repeated_linear(source.Wo,r,r,break_symmetry=True)
        self.position=copy.deepcopy(source.position);self.position.project=repeated_linear(source.position.project,r,1)
    def forward(self,x,coords,active):
        hidden=self.base(x,coords,active,condition=x)
        valid=active[4:]&active[3][None]
        initial=torch.where(active[3,:,None],x[3],torch.zeros_like(x[3]))
        actions=torch.where(valid[...,None],x[4:],torch.zeros_like(x[4:]))
        spatial=torch.tanh(self.Wf(self.initial_norm(initial))+self.position(coords))
        delta=self.Wo(spatial[None]*self.Wz(self.action_norm(actions)))
        return torch.cat((hidden[:4],hidden[4:]+torch.where(valid[...,None],delta,torch.zeros_like(delta))),0)


class CapacityDecoder(nn.Module):
    def __init__(self,reference,code_dim,width,seed=20261102):
        super().__init__()
        if code_dim not in (16,64) or width not in (64,128):raise ValueError('Unregistered capacity arm')
        self.code_dim,self.width=code_dim,width;r=width//64
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.backbone=SpatialAction(reference.base.decoder_backbone,r)
            self.output_norm=repeated_ln(reference.base.output_norm,r)
            self.features=repeated_linear(reference.base.features,1,r,break_symmetry=True)
            self.geometry=repeated_linear(reference.geometry,r,1)
            self.occupancy=repeated_linear(reference.occupancy,1,r,break_symmetry=True)
            self.action=nn.Linear(code_dim,width,bias=False)
            with torch.no_grad():
                self.action.weight[:,:16].copy_(reference.base.quantizer.project_out.weight.detach().repeat(r,1))
                if code_dim>16:nn.init.normal_(self.action.weight[:,16:],std=.02)
        self.register_buffer('output_std',reference.base.output_std.detach().clone())
        self.requires_grad_(True)
    def forward(self,condition,codes):
        if codes.shape!=(9,self.code_dim) or codes.dtype!=torch.float32:raise ValueError('Code shape/dtype mismatch')
        initial=condition.context.repeat(1,1,self.width//64)+self.geometry(condition.geometry)[None]
        latent=self.action(codes);n=len(condition.coordinates)
        sequence=torch.cat((initial,latent[:,None].expand(-1,n,-1)),0)
        active=torch.ones(sequence.shape[:2],device=sequence.device,dtype=torch.bool)
        hidden=self.backbone(sequence,condition.coordinates,active)[4:]
        normalized=self.output_norm(hidden);raw=self.features(normalized)
        features=condition.features[None]+3*self.output_std*torch.tanh(raw)
        logits=torch.where(condition.initial_present,2.,-2.)[None]+self.occupancy(normalized).squeeze(-1)
        return dict(features=features,logits=logits,raw_residual=raw)


class FreeCodes(nn.Module):
    def __init__(self,original,dim):
        super().__init__()
        if original.shape!=(168,9,16):raise ValueError('Expected all original free codes')
        initial=torch.nn.functional.pad(original,(0,dim-16))
        self.values=nn.ParameterList([nn.Parameter(v.clone()) for v in initial])
    def code(self,start):return self.values[start]
    def stack(self):return torch.stack([p.detach().cpu() for p in self.values])

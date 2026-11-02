"""Meaningful mask, supervision-gradient, physical-frame and paired-schedule checks."""
import numpy as np
import torch
from object_masks import color_group,object_loss,regions
from trial_utils import ROOT,read,write,verify,schedule,sha,require


def main():
    torch.set_num_threads(2);p=verify();checks=[]
    image=np.zeros((252,448,3),np.uint8);image[40:50,200:210]=[0,180,0];image[10,300]=[0,180,0];image[30:45,30:45]=[0,180,0]
    mask,stats=color_group(image,p['objects'][1],20)
    require(mask.sum()==100 and stats['removed_pixels']==1,'Component filter or green-triangle exclusion failed');checks.append('color_components_and_spatial_scope')
    bank=torch.zeros((177,2,252,448),dtype=torch.bool);bank[123,1,50,200]=True;bank[126,1,60,220]=True
    region=regions(bank,123,3,3)
    require(region[1,47:54,197:204].all() and region[1,57:64,217:224].all() and region.sum()==98,'Initial/target union or dilation failed')
    require(torch.equal(regions(bank,124,2,3)[1],regions(bank,125,1,3)[1]),'Same physical target has inconsistent masks');checks.append('source_target_union_and_frame_indexing')
    pred=torch.ones((252,448,3),requires_grad=True);target=torch.zeros_like(pred,requires_grad=False)
    value,terms=object_loss(pred,target,region);require(float(value)==1 and terms['present']==[False,True],'Empty group normalization failed')
    value.backward();require(bool((pred.grad[region.any(0)]>0).all()) and bool((pred.grad[~region.any(0)]==0).all()),'Object loss has wrong gradient support');checks.append('nonzero_object_gradient_only_inside_union')
    zero,_=object_loss(pred,target,torch.zeros_like(region));require(float(zero)==0 and torch.isfinite(zero),'Empty mask yields NaN');checks.append('empty_mask_is_finite_zero')
    items,report=schedule(p);require(len(items)==846 and items[0]['step']==2116 and items[-1]['step']==2961,'Continuation indices differ');checks.append('matched_two_sweep_continuation_preserves_parent_prefix')
    # No target masks are passed to a decoder; the training edit is confined to the loss after render.
    import ast
    tree=ast.parse((ROOT/'loss_step.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='decoder']
    require(len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords,'Decoder signature acquired supervision inputs');checks.append('masks_not_added_to_decoder_inputs')
    write(ROOT/'CPU_TESTS.json',dict(status='passed',checks=checks,schedule=report,code_sha256=sha(ROOT/'CODE.sha256')))
    print('PASS: '+', '.join(checks))


if __name__=='__main__':main()

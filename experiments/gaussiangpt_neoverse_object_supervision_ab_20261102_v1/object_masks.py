"""TRAIN-RGB color-group supervision, indexed only by physical source frame."""
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F


def color_group(rgb,spec,min_pixels):
    if rgb.dtype!=np.uint8 or rgb.shape!=(252,448,3):raise ValueError('Expected native-resolution uint8 RGB')
    hsv=np.asarray(Image.fromarray(rgb).convert('HSV'))
    h0,h1=spec['hue_uint8'];x0,y0,x1,y1=spec['roi_xyxy']
    region=np.zeros(rgb.shape[:2],bool);region[y0:y1,x0:x1]=True
    selected=region&(hsv[...,0]>=h0)&(hsv[...,0]<=h1)&(hsv[...,1]>=spec['saturation_min'])&(hsv[...,2]>=spec['value_min'])
    labels,count=ndimage.label(selected,structure=np.ones((3,3),bool))
    sizes=np.bincount(labels.ravel());keep=sizes>=min_pixels;keep[0]=False
    result=keep[labels]
    return result,dict(raw_pixels=int(selected.sum()),kept_pixels=int(result.sum()),components=count,
                      kept_components=int(keep.sum()),removed_pixels=int(selected.sum()-result.sum()))


def regions(masks,start,horizon,radius):
    if masks.dtype!=torch.bool or masks.shape!=(177,2,252,448):raise ValueError('Invalid source mask bank')
    if not 0<=start<168 or not 1<=horizon<=9 or radius!=3:raise ValueError('Invalid frame pair or dilation')
    union=masks[start]|masks[start+horizon]
    return F.max_pool2d(union.float()[None],2*radius+1,stride=1,padding=radius)[0].bool()


def object_loss(image,target,mask):
    if image.shape!=target.shape or image.shape!=(252,448,3) or mask.shape!=(2,252,448) or mask.dtype!=torch.bool:
        raise ValueError('Object loss input mismatch')
    if mask.requires_grad or target.requires_grad:raise ValueError('Supervision must be fixed')
    error=(image-target).abs().mean(-1)
    area=mask.sum((-2,-1));present=area>0
    values=(error[None]*mask).sum((-2,-1))/area.clamp_min(1)
    value=(values*present).sum()/present.sum().clamp_min(1)
    return value,dict(per_color=values.detach().cpu().tolist(),area=area.cpu().tolist(),present=present.cpu().tolist())


class MaskBank:
    def __init__(self,path):
        with np.load(path) as data:
            self.masks=torch.from_numpy(data['masks'].copy())
            if not np.array_equal(data['frame_indices'],np.arange(177)):raise ValueError('Mask frame order differs')
        if self.masks.shape!=(177,2,252,448) or self.masks.dtype!=torch.bool:raise ValueError('Wrong mask layout')

    def get(self,start,horizon,device):
        return regions(self.masks,start,horizon,3).to(device)

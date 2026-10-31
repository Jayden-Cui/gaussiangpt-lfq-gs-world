import torch
from torch.nn import functional as F
from motion_common import render_points
from qmotion import select_support

def calibrated_occupancy_loss(logits,labels):
    positives=labels.sum().clamp_min(1.);weight=((labels.numel()-positives)/positives).clamp(1.,50.)
    return F.binary_cross_entropy_with_logits(logits+weight.log(),labels,pos_weight=weight)

def all_losses(codec,prediction,target_state,target_indices,target_images,vgg,*,force_target=False,rendering=True,max_tokens=6000):
    labels=torch.zeros_like(prediction['occupancy_logits']);labels[target_indices]=1
    terms={'feature':F.mse_loss(prediction['features'][target_indices],target_state['features']),
        'occupancy':calibrated_occupancy_loss(prediction['occupancy_logits'],labels)}
    aux={}
    if rendering:
        coordinates,features,keep=select_support(prediction,target_indices,force_target,max_tokens)
        points=codec.decode_state(coordinates,features)
        image=render_points(points,target_images,0);nearby=render_points(points,target_images,1)
        terms.update(real_l1=(image-target_images['real']).abs().mean(),vgg19=vgg(image,target_images['real']),
            teacher_l1=(image-target_images['targets'][0]).abs().mean(),
            frozen_target_l1=(image-target_state['reconstruction'][0]).abs().mean(),
            workspace_roi_l1=(image[38:227,67:380]-target_images['real'][38:227,67:380]).abs().mean(),
            pseudo_view_l1=(nearby-target_images['targets'][1]).abs().mean())
        predicted=prediction['occupancy_logits']>0
        aux={'predicted_tokens':int(keep.sum()),'raw_predicted_tokens':int(predicted.sum()),
            'forced_additions':int((keep&~predicted).sum()),'image':image}
    return terms,aux


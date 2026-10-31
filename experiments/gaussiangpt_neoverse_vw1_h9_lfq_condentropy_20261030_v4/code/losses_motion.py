import torch
from torch.nn import functional as F
from motion_common import render_points
from qmotion import select_support

def all_losses(codec,prediction,target_state,target_indices,target_images,*,rendering):
    labels=torch.zeros_like(prediction['occupancy_logits']);labels[target_indices]=1
    positive=labels.sum().clamp_min(1)
    weight=((labels.numel()-positive)/positive).clamp(1,50)
    terms=dict(feature=F.mse_loss(prediction['features'][target_indices],target_state['features'])/target_state['features'].square().mean().clamp_min(1.),
        occupancy=F.binary_cross_entropy_with_logits(prediction['occupancy_logits']+weight.log(),labels,pos_weight=weight))
    if rendering:
        # The target support is NEVER given to selection or the frozen decoder.
        coordinates,features,keep=select_support(prediction,max_tokens=6000)
        points=codec.decode_state(coordinates,features)
        image=render_points(points,target_images,0)
        nearby=render_points(points,target_images,1)
        terms.update(future_rgb_l2=F.mse_loss(image,target_images['real']),
            future_codec_l2=F.mse_loss(image,target_state['reconstruction'][0]),
            future_pseudoview_l2=F.mse_loss(nearby,target_images['targets'][1]))
    return terms

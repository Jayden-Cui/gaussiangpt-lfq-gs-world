"""All-ten-frame latent supervision for integration checks.

This is only the latent/occupancy component. It does not replace the required
real-image reconstruction objective, perceptual/GAN choice or GS renderer.
No target support is passed to the decoder.
"""
import torch
from torch.nn import functional as F

def target_indices(candidates,target):
    rows={tuple(c):i for i,c in enumerate(candidates.detach().cpu().tolist())}
    target_rows=[tuple(c) for c in target.detach().cpu().tolist()]
    missing=[c for c in target_rows if c not in rows]
    if missing:raise ValueError('Target support outside initial-only candidate domain')
    return torch.tensor([rows[c] for c in target_rows],device=candidates.device,dtype=torch.long)

def latent_reconstruction_loss(predictions,windows,feature_weight=.1,occupancy_weight=.2):
    if len(predictions)!=len(windows) or not windows:raise ValueError('Batch mismatch')
    if any(len(x)!=10 for x in predictions+windows):raise ValueError('All ten real frames must contribute')
    per_frame=[]
    for predicted,states in zip(predictions,windows):
        for p,target in zip(predicted,states):
            indices=target_indices(p['coordinates'],target['coordinates'])
            labels=torch.zeros_like(p['occupancy_logits']);labels[indices]=1
            pos_weight=((len(labels)-len(indices))/max(len(indices),1))
            pos_weight=labels.new_tensor(pos_weight).clamp(1,50)
            feature=F.mse_loss(p['features'][indices],target['features'])/target['features'].square().mean().clamp_min(1.)
            occupancy=F.binary_cross_entropy_with_logits(p['occupancy_logits']+pos_weight.log(),labels,pos_weight=pos_weight)
            per_frame.append(feature_weight*feature+occupancy_weight*occupancy)
    values=torch.stack(per_frame).reshape(len(windows),10)
    return dict(loss=values.mean(),per_frame=values)


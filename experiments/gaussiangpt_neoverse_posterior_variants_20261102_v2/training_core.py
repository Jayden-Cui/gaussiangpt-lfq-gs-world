"""Teacher-conditioned continuous posterior losses and same-initial-state control.

This module is preparatory: no teacher is selected and no training is launched.
The teacher decoder and native GS codec must be frozen by the caller.
"""
import math
import torch


def static_states(states):
    """Use S0 at each original monotonic timestamp without changing its coordinates."""
    if len(states)!=10:raise ValueError('Expected ten states')
    initial=states[0]
    return [dict(initial,pts=state['pts']) for state in states]


def infer_codes(model,window,*,static=False):
    if model.variant=='C0':
        result=model.forward_cached(window['initial_memories'] if static else window['memories'])
    else:
        states=static_states(window['states']) if static else window['states']
        result=model.forward_states(states)
    if tuple(result.shape)!=(9,16) or not bool(torch.isfinite(result).all()):
        raise RuntimeError('Expected finite continuous [9,16] codes')
    return result


def teacher_scale(codes):
    """One TRAIN-only scalar; the frozen teacher fixes the code coordinate system."""
    if tuple(codes.shape)!=(168,9,16) or not bool(torch.isfinite(codes).all()):
        raise ValueError('Expected all 168 TRAIN-window teacher codes')
    return codes.detach().square().mean().clamp_min(1e-8)


def code_loss(predicted,target,scale):
    if tuple(predicted.shape)!=(9,16) or tuple(target.shape)!=(9,16):raise ValueError('Unexpected code shape')
    if not bool(torch.isfinite(scale)) or float(scale)<=0:raise ValueError('Invalid teacher scale')
    return (predicted-target.detach()).square().mean()/scale


def step(model,decoder,bridge,store,std,optimizer,batch,teacher_codes,scale,p,phase,*,probe_rgb=False):
    """One B4 accumulated update; phase is `distill` or `joint`.

    Joint RGB contains full-image, hand-region and motion-weighted terms from
    the pinned implementation. No ground-truth future support enters decoder.
    """
    from core import image_losses
    from support_decoder import state_loss,select_state
    from runtime import render
    if phase not in ('distill','joint') or len(batch)!=4:raise ValueError('Expected a B4 distill or joint batch')
    if any(x.requires_grad for x in decoder.parameters()):raise RuntimeError('Teacher motion decoder is not frozen')
    if any(x.requires_grad for x in bridge.parameters()):raise RuntimeError('GS codec is not frozen')
    expected={id(x) for x in model.parameters() if x.requires_grad}
    actual=[id(x) for group in optimizer.param_groups for x in group['params']]
    if set(actual)!=expected or len(actual)!=len(expected):raise RuntimeError('Optimizer must own only posterior parameters')
    optimizer.zero_grad(set_to_none=True);rows=[]
    for item in batch:
        start,h=item['start'],item['horizon']
        window=store.get(start);codes=infer_codes(model,window)
        distill=code_loss(codes,teacher_codes[start].to(codes.device),scale.to(codes.device))
        loss=distill;row=dict(start=start,horizon=h,code_loss=float(distill.detach()))
        if phase=='joint':
            prediction=decoder(window['condition'],codes)
            latent,terms=state_loss(prediction,window['targets'],std)
            state=select_state(window['condition'].coordinates,prediction['features'][h-1],
                               prediction['logits'][h-1],p['max_predicted_latent_nodes'])
            sample=store.sample(start,h)
            images,gs,sh=render(bridge,state,sample['record'])
            rgb,rgb_terms=image_losses(images[0],sample['record']['real'],window['roi'],sample['motion'])
            if probe_rgb:
                variables=[model.posterior.queries.weight,model.posterior.project_in.weight]
                if model.encoder is not None:variables.append(model.encoder.input.weight)
                grads=torch.autograd.grad(rgb,variables,retain_graph=True)
                values=[float(grad.norm()) for grad in grads]
                if not all(math.isfinite(v) and v>0 for v in values):raise RuntimeError('RGB gradient did not reach posterior')
                row['rgb_only_gradients']=dict(query=values[0],projection=values[1],
                    encoder_input=values[2] if len(values)>2 else None)
            loss=rgb+.1*latent+.1*distill
            row.update(rgb_loss=float(rgb.detach()),motion_mae=float(rgb_terms['motion_mae'].detach()),
                latent_loss=float(latent.detach()),occupancy_loss=float(terms['occupancy'].detach()),
                feature_loss=float(terms['feature'].detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite posterior loss')
        row['loss']=float(loss.detach());(loss/len(batch)).backward();rows.append(row)
        if phase=='joint':del prediction,latent,terms,state,sample,images,gs,sh,rgb,rgb_terms
        del window,codes,distill,loss
    params=list(model.parameters())
    if any(x.grad is not None and not bool(torch.isfinite(x.grad).all()) for x in params):
        raise RuntimeError('Nonfinite posterior gradient')
    if any(x.grad is not None for x in decoder.parameters()):raise RuntimeError('Teacher decoder acquired gradients')
    gradient_norm=float(torch.nn.utils.clip_grad_norm_(params,p['grad_clip']))
    optimizer.step()
    return dict(windows=rows,preclip_gradient_norm=gradient_norm,clip_triggered=gradient_norm>p['grad_clip'])


@torch.no_grad()
def same_initial_control(model,decoder,bridge,window,sample,h,p):
    """Compare motion and static prefixes using the same S0, target camera and codec.

    This is a diagnostic, not a cross-video action-migration test. Both reference
    images are produced by the frozen codec under the same frame-h camera.
    """
    from support_decoder import select_state
    from runtime import render
    if not 1<=h<=9:raise ValueError('Expected future horizon')
    if model.training or decoder.training or bridge.training:raise RuntimeError('Static control requires evaluation mode')
    moving_codes=infer_codes(model,window,static=False)
    static_codes=infer_codes(model,window,static=True)
    predictions=[decoder(window['condition'],codes) for codes in (moving_codes,static_codes)]
    images=[]
    for prediction in predictions:
        state=select_state(window['condition'].coordinates,prediction['features'][h-1],
                           prediction['logits'][h-1],p['max_predicted_latent_nodes'])
        image,_,_=render(bridge,state,sample['record'],views=(0,1));images.append(image)
    moving_reference,_,_=render(bridge,window['states'][h],sample['record'],views=(0,1))
    static_reference,_,_=render(bridge,window['states'][0],sample['record'],views=(0,1))
    difference=(moving_reference[0]-static_reference[0]).abs().mean(-1)
    weight=(difference/.1).clamp(0,1);denominator=weight.sum().clamp_min(1e-6)
    def weighted(prediction,target):
        return float((weight*(prediction-target).abs().mean(-1)).sum()/denominator)
    return dict(horizon=h,code_rms_difference=float((moving_codes-static_codes).square().mean().sqrt()),
        motion_code_to_motion_reference=weighted(images[0][0],moving_reference[0]),
        motion_code_to_static_reference=weighted(images[0][0],static_reference[0]),
        static_code_to_static_reference=weighted(images[1][0],static_reference[0]),
        static_code_to_motion_reference=weighted(images[1][0],moving_reference[0]),
        gt_motion_weight_sum=float(weight.sum()),same_initial_coordinates_and_features=True,
        same_target_camera=True,reference_renderer='frozen GaussianGPT codec',
        interpretation='Per-window motion-versus-static diagnostic; no action transfer claim')

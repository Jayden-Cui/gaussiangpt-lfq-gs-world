"""Matched main objective plus optional weighted anchor replay."""
import math
import torch


def step(decoder,bank,bridge,store,std,optimizer,batch,anchor,p,replay_weight,*,probe=False):
    from core import image_losses
    from support_decoder import state_loss,select_state
    from runtime import render
    if len(batch)!=4 or replay_weight not in (0.0,0.1):
        raise ValueError('Expected four main examples and the fixed replay weights')
    if anchor['start'] not in (35,123) or not 1<=anchor['horizon']<=9:
        raise ValueError('Unexpected replay example')
    if any(x.requires_grad for x in bridge.parameters()):raise RuntimeError('Native GS codec is not frozen')
    trainable=[x for x in decoder.parameters() if x.requires_grad]
    expected=trainable+list(bank.parameters())
    actual=[x for group in optimizer.param_groups for x in group['params']]
    if len(actual)!=len(expected) or {id(x) for x in actual}!={id(x) for x in expected}:
        raise RuntimeError('Optimizer must contain exactly the decoder and free codes')
    optimizer.zero_grad(set_to_none=True)
    rows=[];replay_row=None
    examples=[(m,0.25,False) for m in batch]
    if replay_weight:examples.append((anchor,replay_weight,True))
    for member,weight,is_replay in examples:
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
        loss=rgb+.1*latent
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite pilot loss')
        row={'start':start,'horizon':h,'loss':float(loss.detach()),'rgb_loss':float(rgb.detach()),
             'latent_loss':float(latent.detach()),'occupancy_loss':float(terms['occupancy'].detach()),
             'feature_loss':float(terms['feature'].detach()),'predicted_nodes':len(state['coordinates']),
             'gs_count':len(gs)}
        if probe and is_replay:
            grads=torch.autograd.grad(rgb,(code,decoder.base.features.weight),retain_graph=True)
            values=[float(x.norm()) for x in grads]
            if not all(math.isfinite(x) and x>0 for x in values):
                raise RuntimeError('Replay RGB gradient did not reach code and decoder')
            row['rgb_to_code_and_decoder_gradients']=values
        (loss*weight).backward()
        if is_replay:replay_row=row
        else:rows.append(row)
        del window,code,prediction,latent,terms,state,sample,images,gs,sh,rgb,rgb_terms,loss
    selected={x['start'] for x in batch}
    if replay_weight:selected.add(anchor['start'])
    for start in bank.starts:
        if start not in selected and bank.code(start).grad is not None:
            raise RuntimeError('Unselected window unexpectedly acquired a gradient')
    if any(x.grad is not None and not bool(torch.isfinite(x.grad).all()) for x in expected):
        raise RuntimeError('Nonfinite decoder or free-code gradient')
    if any(x.grad is not None for x in decoder.parameters() if not x.requires_grad):
        raise RuntimeError('Frozen initial encoder or unused anchor codes acquired gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(expected,p['grad_clip']))
    optimizer.step()
    if any(not bool(torch.isfinite(x).all()) for x in expected):raise RuntimeError('Nonfinite updated parameter')
    return {'windows':rows,'replay':replay_row,'replay_weight':replay_weight,
            'preclip_gradient_norm':norm,'clip_triggered':norm>p['grad_clip']}

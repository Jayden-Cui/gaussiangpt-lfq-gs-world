import hashlib,random
from collections import defaultdict
import torch
from posterior import build_pair
from runtime import digest

TRAIN_PREFIXES=('base.decoder_backbone.','base.output_norm.','base.features.','base.quantizer.project_out.','geometry.','occupancy.')


def build_models(p,device,train_decoder=True):
    decoder,posterior=build_pair(p,'cpu')
    checkpoint=torch.load(p['posterior_checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint['step']!=1200:raise RuntimeError('Wrong posterior warm start')
    posterior.load_state_dict(checkpoint['posterior'],strict=True)
    if digest(posterior)!=p['posterior_tensor_digest']:raise RuntimeError('Posterior warm-start tensor mismatch')
    if train_decoder:
        for name,param in decoder.named_parameters():param.requires_grad_(name.startswith(TRAIN_PREFIXES))
        for module in [decoder.base.decoder_backbone,decoder.base.output_norm,decoder.base.features,
                       decoder.base.quantizer.project_out,decoder.geometry,decoder.occupancy]:module.train()
    return decoder.to(device),posterior.to(device)


def frozen_digest(decoder):
    h=hashlib.sha256()
    for name,value in sorted(decoder.state_dict().items()):
        if name.startswith(TRAIN_PREFIXES):continue
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parameters(decoder,posterior):
    return list(posterior.parameters())+[v for v in decoder.parameters() if v.requires_grad]


def schedule(p):
    if p['starts']!=list(range(168)) or p['batch_size']!=4:raise ValueError('Full adjacent-window protocol required')
    rng=random.Random(p['seed']);order=[]
    while len(order)<p['total_steps']*4:
        epoch=list(p['starts']);rng.shuffle(epoch);order.extend(epoch)
    visits=defaultdict(int);rows=[]
    for step in range(p['total_steps']):
        phase='latent' if step<p['latent_steps'] else 'render';batch=[]
        for start in order[step*4:(step+1)*4]:
            h=1
            if phase=='render':h=1+visits[start]%9;visits[start]+=1
            batch.append(dict(start=start,horizon=h))
        rows.append(dict(step=step+1,phase=phase,batch=batch))
    return rows


def aggregate(rows,controls):
    if not rows:raise ValueError('Empty evaluation')
    metrics={};window_metrics={}
    for control in controls:
        names=list(rows[0]['metrics'][control]['metrics'])
        groups=defaultdict(list)
        for row in rows:groups[row['frame']].append(row['metrics'][control]['metrics'])
        metrics[control]={key:sum(sum(r[key] for r in group)/len(group) for group in groups.values())/len(groups) for key in names}
        window_metrics[control]={key:sum(row['metrics'][control]['metrics'][key] for row in rows)/len(rows) for key in names}
    return dict(unique_frame_equal=metrics,window_prediction_equal=window_metrics,
                unique_frames=len({r['frame'] for r in rows}),window_predictions=len(rows))

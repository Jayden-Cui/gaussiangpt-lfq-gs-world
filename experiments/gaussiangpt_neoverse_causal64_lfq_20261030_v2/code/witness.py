"""Bounded real prefix-causality, integer replay and batch4 render-backward witness."""
import argparse
import gc
import time
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from motion_common import BASE, CLIPS, H, W, read, write, frozen_codec, load_target, Vgg19
from causal_data import load_prefix, load_prefix_pair
from qmotion import build_model
from causality_checks import check_causality, compare
from train_motion import batch_loss, parameter_sha
from contracts import verify_sources


def close(a,b,tolerance=1e-5):
    error=float((a-b).abs().max());assert error<=tolerance,error
    return error


def run(root):
    started=time.monotonic();sources=verify_sources(root,full_cache=True)
    protocol=read(root/'protocol.json');config=protocol['arms'][0]
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    torch.manual_seed(protocol['seed']);torch.use_deterministic_algorithms(False)
    codec=frozen_codec(root);frozen_before=parameter_sha(codec)
    assert isinstance(codec.ae.vq,torch.nn.Identity)
    model=build_model(config).cuda().eval();model_before=parameter_sha(model)
    temporal=check_causality(model,load_prefix(root,CLIPS[0],49,10))
    starts=[0,58,117,175]
    records=[load_prefix_pair(root,CLIPS[0],s,s+1,codec,10) for s in starts]
    source,target,initials,conditions,indices,windows=zip(*records)
    assert all(abs(i.dt-1/29.97)<1e-9 for i in initials)
    with torch.no_grad():
        z=model.encode_motion_batch(windows)
        assert z['z_prequant'].shape==z['z_quantized'].shape==(4,4,64)
        assert z['indices'].shape==(4,4)
        predictions=model.restore_batch(z['z_quantized'],conditions)
        batch_reports=[]
        for b,window in enumerate(windows):
            single=model.encode_prefix(window)
            error=close(z['z_prequant'][b],single['z_prequant'][0])
            assert torch.equal(z['indices'][b],single['indices'][0])
            restored=model.restore(single['z_quantized'],conditions[b])
            fe=close(predictions[b]['features'],restored['features'])
            oe=close(predictions[b]['occupancy_logits'],restored['occupancy_logits'])
            batch_reports.append({'start':starts[b],'prefix_length':len(window),
                'prequant_max_abs':error,'features_max_abs':fe,'occupancy_max_abs':oe})
        inverse=model.quantizer.indices_to_codes(z['indices'],project_out=True)
        inverse_error=close(inverse,z['z_quantized'])
        packet=model.packet(initials[0],z['indices'][0:1].long(),binding='causal64-witness')
        replay=model.replay(packet,codec,binding='causal64-witness')
        packet_error=close(replay['features'],predictions[0]['features'])
        assert torch.equal(replay['coordinates'],predictions[0]['coordinates'])
        close(replay['occupancy_logits'],predictions[0]['occupancy_logits'])
        try:model.replay(dict(packet,future_features=target[0]['features']),codec,binding='causal64-witness')
        except ValueError:pass
        else:raise AssertionError('Future-side field accepted')
        sequence=model.encode_sequence(windows[1])
        assert sequence['indices'].shape==(1,9,4)
        assert sequence['z_prequant'].shape==(1,9,4,64)
        compare({k:sequence[k][:,8] for k in ['z_prequant','indices']},model.encode_prefix(windows[1]))
        tokens=[model.prefix_tokens(model.prepare_window(w),len(w)-1) for w in windows]
        keys=pad_sequence([p[0] for p in tokens],batch_first=True)
        positions=pad_sequence([p[1] for p in tokens],batch_first=True)
        lengths=torch.tensor([len(p[0]) for p in tokens],device='cuda')
        valid=torch.arange(keys.shape[1],device='cuda')[None]<lengths[:,None]
        assert (~valid).any()
        clean=model.posterior_keys(keys,positions,valid)
        poison=keys.clone();poison[~valid]=1000
        badpos=positions.clone();badpos[~valid]=-1000
        padding_error=close(clean,model.posterior_keys(poison,badpos,valid))
    del z,predictions,single,restored,inverse,replay,sequence,tokens,clean,poison,badpos
    keys=keys.detach().requires_grad_(True)
    result=model.posterior_keys(keys,positions,valid)
    grad=torch.autograd.grad((result*torch.linspace(-1,1,64,device='cuda')).sum(),keys)[0]
    assert float(grad[~valid].abs().max())==0 and torch.isfinite(grad).all()
    del keys,positions,result,grad
    gc.collect();model.train();model.zero_grad(set_to_none=True)
    vgg=Vgg19(BASE/'torch_cache').cuda().eval().requires_grad_(False)
    images=[load_target(CLIPS[0],s+1) for s in starts]
    assert all(tuple(im['real'].shape)==(252,448,3) for im in images)
    loss,terms,z,details=batch_loss(model,codec,records,images,vgg,protocol,rendering=True,force_target=True)
    assert torch.isfinite(loss);loss.backward()
    modules=[('temporal_input',model.temporal_input),('temporal_cnn',model.temporal_cnn),
        ('time_position',model.time_position),('endpoint',model.endpoint),('posterior',model.posterior),
        ('lfq_input',model.quantizer.project_in),('lfq_output',model.quantizer.project_out),
        ('restorer',model.restorer),('delta',model.delta),('occupancy',model.occupancy)]
    norms={}
    for name,module in modules:
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients),name
        norm=float(sum(g.square().sum() for g in gradients).sqrt());assert norm>0,name
        norms[name]=norm
    query_norm=float(model.queries.grad.norm());assert query_norm>0
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    assert parameter_sha(codec)==frozen_before and parameter_sha(model)==model_before
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=48
    write(root/'witness.json',{'status':'passed','sources':sources,'elapsed_seconds':time.monotonic()-started,
        'causality':temporal,'float_shape':[4,4,64],'id_shape':[4,4],
        'ten_frame_sequence_float_shape':[1,9,4,64],'ten_frame_sequence_id_shape':[1,9,4],
        'prefix_lengths':[len(w) for w in windows],'attention_key_lengths':lengths.cpu().tolist(),
        'qformer_width':64,'heads':4,'head_width':16,'ffn_width':256,
        'temporal_width':64,'temporal_convolutions':4,'temporal_receptive_frames':13,
        'batch_vs_single':batch_reports,'padding_poison_max_abs':padding_error,'padding_gradient_max_abs':0.,
        'hard_id_inverse_max_abs':inverse_error,'packet_features_max_abs':packet_error,'future_fields_rejected':True,
        'gradient_norms':norms,'query_gradient_norm':query_norm,
        'frozen_state_no_parameter_gradients':True,'frozen_state_parameters_unchanged':True,
        'motion_parameters_unchanged':True,'frozen_state_parameter_sha256':frozen_before,
        'native_render_size':[448,252],'allocator_fraction':.60,'torch_threads':torch.get_num_threads(),
        'peak_allocated_gib':peak,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'lfq_entropy_tokens':16,'loss':float(loss.detach()),'batch_details':details,
        'limitations':protocol['limitations']})
    print('CAUSAL64_WITNESS_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as error:
        write(a.root/'witness_failure.json',{'status':'failed','error':repr(error)})
        raise

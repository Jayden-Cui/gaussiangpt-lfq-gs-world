"""H9 real-data checks; exact gradient staging; all horizons render backwards."""
import argparse,copy,gc,time
from pathlib import Path
import torch
from motion_common import CLIPS,read,write,frozen_codec
from sequence_data import load_record
from qmotion import build_model
from train_motion import parameter_sha,sequence_backward
from losses_motion import all_losses
from contracts import verify_sources
from causality_checks import check_causality,close

def gradient_equivalence(model,codec,record,protocol):
    model.train();model.zero_grad(set_to_none=True)
    z=model.encode_windows([record['window']])
    predictions=model.decode_latents(z['z_quantized'],[record['condition']])[0]
    weights=protocol['train']['warmup_loss']
    loss=z['aux_loss']
    for h,p in enumerate(predictions):
        terms=all_losses(codec,p,record['window'][h+1],record['labels'][h],None,rendering=False)
        loss=loss+sum(weights[k]*v for k,v in terms.items())/9
    loss.backward()
    direct={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    direct_loss=float(loss.detach());del loss,predictions,terms,z
    model.zero_grad(set_to_none=True)
    staged=sequence_backward(model,codec,[record],[CLIPS[0]],protocol,rendering=False)
    max_error=0.;max_relative=0.
    for n,p in model.named_parameters():
        if p.grad is None:assert n not in direct;continue
        a=direct[n];b=p.grad.detach().cpu()
        error=float((a-b).abs().max());max_error=max(max_error,error)
        relative=float((a-b).norm()/a.norm().clamp_min(1e-8));max_relative=max(max_relative,relative)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-3),(n,error,relative)
    assert abs(staged['loss']-direct_loss)<1e-5
    return dict(passed=True,gradient_max_abs=max_error,max_tensor_relative_l2=max_relative,
        direct_loss=direct_loss,staged_loss=staged['loss'],meaning='All 9 reconstruction gradients reach original LFQ/encoder; staging is not truncated backpropagation.')

def run(root):
    started=time.monotonic();sources=verify_sources(root,full_cache=True)
    p=read(root/'protocol.json');config=p['arms'][0]
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    torch.manual_seed(p['seed']);codec=frozen_codec(root);before=parameter_sha(codec)
    model=build_model(config).cuda().eval();motion_before=parameter_sha(model)
    starts=[0,58,117,175];records=[load_record(root,CLIPS[0],s,codec) for s in starts]
    temporal=check_causality(model,records[1]['window'],records[1]['condition'])
    with torch.no_grad():
        z=model.encode_windows([r['window'] for r in records])
        assert z['indices'].shape==(4,9) and z['z_quantized'].shape==(4,9,64)
        singles=[]
        for b,r in enumerate(records):
            single=model.encode_windows([r['window']])
            singles.append(close(z['z_prequant'][b],single['z_prequant'][0]))
            assert torch.equal(z['indices'][b],single['indices'][0])
        inverse_error=close(model.quantizer.indices_to_codes(z['indices'],project_out=True),z['z_quantized'])
        pred=model.decode_latents(z['z_quantized'][:1],[records[0]['condition']])[0]
        packet=model.packet(records[0]['initial'],z['indices'][:1],binding='h9-witness')
        replay=model.replay(packet,codec,binding='h9-witness')
        replay_error=max(close(a['features'],b['features']) for a,b in zip(pred,replay))
        for a,b in zip(pred,replay):
            close(a['occupancy_logits'],b['occupancy_logits']);assert torch.equal(a['coordinates'],b['coordinates'])
        try:model.replay(dict(packet,future_features=records[0]['window'][1]['features']),codec,binding='h9-witness')
        except ValueError:pass
        else:raise AssertionError('Future fields accepted')
    del z,single,pred,replay
    equivalence=gradient_equivalence(model,codec,records[1],p)
    model.zero_grad(set_to_none=True);gc.collect()
    # Exclude entropy here: nonzero gradients must come from future reconstruction.
    output=sequence_backward(model,codec,records,[CLIPS[0]]*4,p,rendering=True,aux_weight=0.)
    modules=[('temporal_input',model.temporal_input),('encoder_cnn',model.temporal_cnn),
        ('decoder_cnn',model.decoder_cnn),('lfq_in',model.quantizer.project_in),('lfq_out',model.quantizer.project_out),
        ('initial_query',model.initial_query),('delta',model.delta),('occupancy',model.occupancy)]
    modules += [(f'qformer_h{h+1}',m) for h,m in enumerate(model.posteriors)]
    norms={}
    for name,module in modules:
        grads=[v.grad for v in module.parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        norms[name]=float(sum(g.square().sum() for g in grads).sqrt());assert norms[name]>0,name
    query_norms=model.queries.grad.flatten(1).norm(dim=1).tolist();assert min(query_norms)>0
    assert min(output['bottleneck_gradient_per_horizon'])>0
    # Isolate RGB supervision from sparse feature/occupancy auxiliary losses.
    model.zero_grad(set_to_none=True)
    rgb_protocol=copy.deepcopy(p)
    rgb_protocol['train']['render_loss']={k:float(k=='future_rgb_l2') for k in p['train']['render_loss']}
    rgb=sequence_backward(model,codec,[records[1]],[CLIPS[0]],rgb_protocol,rendering=True,aux_weight=0.)
    rgb_query_norms=model.queries.grad.flatten(1).norm(dim=1).tolist()
    assert min(rgb_query_norms)>0 and min(rgb['bottleneck_gradient_per_horizon'])>0
    assert all(v.grad is None and not v.requires_grad for v in codec.parameters())
    assert parameter_sha(codec)==before and parameter_sha(model)==motion_before
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=48
    write(root/'witness.json',dict(status='passed',sources=sources,causality=temporal,gradient_staging=equivalence,
        float_shape=[4,9,64],id_shape=[4,9],batch_vs_single_errors=singles,hard_id_inverse_error=inverse_error,
        all9_packet_replay_error=replay_error,future_fields_rejected=True,gradient_norms=norms,
        query_gradient_norms=query_norms,reconstruction_only_bottleneck_gradient_per_horizon=output['bottleneck_gradient_per_horizon'],
        isolated_future_rgb_query_gradient_norms=rgb_query_norms,
        isolated_future_rgb_bottleneck_gradient_per_horizon=rgb['bottleneck_gradient_per_horizon'],
        rendered_targets=36,padded_targets=sum(not v for r in records for v in r['valid']),target_support_forcing=False,
        native_render_size=[448,252],frozen_codec_hash_unchanged=True,frozen_codec_parameter_sha256=before,
        motion_weights_unchanged=True,peak_allocated_gib=peak,elapsed_seconds=time.monotonic()-started,
        trainable_parameters=sum(v.numel() for v in model.parameters()),lfq_entropy_pool_size=36,
        support_coverage=read(root/'support_coverage.json'),limitations=p['limitations']))
    print('H9_LFQ_WITNESS_PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as error:
        write(a.root/'witness_failure.json',dict(status='failed',error=repr(error)));raise

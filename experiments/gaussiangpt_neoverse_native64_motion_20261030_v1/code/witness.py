"""One bounded real-data witness; no optimization, retry, or scientific retuning."""
import argparse
import gc
import time
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from motion_common import BASE, CLIPS, read, write, frozen_codec, load_pair, load_target, Vgg19
from qmotion import build_model, qformer_sequence
from train_motion import batch_loss, parameter_sha
from contracts import verify_sources


def close(a, b, tolerance=1e-5):
    error = float((a-b).abs().max())
    assert error < tolerance, (error, tolerance)
    return error


def run(root):
    started = time.monotonic()
    sources = verify_sources(root, full_cache=True)
    protocol = read(root/'protocol.json'); config = protocol['arms'][0]
    assert config['quantizer'] == 'lfq' and len(protocol['arms']) == 1
    torch.set_num_threads(6); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.60)
    torch.manual_seed(protocol['seed'])
    codec = frozen_codec(root); frozen_before = parameter_sha(codec)
    assert isinstance(codec.ae.vq, torch.nn.Identity)
    model = build_model(config).cuda().eval(); model_before = parameter_sha(model)
    starts = [0, 58, 117, 175]
    records = [load_pair(root, CLIPS[0], s, s+1, codec) for s in starts]
    source, target, initials, conditions, labels = zip(*records)
    dts = [i.dt for i in initials]
    assert all(abs(dt - 1/29.97) < 1e-9 for dt in dts)
    context_items = [model.endpoint_context(s, t, dt) for s, t, dt in zip(source, target, dts)]
    contexts = pad_sequence([v[0] for v in context_items], batch_first=True)
    coordinates = pad_sequence([v[1] for v in context_items], batch_first=True)
    lengths = torch.tensor([len(v[0]) for v in context_items], device='cuda')
    valid = torch.arange(contexts.shape[1], device='cuda')[None] < lengths[:, None]
    assert (~valid).any(), 'Witness must exercise padding'
    with torch.no_grad():
        z = model.encode_motion_batch(source, target, dts)
        assert z['z_prequant'].shape == z['z_quantized'].shape == (4, 4, 64)
        assert z['indices'].shape == (4, 4)
        predictions = model.restore_batch(z['z_quantized'], conditions)
        batch_errors = []
        for b in range(4):
            single = model.encode_motion(source[b], target[b], dts[b])
            error = close(single['z_prequant'][0], z['z_prequant'][b])
            assert torch.equal(single['indices'][0], z['indices'][b])
            single_prediction = model.restore(single['z_quantized'], conditions[b])
            feature_error = close(single_prediction['features'], predictions[b]['features'])
            logit_error = close(single_prediction['occupancy_logits'], predictions[b]['occupancy_logits'])
            batch_errors.append({'start': starts[b], 'prequant_max_abs': error,
                'features_max_abs': feature_error, 'occupancy_max_abs': logit_error})
        clean = model.posterior_padded(contexts, coordinates, valid)
        poisoned = contexts.clone(); poisoned[~valid] = 1000
        bad_coords = coordinates.clone(); bad_coords[~valid] = 12345
        padding_error = close(clean, model.posterior_padded(poisoned, bad_coords, valid))
        # Verify exact reference equations at native64 in the no-padding case.
        keys = model.endpoint(context_items[0][0])[None]
        positions = model.position(context_items[0][1])[None]
        query = model.queries[None]
        reference, _ = model.posterior(keys.transpose(1, 2)[:, :, None, None],
            positions.transpose(1, 2)[:, :, None, None], query)
        equation_error = close(reference, qformer_sequence(model.posterior, keys, positions, query), 1e-6)
        inverse = model.quantizer.indices_to_codes(z['indices'], project_out=True)
        inverse_error = close(inverse, z['z_quantized'])
        packet = model.packet(initials[0], z['indices'][0:1].long(), binding='native64-witness')
        replay = model.replay(packet, codec, binding='native64-witness')
        packet_error = close(replay['features'], predictions[0]['features'])
        assert torch.equal(replay['coordinates'], predictions[0]['coordinates'])
        close(replay['occupancy_logits'], predictions[0]['occupancy_logits'])
        try: model.replay(dict(packet, future_coordinates=target[0]['coordinates']), codec, binding='native64-witness')
        except ValueError: pass
        else: raise AssertionError('Future-coordinate bypass accepted')
    del z, predictions, replay, single, single_prediction, inverse, reference, clean, poisoned, bad_coords
    contexts = contexts.detach().requires_grad_(True)
    output = model.posterior_padded(contexts, coordinates, valid)
    gradient = torch.autograd.grad((output * torch.linspace(-1, 1, 64, device='cuda')).sum(), contexts)[0]
    assert float(gradient[~valid].abs().max()) == 0
    assert torch.isfinite(gradient).all() and float(gradient[valid].abs().sum()) > 0
    del contexts, output, gradient
    model.train(); model.zero_grad(set_to_none=True)
    vgg = Vgg19(BASE/'torch_cache').cuda().eval().requires_grad_(False)
    images = [load_target(CLIPS[0], s+1) for s in starts]
    loss, terms, z, details = batch_loss(model, codec, records, images, vgg, protocol,
        rendering=True, force_target=True)
    assert torch.isfinite(loss)
    aux_gradient = torch.autograd.grad(z['aux_loss'], model.quantizer.project_in.weight, retain_graph=True)[0]
    assert torch.isfinite(aux_gradient).all()
    loss.backward()
    norms = {}
    for name, module in [('endpoint',model.endpoint), ('posterior',model.posterior),
            ('lfq_input',model.quantizer.project_in), ('lfq_output',model.quantizer.project_out),
            ('restorer',model.restorer), ('delta',model.delta), ('occupancy',model.occupancy)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        norm = float(sum(g.float().square().sum() for g in grads).sqrt())
        assert norm > 0, name
        norms[name] = norm
    query_norm = float(model.queries.grad.norm()); assert query_norm > 0
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    assert parameter_sha(codec) == frozen_before and parameter_sha(model) == model_before
    peak = torch.cuda.max_memory_allocated()/2**30; assert peak <= 48
    result = {'status':'passed', 'sources':sources, 'elapsed_seconds':time.monotonic()-started,
        'state_latent_width':64, 'state_quantizer':'Identity', 'state_parameters_unchanged':True,
        'motion_parameters_unchanged':True, 'qformer_width':64, 'attention_heads':4, 'head_width':16,
        'ffn_width':256, 'depth':2, 'batch_size':4, 'z_float_shape':[4,4,64], 'z_id_shape':[4,4],
        'lfq_internal_width':16, 'vocabulary':65536, 'state_shapes':[list(s['features'].shape) for s in source],
        'union_lengths':lengths.cpu().tolist(), 'batch_vs_single':batch_errors,
        'padding_poison_error':padding_error, 'padding_gradient_max_abs':0.,
        'reference_equations_error':equation_error, 'hard_id_inverse_error':inverse_error,
        'packet_features_max_abs':packet_error, 'future_fields_rejected':True,
        'gradient_norms':norms, 'query_gradient_norm':query_norm,
        'lfq_joint_entropy_tokens':16, 'lfq_aux_loss':float(z['aux_loss'].detach()),
        'lfq_aux_input_gradient_norm':float(aux_gradient.norm()),
        'loss':float(loss.detach()), 'terms':{k:float(v.detach()) for k,v in terms.items()},
        'batch_details':details, 'peak_allocated_gib':peak,
        'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'frozen_state_parameter_sha256':frozen_before,
        'interpretation':'Engineering witness only; reconstruction and motion usefulness require full evaluation.'}
    write(root/'witness.json', result)
    print('WITNESS_PASSED', result, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
    try: run(a.root)
    except BaseException as error:
        write(a.root/'witness_failure.json', {'status':'failed','error':repr(error)})
        raise

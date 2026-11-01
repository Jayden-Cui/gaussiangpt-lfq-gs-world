"""Bounded CPU witnesses for code-only training, sampling and oracle selection."""
import sys,types
import torch
from stage_context import ROOT,setup,load_codes,load_decoder,sha,write,explicit
from sampling import audit
from optimize import FreeCodes,make_optimizer
from diagnose import oracle_support
from merge import merge_parts


def main():
    c,helper,orig,p,paths,receipt,evaluation=setup()
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    sampler=audit(c);baseline=load_codes(paths,evaluation)
    if tuple(baseline['correct'].shape)!=(168,9,16):raise RuntimeError('Code shape changed')
    # An Adam parameter with old momentum must remain untouched when its window is absent.
    bank=FreeCodes(torch.ones(2,9,16),[0,1],'cpu');opt=make_optimizer(bank,c)
    bank.code(1).square().sum().backward();opt.step();saved=bank.code(1).detach().clone()
    opt.zero_grad(set_to_none=True);bank.code(0).square().sum().backward();opt.step()
    if bank.code(1).grad is not None or not torch.equal(saved,bank.code(1).detach()):
        raise RuntimeError('An inactive window moved under Adam momentum')
    # Oracle support must select the *predicted* features at the exact target positions.
    condition=types.SimpleNamespace(coordinates=torch.arange(15).reshape(5,3))
    prediction={'features':torch.arange(9*5*64).reshape(9,5,64)}
    indices=torch.tensor([3,0])
    chosen=oracle_support(condition,prediction,2,indices)
    if not torch.equal(chosen['coordinates'],condition.coordinates[indices]) or not torch.equal(chosen['features'],prediction['features'][1,indices]):
        raise RuntimeError('Oracle support substituted features or row positions')
    codes=torch.arange(168,dtype=torch.float32)[:,None,None].expand(168,9,16).clone()
    assembled=merge_parts([(list(range(84,168)),codes[84:]),(list(range(84)),codes[:84])])
    if not torch.equal(assembled,codes):raise RuntimeError('Merged codes reordered windows')
    try:merge_parts([(list(range(84)),codes[:84]),(list(range(83,167)),codes[83:167])])
    except ValueError:pass
    else:raise RuntimeError('Overlapping shards not rejected')
    decoder,modules=load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    fm,_,runtime=modules
    if 'common' in sys.modules:
        raise RuntimeError('Renderer module name was preempted before native codec import')
    evaluator=explicit('_stage_a_evaluator_witness',ROOT/'evaluate.py')
    controls=evaluator.build_controls(codes,baseline,list(range(168)))
    if (not torch.equal(controls['baseline_qformer'],baseline['correct'])
            or not torch.equal(controls['initial_only'],baseline['initial_only'])
            or not torch.equal(controls['swapped_window_codes'][0],codes[1])
            or not torch.equal(controls['swapped_window_codes'][167],codes[166])):
        raise RuntimeError('Control codes or neighbor-window mapping changed')
    from support_decoder import make_condition
    torch.manual_seed(81)
    initial={'coordinates':torch.tensor([[0,0,0],[1,0,0],[0,1,0]],dtype=torch.int32),
             'features':torch.randn(3,64),'pts':0.}
    condition=make_condition(decoder.base,initial,1,200)
    codes=torch.nn.Parameter(baseline['correct'][35].clone())
    output=decoder(condition,codes)
    loss=output['features'][-1].square().mean()+output['logits'][-1].square().mean()
    loss.backward()
    if codes.grad is None or not bool(torch.isfinite(codes.grad).all()) or not float(codes.grad.norm())>0:
        raise RuntimeError('Frozen decoder did not propagate gradient to free code')
    if any(v.requires_grad or v.grad is not None for v in decoder.parameters()):
        raise RuntimeError('Motion decoder not fully frozen')
    if runtime.digest(decoder)!=c['decoder_tensor_digest']:
        raise RuntimeError('Decoder changed during CPU witness')
    changed=codes.detach().clone();changed[-1]+=2.5
    with torch.no_grad():a=decoder(condition,codes.detach());b=decoder(condition,changed)
    torch.testing.assert_close(a['features'][:-1],b['features'][:-1],atol=2e-5,rtol=1e-6)
    torch.testing.assert_close(a['logits'][:-1],b['logits'][:-1],atol=2e-5,rtol=1e-6)
    write(ROOT/'CPU_TESTS.json',dict(status='passed',sampling=sampler,code_sha256=sha(ROOT/'CODE.sha256'),
        decoder_tensor_digest=runtime.digest(decoder),baseline_codes_shape=[168,9,16],
        frozen_decoder_to_code_gradient_norm=float(codes.grad.norm()),inactive_adam_parameter_unchanged=True,
        oracle_uses_predicted_features_at_target_coordinates=True,shard_merge_and_control_mapping=True,
        native_renderer_module_name_unshadowed=True,
        last_code_does_not_change_earlier_outputs=True,
        scope='CPU witnesses on bound weights; actual RGB-to-code gradients are checked on each GPU'))
    print('PASS: complete sampling, 9560 checkpoint, causal free-code gradients, inactive Adam state and oracle selection',flush=True)


if __name__=='__main__':main()

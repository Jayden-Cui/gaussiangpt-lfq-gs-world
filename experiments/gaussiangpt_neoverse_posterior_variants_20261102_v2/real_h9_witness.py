"""Supplemental full-prefix CPU witness on real 10-state GS data."""
from pathlib import Path
import importlib.util,json,resource,sys,time
import torch
from posterior_variants import ContinuousPosterior
from real_data_witness import norm

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=json.loads((ROOT/'variant_config.json').read_text(encoding='utf-8'))
    path=Path(config['stage_a_root'])/'stage_context.py'
    spec=importlib.util.spec_from_file_location('_stage_a_real_h9_context',path)
    context=importlib.util.module_from_spec(spec);sys.modules[spec.name]=context;spec.loader.exec_module(context)
    if (context.sha(path)!=config['stage_context_sha256']
            or context.sha(path.parent/'CODE.sha256')!=config['stage_a_code_sha256']):
        raise RuntimeError('Stage A source binding differs')
    if context.sha(ROOT/'real_data_witness.py')!=context.read(ROOT/'REAL_DATA_WITNESS.json')['witness_script_sha256']:
        raise RuntimeError('Supplemental gradient helper changed')
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if context.sha(ROOT/name)!=digest:raise RuntimeError('Posterior source changed')
    c,helper,orig,p,paths,receipt,evaluation=context.setup()
    decoder,modules=context.load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    fm,_,runtime=modules
    unused,posterior=fm.build_models(p,'cpu',train_decoder=False)
    data=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    posterior.load_state_dict(data['posterior'],strict=True);posterior.eval()
    if runtime.digest(posterior)!=receipt['posterior_digest']:raise RuntimeError('Wrong posterior')
    del unused,data
    store=context.dataset(modules,p,c);window=store.get(35,device='cpu');states=window['states']
    if len(states)!=10:raise RuntimeError('Incomplete real window')
    first_digest=runtime.digest(decoder);result={}
    for variant in ['C1','C2']:
        torch.manual_seed(20261102)
        model=ContinuousPosterior(decoder.base,posterior,variant).eval();began=time.monotonic()
        with torch.no_grad():
            actual=decoder.base.prepare(states);copied=model.encoder.prepare(states)
            if not torch.equal(actual['coordinates'],copied['coordinates']) or not torch.equal(actual['present'],copied['present']):
                raise RuntimeError('Copied full-prefix grid or presence differs')
            max_error=float((actual['hidden']-copied['hidden']).abs().max())
            torch.testing.assert_close(actual['hidden'],copied['hidden'],atol=2e-5,rtol=1e-5)
            before=model.code_for_h(states,7)
            changed=list(states);changed[8]=dict(changed[8],features=changed[8]['features']+3.)
            after=model.code_for_h(changed,7)
            torch.testing.assert_close(before,after,atol=2e-5,rtol=1e-6)
            keys,_,metadata=model.memory_for_h(states,9)
            token_count=len(keys)
        del actual,copied,keys,metadata,before,after
        model.train();model.zero_grad(set_to_none=True)
        z=model.code_for_h(states,9);z.square().mean().backward()
        gradients=dict(encoder_input=norm(model.encoder.input.parameters()),
                       encoder_backbone=norm(model.encoder.encoder_backbone.parameters()),
                       query=norm(model.posterior.queries.parameters()),
                       projection=norm(model.posterior.project_in.parameters()))
        if model.include_time:gradients['time_projection']=norm(model.time_projection.parameters())
        if model.include_change:gradients['delta_projection']=norm(model.delta_projection.parameters())
        if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise RuntimeError('Missing full-prefix gradient on real GS states: '+str(gradients))
        if runtime.digest(decoder)!=first_digest or any(v.grad is not None for v in decoder.parameters()):
            raise RuntimeError('Real-data witness changed the frozen decoder')
        result[variant]=dict(prefix_horizon=9,source_frames=[35,44],input_nodes=[len(s['coordinates']) for s in states],
            memory_tokens=token_count,max_copied_encoder_hidden_error=max_error,gradients=gradients,
            changing_frame_43_does_not_change_h7=True,elapsed_seconds=time.monotonic()-began,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    output=dict(status='passed',source='actual frozen TRAIN state cache, start=35',
        source_cache_sha256=c['bindings']['cache_manifest'],decoder_digest=first_digest,
        source_model_code_sha256=context.sha(ROOT/'CODE.sha256'),
        witness_script_sha256=context.sha(ROOT/'real_h9_witness.py'),variants=result,
        scope='Real h9 CPU gradient and h7 future-perturbation witness; GPU memory and teacher training remain unverified')
    context.write(ROOT/'REAL_H9_WITNESS.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()

"""Supplemental CPU witness on real cached GS states; no training job is launched."""
from pathlib import Path
import importlib.util,json,sys,time
import torch
from posterior_variants import ContinuousPosterior

ROOT=Path(__file__).resolve().parent


def norm(parameters):
    terms=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=json.loads((ROOT/'variant_config.json').read_text(encoding='utf-8'))
    path=Path(config['stage_a_root'])/'stage_context.py'
    spec=importlib.util.spec_from_file_location('_stage_a_real_data_context',path)
    context=importlib.util.module_from_spec(spec);sys.modules[spec.name]=context;spec.loader.exec_module(context)
    if (context.sha(path)!=config['stage_context_sha256']
            or context.sha(path.parent/'CODE.sha256')!=config['stage_a_code_sha256']):
        raise RuntimeError('Stage A source binding differs')
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
        model=ContinuousPosterior(decoder.base,posterior,variant).eval()
        with torch.no_grad():
            actual=decoder.base.prepare(states[:2]);copied=model.encoder.prepare(states[:2])
            if not torch.equal(actual['coordinates'],copied['coordinates']) or not torch.equal(actual['present'],copied['present']):
                raise RuntimeError('Copied real-state grid or presence differs')
            max_error=float((actual['hidden']-copied['hidden']).abs().max())
            torch.testing.assert_close(actual['hidden'],copied['hidden'],atol=2e-5,rtol=1e-5)
            before=model.code_for_h(states,1)
            changed=list(states);changed[2]=dict(changed[2],features=changed[2]['features']+3.)
            after=model.code_for_h(changed,1)
            torch.testing.assert_close(before,after,atol=2e-5,rtol=1e-6)
        model.train();model.zero_grad(set_to_none=True)
        began=time.monotonic();z=model.code_for_h(states,1);z.square().mean().backward()
        gradients=dict(encoder_input=norm(model.encoder.input.parameters()),
                       encoder_backbone=norm(model.encoder.encoder_backbone.parameters()),
                       query=norm(model.posterior.queries.parameters()),
                       projection=norm(model.posterior.project_in.parameters()))
        if model.include_time:gradients['time_projection']=norm(model.time_projection.parameters())
        if model.include_change:gradients['delta_projection']=norm(model.delta_projection.parameters())
        if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise RuntimeError('Missing gradient on real GS states: '+str(gradients))
        if runtime.digest(decoder)!=first_digest or any(v.grad is not None for v in decoder.parameters()):
            raise RuntimeError('Real-data witness changed the frozen decoder')
        result[variant]=dict(prefix_horizon=1,source_frames=[35,36],
            input_nodes=[len(s['coordinates']) for s in states[:2]],
            max_copied_encoder_hidden_error=max_error,gradients=gradients,
            changing_later_frame_does_not_change_code=True,elapsed_seconds=time.monotonic()-began)
    output=dict(status='passed',source='actual frozen TRAIN state cache, start=35',
        source_cache_sha256=c['bindings']['cache_manifest'],decoder_digest=first_digest,
        source_model_code_sha256=context.sha(ROOT/'CODE.sha256'),
        witness_script_sha256=context.sha(ROOT/'real_data_witness.py'),variants=result,
        scope='Real prefix h1 CPU gradient/cause witness; not full-window GPU memory or training success')
    context.write(ROOT/'REAL_DATA_WITNESS.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()

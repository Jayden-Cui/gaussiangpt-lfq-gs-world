"""Exercise the C1 distillation update on four real TRAIN windows, on CPU."""
from pathlib import Path
import importlib.util,json,sys,time
import torch
from posterior_variants import ContinuousPosterior
from training_core import infer_codes,static_states,teacher_scale,step
from real_data_witness import norm

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=json.loads((ROOT/'variant_config.json').read_text(encoding='utf-8'))
    path=Path(config['stage_a_root'])/'stage_context.py'
    spec=importlib.util.spec_from_file_location('_stage_a_training_core_context',path)
    context=importlib.util.module_from_spec(spec);sys.modules[spec.name]=context;spec.loader.exec_module(context)
    if (context.sha(path)!=config['stage_context_sha256']
            or context.sha(path.parent/'CODE.sha256')!=config['stage_a_code_sha256']
            or context.sha(ROOT/'real_data_witness.py')!=context.read(ROOT/'REAL_DATA_WITNESS.json')['witness_script_sha256']):
        raise RuntimeError('Witness dependency changed')
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if context.sha(ROOT/name)!=digest:raise RuntimeError('Posterior source changed')
    c,helper,orig,p,paths,receipt,evaluation=context.setup()
    decoder,modules=context.load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    fm,_,runtime=modules
    unused,posterior=fm.build_models(p,'cpu',train_decoder=False)
    data=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    posterior.load_state_dict(data['posterior'],strict=True);posterior.eval();del unused,data
    if runtime.digest(posterior)!=receipt['posterior_digest']:raise RuntimeError('Wrong posterior')
    store=context.dataset(modules,p,c)
    class CPUStore:
        def get(self,start):return store.get(start,device='cpu')
    actual=CPUStore();window=actual.get(35);sequence=static_states(window['states'])
    for state,original in zip(sequence,window['states']):
        if (not torch.equal(state['coordinates'],window['states'][0]['coordinates'])
                or not torch.equal(state['features'],window['states'][0]['features'])
                or state['pts']!=original['pts']):raise RuntimeError('Static branch leaked future GS state or changed time')
    model=ContinuousPosterior(decoder.base,posterior,'C1').eval()
    with torch.no_grad():
        before=infer_codes(model,window,static=True)
        altered=dict(window,states=[dict(s,features=s['features']+5) if i>0 else s
                                    for i,s in enumerate(window['states'])])
        after=infer_codes(model,altered,static=True)
    torch.testing.assert_close(before,after,atol=2e-5,rtol=1e-6)
    base_codes=context.load_codes(paths,evaluation)['correct'];teacher=base_codes.clone()+.05
    scale=teacher_scale(teacher);decoder_digest=runtime.digest(decoder)
    model.train();previous=model.encoder.input.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99))
    batch=[dict(start=s,horizon=h) for s,h in [(0,1),(35,3),(123,7),(167,9)]]
    began=time.monotonic()
    values=step(model,decoder,torch.nn.Module().eval(),actual,store.frames['std'],optimizer,
                batch,teacher,scale,p,'distill')
    gradients=dict(encoder_input=norm(model.encoder.input.parameters()),
                   encoder_backbone=norm(model.encoder.encoder_backbone.parameters()),
                   queries=norm(model.posterior.queries.parameters()),
                   projection=norm(model.posterior.project_in.parameters()))
    if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in gradients.values()):
        raise RuntimeError('A required distillation gradient is missing')
    if torch.equal(previous,model.encoder.input.weight.detach()):raise RuntimeError('Encoder did not update')
    if runtime.digest(decoder)!=decoder_digest or any(v.grad is not None for v in decoder.parameters()):
        raise RuntimeError('Teacher decoder changed')
    output=dict(status='passed',source='actual cached TRAIN states at four windows; synthetic perturbed teacher codes',
        windows=[0,35,123,167],effective_batch_size=4,phase='distill',teacher_scale=float(scale),
        gradients=gradients,code_only_optimizer_update='passed',static_input_uses_s0_and_original_times=True,
        static_code_ignores_future_features=True,decoder_digest=decoder_digest,
        source_model_code_sha256=context.sha(ROOT/'CODE.sha256'),
        training_core_sha256=context.sha(ROOT/'training_core.py'),
        witness_script_sha256=context.sha(ROOT/'training_core_witness.py'),
        elapsed_seconds=time.monotonic()-began,rows=values['windows'],
        scope='CPU distillation plumbing only; perturbed baseline is not a validated teacher, no RGB/GPU or quality claim')
    context.write(ROOT/'TRAINING_CORE_WITNESS.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()

"""CPU witnesses for C0/C1/C2 using real bound 9560-step modules."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import torch
from posterior_variants import ContinuousPosterior,aligned_fields

ROOT=Path(__file__).resolve().parent


def load_context(config):
    path=Path(config['stage_a_root'])/'stage_context.py'
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['stage_context_sha256']:
        raise RuntimeError('Stage A source changed')
    spec=importlib.util.spec_from_file_location('_stage_a_context_for_c',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    if module.sha(path.parent/'CODE.sha256')!=config['stage_a_code_sha256']:
        raise RuntimeError('Stage A manifest changed')
    return module


def state(cells,values,t):
    return dict(coordinates=torch.tensor(cells,dtype=torch.int32),
                features=torch.tensor(values,dtype=torch.float32)[:,None].expand(-1,64).clone(),pts=float(t))


def check_world_alignment():
    a=state([[2,0,0],[0,0,0]],[20,1],0)
    b=state([[1,0,0],[2,0,0]],[10,25],.033)
    fields=aligned_fields([a,b]);lookup={tuple(row):i for i,row in enumerate(fields['coordinates'].tolist())}
    absent,new,shared=(lookup[k] for k in [(0,0,0),(1,0,0),(2,0,0)])
    if (float(fields['delta_features'][1,absent,0])!=-1 or float(fields['delta_occupancy'][1,absent])!=-1
            or float(fields['delta_features'][1,new,0])!=10 or float(fields['delta_occupancy'][1,new])!=1
            or float(fields['delta_features'][1,shared,0])!=5
            or bool(fields['seen'][0,new]) or not bool(fields['seen'][1,new])
            or not bool(fields['seen'][1,absent]) or bool(fields['present'][1,absent])):
        raise RuntimeError('World-coordinate feature/occupancy differences incorrect')
    reverse=[dict(s,coordinates=s['coordinates'].flip(0),features=s['features'].flip(0)) for s in [a,b]]
    other=aligned_fields(reverse)
    for key in ['coordinates','features','present','delta_features','delta_occupancy','seen']:
        if not torch.equal(fields[key],other[key]):raise RuntimeError('Feature alignment depends on row order')


def synthetic_window():
    channels=torch.arange(64,dtype=torch.float32)*.03
    states=[]
    for t in range(10):
        cells=[[0,0,0],[1,0,0],[t%3,1,0]]
        features=torch.stack([torch.sin(channels+sum(cell)*.11+t*.09) for cell in cells])
        if t%2:cells=list(reversed(cells));features=features.flip(0)
        states.append(dict(coordinates=torch.tensor(cells,dtype=torch.int32),features=features,pts=t/30.))
    return states


def mutated_future(states,after):
    changed=[]
    for i,value in enumerate(states):
        item=dict(value)
        if i>after:
            item['coordinates']=torch.cat((value['coordinates'],torch.tensor([[50+i,2,0]],dtype=torch.int32)))
            item['features']=torch.cat((value['features']+3,torch.full((1,64),7.)),0)
        changed.append(item)
    return changed


def gradient_norm(parameters):
    values=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Posterior variant source changed: '+name)
    config=json.loads((ROOT/'variant_config.json').read_text(encoding='utf-8'))
    context=load_context(config)
    stage,helper,orig,p,paths,receipt,evaluation=context.setup()
    decoder,modules=context.load_decoder(stage,helper,orig,p,paths,receipt,'cpu')
    fm,_,runtime=modules
    _,posterior=fm.build_models(p,'cpu',train_decoder=False)
    checkpoint=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    posterior.load_state_dict(checkpoint['posterior'],strict=True);posterior.eval()
    if runtime.digest(posterior)!=receipt['posterior_digest']:
        raise RuntimeError('9560-step posterior digest differs')
    initial_decoder_digest=runtime.digest(decoder)
    check_world_alignment()
    states=synthetic_window()
    from posterior import make_memories
    memories=make_memories(decoder.base,states)
    c0=ContinuousPosterior(decoder.base,posterior,'C0').eval()
    c1=ContinuousPosterior(decoder.base,posterior,'C1').eval()
    torch.manual_seed(20261102)
    c2_time=ContinuousPosterior(decoder.base,posterior,'C2',include_time=True,include_change=False).eval()
    torch.manual_seed(20261102)
    c2=ContinuousPosterior(decoder.base,posterior,'C2',include_time=True,include_change=True).eval()
    c2_plain=ContinuousPosterior(decoder.base,posterior,'C2',include_time=False,include_change=False).eval()
    with torch.no_grad():
        old=posterior([memories])['codes'][0]
        cached=c0.forward_cached(memories)
        copied=c1.forward_states(states)
    torch.testing.assert_close(old,cached,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(old,copied,atol=2e-5,rtol=1e-5)
    shape_checks={}
    for name,model in [('C1',c1),('C2_time',c2_time),('C2',c2)]:
        with torch.no_grad():all_codes=model.forward_states(states)
        if tuple(all_codes.shape)!=(9,16):raise RuntimeError('Wrong full-window code shape')
        for h in [1,3,7]:
            with torch.no_grad():
                before=model.code_for_h(states,h)
                after=model.code_for_h(mutated_future(states,h),h)
            torch.testing.assert_close(before,after,atol=2e-5,rtol=1e-6)
            if tuple(before.shape)!=(16,):raise RuntimeError('Wrong per-horizon shape')
        shape_checks[name]=list(all_codes.shape)
    keys,positions,time_memory=c2_time.memory_for_h(states,3)
    if not torch.equal(time_memory['mask'],time_memory['present']):raise RuntimeError('Time-only control unexpectedly added absent tokens')
    with torch.no_grad():plain_keys,plain_positions,_=c2_plain.memory_for_h(states,3)
    torch.testing.assert_close(keys,plain_keys,atol=1e-6,rtol=1e-6)
    if not float((positions-plain_positions).abs().max())>0:raise RuntimeError('Time embedding did not alter key positions')
    keys,positions,changed_memory=c2.memory_for_h(states,3)
    if not bool((changed_memory['mask'] & ~changed_memory['present']).any()):raise RuntimeError('Disappearance tokens missing')
    if len(keys)!=len(positions) or keys.shape[1]!=64 or positions.shape[1]!=64:
        raise RuntimeError('Q-Former memory dimension mismatch')
    gradients={}
    for name,model in [('C1',c1),('C2_time',c2_time),('C2',c2)]:
        model.train();model.zero_grad(set_to_none=True)
        weight=model.encoder.input.weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        code=model.code_for_h(states,3);(code.square().mean()).backward()
        norms=dict(encoder_input=gradient_norm(model.encoder.input.parameters()),
                   encoder_backbone=gradient_norm(model.encoder.encoder_backbone.parameters()),
                   query=gradient_norm(model.posterior.queries.parameters()),
                   projection=gradient_norm(model.posterior.project_in.parameters()))
        if model.include_time:norms['time_projection']=gradient_norm(model.time_projection.parameters())
        if model.include_change:norms['delta_projection']=gradient_norm(model.delta_projection.parameters())
        if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in norms.values()):
            raise RuntimeError('No gradient in '+name+': '+str(norms))
        optimizer.step()
        if torch.equal(weight,model.encoder.input.weight.detach()):raise RuntimeError('Encoder copy did not update')
        if runtime.digest(decoder)!=initial_decoder_digest or any(v.grad is not None for v in decoder.parameters()):
            raise RuntimeError('Fixed decoder changed during posterior-only update')
        gradients[name]=norms
    result=dict(status='passed',source_checkpoint_sha256=stage['bindings']['checkpoint'],
        decoder_digest=initial_decoder_digest,source_posterior_digest=runtime.digest(posterior),
        c0_matches_9560_posterior=True,c1_initial_output_matches_c0=True,
        world_alignment_and_disappearance_tokens=True,row_order_invariant=True,
        prefix_causality_horizons=[1,3,7],horizon9_output_shape_verified=True,
        time_only_uses_only_present_cells=True,time_embedding_changes_positions=True,
        gradient_checkpointing_passed=True,trainable_encoder_copy_gradients=gradients,
        decoder_parameters_unchanged=True,shape_checks=shape_checks,
        scope='Small synthetic states using actual frozen source modules; no teacher training or visual claim',
        code_sha256=context.sha(ROOT/'CODE.sha256'))
    (ROOT/'CPU_TESTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

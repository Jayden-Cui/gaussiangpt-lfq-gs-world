"""Full 50-clip codec experiment; reuse bound original model/loss/latent replay."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import numpy as np
import torch
import pool_index as pool

RUN = 'gaussiangpt_neoverse_fullpool_codec_20261031_v1'
SEED, BATCH, FRAMES, EPOCH_UPDATES, STEPS = 20261031, 4, 8850, 2213, 4426
ARMS = ('decoder_only', 'full_codec')
EVAL_CLIP = 'buildingblock_tower_14_0'
LEGACY_WORKER_SHA = 'f210f2f0961d87ed0f4e9d448b54b34118da00e9c958ae32ac1fb70a6335e3f7'
LEGACY_BINDING_SHA = '695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715'
INITIAL_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
LEGACY_ROOT = Path(__file__).resolve().parent.parent / pool.LEGACY_RUN
REMOTE_BASE = '/data/CJH/AGS-V6Direct/outputs/'

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_bound(path, digest, name):
    source = Path(path).read_bytes()
    require(hashlib.sha256(source).hexdigest() == digest, 'Bound implementation changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

legacy = load_bound(LEGACY_ROOT/'worker.py', LEGACY_WORKER_SHA, '_fullpool_original_worker')
LOSS_WEIGHTS = dict(legacy.LOSS_WEIGHTS)

def validate_protocol(p):
    checks = {
        ('seed',):SEED, ('source','checkpoint_sha256'):INITIAL_SHA,
        ('source','legacy_root'):REMOTE_BASE+pool.LEGACY_RUN,
        ('data','pool_root'):REMOTE_BASE+pool.POOL_RUN,
        ('data','train_clips'):sorted(pool.EXPECTED), ('data','eval_only_clips'):[EVAL_CLIP],
        ('data','frames_per_clip'):177, ('data','teacher_points_per_frame'):112896,
        ('data','height'):252, ('data','width'):448, ('data','frame_stride'):1,
        ('model','latent_width'):64, ('model','state_quantization'):False,
        ('model','grid'):.008, ('model','stride'):4, ('model','gaussians_per_voxel'):4,
        ('model','max_input_voxels'):80000, ('model','max_decoder_candidates'):600000,
        ('model','max_decoded_gaussians'):2400000,
        ('train','steps'):STEPS, ('train','batch_size'):BATCH,
        ('train','epoch_seeds'):[SEED,SEED+1], ('train','epochs'):2,
        ('train','force_target_epochs'):[True,False], ('train','learning_rate'):1e-4,
        ('train','betas'):[.9,.95], ('train','weight_decay'):0., ('train','grad_clip'):1.,
        ('train','loss'):LOSS_WEIGHTS, ('train','save_steps'):[EPOCH_UPDATES,STEPS],
        ('train','max_seconds'):7200, ('preflight','updates_per_arm'):2,
        ('preflight','force_target_sequence'):[True,False],
        ('evaluation','records_per_model'):9027, ('evaluation','snapshot_frames'):[0,44,88,132,176],
        ('evaluation','workspace_roi_xyxy'):list(legacy.ROI), ('evaluation','target_forcing'):False,
        ('resources','peak_limit_gib'):48, ('resources','cpu_threads'):6,
    }
    for keys, expected in checks.items():
        actual = p
        for key in keys: actual = actual[key]
        require(actual == expected, 'Protocol drift at '+'.'.join(keys))
    require(type(p['preflight']['max_seconds']) is int and 0 < p['preflight']['max_seconds'] <= 1800,
            'Preflight needs bounded seconds')
    require(type(p['evaluation']['max_seconds']) is int and 0 < p['evaluation']['max_seconds'] <= 14400,
            'Evaluation needs bounded seconds')
    validate_preflight_batches(p['preflight']['batches'])

def validate_preflight_batches(batches):
    require(len(batches) == 2 and all(len(b) == 4 for b in batches), 'Preflight requires two complete B4 batches')
    for batch in batches:
        ids = [(r['clip'],r['frame']) for r in batch]
        require(len(set(ids)) == 4 and all(c in pool.EXPECTED and type(f) is int and 0 <= f < 177 for c,f in ids),
                'Each preflight batch requires four distinct TRAIN frames')

def verify_root(root):
    manifest = read(root/'CODE_MANIFEST.json')
    require(set(p.name for p in root.glob('*.py')) <= set(manifest), 'Unbound run executable')
    for name, digest in manifest.items():
        require(Path(name).name == name and sha(root/name) == digest, 'Run source changed: '+name)
    proof = read(root/'CPU_VALIDATION.json')
    digest = sha(root/'CODE_MANIFEST.json')
    require(proof.get('status') == 'passed' and proof.get('gpu_executed') is False
            and proof.get('code_manifest_sha256') == digest, 'Fresh exact CPU witness required')
    p = read(root/'protocol.json'); validate_protocol(p)
    return p, dict(code_manifest_sha256=digest, protocol_sha256=sha(root/'protocol.json'),
                   cpu_validation_sha256=sha(root/'CPU_VALIDATION.json'),
                   pool_complete_sha256=pool.COMPLETE_SHA, pool_done_sha256=pool.DONE_SHA,
                   legacy_worker_sha256=LEGACY_WORKER_SHA, legacy_sources_sha256=pool.LEGACY_SHA,
                   legacy_binding_sha256=LEGACY_BINDING_SHA, initial_checkpoint_sha256=INITIAL_SHA)

def make_epochs(index):
    require(len(index) == FRAMES and {r.clip for r in index} == set(pool.EXPECTED), 'Full TRAIN pool required')
    plans = [pool.balanced_plan(index, seed, BATCH) for seed in (SEED,SEED+1)]
    for plan in plans:
        pool.assert_coverage(index, plan)
        require(len(plan) == EPOCH_UPDATES and len(plan[-1]) == 2
                and all(len(b) == BATCH for b in plan[:-1]), 'Coverage/batch size drift')
    return plans

def forcing_for_step(step):
    require(type(step) is int and 1 <= step <= STEPS, 'Step outside full two-epoch plan')
    return step <= EPOCH_UPDATES

def accumulated_update(model, optimizer, batch, frame_loader, terms_fn, vgg, force, *, guard=lambda:None):
    require(0 < len(batch) <= BATCH and len({(r.clip,r.frame) for r in batch}) == len(batch),
            'Empty, oversized or duplicate batch')
    require(all(r.current_role == 'TRAIN' and r.clip in pool.EXPECTED for r in batch), 'Evaluation frame in update')
    optimizer.zero_grad(set_to_none=True)
    records = []
    for ref in batch:
        guard()
        payload = frame_loader(ref)
        terms, info = terms_fn(model,payload,vgg,force)
        require(set(terms) == set(LOSS_WEIGHTS), 'Original six loss terms changed')
        loss = sum(LOSS_WEIGHTS[k]*v for k,v in terms.items())
        require(bool(torch.isfinite(loss)), 'Nonfinite loss')
        (loss/len(batch)).backward()
        records.append(dict(clip=ref.clip,frame=ref.frame,source_group=ref.source_group,
            loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            latent_count=int(info['latent_count']),output_count=int(info['output_count'])))
        del payload,terms,info,loss
        guard()
    gradients = legacy.gradient_report(model)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    require(float(norm) > 0, 'Zero update gradient')
    guard();optimizer.step();guard()
    return dict(frames=records,actual_batch_size=len(batch),force_target=force,
        loss=sum(r['loss'] for r in records)/len(batch),
        terms={k:sum(r['terms'][k] for r in records)/len(batch) for k in LOSS_WEIGHTS},
        gradient_norm=float(norm),gradient_groups=gradients)

def load_runtime(root):
    p, evidence = verify_root(root)
    # This old binding must already exist. Never allow old load_verified to create it.
    require((LEGACY_ROOT/'DATA_BINDING.json').is_file() and sha(LEGACY_ROOT/'DATA_BINDING.json') == LEGACY_BINDING_SHA,
            'Existing exact legacy binding required; cannot write into old run')
    sources = load_bound(LEGACY_ROOT/'sources.py',pool.LEGACY_SHA,'_fullpool_runtime_sources')
    index = pool.load_pool_index(Path(p['data']['pool_root']))
    plans = make_epochs(index)
    bundle = sources.load_verified(LEGACY_ROOT)
    require(sha(LEGACY_ROOT/'DATA_BINDING.json') == LEGACY_BINDING_SHA, 'Old binding changed')
    require(sha(bundle.checkpoint) == INITIAL_SHA, 'Initial checkpoint changed')
    summaries = [pool.plan_summary(index,plan) for plan in plans]
    evidence['epoch_plan_sha256'] = [s['plan_sha256'] for s in summaries]
    return SimpleNamespace(root=root,p=p,evidence=evidence,index=index,plans=plans,plan_summaries=summaries,
        old=bundle,sources=sources,by_id={(r.clip,r.frame):r for r in index})

def recheck(x):
    _, current = verify_root(x.root)
    require(all(x.evidence[k] == v for k,v in current.items()), 'Run binding changed')
    for name,pinned in [('queue/DONE.json',pool.DONE_SHA),('POOL_COMPLETE.json',pool.COMPLETE_SHA),
                        ('INPUT_POOL.json',pool.INPUT_POOL_SHA),('CODE_MANIFEST.json',pool.MANIFEST_SHA)]:
        require(sha(Path(x.p['data']['pool_root'])/name) == pinned,'Pool receipt changed')
    x.sources.verify_sources(x.old)
    # Every TRAIN cache is read and hashed on each use; unused payloads are not retained.

def gate(x):
    g = read(x.root/'PREFLIGHT.json')
    require(g['status'] == 'passed' and g['source'] == x.evidence and g.get('initial_arms_identical') is True,
            'Exact two-arm preflight required')
    require(set(g['arms']) == set(ARMS) and all(a['passed'] and a['discarded_updates'] == 2
            and a['production_updates'] == 0 for a in g['arms'].values()),'Incomplete preflight')
    require(len({a['initial_model_sha256'] for a in g['arms'].values()}) == 1,'Unequal preflight initial models')
    return g

def preflight(x):
    path=x.root/'PREFLIGHT.json'
    require(not path.exists(),'Preflight already attempted')
    legacy.write_json(path,dict(status='running',source=x.evidence,goal_complete=False))
    budget=legacy.Budget(x.p['preflight']['max_seconds']);results={};images={};initial=None
    batches=[tuple(x.by_id[r['clip'],r['frame']] for r in b) for b in x.p['preflight']['batches']]
    try:
        for arm in ARMS:
            model=legacy.initial_model(x.old);digest=legacy.digest_snapshot(legacy.snapshot(model))
            require(initial is None or initial == digest,'Initial arm weights differ');initial=digest
            states=[]
            for batch in batches:
                for ref in batch:
                    budget.check();payload=pool.load_frame(ref,device='cuda')
                    image,nearby,packet,replay=legacy.encode_replay(model,payload,x.old.train.render_prediction,
                        dict(clip=ref.clip,frame=ref.frame,phase='discarded_preflight'))
                    identity=(ref.clip,ref.frame)
                    parity=None
                    if identity in images:
                        parity=dict(real=legacy.compare_render(image.cpu(),images[identity][0]),
                                    nearby=legacy.compare_render(nearby.cpu(),images[identity][1]))
                    images[identity]=(image.cpu(),nearby.cpu())
                    states.append(dict(clip=ref.clip,frame=ref.frame,input_voxels=len(payload['representatives']),
                        latent_shape=list(packet['features'].shape),replay=replay,arm_parity=parity))
                    del payload,image,nearby,packet
            legacy.configure_arm(model,arm);frozen=legacy.snapshot(model,encoder_only=True)
            before=legacy.snapshot(model,parameters_only=True);optimizer=legacy.fresh_optimizer(model)
            require(not optimizer.state,'Fresh optimizer required')
            vgg=x.old.perceptual.Vgg19(x.old.codec_root/'torch_cache').cuda();updates=[]
            for batch,force in zip(batches,(True,False)):
                updates.append(accumulated_update(model,optimizer,batch,lambda r:pool.load_frame(r,device='cuda'),
                    x.old.train.terms,vgg,force,guard=budget.check))
                legacy.assert_training_modes(model,arm)
                if arm == 'decoder_only': legacy.assert_frozen(model,frozen)
            changed={legacy.group_name(n) for n,v in legacy.snapshot(model,parameters_only=True).items() if before[n]!=v}
            required=set(legacy.TRAINABLE_DECODER)
            if arm == 'full_codec':required.update(legacy.group_name(n) for n,_ in model.named_parameters() if not legacy.in_decoder(n))
            require(required <= changed,'Missing updated parameter groups')
            require(all(any(u['gradient_groups'].get(g,{}).get('nonzero_grad',0) for u in updates) for g in required),
                    'Missing finite nonzero gradient groups')
            results[arm]=dict(passed=True,initial_model_sha256=digest,discarded_updates=2,production_updates=0,
                actual_batch_sizes=[len(b) for b in batches],states=states,updates=updates,
                changed_groups=sorted(changed),frozen_encoder_verified=arm=='decoder_only',**budget.evidence())
            del model,optimizer,vgg;gc.collect();torch.cuda.empty_cache()
        recheck(x);budget.check()
        legacy.write_json(path,dict(status='passed',source=x.evidence,arms=results,initial_arms_identical=True,
            gpu_witness=True,goal_complete=False,**budget.evidence()))
    except BaseException as error:
        legacy.write_json(path,dict(status='failed',source=x.evidence,arms=results,error=repr(error),goal_complete=False))
        raise

def train(x,arm):
    g=gate(x);folder=x.root/'arms'/arm;folder.mkdir(parents=True,exist_ok=False)
    budget=legacy.Budget(x.p['train']['max_seconds']);completed=0;model=optimizer=None;seen=[set(),set()]
    try:
        model=legacy.initial_model(x.old);initial=legacy.digest_snapshot(legacy.snapshot(model))
        require(initial==g['arms'][arm]['initial_model_sha256'],'Initial model differs from preflight')
        legacy.configure_arm(model,arm);frozen=legacy.snapshot(model,encoder_only=True)
        initial_encoder_parameters=legacy.snapshot(model,encoder_only=True,parameters_only=True)
        optimizer=legacy.fresh_optimizer(model);require(not optimizer.state,'Old optimizer state forbidden')
        vgg=x.old.perceptual.Vgg19(x.old.codec_root/'torch_cache').cuda()
        legacy.write_json(folder/'SOURCE.json',dict(source=x.evidence,arm=arm,preflight_sha256=sha(x.root/'PREFLIGHT.json')))
        legacy.write_json(folder/'PLAN.json',dict(epochs=x.plan_summaries,epoch_seeds=[SEED,SEED+1],forcing=[True,False]))
        with (folder/'trace.jsonl').open('x',encoding='utf-8',buffering=1) as trace:
            for epoch,plan in enumerate(x.plans):
                for batch in plan:
                    step=completed+1
                    row=accumulated_update(model,optimizer,batch,lambda r:pool.load_frame(r,device='cuda'),
                        x.old.train.terms,vgg,forcing_for_step(step),guard=budget.check)
                    completed=step;seen[epoch].update((r.clip,r.frame) for r in batch)
                    legacy.assert_training_modes(model,arm)
                    if arm=='decoder_only' and (step==1 or step%100==0):legacy.assert_frozen(model,frozen)
                    row.update(step=step,epoch=epoch+1,arm=arm,epoch_frames_seen=len(seen[epoch]),**budget.evidence())
                    trace.write(json.dumps(row,allow_nan=False)+'\n')
                    if step==1 or step%100==0:print('FULLPOOL_TRAIN',arm,step,row['loss'],flush=True)
                require(len(seen[epoch])==FRAMES,'Incomplete epoch coverage')
                legacy.atomic_save(dict(schema='fullpool_codec_checkpoint_v1',model=model.state_dict(),
                    optimizer=optimizer.state_dict(),arm=arm,step=completed,source=x.evidence,
                    initial_model_sha256=initial,initial_checkpoint_sha256=INITIAL_SHA,inherited_codec_steps=3000,
                    encoder_sha256=legacy.digest_snapshot(legacy.snapshot(model,encoder_only=True)),
                    eligible_for_final_evaluation=completed==STEPS,failed=False,resumable=False),folder/f'step_{completed:04d}.pt')
        require(completed==STEPS and seen[0]==seen[1] and len(seen[0])==FRAMES,'Incomplete two full epochs')
        if arm=='decoder_only':legacy.assert_frozen(model,frozen)
        else:require(legacy.snapshot(model,encoder_only=True,parameters_only=True)!=initial_encoder_parameters,'Encoder did not update')
        recheck(x);budget.check()
        legacy.write_json(folder/'TRAIN_DONE.json',dict(status='complete',arm=arm,steps=STEPS,source=x.evidence,
            inherited_codec_steps=3000,additional_codec_updates=STEPS,epoch_frames=[len(s) for s in seen],
            total_frame_exposures=2*FRAMES,forcing_updates=EPOCH_UPDATES,unforced_updates=EPOCH_UPDATES,
            logical_batch_size=BATCH,tail_batch_sizes=[2,2],fresh_optimizer_verified=True,
            initial_model_sha256=initial,initial_encoder_sha256=legacy.digest_snapshot(frozen),
            final_encoder_sha256=legacy.digest_snapshot(legacy.snapshot(model,encoder_only=True)),
            frozen_encoder_verified=arm=='decoder_only',checkpoint_sha256=sha(folder/f'step_{STEPS:04d}.pt'),
            first_epoch_checkpoint_sha256=sha(folder/f'step_{EPOCH_UPDATES:04d}.pt'),
            trace_sha256=sha(folder/'trace.jsonl'),plan_sha256=sha(folder/'PLAN.json'),goal_complete=False,**budget.evidence()))
    except BaseException as error:
        failed_checkpoint_saved=False
        if model is not None:
            try:
                legacy.atomic_save(dict(schema='fullpool_codec_checkpoint_v1',model=model.state_dict(),
                    optimizer=optimizer.state_dict() if optimizer is not None else {},arm=arm,step=completed,
                    source=x.evidence,failed=True,eligible_for_final_evaluation=False,resumable=False),
                    folder/'FAILED_DO_NOT_EVALUATE.pt')
                failed_checkpoint_saved=True
            except BaseException:
                pass
        legacy.write_json(folder/'FAILURE.json',dict(status='failed',error=repr(error),last_completed_step=completed,
            eligible_for_final_evaluation=False,failed_checkpoint_saved=failed_checkpoint_saved,
            goal_complete=False,**budget.evidence()))
        raise

def evaluation_model(x,arm):
    path=x.old.checkpoint;step=0
    if arm is not None:
        folder=x.root/'arms'/arm;require(not (folder/'FAILURE.json').exists(),'Failed arm cannot be evaluated')
        done=read(folder/'TRAIN_DONE.json');path=folder/f'step_{STEPS:04d}.pt'
        require(done['status']=='complete' and done['steps']==STEPS and done['source']==x.evidence
                and sha(path)==done['checkpoint_sha256'],'Incomplete/mismatched final checkpoint')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['step']==STEPS and saved['arm']==arm and saved['source']==x.evidence
                and saved['eligible_for_final_evaluation'] and not saved['failed'],'Ineligible checkpoint')
        del saved;step=STEPS
    legacy.seed_all()
    model=x.old.codec.load_codec(x.old.codec_root,path).eval()
    return model,path,step

def evaluate(x,arm):
    gate(x);label=arm or 'baseline';folder=x.root/'evaluations'/label;folder.mkdir(parents=True,exist_ok=False)
    budget=legacy.Budget(x.p['evaluation']['max_seconds']);rows=[];context={}
    try:
        model,checkpoint,step=evaluation_model(x,arm);checkpoint_sha=sha(checkpoint)
        encoder_sha=legacy.digest_snapshot(legacy.snapshot(model,encoder_only=True))
        from PIL import Image
        clips=[*sorted(pool.EXPECTED),EVAL_CLIP]
        with (folder/'targets.jsonl').open('x',buffering=1) as log, \
             (folder/'state_manifest.jsonl').open('x',buffering=1) as manifest, \
             (folder/'screenshot_manifest.jsonl').open('x',buffering=1) as shots:
            for clip in clips:
                state_dir=folder/'states'/clip;state_dir.mkdir(parents=True)
                picture_dir=folder/'screenshots'/clip;picture_dir.mkdir(parents=True)
                for frame in range(177):
                    budget.check();context=dict(clip=clip,frame=frame)
                    payload=(pool.load_frame(x.by_id[clip,frame],device='cuda') if clip in pool.EXPECTED
                             else x.sources.load_frame(x.old.sources[clip],frame,device='cuda'))
                    metadata=dict(**context,arm=label,split='TRAIN' if clip in pool.EXPECTED else 'EVAL-only',
                        source_group=pool.EXPECTED.get(clip,'buildingblock_tower_14'),normalization=payload['normalization'],
                        pts=payload['pts'],source_cache_sha256=payload['_source_cache_sha256'],
                        checkpoint_sha256=checkpoint_sha,encoder_sha256=encoder_sha,source=x.evidence,
                        additional_codec_updates=step,inherited_codec_steps=3000,decoder_target_key=None,fresh_encode=True)
                    path=state_dir/f'{frame:03d}.pt'
                    image,nearby,packet,replay=legacy.encode_replay(model,payload,x.old.train.render_prediction,metadata,path)
                    teacher=payload['targets'][0].clamp(0,1)
                    row=dict(**context,split=metadata['split'],source_group=metadata['source_group'],pts=payload['pts'],
                        codec_real=legacy.image_metrics(image,payload['real']),teacher_real=legacy.image_metrics(teacher,payload['real']),
                        codec_teacher=legacy.image_metrics(image,teacher),pseudo_view_teacher=legacy.image_metrics(nearby,payload['targets'][1].clamp(0,1)),
                        latent_positions=len(packet['coordinates']),latent_channels=64,input_gaussians=int(payload['n_input']),
                        input_voxels=len(payload['representatives']),decoded_gaussians=metadata['decoded_gaussians'],
                        replay=replay,checkpoint_sha256=checkpoint_sha,encoder_sha256=encoder_sha,additional_codec_updates=step)
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                    manifest.write(json.dumps(dict(**metadata,state_path=path.relative_to(folder).as_posix(),
                        state_sha256=sha(path),state_bytes=path.stat().st_size,shape=list(packet['features'].shape),
                        saved_packet_replay=replay),allow_nan=False)+'\n')
                    if frame in legacy.SCREENSHOT_FRAMES:
                        for kind,picture in [('real',payload['real']),('teacher',teacher),('codec',image),('nearby',nearby),('nearby_teacher',payload['targets'][1])]:
                            target=picture_dir/f'{frame:03d}_{kind}.png'
                            Image.fromarray((picture.detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)).save(target)
                            shots.write(json.dumps(dict(**context,kind=kind,path=target.relative_to(folder).as_posix(),
                                bytes=target.stat().st_size,sha256=sha(target)))+'\n')
                    del payload,image,nearby,packet,teacher
                    budget.check()
                print('FULLPOOL_EVAL',label,clip,len(rows),flush=True)
        expected={(c,f) for c in clips for f in range(177)}
        require(len(rows)==9027 and {(r['clip'],r['frame']) for r in rows}==expected,'Incomplete 9027-frame evaluation')
        recheck(x);budget.check()
        def average(selected):
            return {k:{m:float(np.mean([r[k][m] for r in selected])) for m in selected[0][k]}
                    for k in ('codec_real','teacher_real','codec_teacher','pseudo_view_teacher')}
        legacy.write_json(folder/'EVALUATION.json',dict(status='evaluated',arm=label,frames=9027,train_frames=8850,
            exposed_eval_only_frames=177,additional_codec_updates=step,checkpoint_sha256=checkpoint_sha,
            encoder_sha256=encoder_sha,source=x.evidence,summary_by_clip={c:average([r for r in rows if r['clip']==c]) for c in clips},
            summary_train_frames=average([r for r in rows if r['split']=='TRAIN']),
            summary_exposed_eval=average([r for r in rows if r['split']=='EVAL-only']),
            every_packet_saved_replay_passed=True,all_frames_freshly_encoded=True,saved_latent_state_replay=True,
            independent_native_gs_replay=False,decoder_target_key=None,roi_xyxy=list(legacy.ROI),
            targets_sha256=sha(folder/'targets.jsonl'),state_manifest_sha256=sha(folder/'state_manifest.jsonl'),
            screenshot_manifest_sha256=sha(folder/'screenshot_manifest.jsonl'),goal_complete=False,**budget.evidence()))
    except BaseException as error:
        legacy.write_json(folder/'FAILURE.json',dict(status='failed',error=repr(error),context=context,
            frames_complete=len(rows),valid_complete_evaluation=False,goal_complete=False,**budget.evidence()))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=('preflight','train','evaluate','baseline'),required=True)
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    if (args.mode in ('train','evaluate')) != (args.arm is not None):parser.error('Only train/evaluate require --arm')
    root=args.root.resolve(strict=True);require(root.name==RUN and root==Path(__file__).resolve().parent,'Dedicated run required')
    x=load_runtime(root);legacy.setup_gpu()
    def interrupted(signum,frame):raise RuntimeError('Interrupted owned worker')
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='preflight':preflight(x)
        elif args.mode=='train':train(x,args.arm)
        else:evaluate(x,args.arm)
    finally:signal.signal(signal.SIGTERM,previous)

if __name__=='__main__':main()

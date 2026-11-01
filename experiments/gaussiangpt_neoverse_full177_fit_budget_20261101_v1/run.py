"""Bounded paired continuation, full evaluations, and independent native replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from sampling import schedule, audit

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as f:
            f.write(text)


def explicit(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path).resolve():
        raise RuntimeError('Module origin mismatch')
    return module


def setup():
    for line in (ROOT / 'CODE.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Continuation source changed: ' + name)
    c = read(ROOT / 'continuation.json')
    helper_root = Path(c['resume_helper_root'])
    if (sha(helper_root / 'CODE.sha256') != c['resume_helper_code_sha256']
            or sha(helper_root / 'resume.py') != c['resume_helper_entry_sha256']):
        raise RuntimeError('Bound restore helper changed')
    helper = explicit('_full177_resume_helpers', helper_root / 'resume.py')
    binding, parent, p = helper.setup()
    if sha(ROOT / 'protocol.json') != binding['parent_protocol_sha256']:
        raise RuntimeError('Original data/evaluation protocol changed')
    if sha(helper_root / 'SUMMARY.json') != c['baseline_summary_sha256']:
        raise RuntimeError('Completed baseline evidence changed')
    return c, helper, binding, parent, p


def optimizer_digest(value):
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(str(type(x)).encode())
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def training_mode(decoder, posterior, full_model):
    decoder.eval()
    for name, param in decoder.named_parameters():
        param.requires_grad_(name.startswith(full_model.TRAIN_PREFIXES))
    for module in [decoder.base.decoder_backbone, decoder.base.output_norm,
                   decoder.base.features, decoder.base.quantizer.project_out,
                   decoder.geometry, decoder.occupancy]:
        module.train()
    posterior.requires_grad_(True).train()


def restore(c, helper, binding, parent, p, mode, device, modules):
    import torch
    fm, _, runtime = modules
    decoder, posterior, done = helper.restore(binding, parent, p, mode, device, modules)
    training_mode(decoder, posterior, fm)
    checkpoint = torch.load(parent / mode / 'step_2000.pt', map_location='cpu', weights_only=True)
    opt = torch.optim.Adam(fm.parameters(decoder, posterior), lr=c['learning_rate'],
                           betas=tuple(p['betas']), eps=p['eps'])
    opt.load_state_dict(checkpoint['optimizer'])
    if len(opt.param_groups) != 1 or opt.param_groups[0]['lr'] != c['learning_rate']:
        raise RuntimeError('Continuation learning rate differs from final parent rate')
    if optimizer_digest(opt.state_dict()) != optimizer_digest(checkpoint['optimizer']):
        raise RuntimeError('Adam moments or parameter association changed')
    steps = {int(v['step']) for v in opt.state.values()}
    if steps != {c['parent_step']}:
        raise RuntimeError('Optimizer was reset or has an unexpected step')
    if (runtime.digest(decoder) != done['decoder_digest']
            or runtime.digest(posterior) != done['posterior_digest']):
        raise RuntimeError('Restore modified model tensors')
    names = ['posterior.' + n for n, _ in posterior.named_parameters()]
    names += ['decoder.' + n for n, v in decoder.named_parameters() if v.requires_grad]
    if len(names) != len(opt.param_groups[0]['params']):
        raise RuntimeError('Optimizer parameter ownership differs')
    del checkpoint
    return decoder, posterior, opt, done, names


def tests():
    import torch
    c, helper, binding, parent, p = setup()
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    sampling = audit(schedule(c), c)
    modules = helper.model_modules(parent)
    rows = []
    for mode in p['modes']:
        decoder, posterior, opt, done, names = restore(c, helper, binding, parent, p, mode, 'cpu', modules)
        fm, _, runtime = modules
        if fm.frozen_digest(decoder) != done['source']['frozen_component_digest']:
            raise RuntimeError('Frozen state encoder changed')
        evaluator = explicit('_full177_fit_evaluator', parent / 'evaluate.py')
        update_module = explicit('_full177_fit_parent_update', parent / 'worker.py')
        if not callable(evaluator.evaluate) or not callable(update_module.update):
            raise RuntimeError('Bound update/evaluation entry missing')
        rows.append(dict(mode=mode, checkpoint_sha256=done['checkpoint_sha256'],
                         decoder_digest=runtime.digest(decoder), posterior_digest=runtime.digest(posterior),
                         optimizer_digest=optimizer_digest(opt.state_dict()), optimizer_step=2000,
                         trainable_parameter_names=names, evaluation_origin=evaluator.__file__,
                         training_update_origin=update_module.__file__))
        del decoder, posterior, opt
    write(ROOT / 'CPU_TESTS.json', dict(status='passed', sampling=sampling, rows=rows,
          code_manifest_sha256=sha(ROOT / 'CODE.sha256'),
          scope='Exact full-pair coverage, real Adam/checkpoint continuation, and explicit import origins; GPU preflight runs in workers'))
    print('PASS: each of 1512 window/horizon pairs rendered exactly 20 times; both Adam states and models restored', flush=True)


def worker(mode):
    import torch
    c, helper, binding, parent, p = setup()
    if mode not in p['modes']: raise ValueError('Unknown arm')
    out = ROOT / mode; out.mkdir()
    began = time.monotonic(); eval_seconds = 0.; context = dict(phase='initialization', global_step=2000)
    write(out / 'STARTED.json', dict(mode=mode, pid=os.getpid(), parent_step=2000))
    write(out / 'STATUS.json', dict(status='running', **context))
    try:
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose one GPU')
        free, total = torch.cuda.mem_get_info()
        if free < p['minimum_free_gpu_gib'] * 2**30: raise RuntimeError('GPU memory unavailable')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib'] * 2**30 / total)
        modules = helper.model_modules(parent); fm, data, runtime = modules
        decoder, posterior, opt, done, names = restore(c, helper, binding, parent, p, mode, 'cuda', modules)
        witness = read(ROOT / 'CPU_TESTS.json')
        expected = next(row for row in witness['rows'] if row['mode'] == mode)
        if (optimizer_digest(opt.state_dict()) != expected['optimizer_digest']
                or names != expected['trainable_parameter_names']):
            raise RuntimeError('GPU optimizer differs from CPU restore witness')
        store = data.WindowStore(p); std = store.frames['std'].cuda()
        if (sha(parent / 'cache/MANIFEST.json') != done['source']['cache_manifest_sha256']
                or store.manifest['frozen_model_digest'] != done['source']['frozen_component_digest']):
            raise RuntimeError('Training cache/encoder binding differs')
        from runtime_bridge import BoundNativeBridge
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        update_module = explicit('_full177_fit_parent_update', parent / 'worker.py')
        evaluator = explicit('_full177_fit_evaluator', parent / 'evaluate.py')
        items = schedule(c); sampling = audit(items, c)
        if sampling != witness['sampling']: raise RuntimeError('Training schedule differs from witness')
        source = dict(parent_training_root=str(parent), parent_step=2000,
                      warm_checkpoint_sha256=done['checkpoint_sha256'],
                      warm_decoder_digest=done['decoder_digest'], warm_posterior_digest=done['posterior_digest'],
                      warm_optimizer_digest=expected['optimizer_digest'],
                      original_protocol_sha256=binding['parent_protocol_sha256'],
                      continuation_sha256=sha(ROOT / 'continuation.json'), code_sha256=sha(ROOT / 'CODE.sha256'),
                      cache_manifest_sha256=done['source']['cache_manifest_sha256'],
                      frozen_component_digest=done['source']['frozen_component_digest'],
                      codec_sha256=p['codec_checkpoint_sha256'], schedule_sha256=sampling['sha256'],
                      mode=mode, training_frames=177, training_windows=168, code_shape=[4,9,16], quantized=False)
        write(out / 'PROVENANCE.json', dict(source=source, trainable_parameter_names=names))
        def frozen():
            bridge.assert_frozen()
            if fm.frozen_digest(decoder) != done['source']['frozen_component_digest']:
                raise RuntimeError('Frozen state encoder or unused free codes changed')
            if any(v.grad is not None for n, v in decoder.named_parameters() if not n.startswith(fm.TRAIN_PREFIXES)):
                raise RuntimeError('Gradient on frozen component')
        probe = dict(phase='render', batch=[dict(start=s, horizon=9) for s in [0,35,123,167]])
        probe_result = update_module.update(decoder, posterior, bridge, store, std, opt, probe, mode, p, probe=True)
        frozen()
        del decoder, posterior, opt
        decoder, posterior, opt, _, restored_names = restore(c, helper, binding, parent, p, mode, 'cuda', modules)
        if restored_names != names: raise RuntimeError('Preflight reset changed parameter ownership')
        write(out / 'PREFLIGHT.json', dict(status='passed', discarded_updates=1, effective_batch_size=4,
              windows=[0,35,123,167], horizon=9, rgb_to_qformer_and_motion_decoder=True,
              restored_parent_models_and_adam=True, source=source, results=probe_result))
        with (out / 'train.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            for index, item in enumerate(items):
                step = item['step']; context = dict(phase='render', global_step=step-1, sweep=item['sweep'])
                if time.monotonic()-began-eval_seconds > c['training_max_seconds']:
                    raise RuntimeError('Training time budget reached')
                if opt.param_groups[0]['lr'] != c['learning_rate']: raise RuntimeError('Learning rate changed')
                row = update_module.update(decoder, posterior, bridge, store, std, opt, item, mode, p)
                row.update(step=step, additional_step=index+1, sweep=item['sweep'], phase='render',
                           lr=opt.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-began,
                           training_seconds=time.monotonic()-began-eval_seconds)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                if index == 0 or (index+1) % c['progress_every'] == 0:
                    write(out / 'STATUS.json', dict(status='running', phase='render', global_step=step,
                          additional_updates=index+1, total_additional_updates=c['additional_steps'], sweep=item['sweep'],
                          mean_loss=sum(w['loss'] for w in row['windows'])/4,
                          training_seconds=row['training_seconds']), replace=True)
                if step not in c['checkpoints']: continue
                opt.zero_grad(set_to_none=True); frozen()
                checkpoint_path = out / f'step_{step:05d}.pt'
                current_source = dict(source, global_step=step, completed_additional_updates=index+1,
                                      completed_pair_sweeps=item['sweep'])
                runtime.save(checkpoint_path, dict(schema='full177_fit_budget_continuation_v1', mode=mode,
                    step=step, decoder=decoder.state_dict(), posterior=posterior.state_dict(),
                    optimizer=opt.state_dict(), source=current_source, trainable_parameter_names=names))
                decoder_digest, posterior_digest = runtime.digest(decoder), runtime.digest(posterior)
                disk = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                check_decoder, check_posterior = fm.build_models(p, 'cpu', train_decoder=False)
                check_decoder.load_state_dict(disk['decoder']); check_posterior.load_state_dict(disk['posterior'])
                if (runtime.digest(check_decoder) != decoder_digest or runtime.digest(check_posterior) != posterior_digest
                        or optimizer_digest(disk['optimizer']) != optimizer_digest(opt.state_dict())):
                    raise RuntimeError('Checkpoint model/optimizer replay mismatch')
                key = 'memories' if mode == 'posterior' else 'initial_memories'
                decoder.eval(); posterior.eval(); check_posterior.eval().cuda()
                with torch.no_grad():
                    memory = store.get(35)[key]
                    if not torch.equal(posterior([memory])['codes'], check_posterior([memory])['codes']):
                        raise RuntimeError('Checkpoint action-code replay mismatch')
                del memory, disk, check_decoder, check_posterior
                stage = ROOT / f'step_{step:05d}' / mode; stage.mkdir(parents=True)
                write(stage / 'STARTED.json', dict(mode=mode, pid=os.getpid(), global_step=step))
                receipt = dict(status='complete', steps=step, additional_steps=index+1, source=current_source,
                               decoder_digest=decoder_digest, posterior_digest=posterior_digest,
                               checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                               checkpoint_replay='passed', codec_and_state_encoder_frozen=True)
                write(stage / 'TRAIN_DONE.json', receipt)
                context = dict(phase='evaluation', global_step=step)
                write(out / 'STATUS.json', dict(status='running', **context, stage=str(stage)), replace=True)
                started = time.monotonic()
                evaluator.evaluate(decoder, posterior, bridge, store, p, stage,
                                   dict(current_source, trained_checkpoint_sha256=receipt['checkpoint_sha256']), mode)
                eval_seconds += time.monotonic()-started
                frozen()
                if runtime.digest(decoder) != decoder_digest or runtime.digest(posterior) != posterior_digest:
                    raise RuntimeError('Evaluation modified trained tensors')
                write(stage / 'STATUS.json', dict(status='complete', phase='evaluation', native_replay='pending'))
                training_mode(decoder, posterior, fm)
        frozen()
        write(out / 'TRAIN_DONE.json', dict(status='complete', final_step=c['final_step'],
              additional_updates=c['additional_steps'], additional_pair_sweeps=c['sweeps'], source=source,
              checkpoints=c['checkpoints'], unchanged_learning_rate=c['learning_rate'],
              training_seconds=time.monotonic()-began-eval_seconds, evaluation_seconds=eval_seconds,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(out / 'STATUS.json', dict(status='complete', phase='training_and_evaluation',
              global_step=c['final_step'], native_replay='pending'), replace=True)
    except BaseException as error:
        failure = dict(status='failed', mode=mode, context=context, error=repr(error), traceback=traceback.format_exc())
        write(out / 'FAILED.json', failure); write(out / 'STATUS.json', failure, replace=True)
        raise


def replay(mode, step):
    c, helper, binding, parent, p = setup()
    if step not in c['checkpoints'] or mode not in p['modes']: raise ValueError('Unknown replay target')
    module = explicit('_full177_fit_replay', parent / 'replay.py')
    module.ROOT = ROOT / f'step_{step:05d}'
    module.main(mode)


def summarize(c, p):
    baseline = read(Path(c['resume_helper_root']) / 'SUMMARY.json')
    stages = {}; lines = ['# 全177帧：固定预算继续拟合', '',
        '沿用177帧TRAIN视频、168个相邻T10窗口；两组从各自2000步权重和Adam状态继续，追加20轮完整窗口/未来步遍历。',
        '指标对176个不同未来帧等权。所有阶段均固定报告，不按结果选择checkpoint。', '',
        '| 累计步数 | 受训组 | 全图MAE | 运动MAE | PSNR | 支撑IoU | 相对本组2000步运动误差下降 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for step in c['checkpoints']:
        arms = {}
        for mode in p['modes']:
            folder = ROOT / f'step_{step:05d}' / mode
            ev = read(folder / 'evaluation/EVALUATION.json'); rp = read(folder / 'evaluation/REPLAY.json')
            done = read(folder / 'TRAIN_DONE.json')
            if (ev['window_predictions'] != 1512 or ev['unique_future_frames'] != 176
                    or rp['status'] != 'passed' or rp['packets'] != 104 or rp['views'] != 208
                    or done['steps'] != step or done['checkpoint_replay'] != 'passed'):
                raise RuntimeError('Incomplete fixed-stage evidence')
            m = ev['metrics']['unique_frame_equal']; support = ev['support']['unique_frame_equal']
            old = baseline['arms'][mode]['metrics']['unique_frame_equal']['correct']['motion_mae']
            gain = 1-m['correct']['motion_mae']/old
            endpoints = [row for row in ev['rows'] if row['horizon']==9 and row['start'] in p['diagnostic_starts']]
            arms[mode] = dict(metrics=ev['metrics'], support=ev['support'], temporal_segments=ev['temporal_segments'],
                code_stats=ev['code_stats'], endpoints=endpoints, motion_error_reduction_vs_2000=gain,
                initial_input_error_increase=m['initial_only']['motion_mae']/m['correct']['motion_mae']-1,
                swapped_error_increase=m['swapped_window_codes']['motion_mae']/m['correct']['motion_mae']-1,
                train=done, replay={k:v for k,v in rp.items() if k!='rows'})
            v = m['correct']
            lines.append(f"| {step} | {mode} | {v['mae']:.6f} | {v['motion_mae']:.6f} | {v['psnr']:.3f} | {support['correct']['iou']:.4f} | {gain*100:.2f}% |")
        a = arms['posterior']; b = arms['initial_only']
        advantage = 1-a['metrics']['unique_frame_equal']['correct']['motion_mae']/b['metrics']['unique_frame_equal']['correct']['motion_mae']
        proxies = {str(row['start']):row['metrics']['correct']['object_proxy']['iou']
                   for row in a['endpoints'] if row['start'] in [35,123]}
        stages[str(step)] = dict(arms=arms, posterior_advantage_over_trained_initial=advantage,
            diagnostics=dict(posterior_motion_improvement_ge_10pct=a['motion_error_reduction_vs_2000']>=.10,
                posterior_advantage_ge_10pct=advantage>=.10,
                input_ablation_increase_ge_10pct=a['initial_input_error_increase']>=.10,
                endpoint_color_proxy_iou=proxies, both_endpoint_proxy_iou_ge_half=all(x>=.5 for x in proxies.values())))
    sources = [read(ROOT/mode/'TRAIN_DONE.json')['source'] for mode in p['modes']]
    if len({s['schedule_sha256'] for s in sources}) != 1: raise RuntimeError('Paired schedules differ')
    lines += ['', '颜色ROI代理仅作诊断，不等同物体姿态或动作成功；关键动作仍需逐帧目视复核。',
              '全部阶段均是已训练视频的拟合；没有未见动作测试、LFQ/FSQ或AR，也不自动启动下一轮。', '']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,stages=stages,
        training_frames=177,training_windows=168,unique_future_frames=176,
        final_step=c['final_step'],additional_steps_per_arm=c['additional_steps'],
        additional_rgb_visits_per_window_horizon=c['sweeps'],
        primary_checkpoint=c['final_step'],action_success='pending visual review',
        baseline_summary_sha256=c['baseline_summary_sha256'],code_sha256=sha(ROOT/'CODE.sha256'),
        scientific_scope='Fixed-budget TRAIN fit diagnosis, with balanced horizon interleaving; not an isolated step-count ablation'))


def child(mode, phase, c, p, step=None):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['gpu_assignment'][mode]),OMP_NUM_THREADS='6',
               MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    args = [sys.executable,'-B',str(ROOT/'run.py'),phase,'--mode',mode]
    if step is not None: args += ['--step',str(step)]
    label = phase if step is None else f'{phase}_{step}'
    timeout = c['worker_wall_max_seconds'] if phase=='worker' else p['replay_max_seconds']+30
    with (ROOT/f'{mode}_{label}.log').open('x') as log:
        job = subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(ROOT/f'{mode}_PROCESS.json',dict(pid=job.pid,mode=mode,phase=phase,step=step),replace=True)
        try: code = job.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            job.terminate()
            try: job.wait(timeout=15)
            except subprocess.TimeoutExpired: job.kill();job.wait()
            return dict(status='failed',mode=mode,phase=phase,step=step,error='deadline')
    return dict(status='complete' if code==0 else 'failed',mode=mode,phase=phase,step=step,exit_code=code)


def coordinate():
    began = time.monotonic()
    try:
        c, helper, binding, parent, p = setup()
        witness = read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_manifest_sha256']!=sha(ROOT/'CODE.sha256'):
            raise RuntimeError('CPU witness stale')
        available = int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
        cache = read(parent/'cache/MANIFEST.json')
        cache_bytes = sum(v['bytes'] for v in cache['windows'].values())+cache['frames']['bytes']
        if available<max(p['minimum_host_available_gib']*2**30,2*cache_bytes+8*2**30):raise RuntimeError('Host RAM unavailable')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:raise RuntimeError('Disk space unavailable')
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),parent_step=2000,final_step=c['final_step'],
              modes=p['modes'],code_sha256=sha(ROOT/'CODE.sha256'),schedule_sha256=witness['sampling']['sha256']))
        write(ROOT/'STATUS.json',dict(status='running',phase='paired_continuation',parent_step=2000,final_step=c['final_step']))
        def arm(mode):
            result = child(mode,'worker',c,p)
            if result['status']!='complete':return result
            for step in c['checkpoints']:
                result = child(mode,'replay',c,p,step)
                if result['status']!='complete':return result
                write(ROOT/f'step_{step:05d}'/mode/'STATUS.json',dict(status='complete',native_replay='passed'),replace=True)
            write(ROOT/mode/'STATUS.json',dict(status='complete',global_step=c['final_step'],native_replay='passed'),replace=True)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(arm,mode) for mode in p['modes']]
            results = [f.result() for f in futures]
        if any(r['status']!='complete' for r in results):raise RuntimeError('Branch failed: '+str(results))
        summarize(c,p)
        write(ROOT/'STATUS.json',dict(status='complete',final_step=c['final_step'],native_replay='passed',
              action_success='pending visual review',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure = dict(status='failed',error=repr(error),traceback=traceback.format_exc())
        write(ROOT/'FAILED.json',failure);write(ROOT/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['tests','worker','replay','coordinate'])
    parser.add_argument('--mode',choices=['posterior','initial_only'])
    parser.add_argument('--step',type=int)
    args=parser.parse_args()
    if args.phase in ['worker','replay'] and not args.mode:parser.error('--mode required')
    if args.phase=='replay' and args.step is None:parser.error('--step required')
    if args.phase=='tests':tests()
    elif args.phase=='worker':worker(args.mode)
    elif args.phase=='replay':replay(args.mode,args.step)
    else:coordinate()

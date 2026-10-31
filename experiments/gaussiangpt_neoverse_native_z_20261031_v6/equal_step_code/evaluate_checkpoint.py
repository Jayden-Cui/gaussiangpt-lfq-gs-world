"""Evaluate an existing, centrally selected common-step checkpoint once.

No training, optimizer update, checkpoint renaming, synthetic TRAIN_DONE, or
changes to main/postfit code. The selection is based on saved steps, never on
quality. GPU imports are delayed until immutable and terminal evidence passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True

ARMS = ('frozen', 'partial', 'constant')
STEPS = (50, 100, 150)
EXPECTED_TARGETS = {(start, h) for start in range(168) for h in range(1, 10)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def plan_prefix_sha256(plan, step):
    if type(step) is not int or step not in STEPS:
        raise ValueError('Only saved common steps 50/100/150 are admitted')
    starts = plan.get('starts')
    if (not isinstance(starts, list) or len(starts) < step
            or any(not isinstance(batch, list) or len(batch) != 4
                   or any(type(start) is not int or not 0 <= start < 168 for start in batch)
                   for batch in starts[:step])):
        raise ValueError('Invalid completed B4 plan prefix')
    prefix = starts[:step]
    if {start for batch in prefix for start in batch} != set(range(168)):
        raise ValueError('Selected prefix does not cover all168 complete windows')
    return hashlib.sha256(json.dumps(prefix, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_selection(selection, root, step, plans, train_done):
    """CPU-only structural and same-initialization/same-prefix proof."""
    if (selection.get('schema') != 'gs_native_equal_steps_selection_v1'
            or selection.get('status') != 'selected'
            or selection.get('selected_step') != step or step not in STEPS):
        raise ValueError('Central selection does not admit this requested step')
    if not isinstance(selection.get('arms'), dict) or set(selection['arms']) != set(ARMS):
        raise ValueError('Selection must name exactly the three original arms')
    initial_hashes, prefix_hashes, final_steps = [], [], []
    paths = {}
    for arm in ARMS:
        record, plan, train = selection['arms'][arm], plans[arm], train_done[arm]
        if not isinstance(record, dict):
            raise ValueError('Invalid selection arm record')
        final_step = train.get('steps')
        if (train.get('status') != 'complete' or type(final_step) is not int or final_step < step
                or train.get('full_window_coverage') is not True
                or train.get('target_frames_seen') != list(range(177))):
            raise ValueError(f'{arm}: original successful full-coverage training evidence is required')
        expected = (root / arm / f'step_{step:04d}.pt').resolve()
        declared = record.get('checkpoint_path')
        if not isinstance(declared, str) or not Path(declared).is_absolute() or Path(declared).resolve() != expected:
            raise ValueError(f'{arm}: selection must name the exact step checkpoint, never final.pt')
        prefix_hash = plan_prefix_sha256(plan, step)
        initial_hash = plan.get('initial_model_hash')
        if (not isinstance(initial_hash, str) or not initial_hash
                or record.get('initial_model_hash') != initial_hash
                or record.get('plan_prefix_sha256') != prefix_hash
                or record.get('completed_steps') != final_step
                or record.get('checkpoint_steps') != step):
            raise ValueError(f'{arm}: central initialization/step/prefix proof disagrees with original evidence')
        initial_hashes.append(initial_hash); prefix_hashes.append(prefix_hash); final_steps.append(final_step)
        paths[arm] = expected
    if (len(set(initial_hashes)) != 1 or selection.get('initial_model_hash') != initial_hashes[0]
            or len(set(prefix_hashes)) != 1 or selection.get('plan_prefix_sha256') != prefix_hashes[0]):
        raise ValueError('The three arms do not share initialization and the completed sampling prefix')
    if len(set(final_steps)) == 1:
        raise ValueError('Original final steps already match; controller must skip redundant reevaluation')
    return paths


def validate_checkpoint_metadata(checkpoint, config, plan, step):
    if (checkpoint.get('completed_steps') != step or checkpoint.get('config') != config
            or checkpoint.get('plan') != plan.get('starts')):
        raise ValueError('Selected checkpoint internal step/config/plan do not match the central selection')


def validate_targets(rows):
    keys = [(row['start'], row['horizon']) for row in rows]
    if (len(rows) != 1512 or set(keys) != EXPECTED_TARGETS
            or any(type(row['start']) is not int or type(row['horizon']) is not int
                   or row['end'] != row['start'] + row['horizon'] for row in rows)):
        raise ValueError('Evaluation must cover each of168 complete windows and9 real futures exactly once')


def verify_bound_manifest(root, name, expected_hash):
    path = root / name
    if sha(path) != expected_hash:
        raise ValueError('Selection manifest hash mismatch: ' + name)
    for relative, digest in read(path).items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or sha(source) != digest:
            raise ValueError('Immutable manifest entry mismatch: ' + str(relative))


def run(root, arm, step, output, max_seconds):
    begun = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    stage, completed_targets = 'verify_selection_and_original_terminal_evidence', 0

    def check_budget():
        if time.monotonic() - begun >= max_seconds:
            raise TimeoutError(f'Equal-step evaluation exceeded {max_seconds:g} seconds')

    try:
        pipeline_path = root / 'PIPELINE_STATUS.json'
        if read(pipeline_path).get('status') not in ('complete', 'failed'):
            raise RuntimeError('Main PIPELINE_STATUS is not terminal; no evaluation started')
        pipeline_hash = sha(pipeline_path)
        selection_path = output.parent / 'SELECTION.json'
        selection = read(selection_path)
        selection_hash = sha(selection_path)
        sys.path.insert(0, str(root / 'analysis_code'))
        from future_information_probe import verify_main_manifest, load_completed_evidence

        main_hash = verify_main_manifest(root)
        if (selection.get('main_manifest_sha256') != main_hash
                or selection.get('protocol_sha256') != sha(root / 'protocol.json')):
            raise ValueError('Selection does not bind the current immutable main recipe')
        verify_bound_manifest(root, 'POSTFIT_MANIFEST.json', selection.get('postfit_manifest_sha256'))
        verify_bound_manifest(root, 'EQUAL_STEP_MANIFEST.json', selection.get('equal_step_manifest_sha256'))
        # Reuse the existing validator; do not reconstruct or invent completed
        # original results. All three source arms must be complete successes.
        sources = {name: load_completed_evidence(root, name) for name in ARMS}
        plans = {name: source['plan'] for name, source in sources.items()}
        trains = {name: source['train'] for name, source in sources.items()}
        checkpoint_paths = validate_selection(selection, root, step, plans, trains)
        for name in ARMS:
            bound = selection['arms'][name]
            if (bound.get('train_done_sha256') != sources[name]['input_hashes']['TRAIN_DONE.json']
                    or bound.get('plan_sha256') != sources[name]['input_hashes']['PLAN.json']
                    or sha(checkpoint_paths[name]) != bound.get('checkpoint_sha256')):
                raise ValueError(f'{name}: selected checkpoint/source proof hash mismatch')
        available = [candidate for candidate in STEPS
                     if all(trains[name]['steps'] >= candidate
                            and (root/name/f'step_{candidate:04d}.pt').is_file() for name in ARMS)]
        if not available or max(available) != step:
            raise ValueError('Central selection is not the largest common existing saved step')
        check_budget()
        # No GPU import occurs until all original successes and central proof
        # have been checked. build creates an optimizer; it is discarded unused.
        sys.path.insert(0, str(root / 'code'))
        import torch
        import native_experiment as core

        cfg = core.setup(root)
        torch.cuda.reset_peak_memory_stats()
        stage = 'load_selected_checkpoint'
        data = core.Data()
        model, bridge, optimizer = core.build(cfg, data, arm)
        del optimizer
        checkpoint_path = checkpoint_paths[arm]
        checkpoint_hash = selection['arms'][arm]['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        validate_checkpoint_metadata(checkpoint, cfg, plans[arm], step)
        model.load_state_dict(checkpoint['model'], strict=True)
        bridge.codec.load_state_dict(checkpoint['codec'], strict=True)
        del checkpoint
        model.eval().requires_grad_(False); bridge.eval().requires_grad_(False)
        expected_mode = 'constant' if arm == 'constant' else 'posterior'
        if model.code_mode != expected_mode or bridge.codec.ae.decoder.training:
            raise RuntimeError('Selected model mode or native decoder eval contract changed')
        if (bridge.codec.protocol['max_decoder_candidates'] != cfg['resources']['max_decoder_candidates']
                or bridge.codec.protocol['max_decoded_gaussians'] != cfg['resources']['max_decoded_gaussians']
                or len(bridge.codec._guards) != len(bridge.codec.ae.decoder.ups) + 1):
            raise RuntimeError('Native GS guards differ from the immutable experiment')
        if (cfg['data']['frames'] != 177 or cfg['data']['resolution_hw'] != [252,448]
                or cfg['data']['frame_gap'] != 1 or len(data.states) != 177 or len(data.records) != 177):
            raise RuntimeError('Native177-frame/448x252 data contract changed')
        controls = ('correct','shuffle42','shuffle84','shuffle126','adapted_held','held_initial','frozen_target')
        rows, initial_rows = [], []
        stage = 'encode_selected_ids'
        with torch.no_grad():
            chunks = []
            for start in range(0,168,4):
                check_budget()
                chunks.append(model.encode_windows([data.window(s) for s in range(start,start+4)])['indices'])
            ids = torch.cat(chunks,dim=0)
            if ids.shape != (168,9) or not bool(((ids >= 0) & (ids < 65536)).all()):
                raise RuntimeError('Selected LFQ IDs are invalid')
            with (output/'actions.pt').open('xb') as stream:
                torch.save(dict(schema='gs_native_equal_step_actions_v1',
                                evaluation_kind='common_checkpoint_evaluation',
                                ids=ids.cpu(),starts=list(range(168)),horizons=list(range(1,10)),
                                selected_step=step,arm=arm,checkpoint_path=str(checkpoint_path),
                                checkpoint_sha256=checkpoint_hash,selection_sha256=selection_hash,
                                source_frames=177,frame_gap=1,all_windows_complete=True,
                                code_mode=model.code_mode,
                                constant_id=model.constant_id if model.code_mode=='constant' else None,
                                cache_manifest_sha256=bridge.binding['cache_manifest_sha256'],
                                state_source_sha256=bridge.binding['state_checkpoint_sha256'],
                                semantics='Common-initial H9 bundle at selected existing checkpoint; '
                                          'posterior or explicitly marked constant control; not final-model '
                                          'export, flattened physical actions, or complete AR samples'),stream)
            stage = 'evaluate_complete_windows'
            with (output/'targets.jsonl').open('x',encoding='utf-8') as log, \
                 (output/'progress.jsonl').open('x',encoding='utf-8') as progress:
                for start in range(168):
                    check_budget()
                    condition = [core.make_condition(data.states[start])]
                    predictions = {'correct':model.decode_ids(ids[start:start+1],condition)[0]}
                    for offset in (42,84,126):
                        source=(start+offset)%168
                        predictions[f'shuffle{offset}']=model.decode_ids(ids[source:source+1],condition)[0]
                    for sequence in predictions.values():
                        if len(sequence) != 10:
                            raise RuntimeError('Missing full10 outputs')
                        if any(not torch.equal(p['coordinates'],data.states[start]['coordinates'])
                               or p['features'].shape != data.states[start]['features'].shape for p in sequence):
                            raise RuntimeError('Selected decoder changed/dropped initial support')
                    image = bridge.render(predictions['correct'][0],data.records[start])
                    initial = dict(start=start,end=start,horizon=0,correct=data.metrics(image,start,0))
                    image = bridge.render(data.states[start],data.records[start])
                    initial['adapted_held']=data.metrics(image,start,0)
                    initial_rows.append(initial)
                    for h in range(1,10):
                        row=dict(start=start,end=start+h,horizon=h,motion_ids=ids[start].cpu().tolist(),
                                 **data.reference[(start,h)])
                        for control,sequence in predictions.items():
                            image=bridge.render(sequence[h],data.records[start+h])
                            row[control]=data.metrics(image,start,h)
                        image=bridge.render(data.states[start],data.records[start+h])
                        row['adapted_held']=data.metrics(image,start,h)
                        for control in controls:
                            if any(type(value) not in (int,float) or not math.isfinite(value)
                                   for value in row[control].values()):
                                raise RuntimeError('Nonfinite selected evaluation metric')
                        rows.append(row);completed_targets+=1
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                        if torch.cuda.max_memory_allocated()/2**30 > cfg['resources']['peak_limit_gib']:
                            raise RuntimeError('GPU allocation guard exceeded; no point/resolution reduction')
                    del image,predictions
                    log.flush()
                    progress.write(json.dumps(dict(windows=start+1,real_future_exposures=completed_targets,
                                                    elapsed_seconds=time.monotonic()-begun))+'\n')
                    progress.flush()
                    if start%21==0:
                        print('EQUAL_STEP_EVAL',arm,step,start+1,'/168',flush=True)
        stage='validate_and_write_evaluation'
        check_budget();validate_targets(rows)
        if len(initial_rows)!=168 or {r['start'] for r in initial_rows}!=set(range(168)):
            raise RuntimeError('Incomplete h0 reconstruction coverage')
        write_new(output/'initial_reconstruction.json',dict(rows=initial_rows,unique_initial_frames=168,
                  metrics={control:core.mean_rows(initial_rows,control,['psnr','hand_activity_psnr','mae','hand_activity_mae'])
                           for control in ('correct','adapted_held')}))
        metrics={control:core.aggregate(rows,control) for control in controls}
        entropy,distinct=[],[]
        for h in range(9):
            _,counts=ids[:,h].unique(return_counts=True);p=counts.float()/168
            entropy.append(float(-(p*p.log()).sum()));distinct.append(len(counts))
        # Source identities must survive the complete evaluation unchanged.
        if sha(selection_path)!=selection_hash or sha(pipeline_path)!=pipeline_hash:
            raise RuntimeError('Selection or original terminal record changed during evaluation')
        if verify_main_manifest(root)!=main_hash:
            raise RuntimeError('Main source changed during evaluation')
        verify_bound_manifest(root,'POSTFIT_MANIFEST.json',selection['postfit_manifest_sha256'])
        verify_bound_manifest(root,'EQUAL_STEP_MANIFEST.json',selection['equal_step_manifest_sha256'])
        for name,source in sources.items():
            for basename,digest in source['input_hashes'].items():
                if sha(source['directory']/basename)!=digest:
                    raise RuntimeError('Original completed artifact changed: '+name+'/'+basename)
            if sha(checkpoint_paths[name])!=selection['arms'][name]['checkpoint_sha256']:
                raise RuntimeError('Selected checkpoint changed: '+name)
        evaluation=dict(schema='gs_native_equal_step_evaluation_v1',status='complete',
                        evaluation_kind='common_checkpoint_evaluation',
                        scope='single_clip_TRAIN_common_checkpoint_reconstruction',arm=arm,selected_step=step,
                        original_final_steps=trains[arm]['steps'],checkpoint_path=str(checkpoint_path),
                        checkpoint_sha256=checkpoint_hash,selection_sha256=selection_hash,
                        main_manifest_sha256=main_hash,protocol_sha256=selection['protocol_sha256'],
                        postfit_manifest_sha256=selection['postfit_manifest_sha256'],
                        equal_step_manifest_sha256=selection['equal_step_manifest_sha256'],
                        initial_model_hash=selection['initial_model_hash'],plan_prefix_sha256=selection['plan_prefix_sha256'],
                        source_evidence_sha256={name:source['input_hashes'] for name,source in sources.items()},
                        main_terminal_sha256=pipeline_hash,source_binding=bridge.binding,
                        targets_sha256=sha(output/'targets.jsonl'),
                        initial_reconstruction_sha256=sha(output/'initial_reconstruction.json'),
                        actions_sha256=sha(output/'actions.pt'),frozen_reference_sha256=data.reference_hash,
                        windows=168,real_future_exposures=len(rows),unique_future_frames=176,metrics=metrics,
                        initial_reconstruction='initial_reconstruction.json',physical_frames_with_some_reconstruction=177,
                        conditional_entropy_nats=sum(entropy)/9,per_h_entropy_nats=entropy,
                        per_h_distinct_ids=distinct,distinct_bundles=len(torch.unique(ids,dim=0)),
                        optimizer_updates=0,goal_complete=False,
                        elapsed_seconds=time.monotonic()-begun,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                        max_seconds=max_seconds,
                        limitations=['Existing common-step checkpoint evaluation only; original final results remain unchanged',
                                     'Single TRAIN clip, no generalization or AR semantic claim',
                                     'Budget checked at complete-window boundaries; external2500s process bound is controller-owned'])
        write_new(output/'EVALUATION.json',evaluation)
        print(json.dumps(dict(status='complete',arm=arm,selected_step=step,optimizer_updates=0,
                              evaluation=str(output/'EVALUATION.json'))),flush=True)
    except BaseException as error:
        write_new(output/'FAILED.json',dict(status='failed',arm=arm,selected_step=step,
                  evaluation_kind='common_checkpoint_evaluation',stage=stage,error=repr(error),
                  traceback=traceback.format_exc(),real_future_exposures=completed_targets,
                  elapsed_seconds=time.monotonic()-begun,optimizer_updates=0,goal_complete=False))
        raise


def cpu_self_test():
    # Only validate input contracts and enumeration, never fabricate result files.
    root=Path.cwd().resolve()/'_equal_step_contract_fixture'
    batches=[list(range(start,start+4)) for start in range(0,168,4)]*4
    plans={name:dict(starts=batches,initial_model_hash='same_initial') for name in ARMS}
    trains={name:dict(status='complete',steps=steps,full_window_coverage=True,target_frames_seen=list(range(177)))
            for name,steps in zip(ARMS,(150,112,125))}
    prefix=plan_prefix_sha256(plans['frozen'],100)
    selection=dict(schema='gs_native_equal_steps_selection_v1',status='selected',selected_step=100,
                   initial_model_hash='same_initial',plan_prefix_sha256=prefix,
                   arms={name:dict(checkpoint_path=str(root/name/'step_0100.pt'),checkpoint_steps=100,
                                   completed_steps=trains[name]['steps'],initial_model_hash='same_initial',
                                   plan_prefix_sha256=prefix) for name in ARMS})
    assert len(validate_selection(selection,root,100,plans,trains))==3
    wrong=json.loads(json.dumps(selection));wrong['arms']['partial']['checkpoint_path']=str(root/'partial/final.pt')
    try:validate_selection(wrong,root,100,plans,trains)
    except ValueError:pass
    else:raise AssertionError('final.pt masquerading as common step accepted')
    wrong=json.loads(json.dumps(plans));wrong['constant']['starts'][0]=[4,5,6,7]
    try:validate_selection(selection,root,100,wrong,trains)
    except ValueError:pass
    else:raise AssertionError('Different plan prefix accepted')
    rows=[dict(start=start,horizon=h,end=start+h) for start,h in sorted(EXPECTED_TARGETS)]
    validate_targets(rows)
    for bad in (rows[:-1],rows[:-1]+[rows[0]]):
        try:validate_targets(bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete/duplicate targets accepted')
    validate_checkpoint_metadata(dict(completed_steps=100,config={'x':1},plan=batches),{'x':1},plans['partial'],100)
    try:validate_checkpoint_metadata(dict(completed_steps=112,config={'x':1},plan=batches),{'x':1},plans['partial'],100)
    except ValueError:pass
    else:raise AssertionError('Final-step metadata accepted for a different selected step')
    assert 'torch' not in sys.modules
    print('CPU_SELF_TEST_PASSED: selected-step identity, shared init/plan, exact1512 coverage, '
          'checkpoint metadata; no torch/GPU/files')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--step',type=int,choices=STEPS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=float,default=2400.)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:cpu_self_test();return
    if args.root is None or args.arm is None or args.step is None or args.output is None:
        parser.error('--root, --arm, --step and --output are required')
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=2400:
        parser.error('--max-seconds must be positive and no greater than2400')
    root,output=args.root.resolve(),args.output.resolve()
    expected=(root/'diagnostics/equal_steps_v1'/args.arm).resolve()
    if not root.is_dir() or output!=expected:
        parser.error('Output must be ROOT/diagnostics/equal_steps_v1/<arm>')
    if output.exists():parser.error('Output exists; no overwrite, resume, or retry is allowed')
    run(root,args.arm,args.step,output,args.max_seconds)


if __name__=='__main__':main()

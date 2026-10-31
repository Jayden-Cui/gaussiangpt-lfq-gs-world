"""One read-only CPU revalidation of a completed experiment's failed queue.

Never rewrites the old queue, summary, manifest, checkpoints, or evaluation.
Only this new acceptance directory may be written. This is not a GPU replay,
new training, action-Z acceptance, or a replacement for the old FAILED record.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
BASE = Path('/data/CJH/AGS-V6Direct/outputs')
SOURCE = BASE / 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
ROOT = BASE / 'gaussiangpt_neoverse_multisource_acceptance_20261031_v1'
PINS = {
    'CODE_MANIFEST.json': 'b4958856fb50af2c55f4c2fce9eaa2a16c14ca9ec5f87436565a499052efe48a',
    'SUMMARY.json': 'affdd286d2954183e8a0d36a57883894f0165757380ce7c85774513634bd741c',
    'CPU_VALIDATION.json': '7592a3ae70cb2b17ae1415bbbd4c9e597905df7fa617e23b53898a6b7fc16301',
    'queue/FAILED.json': '9192f0e98ed69b16d01f8e403fb29f35efae6a8ed47e27c343185b621817a1d9',
}
EXPECTED_ERROR = 'Manifest omits queue sources/tests/protocol/docs or the exact process helper'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, content):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def readonly_source_audit(event, args):
    """Reject Python filesystem mutations outside the new acceptance root."""
    paths = []
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and not isinstance(path, int):
            paths = [path]
    elif event in ('os.remove', 'os.unlink', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.replace', 'os.link', 'os.symlink'):
        paths = args[:2]
    for path in paths:
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        require(isinstance(path, (str, os.PathLike)) and Path(path).resolve().is_relative_to(ROOT),
                f'Read-only recovery rejected filesystem mutation: {event} {path!r}')


def original_process():
    proc = Path('/proc/439814')
    if not proc.exists():
        return dict(pid=439814, expected_start_ticks='38835617', pid_exists=False,
                    original_identity_live=False)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    same = fields[19] == '38835617'
    require(not same, 'Original queue identity unexpectedly still exists')
    return dict(pid=439814, expected_start_ticks='38835617', pid_exists=True,
                observed_start_ticks=fields[19], original_identity_live=False,
                note='PID reused by a different process; no process was signalled')


def load_pinned(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(script_sha):
    require(Path(__file__).resolve().parent == ROOT, 'Require the dedicated remote acceptance root')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Recovery must hide all CUDA devices')
    require(sha(__file__) == script_sha, 'Acceptance script differs from the deployed pin')
    original_before = original_process()
    for relative, expected in PINS.items():
        require(sha(SOURCE / relative) == expected, f'Pinned evidence changed: {relative}')
    manifest = read(SOURCE / 'CODE_MANIFEST.json')
    for relative, expected in manifest.items():
        path = (SOURCE / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(BASE),
                f'Unsafe manifest member: {relative}')
        require(sha(path) == expected, f'Original source/helper changed: {relative}')
    sys.path.insert(0, str(SOURCE))
    pipeline = load_pinned('_multisource_recovery_pipeline', SOURCE / 'pipeline.py')
    compare = load_pinned('_multisource_recovery_compare', SOURCE / 'compare.py')
    require('torch' not in sys.modules and 'MinkowskiEngine' not in sys.modules,
            'Recovery unexpectedly imported GPU/model runtime modules')
    required = pipeline.REQUIRED | {p.name for p in SOURCE.glob('*.py')} | {p.name for p in SOURCE.glob('*.md')}
    missing = sorted(required - set(manifest))
    require(missing == ['SUMMARY.md'], f'Unexpected unregistered source/docs: {missing}')
    require({p.name for p in SOURCE.glob('*.py')} <= set(manifest), 'Additional unregistered Python source')
    require(all(manifest.get(k) == v for k, v in pipeline.EXTERNAL_HELPERS.items()),
            'Original helper pins differ')
    require(not (SOURCE / 'queue/DONE.json').exists(), 'Original queue unexpectedly has DONE')
    failure = read(SOURCE / 'queue/FAILED.json')
    status = read(SOURCE / 'queue/STATUS.json')
    require(failure.get('status') == status.get('status') == 'failed', 'Original failure status changed')
    require(failure.get('error') == repr(ValueError(EXPECTED_ERROR)), 'Unexpected original failure')
    expected_trace = (
        'Traceback (most recent call last):\n'
        f'  File "{SOURCE / "pipeline.py"}", line 501, in main\n'
        "    execute_stage(root, output, 'compare', helper, outcomes, before_stage)\n"
        f'  File "{SOURCE / "pipeline.py"}", line 434, in execute_stage\n'
        '    before_stage()\n'
        f'  File "{SOURCE / "pipeline.py"}", line 478, in before_stage\n'
        '    verify(root, admission)\n'
        f'  File "{SOURCE / "pipeline.py"}", line 95, in verify\n'
        f"    raise ValueError('{EXPECTED_ERROR}')\n"
        f'ValueError: {EXPECTED_ERROR}\n')
    require(failure.get('traceback') == expected_trace, 'Failure was not the pinned post-comparison verify path')
    try:
        pipeline.verify(SOURCE)
    except ValueError as error:
        require(type(error) is ValueError and str(error) == EXPECTED_ERROR,
                'Current pipeline.verify failed for another reason')
        expected_current_exception = dict(type=type(error).__name__, message=str(error),
                                          traceback=traceback.format_exc())
    else:
        raise ValueError('Current pipeline.verify unexpectedly succeeded')
    cfg = read(SOURCE / 'protocol.json')
    pipeline.protocol_contract(cfg)
    admission = dict(code_manifest_sha256=sha(SOURCE / 'CODE_MANIFEST.json'),
                     protocol_sha256=sha(SOURCE / 'protocol.json'),
                     cpu_verification_sha256=sha(SOURCE / 'CPU_VALIDATION.json'))
    proof = read(SOURCE / 'CPU_VALIDATION.json')
    require(proof.get('status') == 'passed' and proof.get('gpu_executed') is False
            and all(proof.get(k) == admission[k] for k in ('code_manifest_sha256', 'protocol_sha256')),
            'Original CPU proof is not valid for the pinned source')
    require(failure.get('admission') == admission, 'Failed queue admission differs')
    stages = ('source_admission', 'preflight', 'baseline', 'train', 'eval', 'compare')
    expected_jobs = {job[0]: job for stage in stages for job in pipeline.jobs(SOURCE, stage)}
    outcomes = failure.get('outcomes', [])
    require(len(outcomes) == len(expected_jobs) == 8, 'Must retain all eight child outcomes')
    require({row.get('name') for row in outcomes} == set(expected_jobs), 'Duplicate or missing child outcome')
    stage_hashes = {}
    for row in outcomes:
        name, gpu, command, seconds = expected_jobs[row['name']]
        require(row.get('gpu') == gpu and row.get('command') == command
                and row.get('process_seconds') == seconds and row.get('status') == 'complete'
                and type(row.get('exit_code')) is int and row['exit_code'] == 0
                and row.get('timed_out') is False
                and type(row.get('elapsed_seconds')) in (float, int)
                and math.isfinite(row['elapsed_seconds']) and 0 <= row['elapsed_seconds'] < seconds,
                f'Invalid original stage outcome: {name}')
        path = SOURCE / 'queue' / f'{name}_PROCESS.json'
        require(read(path) == row, f'Stage receipt differs from queue outcome: {name}')
        stage_hashes[str(path.relative_to(SOURCE))] = sha(path)
    names = [row['name'] for row in outcomes]
    require(names[:3] == ['source_admission', 'preflight', 'baseline'] and names[-1] == 'compare'
            and set(names[3:5]) == {'train_decoder_only', 'train_full_codec'}
            and set(names[5:7]) == {'eval_decoder_only', 'eval_full_codec'}, 'Original stage order differs')
    source_hashes = pipeline.source_contract(SOURCE, admission)
    require(failure.get('source_hashes') == source_hashes, 'Failed queue source binding differs')
    evidence = pipeline.expected_evidence(SOURCE, source_hashes['data_binding_sha256'])
    stage_hashes['PREFLIGHT.json'] = pipeline.preflight_contract(SOURCE, evidence)
    for arm in pipeline.ARMS:
        stage_hashes[f'arms/{arm}/TRAIN_DONE.json'] = pipeline.train_contract(SOURCE, arm, evidence)
    for label in ('baseline', *pipeline.ARMS):
        stage_hashes[f'evaluations/{label}/EVALUATION.json'] = pipeline.evaluation_contract(SOURCE, label, evidence)
    stored = pipeline.comparison_contract(SOURCE)
    require((SOURCE / 'SUMMARY.md').read_bytes() == compare.markdown(stored).encode('utf-8'),
            'Unregistered Markdown is not the exact pinned comparison output')
    # build_summary, never main: complete CPU rehash/statistics without old-root writes.
    rebuilt = compare.build_summary(SOURCE)
    require(rebuilt == stored, 'Full CPU-rebuilt summary differs semantically from stored SUMMARY')
    require(rebuilt['comparison_eligibility']['eligible'] is True
            and rebuilt['engineering_status'] == 'complete' and rebuilt['goal_complete'] is False,
            'Rebuilt evidence is incomplete or exceeds codec-only scope')
    counts = {}
    for label in ('baseline', *pipeline.ARMS):
        folder = SOURCE / 'evaluations' / label
        counts[label] = {name: len(compare.rows(folder / name))
                         for name in ('targets.jsonl', 'state_manifest.jsonl', 'screenshot_manifest.jsonl')}
        require(counts[label] == {'targets.jsonl': 531, 'state_manifest.jsonl': 531,
                                 'screenshot_manifest.jsonl': 75}, f'Incomplete artifacts: {label}')
        for name in counts[label]:
            stage_hashes[str((folder / name).relative_to(SOURCE))] = sha(folder / name)
    for relative in ('SUMMARY.md', 'queue/STATUS.json', 'queue/SOURCE_ADMISSION.json', 'DATA_BINDING.json'):
        stage_hashes[relative] = sha(SOURCE / relative)
    for relative, expected in {**manifest, **PINS, **stage_hashes}.items():
        require(sha(SOURCE / relative) == expected, f'Evidence changed during revalidation: {relative}')
    require('torch' not in sys.modules and 'MinkowskiEngine' not in sys.modules, 'Unexpected model runtime import')
    return dict(status='accepted', results_revalidated=True,
                original_queue_status='failed', original_failure_preserved=True,
                original_done_created=False, original_failure_sha256=PINS['queue/FAILED.json'],
                original_summary_sha256=PINS['SUMMARY.json'], original_source=str(SOURCE),
                script_sha256=script_sha, original_manifest_sha256=PINS['CODE_MANIFEST.json'],
                original_manifest_entries=manifest, original_evidence_pins=PINS,
                original_process_before=original_before, original_process_after=original_process(),
                expected_current_verify_exception=expected_current_exception,
                precise_failure_cause='Pinned verify dynamically treats generated SUMMARY.md as missing immutable source',
                dynamic_missing_entries=missing, unregistered_python_sources=[],
                summary_markdown_exact_pinned_renderer_match=True, rebuilt_summary_semantically_equal=True,
                stage_outcomes=outcomes, stage_artifact_sha256=stage_hashes, source_hashes=source_hashes,
                complete_stage_count=8, completed_updates={'decoder_only': 300, 'full_codec': 300},
                artifact_counts=counts, verified_source_frames=531, verified_saved_state_packets=1593,
                verified_screenshots=225, gpu_executed=False, training_performed=False,
                model_checkpoints_deserialized=False, independent_gpu_replay_performed=False,
                visual_review_performed=False, goal_complete=False, production_codec_selected=None,
                verification_scope='Original immutable sources and exact failure cause; all CPU source/checkpoint/trace/state/screenshot hashes, worker replay receipts, and deterministic paired statistics. No new GPU replay or action-Z claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script-sha256', required=True)
    args = parser.parse_args()
    require(Path(__file__).resolve().parent == ROOT, 'Must execute in the dedicated remote acceptance directory')
    require(not (ROOT / 'ACCEPTANCE.json').exists(), 'Acceptance already exists; do not rerun')
    write_new(ROOT / 'ATTEMPT.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
              script_sha256=args.script_sha256, automatic_retry=False, gpu_executed=False))
    sys.addaudithook(readonly_source_audit)
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError('Read-only recovery exceeded 300 seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(300)
    try:
        result = run(args.script_sha256)
    except Exception as error:
        result = dict(status='rejected', results_revalidated=False, error=repr(error),
                      traceback=traceback.format_exc(), original_queue_status='failed',
                      script_sha256=sha(__file__), gpu_executed=False, training_performed=False,
                      goal_complete=False, automatic_retry=False)
    finally:
        signal.alarm(0)
    result.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
    write_new(ROOT / 'ACCEPTANCE.json', result)
    print(json.dumps({key: result.get(key) for key in
          ('status', 'results_revalidated', 'original_queue_status', 'complete_stage_count',
           'verified_saved_state_packets', 'verified_screenshots', 'gpu_executed',
           'training_performed', 'elapsed_seconds', 'error')}, ensure_ascii=False))
    print(json.dumps({'acceptance_sha256': sha(ROOT / 'ACCEPTANCE.json')}))
    return 0 if result['results_revalidated'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

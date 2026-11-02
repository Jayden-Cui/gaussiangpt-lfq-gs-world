"""V4 evaluator with a bound raw-state causality check for C0."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from guard import audit_raw_c0

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
EVAL = OUTPUTS / 'gaussiangpt_neoverse_stage_bc_evaluation_prepare_20261102_v1'
POSTERIOR = OUTPUTS / 'gaussiangpt_neoverse_qformer_full177_20261101_v1' / 'posterior.py'
POSTERIOR_SHA256 = '3f905a252e82c76b9548521dea74b28aeea38a257be895bc667ac804f5d4bb15'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources():
    witness = json.loads((ROOT / 'CPU_TESTS.json').read_text(encoding='utf-8'))
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('C0 raw-causality test witness differs')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('C0 causality source changed: ' + name)
    previous = json.loads((EVAL / 'DRIVER_TEST_V4.json').read_text(encoding='utf-8'))
    if (previous.get('status') != 'passed' or previous.get('driver_sha256') != sha(EVAL / 'c_evaluate_v4.py')
            or previous.get('test_sha256') != sha(EVAL / 'driver_tests_v4.py')
            or previous.get('core_manifest_sha256') != sha(EVAL / 'CODE.sha256')):
        raise RuntimeError('Stage-C v4 evaluator witness differs')
    return witness


def get_make_memories():
    module = importlib.import_module('posterior')
    if Path(module.__file__).resolve() != POSTERIOR.resolve() or sha(POSTERIOR) != POSTERIOR_SHA256:
        raise RuntimeError('Unexpected original cached-prefix implementation')
    return module.make_memories


def evaluate_with_guard(evaluator, run_root, step, receipt, *, source_metadata):
    """Wrap one evaluation without changing the existing evaluator's files."""
    saved_load, saved_check = evaluator.load_trained, evaluator.check_prefix_causality
    captured = {}

    def load(*args, **kwargs):
        result = saved_load(*args, **kwargs)
        if len(result) != 6:
            raise RuntimeError('Unexpected v4 model loader result')
        captured['decoder'], captured['runtime'] = result[3], result[0]['runtime']
        return result

    def check(model, store):
        result = saved_check(model, store)
        if result.get('status') != 'passed':
            raise RuntimeError('Existing prefix check did not pass')
        if model.variant == 'C0':
            if 'decoder' not in captured:
                raise RuntimeError('Evaluated teacher decoder was not captured')
            decoder, runtime = captured['decoder'], captured['runtime']
            before_decoder, before_model = runtime.digest(decoder), runtime.digest(model)
            raw = audit_raw_c0(model, decoder.base, store, get_make_memories())
            if runtime.digest(decoder) != before_decoder or runtime.digest(model) != before_model:
                raise RuntimeError('Causality check changed the fixed models')
            result = dict(result, raw_state_causality=raw, wrapper=source_metadata)
        return result

    evaluator.load_trained, evaluator.check_prefix_causality = load, check
    try:
        return evaluator.evaluate(run_root, step, receipt)
    finally:
        evaluator.load_trained, evaluator.check_prefix_causality = saved_load, saved_check


def main(phase, run_root, step, receipt=None, mode=None):
    verify_sources()
    sys.path.insert(0, str(EVAL))
    import c_evaluate_v4 as previous
    if phase == 'replay':
        if mode not in ('C0', 'C1', 'C2'):
            raise ValueError('Replay requires a valid variant')
        return previous.replay(run_root, step, mode)
    if phase != 'evaluate' or receipt is None:
        raise ValueError('Evaluation requires a teacher receipt')
    metadata = dict(policy='c0_raw_state_prefix_check_v1', driver_sha256=sha(ROOT / 'driver_v5.py'),
                    guard_sha256=sha(ROOT / 'guard.py'), code_sha256=sha(ROOT / 'CODE.sha256'),
                    cpu_witness_sha256=sha(ROOT / 'CPU_TESTS.json'),
                    original_prefix_source_sha256=POSTERIOR_SHA256)
    return evaluate_with_guard(previous, run_root, step, receipt, source_metadata=metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('evaluate', 'replay'))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--receipt')
    parser.add_argument('--mode', choices=('C0', 'C1', 'C2'))
    args = parser.parse_args()
    main(args.phase, args.run_root, args.step, args.receipt, args.mode)

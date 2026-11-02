"""Bound stage-C v2 driver wrapper with matched optional time/change initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from initializer import POLICY, build_matched

ROOT = Path(__file__).resolve().parent
C_ROOT = ROOT.parent / 'gaussiangpt_neoverse_stage_c_training_prepare_20261102_v1'
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_sources():
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Matched-initialization CPU witness differs')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Matched-initialization source changed: ' + name)
    previous = read(C_ROOT / 'TRAIN_DRIVER_TEST_V2.json')
    if (previous.get('status') != 'passed' or previous.get('driver_sha256') != sha(C_ROOT / 'c_train_driver_v2.py')
            or previous.get('test_sha256') != sha(C_ROOT / 'train_driver_tests_v2.py')
            or previous.get('core_manifest_sha256') != sha(C_ROOT / 'CODE.sha256')):
        raise RuntimeError('Underlying stage-C v2 driver witness differs')
    return witness


def add_binding_metadata(value, metadata):
    if value.get('model_initial_digest') != metadata.get('model_initial_digest'):
        raise RuntimeError('Stage-C model digest changed before binding')
    if value.get('variant') != metadata.get('variant'):
        raise RuntimeError('Stage-C variant changed before binding')
    result = dict(value)
    result['initializer_wrapper'] = dict(metadata)
    return result


def main(config_path, run_root):
    witness = verify_sources()
    config_path, run_root = Path(config_path), Path(run_root)
    config = read(config_path)
    if config.get('initializer_policy') != POLICY:
        raise RuntimeError('Config must explicitly bind the matched initialization policy')
    if not run_root.is_absolute() or run_root.exists():
        raise RuntimeError('A new absolute run directory is required')
    parent = run_root.parent.resolve(strict=True)
    if parent != OUTPUTS.resolve() and OUTPUTS.resolve() not in parent.parents:
        raise RuntimeError('Run directory must remain within the server outputs workspace')
    sys.path.insert(0, str(C_ROOT))
    import c_train_driver_v2 as legacy
    previous_builder, previous_write = legacy.build_variant, legacy.write
    captured = {}

    def build(module, decoder, posterior, runtime, candidate):
        model = build_matched(previous_builder, module, decoder, posterior, runtime, candidate)
        captured.update(policy=POLICY, variant=candidate['variant'],
            include_time=candidate['include_time'], include_change=candidate['include_change'],
            wrapper_sha256=sha(ROOT / 'driver_v3.py'), initializer_sha256=sha(ROOT / 'initializer.py'),
            wrapper_manifest_sha256=sha(ROOT / 'CODE.sha256'), cpu_witness_sha256=sha(ROOT / 'CPU_TESTS.json'),
            model_initial_digest=runtime.digest(model),
            time_projection_initial_digest=runtime.digest(model.time_projection) if model.include_time else None,
            delta_projection_initial_digest=runtime.digest(model.delta_projection) if model.include_change else None)
        return model

    def write(path, value):
        path = Path(path)
        if path.name == 'BINDING.json' and path.parent.resolve() == run_root.resolve():
            if not captured:
                raise RuntimeError('Initializer metadata was not captured')
            value = add_binding_metadata(value, captured)
        return previous_write(path, value)

    legacy.build_variant, legacy.write = build, write
    try:
        legacy.main(config_path, run_root)
    finally:
        legacy.build_variant, legacy.write = previous_builder, previous_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    main(args.config, args.run_root)

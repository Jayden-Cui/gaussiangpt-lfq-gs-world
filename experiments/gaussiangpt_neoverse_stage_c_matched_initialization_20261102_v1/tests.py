"""Real-model CPU tests for matched initialization and wrapper provenance."""
import json
from pathlib import Path
import sys
import tempfile

import torch
import driver_v3 as driver
from initializer import POLICY, build_matched

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
VARIANTS = OUTPUTS / 'gaussiangpt_neoverse_posterior_variants_20261102_v2'


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or driver.sha(ROOT / name) != digest:
            raise RuntimeError('Matched-initialization source changed: ' + name)
    sys.path.insert(0, str(driver.C_ROOT))
    sys.path.insert(0, str(B_ROOT))
    import c_train_driver_v2 as legacy
    from b_core import load_bound_sources
    bundle = load_bound_sources('cpu')
    binding, runtime = bundle['binding'], bundle['runtime']
    decoder, posterior = bundle['decoder'], bundle['posterior']
    module = binding.explicit('_matched_initialization_test_variants', VARIANTS / 'posterior_variants.py')
    common = dict(shared_posterior_digest=runtime.digest(posterior), model_seed=20261102,
                  initializer_policy=POLICY)
    def config(variant, time, change):
        return dict(common, variant=variant, include_time=time, include_change=change)
    time_config, full_config = config('C2', True, False), config('C2', True, True)
    time_only = legacy.build_variant(module, decoder, posterior, runtime, time_config)
    legacy_full = legacy.build_variant(module, decoder, posterior, runtime, full_config)
    legacy_rng = torch.random.get_rng_state().clone()
    matched_full = build_matched(legacy.build_variant, module, decoder, posterior, runtime, full_config)
    matched_rng = torch.random.get_rng_state().clone()
    old_difference = time_only.time_projection.weight - legacy_full.time_projection.weight
    if torch.equal(time_only.time_projection.weight, legacy_full.time_projection.weight):
        raise RuntimeError('Expected old initialization difference was not reproduced')
    if not torch.equal(time_only.time_projection.weight, matched_full.time_projection.weight):
        raise RuntimeError('Shared time projection is still different')
    if not torch.equal(legacy_rng, matched_rng):
        raise RuntimeError('Matched builder changed the legacy CPU RNG state')
    if (runtime.digest(time_only.posterior) != runtime.digest(matched_full.posterior)
            or runtime.digest(time_only.encoder) != runtime.digest(matched_full.encoder)):
        raise RuntimeError('Shared posterior or encoder differs')
    old_state, new_state = legacy_full.state_dict(), matched_full.state_dict()
    if set(old_state) != set(new_state):
        raise RuntimeError('Matched model parameter set differs')
    for name in old_state:
        if not name.startswith('time_projection.') and not torch.equal(old_state[name], new_state[name]):
            raise RuntimeError('Unexpected change outside shared time projection: ' + name)
    unchanged = {}
    for name, cfg in [('C0', config('C0', False, False)), ('C1', config('C1', False, False)),
                      ('C2_time', time_config)]:
        old = legacy.build_variant(module, decoder, posterior, runtime, cfg)
        new = build_matched(legacy.build_variant, module, decoder, posterior, runtime, cfg)
        if runtime.digest(old) != runtime.digest(new):
            raise RuntimeError('Non-full-C2 model changed: ' + name)
        unchanged[name] = True
    example = {'variant': 'C2', 'model_initial_digest': 'fixed'}
    metadata = {'variant': 'C2', 'model_initial_digest': 'fixed', 'policy': POLICY}
    augmented = driver.add_binding_metadata(example, metadata)
    if augmented.get('initializer_wrapper') != metadata or 'initializer_wrapper' in example:
        raise RuntimeError('Wrapper provenance was not added immutably')
    try:
        driver.add_binding_metadata(example, dict(metadata, model_initial_digest='different'))
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Mismatched initialization provenance was accepted')

    # Exercise the actual wrapper and binding hook without starting a trainer.
    original_main, original_verify, original_outputs, original_sha = legacy.main, driver.verify_sources, driver.OUTPUTS, driver.sha
    original_builder, original_write = legacy.build_variant, legacy.write
    with tempfile.TemporaryDirectory(prefix='gs_c_matched_wrapper_') as temp:
        temporary = Path(temp)
        config_path = temporary / 'config.json'
        config_path.write_text(json.dumps(full_config), encoding='utf-8')
        target = temporary / 'run'
        def fake_legacy_main(path, out):
            candidate = json.loads(Path(path).read_text(encoding='utf-8'))
            model = legacy.build_variant(module, decoder, posterior, runtime, candidate)
            Path(out).mkdir()
            legacy.write(Path(out) / 'BINDING.json', dict(variant='C2', model_initial_digest=runtime.digest(model)))
        try:
            legacy.main = fake_legacy_main
            driver.verify_sources = lambda: {'status': 'passed'}
            driver.OUTPUTS = temporary
            driver.sha = lambda path: ('0' * 64 if Path(path) == ROOT / 'CPU_TESTS.json' else original_sha(path))
            driver.main(config_path, target)
            recorded = json.loads((target / 'BINDING.json').read_text(encoding='utf-8'))
            if (recorded['initializer_wrapper']['policy'] != POLICY
                    or recorded['initializer_wrapper']['time_projection_initial_digest'] != runtime.digest(time_only.time_projection)
                    or recorded['initializer_wrapper']['model_initial_digest'] != recorded['model_initial_digest']):
                raise RuntimeError('Driver wrapper did not bind matched initialization')
        finally:
            legacy.main, driver.verify_sources, driver.OUTPUTS, driver.sha = original_main, original_verify, original_outputs, original_sha
        if legacy.build_variant is not original_builder or legacy.write is not original_write:
            raise RuntimeError('Wrapper did not restore the underlying driver')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU check unexpectedly initialized CUDA')
    result = dict(status='passed', code_sha256=driver.sha(ROOT / 'CODE.sha256'),
        base_driver_sha256=driver.sha(driver.C_ROOT / 'c_train_driver_v2.py'),
        variants_manifest_sha256=driver.sha(VARIANTS / 'CODE.sha256'),
        original_time_projection_max_abs_difference=float(old_difference.abs().max()),
        original_time_projection_rms_difference=float(old_difference.square().mean().sqrt()),
        matched_time_projection_bitwise_equal=True, shared_q_and_encoder_identical=True,
        legacy_delta_and_other_parameters_unchanged=True, legacy_cpu_rng_state_unchanged=True,
        c0_c1_and_time_only_unchanged=unchanged, wrapper_binding_hook_passed=True,
        gpu_used=False, teacher_changed=False, training_performed=False,
        scope='Real frozen base modules on CPU; simulated wrapper call; no reviewed teacher or GPU training')
    with (ROOT / 'CPU_TESTS.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: identical shared time initialization, unchanged Q/encoder/delta/RNG, wrapper provenance')


if __name__ == '__main__':
    main()

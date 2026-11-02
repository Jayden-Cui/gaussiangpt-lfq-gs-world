"""Check the fixed anchor/horizon schedule and guarded training budget."""
import copy
from pathlib import Path
import worker


def must_fail(fn):
    try:
        fn()
    except (RuntimeError, ValueError, KeyError):
        return
    raise RuntimeError('Invalid schedule or training budget accepted')


def main():
    config = worker.read(Path(__file__).resolve().parent/'protocol.json')
    steps, report = worker.validate_config(config)
    assert len(steps) == 360 and report['per_window_horizon_visits'] == 40
    assert worker.schedule(40, config['sampler_seed']) == steps
    altered = copy.deepcopy(steps)
    altered[1]['horizon'] = altered[0]['horizon']
    must_fail(lambda: worker.audit_schedule(altered, 40))
    for key, value in (('decoder_frozen', False), ('quantized', True), ('starts', [35]),
                       ('checkpoint_updates', [360]), ('code_learning_rate', .01)):
        candidate = dict(config, **{key: value})
        must_fail(lambda: worker.validate_config(candidate))
    print('PASS: deterministic 360-step coverage, frozen decoder and fixed continuous-code budget')


if __name__ == '__main__':
    main()

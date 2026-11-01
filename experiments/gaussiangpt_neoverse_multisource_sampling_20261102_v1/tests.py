"""Check exact source-balanced coverage, aggregation and real source binding."""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch

from sampling import SOURCE_ROOT, audit_schedule, build_schedule, code_index, evaluation_pairs
from source_store import SourceStore, TRAIN, EVAL, CODEC_SHA
from aggregate import physical_frame_macro

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def expect_error(function, types=(RuntimeError, ValueError, PermissionError)):
    try:
        function()
    except types:
        return
    raise AssertionError('Invalid cross-source example was accepted')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Sampling source changed')
    prior = json.loads((SOURCE_ROOT / 'CPU_TESTS.json').read_text(encoding='utf-8'))
    if prior.get('status') != 'passed' or prior.get('code_sha256') != sha(SOURCE_ROOT / 'CODE.sha256'):
        raise RuntimeError('Real multi-source loader witness changed')
    for line in (SOURCE_ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(SOURCE_ROOT / name) != digest:
            raise RuntimeError('Bound data source changed: ' + name)

    reports = {}
    for phase in ('distill', 'joint'):
        schedule = build_schedule(phase, 2, 20261102)
        report = audit_schedule(schedule, phase, 2)
        if schedule != build_schedule(phase, 2, 20261102) or schedule == build_schedule(phase, 2, 20261103):
            raise RuntimeError('Sampler reproducibility or seed sensitivity failed')
        contaminated = copy.deepcopy(schedule)
        contaminated[0]['batch'][0]['clip'] = EVAL[0]
        expect_error(lambda: audit_schedule(contaminated, phase, 2))
        incomplete = copy.deepcopy(schedule)
        incomplete[0]['batch'][0] = copy.deepcopy(incomplete[0]['batch'][1])
        expect_error(lambda: audit_schedule(incomplete, phase, 2))
        reports[phase] = report
    assert code_index(TRAIN[0], 0) == (0, 0) and code_index(TRAIN[1], 167) == (1, 167)
    expect_error(lambda: code_index(EVAL[0], 0))
    expect_error(lambda: code_index(TRAIN[0], 168))
    pairs = evaluation_pairs(EVAL[0])
    assert len(pairs) == 1512 and pairs[0]['frame'] == 1 and pairs[-1]['frame'] == 176
    expect_error(lambda: evaluation_pairs(TRAIN[0]))

    rows = [{'clip': clip, 'start': start, 'horizon': h, 'frame': start + h,
             'metrics': {'m': (start + h) ** 2 / 100000 + index}}
            for index, clip in enumerate(TRAIN) for start in range(168) for h in range(1, 10)]
    aggregate = physical_frame_macro(rows, TRAIN, 'm')
    base = sum(frame ** 2 / 100000 for frame in range(1, 177)) / 176
    if not math.isclose(aggregate['source_macro'], base + 0.5, abs_tol=1e-12):
        raise RuntimeError('Source/physical-frame macro differs from analytic answer')
    window_mean = sum(row['metrics']['m'] for row in rows) / len(rows)
    if math.isclose(window_mean, aggregate['source_macro'], abs_tol=1e-6):
        raise RuntimeError('Fixture does not distinguish window and physical-frame weighting')
    expect_error(lambda: physical_frame_macro(rows[:-1], TRAIN, 'm'))
    expect_error(lambda: physical_frame_macro(rows + [rows[0]], TRAIN, 'm'))
    contaminated = copy.deepcopy(rows)
    contaminated[0]['clip'] = EVAL[0]
    expect_error(lambda: physical_frame_macro(contaminated, TRAIN, 'm'))

    store = SourceStore()
    checks = []
    for index, clip in enumerate((*TRAIN, *EVAL)):
        purpose = 'train' if clip in TRAIN else 'eval'
        start = (0, 26, 167)[index]
        window = store.load_window(clip, start, purpose=purpose, device='cpu')
        if (window['source_frames'] != list(range(start, start + 10)) or window['frame_stride'] != 1
                or window['quantized'] is not False or len(window['states']) != 10
                or any(x['features'].shape[1] != 64 for x in window['states'])):
            raise RuntimeError('Real source window did not match the sampler')
        checks.append({'clip': clip, 'purpose': purpose, 'start': start,
                       'first_frame': start, 'last_frame': start + 9})
    expect_error(lambda: store.load_window(EVAL[0], 0, purpose='train'))
    result = {'status': 'passed', 'code_sha256': sha(ROOT / 'CODE.sha256'),
              'source_loader_code_sha256': prior['code_sha256'], 'codec_sha256': CODEC_SHA,
              'schedule_reports': reports, 'real_source_windows': checks,
              'eval_source_training_rejected': True, 'source_frame_macro_analytic_check': True,
              'overlap_weighting_differs_from_naive_window_mean': True,
              'eval_video_historically_exposed': True, 'gpu_training_performed': False,
              'scope': 'Sampling and real-data plumbing only; no cross-source action result or trained code bank'}
    with (ROOT / 'CPU_TESTS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: balanced train-only B4 schedules, physical-frame macro and real source windows')


if __name__ == '__main__':
    main()

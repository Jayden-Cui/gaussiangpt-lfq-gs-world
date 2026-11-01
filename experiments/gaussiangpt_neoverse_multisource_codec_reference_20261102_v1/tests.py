"""CPU test for frame identity and no accidental evaluation-source selection."""
import json
from pathlib import Path
import tempfile
import reference


def expect_error(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError('Invalid frame binding was accepted')


def main():
    from source_store import PINS, CODEC_SHA, TRAIN
    historic_path = reference.MULTISOURCE_CODEC / 'EVALUATION.json'
    if reference.sha(historic_path) != PINS['evaluations/baseline/EVALUATION.json']:
        raise RuntimeError('Historic codec evaluation binding changed')
    historic = reference.read(historic_path)
    targets = reference.MULTISOURCE_CODEC / 'targets.jsonl'
    if (historic.get('status') != 'evaluated' or historic.get('checkpoint_sha256') != CODEC_SHA
            or reference.sha(targets) != historic.get('targets_sha256')
            or reference.find_frame(targets, TRAIN[1], 35).get('frame') != 35):
        raise RuntimeError('Real historical codec status or target frame differs')
    with tempfile.TemporaryDirectory(prefix='codec_reference_test_') as directory:
        path = Path(directory) / 'rows.jsonl'
        rows = [{'clip': 'train', 'frame': 35, 'split': 'TRAIN'},
                {'clip': 'eval', 'frame': 35, 'split': 'EVAL'}]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        assert reference.find_frame(path, 'train', 35) == rows[0]
        expect_error(lambda: reference.find_frame(path, 'eval', 35))
        expect_error(lambda: reference.find_frame(path, 'train', 34))
        path.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[0]) + '\n', encoding='utf-8')
        expect_error(lambda: reference.find_frame(path, 'train', 35))
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or reference.sha(root / name) != digest:
            raise RuntimeError('Reference source changed')
    reference.write(root / 'CPU_TESTS.json', {'status': 'passed',
                    'code_sha256': reference.sha(root / 'CODE.sha256'),
                    'rejects_eval_missing_and_duplicate_frames': True,
                    'actual_historic_evaluated_status_and_hash_checked': True,
                    'scope': 'Synthetic frame guards and real historic metadata; no GPU result'})
    print('PASS: training-source frame identity and duplicate/eval rejection')


if __name__ == '__main__':
    main()

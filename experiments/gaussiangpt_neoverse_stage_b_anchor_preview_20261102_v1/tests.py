"""CPU tests for checkpoint identity and the two-window preview's evidence."""
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch
import preview


def expect_error(function):
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid input was unexpectedly accepted')


def main():
    source = {'review': 'bound', 'quantized': False}
    item = {'schema': 'stage_b_shared_decoder_free_codes_v1', 'sweep': 5, 'starts': list(range(168)),
            'source': source, 'decoder_digest': 'decoder', 'codes': torch.zeros(168, 9, 16)}
    metadata = {'status': 'complete', 'sweep': 5, 'sha256': 'checkpoint', 'decoder_digest': 'decoder'}
    assert preview.validate_checkpoint(item, metadata, source, 5, 'checkpoint').shape == (168, 9, 16)
    expect_error(lambda: preview.validate_checkpoint(item, metadata, source, 5, 'wrong'))
    wrong = dict(item, starts=list(reversed(range(168))))
    expect_error(lambda: preview.validate_checkpoint(wrong, metadata, source, 5, 'checkpoint'))
    expect_error(lambda: preview.validate_checkpoint(item, metadata, {'different': True}, 5, 'checkpoint'))
    rows = [{'start': s, 'horizon': h, 'metrics': {'correct': {'metrics': {'motion_mae': h / 10}}}}
            for s in (35, 123) for h in range(1, 10)]
    assert abs(preview.anchor_mean(rows) - 0.5) < 1e-9
    expect_error(lambda: preview.anchor_mean(rows[:-1]))
    with tempfile.TemporaryDirectory(prefix='b_preview_test_') as directory:
        path = Path(directory) / 'pair.png'
        left = torch.zeros(252, 448, 3)
        right = torch.ones(252, 448, 3)
        preview.save_pair(path, left, right)
        with Image.open(path) as image:
            assert image.size == (896, 252) and image.getpixel((0, 0)) == (0, 0, 0)
            assert image.getpixel((448, 0)) == (255, 255, 255)
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or preview.sha(root / name) != digest:
            raise RuntimeError('Preview code changed')
    preview.write(root / 'CPU_TESTS.json', {'status': 'passed',
                  'code_sha256': preview.sha(root / 'CODE.sha256'),
                  'rejects_changed_checkpoint_or_code_order': True,
                  'two_windows_nine_future_steps': True, 'unlabelled_pair_crop': True,
                  'scope': 'CPU synthetic control-flow tests; no motion-quality claim'})
    print('PASS: source-bound checkpoint, exact anchor coverage and unlabelled image pairs')


if __name__ == '__main__':
    main()

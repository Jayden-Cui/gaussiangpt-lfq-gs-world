"""CPU checks for paired metric and unlabelled triplet image semantics."""
from pathlib import Path
import torch
import swap


def expect_error(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError('Invalid paired result was accepted')


def main():
    rows = [{'metrics': {'baseline': {'metrics': {'mae': .1}},
                         'decoder_only': {'metrics': {'mae': .2}}}} for _ in range(18)]
    assert abs(swap.mean_metric(rows, 'baseline', 'mae') - .1) < 1e-9
    assert abs(swap.mean_metric(rows, 'decoder_only', 'mae') - .2) < 1e-9
    expect_error(lambda: swap.mean_metric(rows[:-1], 'baseline', 'mae'))
    image = swap.image_triplet(torch.zeros(252, 448, 3), torch.ones(252, 448, 3),
                               torch.full((252, 448, 3), .5))
    assert image.size == (1344, 252)
    assert image.getpixel((0, 0)) == (0, 0, 0) and image.getpixel((448, 0)) == (255, 255, 255)
    assert image.getpixel((896, 0)) == (128, 128, 128)
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or swap.sha(root / name) != digest:
            raise RuntimeError('Decoder swap source changed')
    swap.write(root / 'CPU_TESTS.json', {'status': 'passed', 'code_sha256': swap.sha(root / 'CODE.sha256'),
               'paired_18_frame_aggregation': True, 'unlabelled_three_column_order': True,
               'scope': 'Synthetic CPU guards; no codec or model selection'})
    print('PASS: paired metric aggregation and source/original/adapted image order')


if __name__ == '__main__':
    main()

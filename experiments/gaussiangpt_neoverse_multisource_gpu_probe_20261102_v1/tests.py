"""CPU shape and source guards for the cross-source GPU plumbing probe."""
from pathlib import Path
import torch
import probe


def expect_error(function):
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid window or image was accepted')


def main():
    state = {'coordinates': torch.zeros(3, 3, dtype=torch.int32), 'features': torch.zeros(3, 64)}
    window = {'clip': 'training_clip', 'start': 26, 'states': [state] * 10,
              'source_frames': list(range(26, 36)), 'targets': {'indices': [None] * 9}}
    probe.validate_window(window, 'training_clip', 26)
    expect_error(lambda: probe.validate_window(window, 'other_clip', 26))
    expect_error(lambda: probe.validate_window(dict(window, source_frames=list(range(10))), 'training_clip', 26))
    expect_error(lambda: probe.validate_window(dict(window, states=[dict(state, features=torch.zeros(3, 63))] * 10),
                                               'training_clip', 26))
    left, right = torch.zeros(252, 448, 3), torch.ones(252, 448, 3)
    image = probe.image_pair(left, right)
    assert image.size == (896, 252) and image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((448, 0)) == (255, 255, 255)
    expect_error(lambda: probe.image_pair(torch.zeros(1, 3), right))
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or probe.sha(root / name) != digest:
            raise RuntimeError('Probe code changed')
    probe.write(root / 'CPU_TESTS.json', {'status': 'passed', 'code_sha256': probe.sha(root / 'CODE.sha256'),
                'rejects_wrong_source_and_nonadjacent_window': True, 'checks_native_64_shape': True,
                'source_prediction_pair_order': True,
                'scope': 'Synthetic CPU guards; no GPU, model training or action-quality claim'})
    print('PASS: source/frame/shape guards and unlabelled source-prediction image')


if __name__ == '__main__':
    main()

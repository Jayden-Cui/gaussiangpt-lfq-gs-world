"""CPU tests for coordinate-based sparse state comparison."""
from pathlib import Path
import numpy as np
import torch
import audit


def expect_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Invalid sparse state accepted')


def main():
    coordinates = np.array([[2, 0, 1], [0, 0, 0], [1, 4, 5]], dtype=np.int32)
    features = np.arange(192, dtype=np.float32).reshape(3, 64)
    permutation = np.array([2, 0, 1])
    result = audit.compare_states(coordinates, features, coordinates[permutation], features[permutation])
    assert result['same_coordinate_set'] and not result['same_raw_row_order']
    assert result['matched_nodes'] == 3 and result['feature_max_abs'] == 0
    changed = features[permutation].copy()
    changed[0, 0] += .25
    result = audit.compare_states(coordinates, features, coordinates[permutation], changed)
    assert result['feature_max_abs'] == .25 and result['feature_mean_abs'] > 0
    partial = audit.compare_states(coordinates, features, coordinates[:2], features[:2])
    assert not partial['same_coordinate_set'] and partial['matched_nodes'] == 2
    duplicate = coordinates.copy()
    duplicate[1] = duplicate[0]
    expect_error(lambda: audit.canonical(duplicate, features))
    expect_error(lambda: audit.canonical(coordinates, np.zeros((3, 63), dtype=np.float32)))
    synthetic = {'ae.encoder.w': torch.tensor([1.0]), 'ae.decoder.w': torch.tensor([2.0])}
    assert set(audit.encoder_snapshot(synthetic)) == {'ae.encoder.w'}
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or audit.sha(root / name) != digest:
            raise RuntimeError('Audit source changed')
    import json
    with (root / 'CPU_TESTS.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'passed', 'code_sha256': audit.sha(root / 'CODE.sha256'),
                   'coordinate_permutation_and_partial_overlap_checked': True,
                   'duplicate_world_coordinates_rejected': True, 'encoder_prefix_selection_checked': True,
                   'scope': 'Synthetic CPU tests; no production codec selected'}, stream, indent=2)
        stream.write('\n')
    print('PASS: world-coordinate alignment, overlap and frozen-encoder selection')


if __name__ == '__main__':
    main()

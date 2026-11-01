"""Small adversarial masks for the proxy sensitivity calculation."""
import numpy as np
from sensitivity import component_stats, measure


def main():
    mask = np.zeros((40, 200), dtype=bool)
    mask[10:20, 10:20] = True
    mask[10, 180] = True
    info, largest = component_stats(mask)
    assert info['pixels'] == 101 and info['largest_pixels'] == 100
    assert info['diagonal_inflation'] > 10 and largest[10, 180] == 0
    prediction = np.zeros_like(mask)
    prediction[10:20, 15:25] = True
    result = measure(mask, prediction)
    assert result['same_center_over_largest_target_bbox'] > result['original_center_over_all_bbox'] * 10
    assert not result['target_is_fragmented']
    split = np.zeros_like(mask)
    split[10:20, 10:20] = True
    split[10:20, 50:60] = True
    assert measure(split, prediction)['target_is_fragmented']
    empty = np.zeros_like(mask)
    assert measure(empty, prediction)['original_center_over_all_bbox'] is None
    print('PASS: detached outlier, fragmented target, empty target')


if __name__ == '__main__':
    main()

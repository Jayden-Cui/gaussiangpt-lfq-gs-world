"""Focused geometric and source-binding checks for axis diagnostics."""
import numpy as np
import axis_audit as audit


def main():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 5:31] = True
    a, b = (5, 20), (30, 20)
    assert all(value == 1 for value in audit.coverage(mask, a, b).values())
    shifted = audit.coverage(mask, (5, 24), (30, 24))
    assert shifted == {'2': 0., '4': 1., '8': 1.}
    assert all(value == 0 for value in audit.coverage(np.zeros_like(mask), a, b).values())
    for args in (((-1, 2), (3, 2)), ((2, 2), (2, 2))):
        try:
            audit.coverage(mask, *args)
        except ValueError:
            pass
        else:
            raise RuntimeError('Invalid axis accepted')
    reference = audit.read(audit.REFERENCE)
    audit.validate_reference(reference, audit.BASELINE)
    image = np.zeros((252, 448, 3), dtype=np.uint8)
    row = dict(reference['rows'][0], source_target_rgb_sha256=audit.rgb_sha(image))
    audit.assert_source(image, row)
    image[0, 0, 0] = 1
    try:
        audit.assert_source(image, row)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Modified source pixels accepted')
    print('PASS: axis distance, missing mask, out-of-image and source-pixel binding')


if __name__ == '__main__':
    main()

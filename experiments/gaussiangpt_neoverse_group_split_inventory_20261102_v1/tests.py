"""Boundary cases for source-group train/evaluation isolation."""
from inventory import parse_name, validate_partition


def row(name, digest):
    group, start = parse_name(name + '.mp4')
    return dict(clip=name, source_group=group, encoded_start_index=start, source_video_sha256=digest * 64)


def rejected(fn, part):
    try:
        fn()
    except ValueError as error:
        if part not in str(error):
            raise
    else:
        raise AssertionError('Expected split rejection: ' + part)


def main():
    a, b, c = row('buildingblock_tower_17_0', 'a'), row('buildingblock_tower_17_176', 'b'), row('buildingblock_tower_14_0', 'c')
    roles = {a['clip']: 'TRAIN', b['clip']: 'TRAIN', c['clip']: 'EVAL'}
    assert validate_partition([a, b, c], roles) == {'buildingblock_tower_17': 'TRAIN', 'buildingblock_tower_14': 'EVAL'}
    rejected(lambda: validate_partition([a, b, c], {a['clip']: 'TRAIN', b['clip']: 'EVAL'}), 'same source group')
    duplicate = dict(c, source_video_sha256=a['source_video_sha256'])
    rejected(lambda: validate_partition([a, duplicate], {a['clip']: 'TRAIN', c['clip']: 'EVAL'}), 'Byte-identical')
    rejected(lambda: validate_partition([a, a], {a['clip']: 'TRAIN'}), 'Duplicate clip')
    rejected(lambda: parse_name('unrelated.mp4'), 'Unexpected')
    print('PASS: same-group training allowed; cross-role group and duplicate-video leaks rejected')


if __name__ == '__main__':
    main()

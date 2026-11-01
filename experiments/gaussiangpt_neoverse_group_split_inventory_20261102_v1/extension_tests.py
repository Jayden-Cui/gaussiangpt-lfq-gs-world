"""Verify that incremental split proposals cannot erase prior assignments."""
from extension_guard import validate_extension
from tests import row, rejected


def main():
    train = row('buildingblock_tower_17_176', 'a')
    same_train_group = row('buildingblock_tower_17_0', 'b')
    evaluation = row('buildingblock_tower_14_0', 'c')
    same_eval_group = row('buildingblock_tower_14_176', 'd')
    rows = [train, same_train_group, evaluation, same_eval_group]
    existing = {train['clip']: 'TRAIN', evaluation['clip']: 'EVAL'}
    accepted, groups = validate_extension(rows, existing, {same_train_group['clip']: 'TRAIN', same_eval_group['clip']: 'EVAL'})
    assert len(accepted) == 4 and groups['buildingblock_tower_17'] == 'TRAIN'
    rejected(lambda: validate_extension(rows, existing, {same_train_group['clip']: 'EVAL'}), 'same source group')
    rejected(lambda: validate_extension(rows, existing, {same_eval_group['clip']: 'TRAIN'}), 'same source group')
    rejected(lambda: validate_extension(rows, existing, {evaluation['clip']: 'TRAIN'}), 'Existing clip assignment')
    duplicate = row('buildingblock_tower_19_0', 'a')
    rejected(lambda: validate_extension(rows + [duplicate], existing, {duplicate['clip']: 'EVAL'}), 'Byte-identical')
    print('PASS: incremental proposals preserve old assignments and reject group/video leakage')


if __name__ == '__main__':
    main()

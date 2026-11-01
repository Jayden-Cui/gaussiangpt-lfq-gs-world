"""Exercise teacher binding, exhaustive coverage and group/horizon aggregation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from report import compare, read, sha

ROOT = Path(__file__).resolve().parent
GROUP_PATH = ROOT.parent / 'gaussiangpt_neoverse_motion_groups_20261102_v1/MOTION_GROUPS.json'


def expect_rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid C/teacher result was accepted')


def main():
    p, groups = read(ROOT / 'protocol.json'), read(GROUP_PATH)
    group_sha = sha(GROUP_PATH)
    teacher_sha, digest, codec, codes = 'a'*64, 'b'*64, 'c'*64, 'd'*64
    common = {'decoder_digest': digest, 'codec_sha256': codec, 'cache_manifest_sha256': p['cache_manifest_sha256']}
    teacher_source = dict(common, codes_sha256=codes)
    student_source = dict(common, schema='fixed_teacher_posterior_evaluation_v1', variant='C1',
                          quantized=False, teacher_evaluation_sha256=teacher_sha, teacher_codes_sha256=codes)
    teacher_rows, student_rows = [], []
    for start in range(168):
        for h in range(1, 10):
            value = .01 + .0001*start + .001*h
            teacher_metric = {'metrics': {'mae': value/2, 'motion_mae': value}, 'object_proxy': {'tag': start}}
            controls = {'teacher_free': copy.deepcopy(teacher_metric)}
            for control, scale in (('correct', 2), ('initial_only', 3), ('swapped_window_codes', 4), ('actual_target', .5)):
                controls[control] = {'metrics': {'mae': scale*value/2, 'motion_mae': scale*value}}
            teacher_rows.append({'start': start, 'horizon': h, 'frame': start+h, 'metrics': {'correct': teacher_metric}})
            student_rows.append({'start': start, 'horizon': h, 'frame': start+h, 'metrics': controls})
    shared = dict(status='complete', window_predictions=1512, unique_future_frames=176,
                  all_source_frames_preserved=True, decoder_future_geometry_input=False)
    teacher = dict(shared, source=teacher_source, rows=teacher_rows)
    student = dict(shared, source=student_source, rows=student_rows)
    result = compare(groups, p, group_sha, student, teacher, teacher_sha)
    assert result['teacher_row_metrics_match'] == 1512
    for name, item in result['groups'].items():
        assert len(item['window_starts']) == 42
        assert item['controls']['correct']['motion_mae']['window_predictions'] == 378
        assert abs(item['student_over_teacher_motion_mae']['physical_frame_equal'] - 2) < 1e-12
        assert all(abs(x-2) < 1e-12 for x in item['student_over_teacher_motion_mae']['by_horizon'].values())
    highest = result['groups']['highest']
    chosen = [r for r in student_rows if r['start'] in highest['window_starts']]
    frames = sorted({r['frame'] for r in chosen})
    manual = sum(sum(r['metrics']['correct']['metrics']['motion_mae'] for r in chosen if r['frame'] == frame)
                 / sum(r['frame'] == frame for r in chosen) for frame in frames) / len(frames)
    assert abs(manual-highest['controls']['correct']['motion_mae']['physical_frame_equal_within_group']) < 1e-12
    shuffled = dict(student, rows=list(reversed(student_rows)))
    assert compare(groups, p, group_sha, shuffled, teacher, teacher_sha)['groups'] == result['groups']
    expect_rejected(lambda: compare(groups, p, '0'*64, student, teacher, teacher_sha))
    expect_rejected(lambda: compare(groups, p, group_sha, dict(student, rows=student_rows[:-1]+[student_rows[0]]), teacher, teacher_sha))
    expect_rejected(lambda: compare(groups, p, group_sha, student, teacher, 'e'*64))
    wrong_source = dict(student, source=dict(student_source, decoder_digest='e'*64))
    expect_rejected(lambda: compare(groups, p, group_sha, wrong_source, teacher, teacher_sha))
    changed = copy.deepcopy(student_rows)
    changed[0]['metrics']['teacher_free']['metrics']['motion_mae'] += .1
    expect_rejected(lambda: compare(groups, p, group_sha, dict(student, rows=changed), teacher, teacher_sha))
    changed = copy.deepcopy(student_rows)
    changed[0]['metrics']['correct']['metrics']['motion_mae'] = float('nan')
    expect_rejected(lambda: compare(groups, p, group_sha, dict(student, rows=changed), teacher, teacher_sha))
    expect_rejected(lambda: compare(groups, p, group_sha, dict(student, decoder_future_geometry_input=True), teacher, teacher_sha))
    # Exercise the real CLI with its source witness and file hashes in isolation.
    with tempfile.TemporaryDirectory(prefix='c_motion_groups_cli_') as folder:
        root = Path(folder)
        names = ('protocol.json', 'report.py', 'tests.py')
        for name in names:
            shutil.copy2(ROOT/name, root/name)
        (root/'CODE.sha256').write_text(''.join(
            hashlib.sha256((root/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in names), encoding='utf8')
        (root/'CPU_TESTS.json').write_text(json.dumps({'status': 'passed', 'code_sha256': sha(root/'CODE.sha256')}),
                                           encoding='utf8')
        teacher_path, student_path, output = root/'teacher.json', root/'student.json', root/'output.json'
        teacher_path.write_text(json.dumps(teacher, allow_nan=False), encoding='utf8')
        student_for_cli = dict(student, source=dict(student_source, teacher_evaluation_sha256=sha(teacher_path)))
        student_path.write_text(json.dumps(student_for_cli, allow_nan=False), encoding='utf8')
        command = [sys.executable, '-B', str(root/'report.py'), '--groups', str(GROUP_PATH),
                   '--student', str(student_path), '--teacher', str(teacher_path), '--output', str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        saved = read(output)
        assert saved['teacher_row_metrics_match'] == 1512 and saved['student_evaluation_sha256'] == sha(student_path)
        assert saved['teacher_evaluation_sha256'] == sha(teacher_path)
        assert subprocess.run(command, capture_output=True).returncode != 0
    print('PASS: all 1512 matching teacher rows, four groups, physical-frame and nine-horizon aggregation')
    print('PASS: real CLI, no-overwrite, source, changed metric, duplicate, nonfinite and future-geometry guards')
    print('Scope: synthetic posterior/teacher values, not a trained C model or real performance')


if __name__ == '__main__':
    main()

"""CPU checks for source-only strata and frame-equal aggregation."""
import math
import json
import tempfile
from pathlib import Path
import motion_groups as m


def raises(fn):
    try:
        fn()
    except (KeyError, ValueError):
        return
    raise AssertionError('Invalid input was accepted')


scores = {s: float(s // 3)/100 for s in range(168)}
groups = m.partition(scores)
assert all(len(v) == 42 for v in groups.values())
assert [x for name in m.NAMES for x in groups[name]] == list(range(168))
raises(lambda: m.partition({0: 0.0}))
raises(lambda: m.partition({s: (float('nan') if s == 2 else 0.0) for s in range(168)}))
starts = set(groups['lowest'])
rows = []
for s in starts:
    for h in range(1,10):
        value = float(s+h)
        rows.append({'start': s, 'horizon': h, 'frame': s+h,
            'metrics': {'correct': {'metrics': {'mae': value, 'motion_mae': value}}}})
out = m.summarize(rows, starts, 'correct')['mae']
assert out['window_predictions'] == 378
assert out['unique_physical_frames'] == 50
assert math.isclose(out['window_prediction_equal'], 25.5)
assert math.isclose(out['physical_frame_equal_within_group'], 25.5)
assert math.isclose(out['h9_window_equal'], 29.5)
raises(lambda: m.summarize(rows[:-1], starts, 'correct'))
rows[0]['metrics']['correct']['metrics']['mae'] = float('nan')
raises(lambda: m.summarize(rows, starts, 'correct'))

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    protocol = m.read(m.ROOT/'protocol.json')
    group_file = root/'groups.json'
    group_file.write_text(json.dumps({
        'status': 'complete', 'protocol_sha256': m.sha(m.ROOT/'protocol.json'),
        'code_sha256': m.sha(m.ROOT/'motion_groups.py'),
        'cache_manifest_sha256': protocol['cache_manifest_sha256'],
        'frame_cache_sha256': protocol['frame_cache_sha256'],
        'scores_by_start': {str(k): v for k,v in scores.items()}, 'groups': groups}), encoding='utf8')
    eval_file = root/'eval.json'
    output = root/'report.json'
    all_rows = []
    for s in range(168):
        for h in range(1,10):
            metrics = {'metrics': {'mae': float(s+h), 'motion_mae': float(s+h)}}
            all_rows.append({'start': s, 'horizon': h, 'frame': s+h,
                             'metrics': {'correct': metrics, 'actual_target': metrics}})
    payload = {'status': 'complete', 'window_predictions': 1512,
               'source': {'cache_manifest_sha256': protocol['cache_manifest_sha256']}, 'rows': all_rows}
    eval_file.write_text(json.dumps(payload), encoding='utf8')
    m.evaluate(group_file, eval_file, output)
    report = m.read(output)
    assert report['status'] == 'complete'
    assert math.isclose(report['groups']['lowest']['correct']['mae']['h9_window_equal'],29.5)
    final_row = payload['rows'][-1]
    payload['rows'][-1] = payload['rows'][0]
    eval_file.write_text(json.dumps(payload), encoding='utf8')
    raises(lambda: m.evaluate(group_file, eval_file, root/'duplicate.json'))
    payload['rows'][-1] = final_row
    payload['source']['cache_manifest_sha256'] = 'wrong source'
    eval_file.write_text(json.dumps(payload), encoding='utf8')
    raises(lambda: m.evaluate(group_file, eval_file, root/'wrong_source.json'))
print('PASS: partition, stable ties, full coverage, frame aggregation, provenance and nonfinite guards')

"""Reject incomplete, mismatched, nonfinite or self-accepting axis reports."""
import copy
import json
from pathlib import Path
import watcher


def must_fail(fn):
    try:
        fn()
    except (RuntimeError, KeyError):
        return
    raise RuntimeError('Bad axis report unexpectedly accepted')


def main():
    reference = json.loads((watcher.AXIS / 'source_axis_reference.json').read_text(encoding='utf-8'))
    source = dict(sweep=5, quantized=False)
    evaluation = dict(source=source)
    values = {'2': .2, '4': .5, '8': .9}
    rows = [dict(start=r['start'], horizon=r['horizon'], object=r['object'],
        source_target_rgb_sha256=r['source_target_rgb_sha256'],
        axis_color_coverage_by_radius_px={name: dict(values) for name in
          ('source', 'stage_b_free_motion', 'actual_state_codec')}) for r in reference['rows']]
    report = dict(status='complete', schema='visible_source_axis_diagnostic_v1',
        evaluation_sha256='e' * 64, reference_sha256='r' * 64, evaluation_source=source,
        manual_motion_review_required=True, no_acceptance_decision=True, rows=rows)
    def verify(candidate, sweep=5):
        watcher.validate_report(candidate, evaluation, 'e' * 64, reference, 'r' * 64, sweep)
    verify(report)
    must_fail(lambda: verify(report, 10))
    for mutation in ('drop', 'duplicate', 'source', 'nonfinite', 'nonmonotonic', 'acceptance'):
        candidate = copy.deepcopy(report)
        if mutation == 'drop': candidate['rows'].pop()
        elif mutation == 'duplicate': candidate['rows'][1] = copy.deepcopy(candidate['rows'][0])
        elif mutation == 'source': candidate['rows'][0]['source_target_rgb_sha256'] = '0' * 64
        elif mutation == 'nonfinite': candidate['rows'][0]['axis_color_coverage_by_radius_px']['source']['2'] = float('nan')
        elif mutation == 'nonmonotonic': candidate['rows'][0]['axis_color_coverage_by_radius_px']['source']['2'] = .8
        else: candidate['no_acceptance_decision'] = False
        must_fail(lambda: verify(candidate))
    print('PASS: distinct frames, source binding, sensitivity, correct checkpoint and no automatic acceptance')


if __name__ == '__main__':
    main()

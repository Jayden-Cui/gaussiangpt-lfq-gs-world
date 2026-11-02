"""Verify local stage-B review artifacts against completed server receipts."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main(output=None):
    transfer = read(ROOT / 'REMOTE_TRANSFER_SHA.json')
    if transfer.get('schema') != 'stage_b_local_transfer_sha_v1':
        raise RuntimeError('Unexpected transfer manifest')
    for name, item in transfer['files'].items():
        path = (ROOT / name).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            raise RuntimeError('Missing or unsafe imported file: ' + name)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Imported file changed: ' + name)
    summary = read(ROOT / 'SUMMARY.json')
    train = read(ROOT / 'TRAIN_DONE.json')
    if (summary.get('status') != 'complete' or summary.get('stage') != 'B'
            or summary.get('manual_dynamic_review_required') is not True
            or summary.get('action_codec_complete') is not False
            or summary['training_sha256'] != sha(ROOT / 'TRAIN_DONE.json')
            or train.get('status') != 'complete' or train.get('updates') != 7560
            or train.get('sweeps') != 20 or train.get('code_shape') != [168, 9, 16]
            or train.get('codec_and_initial_encoder_frozen') is not True):
        raise RuntimeError('Stage-B training receipt differs')
    if len(summary['results']) != 3 or {r['sweep'] for r in summary['results']} != {5, 10, 20}:
        raise RuntimeError('Expected all three registered checkpoints')
    checks = []
    for row in summary['results']:
        sweep = row['sweep']
        folder = ROOT / ('sweep_%02d' % sweep)
        evaluation, replay = read(folder / 'EVALUATION.json'), read(folder / 'REPLAY.json')
        metrics, causal = read(folder / 'STAGE_METRICS.json'), read(folder / 'CAUSALITY.json')
        if (sha(folder / 'EVALUATION.json') != row['evaluation_sha256']
                or sha(folder / 'REPLAY.json') != row['replay']['sha256']
                or sha(folder / 'PACKETS.json') != evaluation['packets_sha256']
                or replay.get('packets_sha256') != evaluation['packets_sha256']
                or replay.get('status') != 'passed' or replay.get('packets') != 104
                or replay.get('views') != 208 or replay.get('encoder_or_codec_loaded') is not False
                or evaluation.get('window_predictions') != 1512 or evaluation.get('unique_future_frames') != 176
                or evaluation['source'].get('sweep') != sweep or evaluation['source'].get('quantized') is not False
                or causal.get('status') != 'passed' or metrics.get('status') != 'complete'
                or metrics.get('sweep') != sweep or metrics.get('manual_visual_review_required') is not True):
            raise RuntimeError('Evaluation, replay or causality differs for sweep ' + str(sweep))
        future = [r for r in evaluation['rows'] if r['horizon'] > 0]
        pairs = {(r['start'], r['horizon']) for r in future}
        if len(future) != 1512 or pairs != {(s, h) for s in range(168) for h in range(1, 10)}:
            raise RuntimeError('Evaluation does not cover all 1512 pairs')
        correct = evaluation['metrics']['unique_frame_equal']['correct']
        if metrics['metrics']['unique_frame_equal']['correct'] != correct:
            raise RuntimeError('Metrics and raw evaluation disagree')
        checks.append(dict(sweep=sweep, evaluation_sha256=sha(folder / 'EVALUATION.json'),
            replay_sha256=sha(folder / 'REPLAY.json'), motion_mae=correct['motion_mae'],
            anchor_motion_mae=metrics['anchor_motion_mae'],
            original_anchor_motion_mae=metrics['original_two_window_anchor_motion_mae'],
            feature_prefix_error=causal['feature_max_abs_error'],
            occupancy_prefix_error=causal['occupancy_max_abs_error']))
    contacts_path = ROOT / 'REVIEW_ASSETS/CONTACTS.json'
    contacts = read(contacts_path)
    if (sha(contacts_path) != summary['visual_review_assets_sha256'] or contacts['status'] != 'complete'
            or contacts['evaluation_sha256'] != next(x['evaluation_sha256'] for x in checks if x['sweep'] == 20)
            or set(contacts['starts']) != {0, 35, 56, 84, 112, 123, 140, 167}):
        raise RuntimeError('Visual contact receipt differs')
    for row in contacts['rows']:
        if sha(ROOT / 'REVIEW_ASSETS' / row['file']) != row['sha256']:
            raise RuntimeError('Visual contact image changed')
    for folder, suffix in (('axis', 'AXIS_DIAGNOSTIC.json'), ('groups', None)):
        watch_path = ROOT / ('WATCH_RESULT.json' if folder == 'axis' else 'groups/WATCH_RESULT.json')
        watch = read(watch_path)
        if watch.get('status') != 'complete' or watch.get('B_summary_sha256') != sha(ROOT / 'SUMMARY.json'):
            raise RuntimeError('Derived report watcher differs: ' + folder)
        records = watch['reports'] if folder == 'axis' else watch['grouped_reports']
        if len(records) != 3 or {x['sweep'] for x in records} != {5, 10, 20}:
            raise RuntimeError('Derived reports incomplete: ' + folder)
        for row in records:
            sweep = row['sweep']
            path = (ROOT / folder / ('result_b_sweep_%02d' % sweep) / suffix if suffix else
                    ROOT / folder / ('STAGE_B_SWEEP%02d.json' % sweep))
            if sha(path) != row['sha256']:
                raise RuntimeError('Derived report hash differs: ' + str(path))
            report = read(path)
            expected_eval = next(x['evaluation_sha256'] for x in checks if x['sweep'] == sweep)
            if report.get('status') != 'complete' or report.get('evaluation_sha256') != expected_eval:
                raise RuntimeError('Derived report evaluation binding differs')
    result = dict(status='passed', transfer_manifest_sha256=sha(ROOT / 'REMOTE_TRANSFER_SHA.json'),
        summary_sha256=sha(ROOT / 'SUMMARY.json'), training_sha256=sha(ROOT / 'TRAIN_DONE.json'),
        imported_files=len(transfer['files']), checkpoints=sorted(checks, key=lambda x: x['sweep']),
        verified_contact_windows=sorted(contacts['starts']), independent_replay_per_checkpoint='104 GS packets, 208 views',
        interpretation='Integrity and scope checks only; dynamic quality requires separate visual review')
    if output is not None:
        with Path(output).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Optional new output file; omit for verification only')
    args = parser.parse_args()
    main(args.output)

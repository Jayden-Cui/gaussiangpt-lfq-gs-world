"""Verify the completed two-window fixed-decoder refinement and its exports."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parent
RESULT = ROOT/'result'
B = ROOT.parent/'gaussiangpt_neoverse_action_stage_b_20261102_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main(output=None):
    transfer = read(RESULT/'TRANSFER_SHA.json')
    if transfer.get('schema') != 'anchor_code_refinement_transfer_v1':
        raise RuntimeError('Unexpected transfer manifest')
    for name, item in transfer['files'].items():
        path = (RESULT/name).resolve()
        if RESULT.resolve() not in path.parents or not path.is_file():
            raise RuntimeError('Missing or unsafe result file')
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Transferred result changed: '+name)
    done, preflight, started = (read(RESULT/name) for name in ('DONE.json','PREFLIGHT.json','STARTED.json'))
    launch = read(ROOT/'LAUNCHED.json')
    if (done.get('status') != 'complete' or done.get('updates') != 360
            or done.get('checkpoints') != [90, 180, 360] or done.get('decoder_and_codec_frozen') is not True
            or done.get('manual_visual_review_required') is not True or done.get('full_action_codec_complete') is not False
            or done['source'].get('code_sha256') != sha(ROOT/'CODE.sha256')
            or done['source'].get('B_evaluation_sha256') != sha(B/'sweep_20/EVALUATION.json')
            or done['source'].get('B_summary_sha256') != sha(B/'SUMMARY.json')
            or done['source'].get('quantized') is not False or started['pid'] != launch['pid']
            or launch['preflight_sha256'] != sha(RESULT/'PREFLIGHT.json')
            or preflight.get('status') != 'passed' or preflight.get('codes_and_optimizer_restored') is not True
            or preflight.get('rgb_to_each_code_gradient_verified') is not True):
        raise RuntimeError('Completed training or provenance differs')
    lines = [json.loads(line) for line in (RESULT/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(lines) != 360 or [r['step'] for r in lines] != list(range(1,361)):
        raise RuntimeError('Training updates are incomplete')
    for sweep in range(1,41):
        values = [r for r in lines if r['sweep'] == sweep]
        if sorted(x['horizon'] for x in values) != list(range(1,10)):
            raise RuntimeError('A window/horizon training visit is missing')
        for row in values:
            if {r['start'] for r in row['windows']} != {35,123} or any(
                    r['horizon'] != row['horizon'] or not math.isfinite(r['loss']) for r in row['windows']):
                raise RuntimeError('Training batch or loss differs')
    records = []
    for step in (90,180,360):
        meta = read(RESULT/('checkpoint_%05d.json' % step))
        path, evpath = RESULT/('codes_%05d.pt' % step), RESULT/('step_%05d/EVALUATION.json' % step)
        if (meta.get('status') != 'complete' or meta.get('step') != step
                or meta['codes_sha256'] != sha(path) or meta['evaluation_sha256'] != sha(evpath)):
            raise RuntimeError('Checkpoint integrity differs')
        codes = torch.load(path, map_location='cpu', weights_only=True)
        if (codes.get('schema') != 'two_anchor_continuous_codes_v1' or codes.get('step') != step
                or codes.get('starts') != [35,123] or tuple(codes['codes'].shape) != (2,9,16)
                or not bool(torch.isfinite(codes['codes']).all()) or codes['source'] != done['source']):
            raise RuntimeError('Checkpoint codes or model binding differ')
        evaluation = read(evpath)
        pairs = {(r['start'],r['horizon']) for r in evaluation['rows']}
        if (evaluation.get('status') != 'complete' or evaluation.get('step') != step
                or len(evaluation['rows']) != 18 or pairs != {(s,h) for s in (35,123) for h in range(1,10)}
                or evaluation.get('oracle_target_support_used') is not False
                or evaluation.get('manual_visual_review_required') is not True
                or evaluation['per_window'] != meta['per_window']):
            raise RuntimeError('Anchor evaluation coverage or values differ')
        records.append(dict(step=step,codes_sha256=sha(path),evaluation_sha256=sha(evpath),
                            per_window=evaluation['per_window']))
    if done['final_codes_sha256'] != sha(RESULT/'codes_00360.pt'):
        raise RuntimeError('Final code receipt differs')
    contacts = read(B/'REVIEW_ASSETS/CONTACTS.json')
    for start in (35,123):
        entry = next(r for r in contacts['rows'] if r['start'] == start)
        frame = next(x for x in entry['frames'] if x['horizon'] == 0)
        if sha(RESULT/('initial/s%03d_frame_00.png' % start)) != frame['sha256']:
            raise RuntimeError('Initial frame differs from stage-B source')
    result = dict(status='passed', transfer_sha256=sha(RESULT/'TRANSFER_SHA.json'),
        done_sha256=sha(RESULT/'DONE.json'), code_sha256=sha(ROOT/'CODE.sha256'),
        imported_files=len(transfer['files']), checkpoints=records, training_updates=360,
        samples_per_horizon_per_window=40, windows=[35,123], decoder_and_codec_frozen=True,
        scope='Integrity and two-window coverage only; visual and action quality are separate')
    if output:
        with Path(output).open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',help='Optional new result file')
    args=parser.parse_args()
    main(args.output)

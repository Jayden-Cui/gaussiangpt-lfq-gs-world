"""Check complete base coverage, replay balance and original base order."""
import collections
import importlib.util
from pathlib import Path
import tempfile
import json
from PIL import Image
import sampling
import worker
import coordinator


def main():
    root=Path(__file__).resolve().parent
    p=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    rows=sampling.schedule(p['sweeps'],p['base_sampler_seed'],p['replay_sampler_seed'])
    report=sampling.audit(rows,p['sweeps'])
    _,worker_report=worker.validate_protocol(p)
    assert worker_report==report
    assert report['updates']==2115 and report['total_visits_per_anchor_pair']==55
    assert rows==sampling.schedule(p['sweeps'],p['base_sampler_seed'],p['replay_sampler_seed'])
    path=root.parent/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1/b_sampling.py'
    spec=importlib.util.spec_from_file_location('old_b_sampler',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original=old.schedule(p['sweeps'],p['base_sampler_seed'])
    for sweep in range(1,p['sweeps']+1):
        old_pairs=[(x['start'],x['horizon']) for r in original if r['sweep']==sweep for x in r['batch']]
        new_pairs=[(x['start'],x['horizon']) for r in rows if r['sweep']==sweep and r['kind']=='base' for x in r['batch']]
        assert old_pairs==new_pairs
    changed=[dict(r,batch=[dict(x) for x in r['batch']]) for r in rows]
    replay=next(r for r in changed if r['kind']=='replay')
    replay['batch'][0]['start']=0
    try:sampling.audit(changed,p['sweeps'])
    except RuntimeError:pass
    else:raise RuntimeError('Invalid anchor replay was accepted')
    changed_p=dict(p,quantized=True)
    try:worker.validate_protocol(changed_p)
    except RuntimeError:pass
    else:raise RuntimeError('Quantized pilot protocol was accepted')
    with tempfile.TemporaryDirectory(prefix='anchor_replay_review_') as temporary:
        base=Path(temporary);evaluation=base/'evaluation';evaluation.mkdir()
        source=dict(status='complete',window_predictions=1512,source={'sweep':5},packets_sha256='p')
        replay=dict(status='passed',packets=104,views=208,packets_sha256='p')
        (evaluation/'EVALUATION.json').write_text(json.dumps(source),encoding='utf-8')
        (evaluation/'REPLAY.json').write_text(json.dumps(replay),encoding='utf-8')
        image=Image.new('RGB',(1792,276),'white')
        image.paste((255,0,0),(0,24,448,276));image.paste((0,0,255),(448,24,896,276))
        for start in (0,35,56,84,112,123,140,167):
            folder=evaluation/'snapshots'/('s%03d'%start);folder.mkdir(parents=True)
            for h in (0,3,6,9):image.save(folder/('frame_%02d.png'%h))
        result=coordinator.review_assets(evaluation,base/'review')
        assert result['labels_in_image'] is False and len(result['rows'])==8
        with Image.open(base/'review/s035_contact.png') as contact:
            assert contact.getpixel((0,0))==(255,0,0) and contact.getpixel((448,0))==(0,0,255)
        replay['packets']=103
        (evaluation/'REPLAY.json').write_text(json.dumps(replay),encoding='utf-8')
        try:coordinator.review_assets(evaluation,base/'rejected')
        except RuntimeError:pass
        else:raise RuntimeError('Incomplete replay was accepted')
    print('PASS: balanced full-clip replay, original base order, protocol guards and review-panel/replay checks')


if __name__=='__main__':main()

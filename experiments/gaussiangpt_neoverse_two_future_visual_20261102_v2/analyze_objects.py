"""Compute existing fixed-color object proxies from completed paired-frame PNGs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SOURCE = EXPERIMENTS/'gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1/result/step_00360'
CORE = EXPERIMENTS/'gaussiangpt_neoverse_qformer_support_20261101_v1/core.py'
CONFIG = EXPERIMENTS/'gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT/'result/MANIFEST.json').read_text(encoding='utf8'))
    evaluation = SOURCE/'EVALUATION.json'
    if manifest['source_evaluation_sha256'] != sha(evaluation):
        raise RuntimeError('Visual assets and evaluation differ')
    code = importlib.util.spec_from_file_location('_existing_object_proxy', CORE)
    core = importlib.util.module_from_spec(code)
    code.loader.exec_module(core)
    specs = json.loads(CONFIG.read_text(encoding='utf8'))['object_proxies']
    recorded = {(r['start'],r['horizon']):r['source_sha256'] for r in manifest['frames']}
    summary = {}
    for start in (35,123):
        spec = specs[str(start)]
        entries = {'motion':[], 'static':[]}
        for h in range(1,10):
            path = SOURCE/f's{start:03d}'/f'frame_{h:02d}.png'
            if sha(path) != recorded[(start,h)]:
                raise RuntimeError('Source visual frame changed')
            image = np.asarray(Image.open(path).convert('RGB'))
            if image.shape != (252,2240,3):
                raise ValueError('Unexpected paired image shape')
            pieces = [image[:,448*i:448*(i+1)] for i in range(5)]
            for branch, target, pred in (('motion',pieces[0],pieces[2]),('static',pieces[3],pieces[4])):
                proxy = core.object_proxy(pred,target,spec)
                yy,xx = np.nonzero(core.color_mask(target,spec))
                diagonal = float(np.hypot(xx.max()-xx.min()+1, yy.max()-yy.min()+1)) if len(xx)>=10 else None
                center = proxy['centroid_error_px']
                entries[branch].append({'horizon':h, 'target_visible':proxy['target']['valid'],
                    'prediction_visible':proxy['prediction']['valid'], 'iou':proxy['iou'],
                    'center_error_over_bbox_diagonal':center/diagonal if center is not None and diagonal else None,
                    'axis_error_degrees':proxy['axis_error_degrees']})
        summary[str(start)] = {}
        for branch, rows in entries.items():
            visible = [r for r in rows if r['target_visible']]
            kept = sum(r['prediction_visible'] for r in visible)
            center = sum(r['center_error_over_bbox_diagonal'] is not None and
                         r['center_error_over_bbox_diagonal']<=.1 for r in visible)
            angles = [r['axis_error_degrees'] for r in visible if r['axis_error_degrees'] is not None]
            run = longest = 0
            for r in rows:
                run = run+1 if r['target_visible'] and not r['prediction_visible'] else 0
                longest = max(run,longest)
            summary[str(start)][branch] = {'target_visible_frames':len(visible),'prediction_visible_frames':kept,
                'retention_fraction':kept/len(visible) if visible else None,
                'center_within_10pct_fraction':center/len(visible) if visible else None,
                'median_axis_error_degrees':statistics.median(angles) if angles else None,
                'maximum_consecutive_missing_frames':longest,'frames':rows}
    result = {'status':'complete','source_evaluation_sha256':sha(evaluation),'visual_manifest_sha256':sha(ROOT/'result/MANIFEST.json'),
              'proxy_code_sha256':sha(CORE),'proxy_config_sha256':sha(CONFIG),'summary':summary,
              'scope':'Fixed HSV and ROI image proxy on known training windows; not tracked 3D pose or action success'}
    output = ROOT/'result/OBJECT_PROXY.json'
    with output.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({s:{b:{k:v for k,v in d.items() if k!='frames'} for b,d in branches.items()}
                      for s,branches in summary.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

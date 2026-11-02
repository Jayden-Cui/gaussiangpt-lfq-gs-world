"""Audit existing TRAIN-pool metric receipts; no codec inference or LDM training."""
import hashlib
import json
from pathlib import Path
import sys
import time

from analysis_core import clip_summary, summarize

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
POOL=OUTPUTS/'gaussiangpt_neoverse_fulltrain_pool_20261031_v1'
INDEX=OUTPUTS/'gaussiangpt_neoverse_fullpool_codec_20261031_v1/pool_index.py'
HIST=OUTPUTS/'gaussiangpt_neoverse_multisource_codec_20261031_v1/evaluations/baseline'
ORIGINAL='buildingblock_tower_17_176'
CODEC='a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
MODEL='2dc414eaeb1fc03bfc3a77fe8d7cc607a23502ba48e6675bbc1816bcb304fb41'
PINS={'pool_complete':'8ffe7055cba065a49c233245b106631665dcd710b3ebee2fd08368ab543224df',
      'pool_done':'920a7f7c030733b2ff6d1be62aafb243bf581ea0a9bcc53ba3fe775b3d0f694b',
      'pool_index':'8577a94ad505c863503aae5c1424f3a8b330cae90b67b25f7c8edfde35196f99',
      'historical_evaluation':'68a5b3da1c122c0339f3171e53698b8c54a55dbbc051bf4987bfd0d04b58b88a',
      'historical_targets':'465d4a65070f7b5a00b379eb0f67105552f65f83c17e0063f4974a0e1e7ce16b'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_own_source():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise RuntimeError('CPU test witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Source changed: '+name)


def bound_json(path, records, pool_index):
    item=pool_index.record(path,records)
    if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
        raise RuntimeError('Bound metric source changed: '+str(path))
    return read(path)


def main():
    verify_own_source();began=time.monotonic()
    for name,path in [('pool_complete',POOL/'POOL_COMPLETE.json'),('pool_done',POOL/'queue/DONE.json'),
                      ('pool_index',INDEX),('historical_evaluation',HIST/'EVALUATION.json'),
                      ('historical_targets',HIST/'targets.jsonl')]:
        if sha(path)!=PINS[name]:raise RuntimeError('Pinned source changed: '+name)
    sys.path.insert(0,str(INDEX.parent));import pool_index
    complete,done=read(POOL/'POOL_COMPLETE.json'),read(POOL/'queue/DONE.json')
    if (complete.get('status')!='complete' or done.get('status')!='complete'
            or done.get('pool_complete_sha256')!=PINS['pool_complete']
            or complete.get('training_performed') is not False):
        raise RuntimeError('Training-pool terminal receipt differs')
    clip_meta=complete['clips'];expected=pool_index.EXPECTED
    if (len(clip_meta)!=50 or {x['clip_id'] for x in clip_meta}!=set(expected)
            or any(x['current_role']!='TRAIN' for x in clip_meta)
            or len(set(expected.values()))!=19):
        raise RuntimeError('Expected the pre-existing 50 training clips and 19 groups')
    records=pool_index.table(complete['files'])
    old=read(HIST/'EVALUATION.json')
    if (old.get('status')!='evaluated' or old.get('checkpoint_sha256')!=CODEC
            or old.get('targets_sha256')!=PINS['historical_targets']):
        raise RuntimeError('Historical baseline differs')
    historical=[]
    with (HIST/'targets.jsonl').open(encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            if item.get('clip')==ORIGINAL:historical.append(item)
    original=clip_summary(historical,ORIGINAL,expected[ORIGINAL],historical=True)
    if (any(r.get('checkpoint_sha256')!=CODEC for r in historical)
            or abs(original['codec_real']['mae_mean']-old['summary_by_clip'][ORIGINAL]['codec_real']['mae'])>1e-10):
        raise RuntimeError('Original training clip baseline differs')
    items=[original];quality_sources=[]
    for clip,group in sorted(expected.items()):
        if clip==ORIGINAL:continue
        directory,_,_,quality_path=pool_index.layout(POOL,clip)
        summary_path=directory/'SUMMARY.json';inputs_path=directory/'inputs_hashes.json'
        summary=bound_json(summary_path,records,pool_index)
        inputs=bound_json(inputs_path,records,pool_index)
        quality=bound_json(quality_path,records,pool_index)
        if (summary.get('status')!='complete' or summary.get('clip')!=clip or summary.get('frames')!=177
                or summary.get('codec_frozen') is not True or summary.get('training_performed') is not False
                or summary.get('codec_state_sha256_before_and_after')!=MODEL
                or not any(x.get('sha256')==CODEC for x in inputs.get('static_files',[]))):
            raise RuntimeError('Wrong frozen codec or prepared clip: '+clip)
        frames=quality['frames']
        if any(r.get('source_count')!=112896 or abs(r.get('pts_seconds',-1)-r['frame']/29.97)>1e-9 for r in frames):
            raise RuntimeError('Frame source point count or timing differs: '+clip)
        item=clip_summary(frames,clip,group)
        if abs(item['codec_real']['mae_mean']-summary['quality']['codec_real']['mae'])>2e-6:
            raise RuntimeError('Frame and bound summary differ: '+clip)
        items.append(item)
        quality_sources.append(dict(clip=clip,summary_sha256=sha(summary_path),
                                    quality_sha256=sha(quality_path),inputs_sha256=sha(inputs_path)))
    summary=summarize(items)
    if summary['clips']!=50 or summary['frames']!=8850 or summary['source_groups']!=19:
        raise RuntimeError('Aggregate scope differs')
    result=dict(status='complete',scope='Existing TRAIN pool only; correlated frames and overlapping clips',
        codec_sha256=CODEC,initial_model_digest=MODEL,source_pins=PINS,
        source_code_sha256=sha(ROOT/'CODE.sha256'),summary=summary,clips=sorted(items,key=lambda x:x['clip']),
        bound_quality_sources=quality_sources,original_clip_metrics='previously bound historical evaluation',
        threshold_005='descriptive count only; not a motion acceptance threshold',
        states_and_images_reloaded=False,reconstruction_recomputed=False,gpu_used=False,training_performed=False,
        eval_video_metrics_selected=False,action_model_evaluated=False,
        elapsed_seconds=time.monotonic()-began)
    with (ROOT/'TRAIN_POOL_QUALITY.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='complete',summary=summary,elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()

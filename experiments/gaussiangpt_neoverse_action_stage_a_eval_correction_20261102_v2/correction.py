"""Create a traceable correction without modifying training or old evaluations."""
import copy
import hashlib
import json
from pathlib import Path

from correction_utils import (CONTROLS,SUPPORT_CONTROLS,aggregate,by_pair,
                              corrected_packet_rows,corrected_rows)

ROOT=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def setup():
    config=read(ROOT/'repair_config.json')
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Correction code changed: '+name)
    training=Path(config['training_root']);baseline=Path(config['baseline_root'])
    if sha(training/'CODE.sha256')!=config['training_code_sha256']:raise RuntimeError('Original training code changed')
    for name,key in [('EVALUATION.json','baseline_evaluation_sha256'),('action_codes.pt','baseline_codes_sha256'),
                     ('PACKETS.json','baseline_packets_sha256'),('REPLAY.json','baseline_replay_sha256')]:
        if sha(baseline/name)!=config[key]:raise RuntimeError('Baseline artifact changed: '+name)
    old=read(baseline/'EVALUATION.json');packets=read(baseline/'PACKETS.json');replay=read(baseline/'REPLAY.json')
    if (old['status']!='complete' or old['mode']!='posterior' or old['window_predictions']!=1512
            or old['source']['trained_checkpoint_sha256']!=config['checkpoint_sha256']
            or old['source']['cache_manifest_sha256']!=config['cache_manifest_sha256']
            or old['action_codes']['sha256']!=config['baseline_codes_sha256']):
        raise RuntimeError('Baseline evaluation provenance mismatch')
    if (replay['status']!='passed' or replay['packets']!=104 or replay['views']!=208
            or replay['packets_sha256']!=config['baseline_packets_sha256']
            or packets['source']!=old['source'] or packets['mode']!='posterior'):
        raise RuntimeError('Baseline independent native replay mismatch')
    return config,training,baseline,old,packets,replay


def write_panels(out,stage,baseline,packet_rows,starts):
    """Replace only the aliased third panel by the matching original control."""
    import numpy as np
    import torch
    from PIL import Image
    reference={row['name']:row for row in packet_rows}
    for start in starts:
        old=reference[f's{start:03d}_h09_initial_only']['reference']
        path=baseline/'references'/old['file']
        if path.stat().st_size!=old['bytes'] or sha(path)!=old['sha256']:
            raise RuntimeError('Baseline native reference changed: '+str(path))
        native=torch.load(path,map_location='cpu',weights_only=True)
        if tuple(native.shape)!=(2,252,448,3):raise RuntimeError('Unexpected control image shape')
        dst=out/'snapshots'/f's{start:03d}';dst.mkdir(parents=True)
        for horizon in range(10):
            relative=Path('snapshots')/f's{start:03d}'/f'frame_{horizon:02d}.png'
            with Image.open(stage/relative) as a,Image.open(baseline/relative) as b:
                if a.size!=(1792,276) or b.size!=a.size:raise RuntimeError('Unexpected diagnostic panel shape')
                if horizon==9:
                    pixels=np.asarray(b.crop((896,24,1344,276)))
                    expected=(native[0].numpy().clip(0,1)*255).round().astype(np.uint8)
                    if not np.array_equal(pixels,expected):raise RuntimeError('Initial panel disagrees with native reference')
                result=a.copy();result.paste(b.crop((896,0,1344,276)),(896,0));result.save(dst/f'frame_{horizon:02d}.png')
        for view in (0,1):
            name=f's{start:03d}_comparison_v{view}.png'
            with Image.open(stage/name) as a,Image.open(baseline/name) as b:
                if a.size!=(1792,276) or b.size!=a.size:raise RuntimeError('Unexpected endpoint panel shape')
                pixels=np.asarray(b.crop((896,24,1344,276)))
                expected=(native[view].numpy().clip(0,1)*255).round().astype(np.uint8)
                if not np.array_equal(pixels,expected):raise RuntimeError('Endpoint panel disagrees with native reference')
                result=a.copy();result.paste(b.crop((896,0,1344,276)),(896,0));result.save(out/name)


def correct_one(sweep,c,training,baseline,old,old_packets,baseline_codes):
    import torch
    stage=training/f'sweep_{sweep:02d}/free_z/evaluation'
    result=read(stage/'EVALUATION.json');packets=read(stage/'PACKETS.json');replay=read(stage/'REPLAY.json')
    meta=read(training/f'sweep_{sweep:02d}/MERGED.json')
    merged=training/f'sweep_{sweep:02d}/merged_codes.pt'
    metrics=read(training/f'sweep_{sweep:02d}/free_z/STAGE_METRICS.json')
    if (result['status']!='complete' or result['mode']!='free_z' or result['window_predictions']!=1512
            or result['source']['checkpoint_sha256']!=c['checkpoint_sha256']
            or result['source']['decoder_digest']!=c['decoder_tensor_digest']
            or result['source']['cache_manifest_sha256']!=c['cache_manifest_sha256']
            or result['source']['stage_a_code_sha256']!=c['training_code_sha256']
            or result['source']['merged_codes_sha256']!=sha(merged)
            or sha(merged)!=meta['sha256'] or result['source']['sweep']!=sweep):
        raise RuntimeError('Stage A evaluation source mismatch')
    if (replay['status']!='passed' or replay['packets']!=104 or replay['views']!=208
            or replay['packets_sha256']!=sha(stage/'PACKETS.json')
            or result['packets_sha256']!=sha(stage/'PACKETS.json')
            or packets['mode']!='free_z' or packets['source']!=result['source']):
        raise RuntimeError('Stage A native replay or packets changed')
    code_path=stage/'action_codes.pt'
    if sha(code_path)!=result['action_codes']['sha256']:raise RuntimeError('Saved action codes changed')
    codes=torch.load(code_path,map_location='cpu',weights_only=True)
    if (codes['source']!=result['source'] or codes['starts']!=list(range(168)) or codes['shape']!=[168,9,16]
            or codes['quantized'] or not torch.equal(codes['controls']['initial_only'],baseline_codes['controls']['initial_only'])
            or not torch.equal(codes['controls']['correct'],torch.load(merged,map_location='cpu',weights_only=True)['codes'])):
        raise RuntimeError('Code tensors do not match the bound controls or merged training output')
    expected={(s,h) for s in range(168) for h in range(1,10)}
    rows,worst=corrected_rows(result['rows'],old['rows'],expected,c['target_reference_mae_tolerance'])
    future=[r for r in rows if r['horizon']>0]
    if len(future)!=1512 or {r['frame'] for r in future}!=set(range(1,177)):
        raise RuntimeError('Corrected metric coverage differs')
    summary=aggregate(future,CONTROLS)
    support_rows=[dict(frame=r['frame'],metrics={name:dict(metrics=value) for name,value in r['support'].items()}) for r in future]
    support=aggregate(support_rows,SUPPORT_CONTROLS)
    segments={f'{lo:03d}_{hi:03d}':aggregate([r for r in future if lo<=r['frame']<=hi],CONTROLS)
              for lo,hi in [(1,44),(45,88),(89,132),(133,176)]}
    horizons={str(h):aggregate([r for r in future if r['horizon']==h],CONTROLS)['window_prediction_equal']
              for h in range(1,10)}
    out=ROOT/f'sweep_{sweep:02d}';out.mkdir()
    source_sha=sha(stage/'EVALUATION.json')
    corrected=copy.deepcopy(result)
    corrected.update(rows=rows,metrics=summary,support=support,temporal_segments=segments,
        correction=dict(status='applied',defect='Original free_z initial_only prediction reused optimized-code prediction',
          method='Copy bound 9560 posterior repeated-S0 metrics and support for each identical start/horizon; no new rendering claimed',
          baseline_evaluation_sha256=c['baseline_evaluation_sha256'],baseline_code_sha256=c['baseline_codes_sha256'],
          original_evaluation_sha256=source_sha,original_replay_sha256=sha(stage/'REPLAY.json'),
          same_initial_code_exact=True,same_checkpoint_decoder_and_cache=True,
          max_actual_target_mae_difference=worst,actual_target_tolerance=c['target_reference_mae_tolerance'],
          original_initial_only_packets_and_panels_invalid=True,
          corrected_packets='CORRECTED_PACKETS.json',corrected_panels='snapshots/'))
    # The original packet manifest contains mislabelled repeated-S0 packets.
    corrected['raw_original_packets_sha256']=corrected.pop('packets_sha256')
    corrected['packets_sha256']=None
    write(out/'EVALUATION.json',corrected)
    references=corrected_packet_rows(packets['rows'],old_packets['rows'],c['diagnostic_starts'],stage,baseline)
    write(out/'CORRECTED_PACKETS.json',dict(status='reference_manifest',rows=references,
          stage_replay_sha256=sha(stage/'REPLAY.json'),baseline_replay_sha256=c['baseline_replay_sha256'],
          note='Each referenced packet was independently replayed in its origin; the original stage-A initial-only packets are excluded.'))
    corrected_metrics=copy.deepcopy(metrics)
    corrected_metrics.update(metrics=summary,support=support,by_horizon=horizons,
        correction='initial_only is sourced from the matching independently replayed 9560 baseline; other modes unchanged',
        original_stage_metrics_sha256=sha(training/f'sweep_{sweep:02d}/free_z/STAGE_METRICS.json'))
    write(out/'STAGE_METRICS.json',corrected_metrics)
    write_panels(out,stage,baseline,old_packets['rows'],c['diagnostic_starts'])
    write(out/'STATUS.json',dict(status='complete',sweep=sweep,window_predictions=len(future),
          corrected_initial_only=True,correction_code_sha256=sha(ROOT/'CODE.sha256'),
          corrected_evaluation_sha256=sha(out/'EVALUATION.json'),
          corrected_packet_manifest_sha256=sha(out/'CORRECTED_PACKETS.json'),
          clean_source_vs_free_videos=str(stage/'videos')))
    return corrected_metrics


def main():
    import torch
    c,training,baseline,old,old_packets,old_replay=setup()
    witness=read(ROOT/'CPU_TESTS.json')
    if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256'):
        raise RuntimeError('Correction CPU tests missing or changed')
    old_summary=read(training/'SUMMARY.json');old_status=read(training/'STATUS.json')
    if (old_summary['status']!='complete' or old_status['status']!='complete'
            or old_summary['code_sha256']!=c['training_code_sha256']):
        raise RuntimeError('Original stage A has not completed successfully')
    baseline_codes=torch.load(baseline/'action_codes.pt',map_location='cpu',weights_only=True)
    if baseline_codes['starts']!=list(range(168)) or baseline_codes['shape']!=[168,9,16]:
        raise RuntimeError('Baseline code scope differs')
    results={}
    for sweep in c['sweeps']:
        write(ROOT/'STATUS.json',dict(status='running',phase='correcting_completed_results',sweep=sweep),replace=True)
        results[str(sweep)]=correct_one(sweep,c,training,baseline,old,old_packets,baseline_codes)
    final=results[str(c['sweeps'])]
    table=['# 阶段 A：修正重复初态对照后的结果','',
      '训练数据、码、运动解码器和原有输出均未修改。重复初态的指标与支撑来自同一9560步模型、同一初态和相机的历史独立评估。',
      '原始阶段 A 的 initial_only 数值、图像和对应 GS 包应视为无效；本目录给出修正版本及原生包来源。',
      '','| 轮数 | 自由码运动 MAE | 原 Q-Former 运动 MAE | 重复初态运动 MAE |',
      '|---:|---:|---:|---:|']
    for sweep in c['sweeps']:
        values=results[str(sweep)]['metrics']['unique_frame_equal']
        table.append(f"| {sweep} | {values['correct']['motion_mae']:.6f} | {values['baseline_qformer']['motion_mae']:.6f} | {values['initial_only']['motion_mae']:.6f} |")
    table+=['','需检查主要物体与运动的可视结果后决定阶段 B 或 C；本阶段不等于动作码训练完成。']
    (ROOT/'SUMMARY.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',action_codec_complete=False,manual_visual_review_required=True,
          stages=results,final_sweep=c['sweeps'],stage_a_anchor_proxy_pass=final['preliminary_known_anchor_proxy_pass'],
          correction_code_sha256=sha(ROOT/'CODE.sha256'),original_summary_sha256=sha(training/'SUMMARY.json'),
          scope='Correction of offline free-code fitting on one training video; no AR or cross-source claim'))


if __name__=='__main__':main()

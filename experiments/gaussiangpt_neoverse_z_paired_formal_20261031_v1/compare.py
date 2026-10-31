"""Pair final arm results by physical frame and compute fixed block bootstrap."""
from __future__ import annotations
import json
import random
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image,ImageDraw
from codec_binding import require,sha
from formal_common import verify_sources,write_new
from evaluate import aggregate
from runner_core import candidate_plan


def bootstrap_difference(first,second,*,seed=20261031,block=16,repetitions=2000):
    """first minus second; inputs already averaged within each physical frame."""
    a,b=np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
    require(a.shape==b.shape==(176,) and np.isfinite(a).all() and np.isfinite(b).all(),'Need 176 paired finite frames')
    dif=a-b;rng=random.Random(seed);sample=[]
    for _ in range(repetitions):
        indices=[]
        while len(indices)<len(dif):
            start=rng.randrange(len(dif))
            indices.extend((start+j)%len(dif) for j in range(block))
        sample.append(float(dif[indices[:len(dif)]].mean()))
    return dict(mean=float(dif.mean()),ci95=[float(x) for x in np.quantile(sample,[.025,.975])],
                method='paired circular moving-block bootstrap',block_frames=block,repetitions=repetitions,seed=seed)


def load_arm(root,mode,source,p):
    out=root/'arms'/mode;done=json.loads((out/'TRAIN_DONE.json').read_text())
    require(done['status']=='complete' and done['mode']==mode and done['steps']==800 and done['source']==source
            and done['plan_sha256']==p['plan_sha256'] and sha(out/'step_0800.pt')==done['checkpoint_sha256']
            and sha(out/'train_metrics.jsonl')==done['metrics_sha256'],'Training receipt mismatch')
    train=[json.loads(x) for x in (out/'train_metrics.jsonl').read_text().splitlines()]
    plan=candidate_plan()['batches']
    require(len(train)==800 and all(r['step']==i+1 and r['starts']==plan[i] and r['action_mode']==mode
                                    for i,r in enumerate(train)),'Training step/plan mismatch')
    evroot=out/'evaluation';ev=json.loads((evroot/'EVALUATION.json').read_text())
    replay=json.loads((evroot/'NATIVE_REPLAY.json').read_text())
    require(ev['status']=='complete' and ev['mode']==mode and ev['source']==source and ev['steps']==800
            and ev['checkpoint_sha256']==done['checkpoint_sha256'] and ev['final_codec_digest']==done['final_codec_digest']
            and ev['windows']==168 and ev['future_exposures']==1512 and ev['physical_future_frames']==176
            and ev['latent_replay']['frames']==177 and ev['latent_replay']['failed_frames']==0
            and sha(evroot/'latent_replay.jsonl')==ev['latent_replay']['rows_sha256']
            and sha(evroot/'targets.jsonl')==ev['targets_sha256'],'Evaluation receipt mismatch')
    require(replay['status']=='complete' and replay['replay_passed_all'] is True and replay['packets']==80
            and replay['views']==160 and replay['encoder_or_codec_loaded'] is False
            and replay['producer_pid']!=replay['replay_pid'] and replay['evaluation_sha256']==sha(evroot/'EVALUATION.json')
            and replay['native_manifest_sha256']==sha(evroot/'native_manifest.json')
            and replay['checkpoint_sha256']==done['checkpoint_sha256'],'Independent native replay incomplete')
    rows=[json.loads(x) for x in (evroot/'targets.jsonl').read_text().splitlines()]
    recomputed=aggregate(rows,p['evaluation_controls'])
    require(recomputed==ev['metrics'],'Frame-level metrics do not match raw rows')
    return done,ev,replay


def create_videos(root):
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:return dict(status='unavailable',reason='ffmpeg not found; PNG frames retained')
    folder=root/'comparison_videos';folder.mkdir(exist_ok=False)
    labels=['Real','Continuous Z','Continuous shuffle84','LFQ Z','LFQ shuffle84']
    outputs={}
    for start in (35,123):
        frames=folder/f's{start:03d}_frames';frames.mkdir()
        for h in range(10):
            base=f's{start:03d}_h{h:02d}'
            paths=[root/'arms/continuous16/evaluation/snapshots'/f'{base}_real_v0.png',
                   root/'arms/continuous16/evaluation/snapshots'/f'{base}_correct_v0.png',
                   root/'arms/continuous16/evaluation/snapshots'/f'{base}_shuffle84_v0.png',
                   root/'arms/lfq16/evaluation/snapshots'/f'{base}_correct_v0.png',
                   root/'arms/lfq16/evaluation/snapshots'/f'{base}_shuffle84_v0.png']
            canvas=Image.new('RGB',(448*5,280),'white');draw=ImageDraw.Draw(canvas)
            for i,path in enumerate(paths):
                image=Image.open(path).convert('RGB');require(image.size==(448,252),'Snapshot resolution mismatch')
                draw.text((i*448+4,4),labels[i],fill='black');canvas.paste(image,(i*448,28))
            canvas.save(frames/f'{h:02d}.png')
        video=folder/f's{start:03d}_slow5fps.mp4'
        command=[ffmpeg,'-nostdin','-loglevel','error','-y','-framerate','5','-i',str(frames/'%02d.png'),
                 '-c:v','libx264','-pix_fmt','yuv420p',str(video)]
        try:
            subprocess.run(command,check=True,timeout=180)
        except (subprocess.SubprocessError,OSError) as error:
            return dict(status='failed',reason=str(error),png_frames_retained=True,completed=outputs)
        outputs[str(start)]=dict(path=str(video.relative_to(root)),sha256=sha(video),frames=10,
                                 playback_fps=5,source_fps_about=29.97)
    return dict(status='complete',videos=outputs,playback_is_slowed_down=True)


def main():
    root=Path(__file__).resolve().parent;p,source=verify_sources(root)
    arms={mode:load_arm(root,mode,source,p) for mode in p['arms']}
    a,b=arms['continuous16'][0],arms['lfq16'][0]
    require(a['initial_rng']==b['initial_rng'] and a['plan_sha256']==b['plan_sha256'],
            'The two arms did not share RNG/plan')
    metrics=('mae','hand_activity_mae','motion_mae','object_box_mae')
    comparison={}
    for key in metrics:
        c=arms['continuous16'][1]['metrics']['correct']['by_physical_frame']
        l=arms['lfq16'][1]['metrics']['correct']['by_physical_frame']
        comparison[key]=bootstrap_difference([l[str(i)][key] for i in range(1,177)],
                                               [c[str(i)][key] for i in range(1,177)])
    controls={}
    for mode in p['arms']:
        controls[mode]={};values=arms[mode][1]['metrics'];correct=values['correct']['by_physical_frame']
        for control in ('fixed_bundle','fixed_single','shuffle42','shuffle84','shuffle126','blinded_posterior','adapted_held'):
            other=values[control]['by_physical_frame']
            controls[mode][control]={k:bootstrap_difference([other[str(i)][k] for i in range(1,177)],
                                                           [correct[str(i)][k] for i in range(1,177)]) for k in metrics}
    videos=create_videos(root)
    write_new(root/'SUMMARY.json',dict(status='complete',source=source,steps_per_arm=800,
        physical_future_frames=176,future_exposures_per_arm=1512,scope=p['scope'],
        continuous_vs_lfq_mae= comparison,comparison_sign='LFQ minus continuous; positive means lower continuous error',
        correct_vs_control_mae=controls,control_sign='control minus correct; positive favors correct code',
        metrics={mode:arms[mode][1]['metrics'] for mode in p['arms']},
        action_statistics={mode:arms[mode][1]['action_statistics'] for mode in p['arms']},
        checkpoints={mode:arms[mode][0]['checkpoint_sha256'] for mode in p['arms']},
        native_replay={mode:arms[mode][2] for mode in p['arms']},videos=videos,
        visual_review='pending',action_success_claimed=False,goal_complete=False))
    lines=['# 连续 Z 与 LFQ 对照汇总','','两臂各800步；下面误差先在同一物理帧内平均，再对176个不同未来帧等权。',
           '', '|指标|连续 Z|LFQ|LFQ−连续 Z（95%区间）|','|---|---:|---:|---:|']
    for key in metrics:
        c=arms['continuous16'][1]['metrics']['correct']['frame_uniform'][key]
        l=arms['lfq16'][1]['metrics']['correct']['frame_uniform'][key]
        d=comparison[key];lines.append(f'|{key}|{c:.6f}|{l:.6f}|{d["mean"]:.6f} [{d["ci95"][0]:.6f}, {d["ci95"][1]:.6f}]|')
    lines += ['','区间使用16帧循环块、2000次配对bootstrap；单一训练片段的后验重建不证明AR预测或跨来源泛化。',
              '最终codec潜状态重放及关键窗口native GS独立重放均通过；动作画面仍待人工或模型视觉复核。']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__': main()

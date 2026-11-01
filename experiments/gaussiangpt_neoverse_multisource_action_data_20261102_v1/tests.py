"""Validate all source files and prevent evaluation-video training access."""
import hashlib
import json
from pathlib import Path
import time
import torch

from source_store import SourceStore,TRAIN,EVAL,CODEC_SHA,sha

ROOT=Path(__file__).resolve().parent


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Loader code changed')
    store=SourceStore();counts={}
    for clip in (*TRAIN,*EVAL):
        for frame in range(177):
            state,raw,row,expected=store.paths(clip,frame)
            if (state.stat().st_size!=row['state_bytes'] or sha(state)!=row['state_sha256']
                    or raw.stat().st_size!=expected['bytes'] or sha(raw)!=expected['sha256']):
                raise RuntimeError('Frame bytes differ: '+str((clip,frame)))
        counts[clip]=177
    for clip in (*TRAIN,*EVAL):
        purpose='train' if clip in TRAIN else 'eval'
        for start in (0,80,167):
            window=store.load_window(clip,start,purpose=purpose)
            if (window['source_frames']!=list(range(start,start+10)) or window['split']!=('TRAIN' if clip in TRAIN else 'EVAL')
                    or any(x['features'].shape[1]!=64 for x in window['states'])):
                raise RuntimeError('Window shape, time or source split differs')
            times=[x['pts'] for x in window['states']]
            if any(abs((b-a)-1/29.97)>1e-6 for a,b in zip(times,times[1:])):
                raise RuntimeError('Not adjacent original video frames')
        store._read_frame.cache_clear()
    try:store.load_window(EVAL[0],0,purpose='train')
    except PermissionError:pass
    else:raise RuntimeError('Evaluation source was accepted for training')
    result=dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),sources=counts,
      verified_state_files=531,verified_original_cache_files=531,full_file_sha256_checked=True,
      sampled_windows_per_source=3,window_frames=10,frame_stride=1,
      continuous_features='float32 [N,64]',camera_shape=[2,4,4],image_shape=[252,448,3],
      codec_sha256=CODEC_SHA,source_level_train_eval_separation=True,
      eval_source_training_read_rejected=True,eval_source_historically_exposed=True,
      elapsed_seconds=time.monotonic()-began,
      scope='Data integrity and loader only; no cross-video action training or evaluation')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: 531 bound continuous states, 531 image/camera caches, adjacent windows and source separation')


if __name__=='__main__':main()

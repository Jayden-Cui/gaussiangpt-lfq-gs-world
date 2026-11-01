"""Bound multi-source state/camera loader with source-level split isolation."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import torch


ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_multisource_codec_20261031_v1')
PINS={'protocol.json':'855e2904f52193498428f9a4ad0f602f3dacf14e54ca33c04b4a456a8639141b',
      'DATA_BINDING.json':'695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715',
      'evaluations/baseline/state_manifest.jsonl':'4eaa25cc6d2759fc7a03ed96bb7a3041c0c3e2a60aeeba616b3e23da45ad963b',
      'evaluations/baseline/EVALUATION.json':'68a5b3da1c122c0339f3171e53698b8c54a55dbbc051bf4987bfd0d04b58b88a'}
CODEC_SHA='a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
TRAIN=('buildingblock_tower_17_176','buildingblock_tower_10_176')
EVAL=('buildingblock_tower_14_0',)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class SourceStore:
    def __init__(self):
        for relative,expected in PINS.items():
            if sha(ROOT/relative)!=expected:raise RuntimeError('Pinned source metadata changed: '+relative)
        protocol=read(ROOT/'protocol.json');binding=read(ROOT/'DATA_BINDING.json')
        evaluation=read(ROOT/'evaluations/baseline/EVALUATION.json')
        if (protocol['data']['train_clips']!=list(TRAIN) or protocol['data']['eval_only_clips']!=list(EVAL)
                or binding['initial_codec_sha256']!=CODEC_SHA or evaluation['checkpoint_sha256']!=CODEC_SHA
                or evaluation['frames']!=531 or not evaluation['all_frames_freshly_encoded']
                or binding['state_quantization']!='Identity; continuous float32 Nx64'):
            raise RuntimeError('Unexpected source split, codec or quantization')
        self.roots={k:Path(v) for k,v in protocol['data']['roots'].items()}
        self.binding={item['path']:item for item in binding['files']}
        if len(self.binding)!=len(binding['files']):raise RuntimeError('Duplicate source binding path')
        self.rows={}
        with (ROOT/'evaluations/baseline/state_manifest.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);key=(row['clip'],row['frame'])
                if key in self.rows:raise RuntimeError('Duplicate source frame')
                self.rows[key]=row
        expected={(clip,frame) for clip in (*TRAIN,*EVAL) for frame in range(177)}
        if set(self.rows)!=expected:raise RuntimeError('Incomplete per-source states')
        for clip in (*TRAIN,*EVAL):
            mapping=read(self.roots[clip]/'mappings'/f'{clip}.json')
            if mapping['source_frame_indices']!=list(range(177)) or len(mapping['source_pts_seconds'])!=177:
                raise RuntimeError('Source frame mapping differs: '+clip)
            for frame in range(177):
                row=self.rows[(clip,frame)]
                split='TRAIN' if clip in TRAIN else 'EVAL'
                if (row['split']!=split or row['checkpoint_sha256']!=CODEC_SHA or not row['fresh_encode']
                        or row['shape'][1]!=64 or abs(row['pts']-mapping['source_pts_seconds'][frame])>1e-7):
                    raise RuntimeError('Frame/source binding mismatch: '+str((clip,frame)))

    def paths(self,clip,frame):
        if (clip,frame) not in self.rows:raise ValueError('Unknown source or frame')
        row=self.rows[(clip,frame)]
        state=ROOT/'evaluations/baseline'/row['state_path']
        raw=self.roots[clip]/'cache'/clip/f'{frame:03d}.pt'
        expected=self.binding.get(str(raw))
        if expected is None or expected['sha256']!=row['source_cache_sha256']:
            raise RuntimeError('Teacher cache binding mismatch')
        return state,raw,row,expected

    def authorize(self,clip,purpose):
        if purpose not in ('train','eval'):raise ValueError('Specify train or eval')
        if clip not in (*TRAIN,*EVAL):raise ValueError('Unknown clip')
        if purpose=='train' and clip not in TRAIN:
            raise PermissionError('Evaluation-source frames may not enter training')

    @lru_cache(maxsize=32)
    def _read_frame(self,clip,frame):
        state_path,raw_path,row,expected=self.paths(clip,frame)
        if (state_path.stat().st_size!=row['state_bytes'] or sha(state_path)!=row['state_sha256']
                or raw_path.stat().st_size!=expected['bytes'] or sha(raw_path)!=expected['sha256']):
            raise RuntimeError('State or teacher cache changed: '+str((clip,frame)))
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        raw=torch.load(raw_path,map_location='cpu',weights_only=True)
        coords,features=state['coordinates'],state['features']
        if (coords.dtype!=torch.int32 or coords.ndim!=2 or coords.shape[1]!=3
                or features.dtype!=torch.float32 or tuple(features.shape)!=(len(coords),64)
                or not bool(torch.isfinite(features).all()) or list(features.shape)!=row['shape']):
            raise RuntimeError('Invalid continuous sparse state')
        timestamp=raw['metrics']['pts_seconds']
        if (raw['clip']!=clip or raw['frame']!=frame or abs(timestamp-row['pts'])>1e-7
                or abs(raw.get('pts',timestamp)-row['pts'])>1e-7
                or raw['n_input']!=112896 or tuple(raw['cameras'].shape)!=(2,4,4)
                or tuple(raw['intrinsics'].shape)!=(4,) or tuple(raw['real'].shape)!=(252,448,3)
                or tuple(raw['targets'].shape)!=(2,252,448,3)
                or not raw['normalization']['fixed_for_all_frames']):
            raise RuntimeError('Frame image, camera or normalization mismatch')
        for key in ('cameras','intrinsics','real','targets'):
            if not bool(torch.isfinite(raw[key]).all()):raise RuntimeError('Nonfinite '+key)
        return state,raw,row

    def load_frame(self,clip,frame,*,purpose,device='cpu'):
        self.authorize(clip,purpose)
        state,raw,row=self._read_frame(clip,frame)
        return dict(state=dict(coordinates=state['coordinates'].to(device),features=state['features'].to(device),pts=row['pts']),
            sample=dict(record=dict(cameras=raw['cameras'].to(device),intrinsics=raw['intrinsics'].to(device),
                                    real=raw['real'].to(device),near=raw['near'],far=raw['far']),
                        teacher_targets=raw['targets'].to(device)),
            clip=clip,frame=frame,split=row['split'],state_sha256=row['state_sha256'],
            source_cache_sha256=row['source_cache_sha256'])

    def load_window(self,clip,start,*,purpose,device='cpu'):
        self.authorize(clip,purpose)
        if type(start)!=int or not 0<=start<=167:raise ValueError('Expected full adjacent ten-frame window')
        values=[self.load_frame(clip,frame,purpose=purpose,device=device) for frame in range(start,start+10)]
        return dict(clip=clip,start=start,split=values[0]['split'],
                    states=[x['state'] for x in values],samples=[x['sample'] for x in values],
                    source_frames=[x['frame'] for x in values],
                    fps=29.97,frame_stride=1,quantized=False)

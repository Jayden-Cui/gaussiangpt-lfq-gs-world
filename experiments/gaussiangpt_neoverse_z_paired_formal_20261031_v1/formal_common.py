"""Shared bound sources, initialization and safety checks for the paired run."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from codec_binding import CHECKPOINT_SHA, require, sha
from model_adapter import load_pinned, attach_adapters
from runtime_bridge import BoundNativeBridge, _bound_module
from runtime_data import BoundData
from runner_core import candidate_plan

BASE=Path('/data/CJH/AGS-V6Direct/outputs')
PREVIOUS=BASE/'gaussiangpt_neoverse_z_ablation_20261031_v1'
ARM=BASE/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline'
NATIVE=BASE/'gaussiangpt_neoverse_native_codec_replay_20261031_v1/results'
TEMPORAL=BASE/'gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py'
TEACHER=BASE/'gaussiangpt_neoverse_gs_supervision_20261031_v10b/loss_step.py'
NULL_BIAS=BASE/'gaussiangpt_neoverse_step_acceleration_20261031_v2/null_key_bias.py'
EXPECTED_PREFLIGHT='2afbcec95f1d5a332e9690f1b90bed74f41926ec2952b64fba58c47526cd9a14'
EXPECTED_INIT='6085c5cd60062c3006f2f08ef81d54ee79734d9d35d1c7daa1f42051e40ae3a1'
EXPECTED_PREFLIGHT_MANIFEST='338dc1d5e102f1827a5812ee365206cf75a5c3051f2497b2abfdff42c71178bd'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def atomic_checkpoint(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    require(not path.exists() and not temporary.exists(),'Checkpoint output already exists')
    with temporary.open('xb') as stream:
        torch.save(value,stream)
        stream.flush()
    temporary.replace(path)
    return sha(path)


def model_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_sources(root):
    root=Path(root).resolve()
    manifest=root/'CODE.sha256'
    rows=[line.split('  ',1) for line in manifest.read_text().splitlines() if line]
    require(rows and len({name for _,name in rows})==len(rows),'Malformed code manifest')
    for digest,name in rows:
        require(Path(name).name==name and sha(root/name)==digest,'Formal code drift: '+name)
    require(sha(PREVIOUS/'PREFLIGHT.sha256')==EXPECTED_PREFLIGHT_MANIFEST,'Preflight manifest changed')
    previous={name:digest for digest,name in (line.split('  ',1) for line in (PREVIOUS/'PREFLIGHT.sha256').read_text().splitlines() if line)}
    for name in ('action_bottleneck.py','initial_norm.py','model_adapter.py','codec_binding.py','teacher_binding.py',
                 'runtime_data.py','runtime_bridge.py','paired_step.py','runner_core.py'):
        require(sha(root/name)==previous[name],'Preflight-tested implementation changed: '+name)
    require(sha(PREVIOUS/'preflight_v1/RESULT.json')==EXPECTED_PREFLIGHT,'Preflight result changed')
    result=read(PREVIOUS/'preflight_v1/RESULT.json')
    require(result['status']=='passed' and result['formal_training'] is False and result['source_frames']==177
            and result['batch_size']==4 and result['frames_per_window']==10 and result['frame_gap']==1
            and result['checkpoint_sha256']==CHECKPOINT_SHA and result['discarded_updates_per_arm']==2
            and all(result['arms'][mode]['discarded_updates']==2 and result['arms'][mode]['formal_updates']==0
                    for mode in ('continuous16','lfq16')), 'Matched GPU preflight did not pass')
    checkpoint=PREVIOUS/'preflight_v1/common_init_preflight.pt'
    require(sha(checkpoint)==EXPECTED_INIT and result['common_preflight_checkpoint_sha256']==EXPECTED_INIT,
            'Preflight common initialization changed')
    protocol=read(root/'protocol.json')
    required=dict(schema='paired_continuous_lfq_formal_v1',clip='buildingblock_tower_17_176',source_frames=177,
        frame_gap=1,window_frames=10,window_starts=168,batch_size=4,steps_per_arm=800,
        arms=['continuous16','lfq16'],checkpoints=[200,400,600,800],
        plan_sha256=candidate_plan()['plan_sha256'],preflight_result_sha256=EXPECTED_PREFLIGHT,
        common_checkpoint_sha256=EXPECTED_INIT,baseline_codec_sha256=CHECKPOINT_SHA,
        common_model_sha256=result['initial_model_sha256'],s_ref=result['calibration']['s_ref'],
        decoder_precision='float64',resolution_hw=[252,448],minimum_free_gpu_gib=32,allocator_limit_gib=24)
    require(all(protocol.get(k)==v and type(protocol.get(k)) is type(v) for k,v in required.items()),'Formal protocol drift')
    return protocol,dict(code_manifest_sha256=sha(manifest),protocol_sha256=sha(root/'protocol.json'),
        preflight_sha256=EXPECTED_PREFLIGHT,common_init_sha256=EXPECTED_INIT,baseline_codec_sha256=CHECKPOINT_SHA,
        native_replay_sha256=result['native_replay_sha256'],state_manifest_sha256=result['state_manifest_sha256'],
        teacher_assets_sha256=result['teacher_assets_sha256'],common_model_sha256=result['initial_model_sha256'])


def gpu_setup(p):
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one GPU')
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    require(free>=p['minimum_free_gpu_gib']*2**30,'Insufficient free GPU memory; other processes remain untouched')
    torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
    torch.cuda.reset_peak_memory_stats()
    return {'free_gib':free/2**30,'total_gib':total/2**30,'allocator_cap_gib':p['allocator_limit_gib']}


def guard(p,started,phase):
    limit=p[phase+'_max_seconds']
    require(time.monotonic()-started<limit,phase+' time budget exceeded')
    require(torch.cuda.max_memory_reserved()<=p['allocator_limit_gib']*2**30, 'Allocator budget exceeded')


def build(p,source,mode,*,training):
    require(mode in p['arms'],'Unknown arm')
    api=load_pinned(BASE)
    temporal=_bound_module('_z_formal_temporal',TEMPORAL,'6a1de737b3e87d88c6678352236258c031f9e8d66f17d79a04e2b944560562ef')
    teacher=_bound_module('_z_formal_teacher',TEACHER,'ce304e2d861e303f1780c1267cf05146e7d5d4a13f8b30f54b28368608ea16ad')
    null_bias=_bound_module('_z_formal_null_bias',NULL_BIAS,'7ef41a420de732a11f6938303f18bfc69cc7655c85b2ad870a1105a1585f83fb')
    data=BoundData(ARM,NATIVE,device='cuda')
    require(data.native_evidence['native_replay_sha256']==source['native_replay_sha256']
            and data.state_evidence['state_manifest_sha256']==source['state_manifest_sha256']
            and data.teacher_evidence['teacher_assets_sha256']==source['teacher_assets_sha256'],'Data binding changed')
    model=api.AnchoredLDM(blocks=4,checkpoint_blocks=True,residual_scale=3.,inv_temperature=1.,code_mode='posterior').cuda().eval()
    api.attach_spatial_action(model,True)
    attach_adapters(model,mode=mode,s_ref=p['s_ref'])
    init=torch.load(PREVIOUS/'preflight_v1/common_init_preflight.pt',map_location='cpu',weights_only=True)
    require(init['schema']=='preflight_common_init_v1' and init['formal_training'] is False
            and init['checkpoint_sha256']==CHECKPOINT_SHA and init['common_model_sha256']==source['common_model_sha256']
            and init['calibration']['s_ref']==p['s_ref'],'Common initialization identity differs')
    model.load_state_dict(init['model'],strict=True);del init
    require(model_digest(model)==source['common_model_sha256'],'Model differs from common preflight initialization')
    bridge=BoundNativeBridge(partial=True,device='cuda')
    expected=read(PREVIOUS/'preflight_v1/RESULT.json')['initial_codec_digest']
    require(bridge.precision.report['mode_state_digest']==expected,'Native codec initialization differs')
    model.train(training);bridge.train(training)
    return dict(api=api,temporal=temporal,teacher=teacher,null_bias=null_bias,data=data,model=model,bridge=bridge)


def seed_common(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    return dict(torch_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                torch_cuda_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())

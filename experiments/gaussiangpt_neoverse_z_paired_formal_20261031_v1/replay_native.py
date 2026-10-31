"""Independent post-training replay of saved GS/SH/cameras; no model loaded."""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from native_packet import require,sha,read_checked,write_new_json

COMMON=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')
COMMON_SHA='38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'
CONTROLS=('correct','shuffle84','fixed_bundle','decoded_actual_target_state')


def main(mode, smoke=False):
    root=Path(__file__).resolve().parent
    out=root/'integration_smoke'/mode if smoke else root/'arms'/mode/'evaluation'
    started=time.monotonic()
    try:
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Model already loaded')
        require(sha(COMMON)==COMMON_SHA,'Renderer source changed')
        evaluation=json.loads((out/'EVALUATION.json').read_text())
        require(evaluation['status']=='complete' and evaluation['mode']==mode
                and sha(out/'native_manifest.json')==evaluation['native_manifest_sha256'],'Evaluation manifest mismatch')
        manifest=json.loads((out/'native_manifest.json').read_text())
        rows=manifest['packets']
        expected=({(s,9,'correct') for s in (35,123)} if smoke else
                  {(s,h,c) for s in (35,123) for h in range(10) for c in CONTROLS})
        require(manifest['schema']=='z_native_packet_manifest_v1' and len(rows)==len(expected)
                and {(r['start'],r['horizon'],r['control']) for r in rows}==expected
                and manifest['checkpoint_sha256']==evaluation['checkpoint_sha256']
                and manifest['final_codec_digest']==evaluation['final_codec_digest'],'Native coverage/source mismatch')
        spec=importlib.util.spec_from_file_location('_z_native_replay_common',COMMON)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one GPU')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info();require(free>=16*2**30,'Insufficient actual free GPU memory')
        torch.cuda.set_per_process_memory_fraction(6*2**30/total);torch.cuda.reset_peak_memory_stats()
        results=[];failed=[]
        with (out/'native_replay.jsonl').open('x',encoding='utf8',buffering=1) as log,torch.no_grad():
            for row in rows:
                require(time.monotonic()-started<3500,'Native replay time budget exceeded')
                name=row['packet_file'];ref_name=row['reference_file']
                require(Path(name).name==name and Path(ref_name).name==ref_name,'Unexpected packet path')
                raw=read_checked(out/'native_packets'/name,row['packet']['bytes'],row['packet']['sha256'])
                packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                keys=('start','horizon','frame','control','mode','pts','normalization','model_checkpoint_sha256',
                      'final_codec_digest','target_state_reference','posterior_reconstruction')
                require(packet['schema']=='z_action_native_packet_v1' and all(packet[k]==row[k] for k in keys)
                        and packet['model_checkpoint_sha256']==manifest['checkpoint_sha256']
                        and packet['final_codec_digest']==manifest['final_codec_digest']
                        and packet['frame']==packet['start']+packet['horizon'] and packet['resolution_hw']==[252,448],
                        'Native packet metadata differs')
                gs,sh,cameras,intrinsics=packet['gs'],packet['sh'],packet['cameras'],packet['intrinsics']
                for value,shape in ((gs,(row['gs_count'],14)),(sh,(row['gs_count'],4,3)),
                                    (cameras,(2,4,4)),(intrinsics,(4,))):
                    require(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and tuple(value.shape)==shape
                            and bool(torch.isfinite(value).all()),'Malformed native tensor')
                ref_bytes=read_checked(out/'native_refs'/ref_name,row['reference']['bytes'],row['reference']['sha256'])
                reference=torch.load(io.BytesIO(ref_bytes),map_location='cpu',weights_only=True)
                require(reference.dtype==torch.float32 and tuple(reference.shape)==(2,252,448,3)
                        and bool(torch.isfinite(reference).all()),'Malformed reference')
                gs,sh,cameras,intrinsics,reference=gs.cuda(),sh.cuda(),cameras.cuda(),intrinsics.cuda(),reference.cuda()
                views=[]
                for view in range(2):
                    image=common.render(gs,cameras[view],intrinsics,packet['near'],packet['far'],sh)[0].clamp(0,1)
                    diff=image-reference[view];mae=float(diff.abs().mean());mse=float(diff.square().mean())
                    psnr=-10*math.log10(max(mse,1e-12))
                    views.append(dict(view=view,mae=mae,psnr=psnr,passed=mae<=1e-5 and psnr>=60))
                item=dict(start=row['start'],horizon=row['horizon'],control=row['control'],views=views,
                          passed=all(v['passed'] for v in views))
                if not item['passed']: failed.append([row['start'],row['horizon'],row['control']])
                results.append(item);log.write(json.dumps(item,allow_nan=False)+'\n')
                del gs,sh,cameras,intrinsics,reference,packet,image,diff
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Replay imported a model')
        write_new_json(out/'NATIVE_REPLAY.json',dict(status='complete' if not failed else 'threshold_failed',mode=mode,
            producer_pid=json.loads((out/'STARTED.json').read_text())['pid'],replay_pid=os.getpid(),
            encoder_or_codec_loaded=False,packets=len(rows),views=2*len(rows),failed=failed,replay_passed_all=not failed,
            diagnostic_smoke=smoke,
            max_mae=max(v['mae'] for row in results for v in row['views']),
            min_psnr=min(v['psnr'] for row in results for v in row['views']),
            evaluation_sha256=sha(out/'EVALUATION.json'),native_manifest_sha256=sha(out/'native_manifest.json'),
            rows_sha256=sha(out/'native_replay.jsonl'),checkpoint_sha256=manifest['checkpoint_sha256'],
            final_codec_digest=manifest['final_codec_digest'],elapsed_seconds=time.monotonic()-started))
        if failed: raise SystemExit(2)
    except Exception as error:
        write_new_json(out/'NATIVE_REPLAY_FAILED.json',dict(status='failed',mode=mode,error=repr(error),
                                                            traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['continuous16','lfq16'])
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    main(args.mode,args.smoke)

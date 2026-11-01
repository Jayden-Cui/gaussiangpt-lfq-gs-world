"""Cold native-GS replay: imports the renderer but never a state/motion codec."""
from __future__ import annotations
import argparse
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from io_utils import ROOT, read, write, sha, verify

COMMON = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')
COMMON_SHA = '38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'


def require(ok,message):
    if not ok: raise RuntimeError(message)


def checked_load(root,entry):
    path = root/entry['file']
    require(Path(entry['file']).name==entry['file'],'Invalid artifact path')
    require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Artifact bytes changed')
    return torch.load(io.BytesIO(path.read_bytes()),map_location='cpu',weights_only=True)


def main(start):
    p=verify();out=ROOT/f's{start:03d}';began=time.monotonic()
    try:
        result=read(out/'RESULT.json');manifest=read(out/'PACKETS.json')
        require(start in p['starts'] and result['status']=='complete' and result['codec_frozen'], 'Fit incomplete')
        require(sha(out/'PACKETS.json')==result['packets_sha256'],'Manifest mismatch')
        require(manifest['provenance']==result['provenance'],'Provenance mismatch')
        require(len(manifest['rows'])==5 and {r['name'] for r in manifest['rows']}==
            {'held_initial','projected_initial_features','decoded_actual_target_state','bounded_final','unbounded_final'},
            'Packet coverage mismatch')
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Codec loaded during replay')
        require(sha(COMMON)==COMMON_SHA,'Renderer changed')
        spec=importlib.util.spec_from_file_location('_free_feature_replay_renderer',COMMON)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one GPU')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info()
        require(free>=8*2**30,'Replay GPU memory unavailable')
        torch.cuda.set_per_process_memory_fraction(6*2**30/total)
        rows=[]
        with torch.no_grad():
            for row in manifest['rows']:
                require(time.monotonic()-began<p['replay_max_seconds'],'Replay time budget exceeded')
                packet=checked_load(out,row['packet']);reference=checked_load(out,row['reference'])
                require(packet['schema']=='free_feature_native_packet_v1' and packet['provenance']==manifest['provenance']
                        and packet['name']==row['name'] and packet['start']==start and packet['frame']==start+9,
                        'Packet metadata mismatch')
                shapes={'gs':(row['gs_count'],14),'sh':(row['gs_count'],4,3),'cameras':(2,4,4),'intrinsics':(4,)}
                for key,shape in shapes.items():
                    t=packet[key]
                    require(t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'Bad '+key)
                require(reference.dtype==torch.float32 and tuple(reference.shape)==(2,252,448,3)
                        and bool(torch.isfinite(reference).all()),'Bad reference render')
                values={key:packet[key].cuda() for key in shapes}
                views=[]
                for view in (0,1):
                    image=common.render(values['gs'],values['cameras'][view],values['intrinsics'],
                                        packet['near'],packet['far'],values['sh'])[0].clamp(0,1).cpu()
                    difference=image-reference[view]
                    mae=float(difference.abs().mean())
                    psnr=-10*math.log10(max(float(difference.square().mean()),1e-12))
                    views.append(dict(view=view,mae=mae,psnr=psnr,
                                      passed=mae<=p['replay_mae_max'] and psnr>=p['replay_psnr_min']))
                rows.append(dict(name=row['name'],views=views,passed=all(v['passed'] for v in views)))
                del values,packet,reference,image,difference
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Replay imported codec')
        passed=all(row['passed'] for row in rows)
        write(out/'REPLAY.json',dict(status='passed' if passed else 'failed',start=start,packets=len(rows),
            rows=rows,producer_pid=read(out/'STARTED.json')['pid'],replay_pid=os.getpid(),encoder_or_codec_loaded=False,
            packets_sha256=sha(out/'PACKETS.json'),max_mae=max(v['mae'] for r in rows for v in r['views']),
            min_psnr=min(v['psnr'] for r in rows for v in r['views']),elapsed_seconds=time.monotonic()-began))
        require(passed,'Independent replay threshold failure')
    except BaseException as error:
        write(out/'REPLAY_FAILED.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,required=True)
    main(parser.parse_args().start)

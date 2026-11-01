"""Independent native renderer only; no state/motion model is imported."""
import argparse,importlib.util,io,math,os,time,traceback
from pathlib import Path
import torch
from io_utils import ROOT,read,write,sha,verify

COMMON=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')
COMMON_SHA='38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'


def require(ok,message):
    if not ok:raise RuntimeError(message)

def checked_load(path,entry):
    require(path.name==entry['file'] and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Packet bytes changed')
    return torch.load(io.BytesIO(path.read_bytes()),map_location='cpu',weights_only=True)

def main(mode):
    import sys
    p=verify();out=ROOT/mode/'evaluation';began=time.monotonic()
    try:
        result=read(out/'EVALUATION.json');manifest=read(out/'PACKETS.json')
        require(result['status']=='complete' and sha(out/'PACKETS.json')==result['packets_sha256'],'Evaluation incomplete')
        require(manifest['source']==result['source'] and manifest['mode']==mode,'Source mismatch')
        expected={f's{s:03d}_h{h:02d}_correct' for s in p['starts'] for h in range(10)}
        expected|={f's{s:03d}_h09_{name}' for s in p['starts'] for name in p['packet_endpoint_controls']}
        require(len(manifest['rows'])==len(expected) and {r['name'] for r in manifest['rows']}==expected,'Incomplete packet coverage')
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Codec loaded')
        require(sha(COMMON)==COMMON_SHA,'Renderer changed')
        spec=importlib.util.spec_from_file_location('_learned_support_renderer',COMMON)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one GPU')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info();require(free>=8*2**30,'Insufficient replay memory')
        torch.cuda.set_per_process_memory_fraction(6*2**30/total)
        rows=[]
        with torch.no_grad():
            for row in manifest['rows']:
                require(time.monotonic()-began<p['replay_max_seconds'],'Replay deadline')
                packet=checked_load(out/'packets'/row['packet']['file'],row['packet'])
                reference=checked_load(out/'references'/row['reference']['file'],row['reference'])
                require(packet['schema']=='learned_support_native_v1' and packet['name']==row['name']
                        and packet['source']==manifest['source'] and packet['mode']==mode,'Packet provenance mismatch')
                shapes={'gs':(row['gs_count'],14),'sh':(row['gs_count'],4,3),'cameras':(2,4,4),'intrinsics':(4,)}
                for name,shape in shapes.items():
                    v=packet[name];require(v.dtype==torch.float32 and tuple(v.shape)==shape and bool(torch.isfinite(v).all()),'Bad '+name)
                require(tuple(reference.shape)==(2,252,448,3) and bool(torch.isfinite(reference).all()),'Bad saved image')
                v={k:packet[k].cuda() for k in shapes};views=[]
                for view in (0,1):
                    image=common.render(v['gs'],v['cameras'][view],v['intrinsics'],packet['near'],packet['far'],v['sh'])[0].clamp(0,1).cpu()
                    difference=image-reference[view];mae=float(difference.abs().mean())
                    psnr=-10*math.log10(max(float(difference.square().mean()),1e-12))
                    views.append(dict(view=view,mae=mae,psnr=psnr,passed=mae<=p['replay_mae_max'] and psnr>=p['replay_psnr_min']))
                rows.append(dict(name=row['name'],views=views,passed=all(q['passed'] for q in views)))
                del v,packet,reference,image,difference
        require('codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules,'Replay imported a codec')
        passed=all(r['passed'] for r in rows)
        write(out/'REPLAY.json',dict(status='passed' if passed else 'failed',mode=mode,packets=len(rows),views=2*len(rows),
            rows=rows,producer_pid=read(ROOT/mode/'STARTED.json')['pid'],replay_pid=os.getpid(),encoder_or_codec_loaded=False,
            max_mae=max(v['mae'] for r in rows for v in r['views']),min_psnr=min(v['psnr'] for r in rows for v in r['views']),
            elapsed_seconds=time.monotonic()-began,packets_sha256=sha(out/'PACKETS.json')))
        require(passed,'Native replay mismatch')
    except BaseException as e:
        write(out/'REPLAY_FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True);main(parser.parse_args().mode)

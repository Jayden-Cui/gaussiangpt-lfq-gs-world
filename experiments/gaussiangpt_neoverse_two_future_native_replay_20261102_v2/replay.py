"""Replay saved native GS in a fresh renderer-only process, with no codec/model."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback

from replay_utils import ROOT, expected_names, read, sha, validate_protocol, verify_self, write

RENDERER = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')


def checked_load(torch, folder, entry):
    path = folder/entry['file']
    if path.name != entry['file'] or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise RuntimeError('Saved packet/reference bytes changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def run():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    verify_self()
    cfg=read(ROOT/'protocol.json')
    validate_protocol(cfg)
    out=ROOT/'result'
    began,phase=time.monotonic(),'binding'
    write(out/'REPLAY_STARTED.json',{'pid':os.getpid(),'code_sha256':sha(ROOT/'CODE.sha256'),
          'protocol_sha256':sha(ROOT/'protocol.json')})
    try:
        if 'codec' in sys.modules or 'MinkowskiEngine' in sys.modules:
            raise RuntimeError('A codec/model was already imported')
        produced,manifest=read(out/'PRODUCED.json'),read(out/'PACKETS.json')
        if (produced['status']!='complete' or produced['packets']!=cfg['packets'] or produced['views']!=cfg['views']
                or produced['packets_sha256']!=sha(out/'PACKETS.json') or manifest['status']!='complete'
                or manifest['decoder_future_geometry_input'] is not False
                or manifest['producer_pid']==os.getpid() or len(manifest['rows'])!=36
                or {r['name'] for r in manifest['rows']}!=expected_names(cfg)
                or manifest['source']['checkpoint_sha256']!=cfg['checkpoint_sha256']
                or manifest['source']['decoder_digest']!=cfg['decoder_digest']):
            raise RuntimeError('Produced packets incomplete or incorrect source')
        if sha(RENDERER)!=cfg['native_renderer_sha256']:
            raise RuntimeError('Native renderer source changed')
        spec=importlib.util.spec_from_file_location('_paired_native_renderer',RENDERER)
        renderer=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Expose exactly one GPU for independent renderer')
        free,total=torch.cuda.mem_get_info()
        if free<cfg['minimum_free_gpu_gib']*2**30:
            raise RuntimeError('Insufficient free GPU memory for replay')
        torch.cuda.set_per_process_memory_fraction(cfg['replay_allocator_limit_gib']*2**30/total)
        rows=[]
        phase='native_render'
        with torch.no_grad():
            for item in manifest['rows']:
                if time.monotonic()-began>cfg['replay_timeout_seconds']:
                    raise TimeoutError('Native replay exceeded budget')
                packet=checked_load(torch,out/'packets',item['packet'])
                reference=checked_load(torch,out/'references',item['reference'])
                if (packet['schema']!='two_future_native_packet_v1' or packet['name']!=item['name']
                        or packet['source']!=manifest['source'] or packet['branch']!=item['branch']
                        or packet['start']!=item['start'] or packet['horizon']!=item['horizon']):
                    raise RuntimeError('Packet identity or binding differs')
                shapes={'gs':(item['gs_count'],14),'sh':(item['gs_count'],4,3),'cameras':(2,4,4),'intrinsics':(4,)}
                for name,shape in shapes.items():
                    tensor=packet[name]
                    if tensor.dtype!=torch.float32 or tuple(tensor.shape)!=shape or not bool(torch.isfinite(tensor).all()):
                        raise ValueError('Invalid native packet tensor: '+name)
                if (reference.dtype!=torch.float32 or tuple(reference.shape)!=(2,252,448,3)
                        or not bool(torch.isfinite(reference).all())):
                    raise ValueError('Invalid saved reference')
                device={name:packet[name].cuda() for name in shapes}
                views=[]
                for v in (0,1):
                    image=renderer.render(device['gs'],device['cameras'][v],device['intrinsics'],
                        packet['near'],packet['far'],device['sh'])[0].clamp(0,1).cpu()
                    delta=image-reference[v]
                    mae=float(delta.abs().mean())
                    psnr=-10*math.log10(max(float(delta.square().mean()),1e-12))
                    views.append({'view':v,'mae':mae,'psnr':psnr,
                        'passed':mae<=cfg['replay_mae_max'] and psnr>=cfg['replay_psnr_min']})
                rows.append({'name':item['name'],'start':item['start'],'horizon':item['horizon'],
                             'branch':item['branch'],'views':views,'passed':all(v['passed'] for v in views)})
                del packet,reference,device,image,delta
        if 'codec' in sys.modules or 'MinkowskiEngine' in sys.modules:
            raise RuntimeError('Codec/model was imported in replay process')
        passed=len(rows)==36 and all(r['passed'] for r in rows)
        write(out/'REPLAY.json',{'status':'passed' if passed else 'failed','packets':len(rows),'views':2*len(rows),
            'rows':rows,'packets_sha256':sha(out/'PACKETS.json'),'producer_pid':manifest['producer_pid'],
            'replay_pid':os.getpid(),'encoder_or_codec_loaded':False,'renderer_sha256':sha(RENDERER),
            'max_mae':max(v['mae'] for r in rows for v in r['views']),
            'min_psnr':min(v['psnr'] for r in rows for v in r['views']),
            'elapsed_seconds':time.monotonic()-began,'limits':cfg['limits']})
        if not passed:
            raise RuntimeError('At least one native replay view failed')
        print('PASS: 36 packets and 72 views independently replayed',flush=True)
    except BaseException as error:
        write(out/'REPLAY_FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'no_automatic_retry':True})
        raise


if __name__=='__main__':
    run()

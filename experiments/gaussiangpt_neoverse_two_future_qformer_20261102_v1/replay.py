"""Independently replay final Q outputs using only saved native GS and renderer."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback

from q_utils import ROOT, expected_packet_names, read, sha, validate_protocol, verify_self, write

RENDERER=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')


def checked_load(torch,folder,entry):
    path=folder/entry['file']
    if path.name!=entry['file'] or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
        raise RuntimeError('Q native packet or reference changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def run():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    verify_self();cfg=read(ROOT/'protocol.json');validate_protocol(cfg)
    out=ROOT/'result/step_01194';began=time.monotonic();phase='source_binding'
    write(out/'REPLAY_STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
         'code_sha256':sha(ROOT/'CODE.sha256')})
    try:
        if 'codec' in sys.modules or 'MinkowskiEngine' in sys.modules:
            raise RuntimeError('Codec/model unexpectedly imported')
        evaluation=read(out/'EVALUATION.json');manifest=read(out/'PACKETS.json');source=manifest['source']
        if (evaluation['status']!='complete' or evaluation['step']!=1194
                or evaluation['packets_sha256']!=sha(out/'PACKETS.json')
                or evaluation['source']!=source or manifest['status']!='complete'
                or manifest['quantized'] is not False or manifest['decoder_future_geometry_input'] is not False
                or manifest['producer_pid']==os.getpid() or len(manifest['rows'])!=36
                or {r['name'] for r in manifest['rows']}!=expected_packet_names(cfg)
                or source['teacher_checkpoint_sha256']!=cfg['teacher_checkpoint_sha256']
                or source['teacher_decoder_digest']!=cfg['teacher_decoder_digest']
                or source['posterior_checkpoint_sha256']!=sha(ROOT/'result/checkpoint_01194.pt')):
            raise RuntimeError('Q native packet coverage or source binding differs')
        if sha(RENDERER)!=cfg['native_renderer_sha256']:raise RuntimeError('Native renderer source changed')
        spec=importlib.util.spec_from_file_location('_q_native_renderer',RENDERER)
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Expose exactly one GPU for native replay')
        free,total=torch.cuda.mem_get_info()
        if free<cfg['minimum_free_gpu_gib']*2**30:raise RuntimeError('Insufficient free GPU memory')
        torch.cuda.set_per_process_memory_fraction(6*2**30/total)
        rows=[];phase='native_render'
        with torch.no_grad():
            for item in manifest['rows']:
                if time.monotonic()-began>cfg['maximum_replay_seconds']:
                    raise TimeoutError('Q native replay exceeded budget')
                packet=checked_load(torch,out/'packets',item['packet'])
                reference=checked_load(torch,out/'references',item['reference'])
                if (packet['schema']!='two_future_qformer_native_packet_v1' or packet['name']!=item['name']
                        or packet['source']!=source or packet['branch']!=item['branch']
                        or packet['start']!=item['start'] or packet['horizon']!=item['horizon']):
                    raise RuntimeError('Q packet metadata differs')
                shapes={'gs':(item['gs_count'],14),'sh':(item['gs_count'],4,3),'cameras':(2,4,4),'intrinsics':(4,)}
                for name,shape in shapes.items():
                    t=packet[name]
                    if t.dtype!=torch.float32 or tuple(t.shape)!=shape or not bool(torch.isfinite(t).all()):
                        raise ValueError('Invalid Q packet tensor: '+name)
                if reference.dtype!=torch.float32 or tuple(reference.shape)!=(2,252,448,3) or not bool(torch.isfinite(reference).all()):
                    raise ValueError('Invalid Q image reference')
                values={name:packet[name].cuda() for name in shapes};views=[]
                for v in (0,1):
                    image=renderer.render(values['gs'],values['cameras'][v],values['intrinsics'],
                        packet['near'],packet['far'],values['sh'])[0].clamp(0,1).cpu()
                    diff=image-reference[v];mae=float(diff.abs().mean())
                    psnr=-10*math.log10(max(float(diff.square().mean()),1e-12))
                    views.append({'view':v,'mae':mae,'psnr':psnr,
                        'passed':mae<=cfg['native_replay_mae_max'] and psnr>=cfg['native_replay_psnr_min']})
                rows.append({'name':item['name'],'start':item['start'],'horizon':item['horizon'],
                             'branch':item['branch'],'views':views,'passed':all(v['passed'] for v in views)})
                del packet,reference,values,image,diff
        if 'codec' in sys.modules or 'MinkowskiEngine' in sys.modules:
            raise RuntimeError('Codec/model was imported in Q replay')
        passed=len(rows)==36 and all(r['passed'] for r in rows)
        write(out/'REPLAY.json',{'status':'passed' if passed else 'failed','packets':len(rows),'views':len(rows)*2,
            'rows':rows,'packets_sha256':sha(out/'PACKETS.json'),'producer_pid':manifest['producer_pid'],
            'replay_pid':os.getpid(),'encoder_or_codec_loaded':False,'renderer_sha256':sha(RENDERER),
            'max_mae':max(v['mae'] for r in rows for v in r['views']),
            'min_psnr':min(v['psnr'] for r in rows for v in r['views']),
            'elapsed_seconds':time.monotonic()-began,'scope':cfg['scope']})
        if not passed:raise RuntimeError('At least one Q native replay view failed')
        print('PASS: all Q GS packets and 72 views independently replayed',flush=True)
    except BaseException as error:
        write(out/'REPLAY_FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'no_automatic_retry':True})
        raise


if __name__=='__main__':
    run()

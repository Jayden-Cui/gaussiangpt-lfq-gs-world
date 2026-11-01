"""Export the exact paired model predictions as GS and two-view image packets."""
import os
from pathlib import Path
import sys
import time
import traceback

from common import ROOT, expected_names, read, sha, validate_protocol, verify_self, write

OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
BROOT = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PREVIEW = OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'


def run():
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    verify_self()
    cfg = read(ROOT/'protocol.json')
    validate_protocol(cfg)
    out = ROOT/'result'
    out.mkdir()
    (out/'packets').mkdir()
    (out/'references').mkdir()
    began, phase = time.monotonic(), 'source_binding'
    write(out/'PRODUCER_STARTED.json', {'pid':os.getpid(),'code_sha256':sha(ROOT/'CODE.sha256'),
          'protocol_sha256':sha(ROOT/'protocol.json'),'training_updates':0})
    try:
        source = Path(cfg['source_root'])
        summary_path = source/'result/SUMMARY.json'
        eval_path = source/'result/step_00360/EVALUATION.json'
        checkpoint_path = source/'result/checkpoint_00360.pt'
        for path,key in ((summary_path,'source_summary_sha256'),(eval_path,'source_evaluation_sha256'),
                         (checkpoint_path,'checkpoint_sha256'),(source/'CODE.sha256','source_code_sha256')):
            if sha(path) != cfg[key]:
                raise ValueError('Paired source changed: '+str(path))
        summary, evaluation = read(summary_path), read(eval_path)
        done = read(source/'DONE.json')
        if (summary['status'] != 'complete' or summary['updates'] != 360
                or done['status'] != 'complete' or done['summary_sha256'] != sha(summary_path)
                or summary['checkpoints']['360'] != sha(checkpoint_path)
                or evaluation['status'] != 'complete' or evaluation['step'] != 360
                or evaluation['summary'] != summary['summaries']['360']):
            raise RuntimeError('Paired experiment is incomplete or source files disagree')
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if (checkpoint['schema'] != 'two_window_same_initial_motion_static_free_codes_v1'
                or checkpoint['step'] != 360 or checkpoint['windows'] != cfg['windows']
                or checkpoint['decoder_digest'] != cfg['decoder_digest']
                or checkpoint['protocol_sha256'] != sha(source/'protocol.json')):
            raise RuntimeError('Paired checkpoint metadata differs')
        for key in ('motion_codes','static_codes'):
            x=checkpoint[key]
            if x.dtype!=torch.float32 or tuple(x.shape)!=(2,9,16) or not bool(torch.isfinite(x).all()):
                raise ValueError('Invalid continuous teacher codes')
        sys.path.insert(0,str(PREVIEW))
        import preview
        preview.verify_sources()
        sys.path.insert(0,str(BROOT))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle=load_bound_sources('cpu')
        runtime,store,decoder,fm,p=(bundle[k] for k in ('runtime','store','decoder','fm','p'))
        if (sha(bundle['paths']['cache_manifest']) != cfg['cache_manifest_sha256']
                or p['codec_checkpoint_sha256'] != cfg['codec_sha256']):
            raise RuntimeError('Source state cache or codec differs')
        decoder.load_state_dict(checkpoint['decoder'],strict=True)
        if runtime.digest(decoder)!=cfg['decoder_digest'] or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
            raise RuntimeError('Trained decoder or frozen initial encoder differs')
        phase='gpu_setup'
        setup_gpu_once(dict(p,allocator_limit_gib=cfg['producer_allocator_limit_gib'],
                            minimum_free_gpu_gib=cfg['minimum_free_gpu_gib']))
        decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from runtime import render,cpu,save
        from support_decoder import select_state
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        source_binding={'checkpoint_sha256':sha(checkpoint_path),'decoder_digest':cfg['decoder_digest'],
            'codec_sha256':cfg['codec_sha256'],'cache_manifest_sha256':cfg['cache_manifest_sha256'],
            'source_evaluation_sha256':sha(eval_path),'quantized':False}
        rows=[]
        phase='producing_packets'
        with torch.no_grad():
            for i,start in enumerate(cfg['windows']):
                w=store.get(start)
                predictions={b:decoder(w['condition'],checkpoint[f'{b}_codes'][i].cuda()) for b in cfg['branches']}
                for h in cfg['horizons']:
                    if time.monotonic()-began>cfg['producer_timeout_seconds']:
                        raise TimeoutError('Packet production exceeded budget')
                    sample=store.sample(start,h)
                    raw=sample['record']
                    record={k:raw[k] for k in ('cameras','intrinsics','near','far')}
                    png=source/f'result/step_00360/s{start:03d}/frame_{h:02d}.png'
                    panel=np.asarray(Image.open(png).convert('RGB'))
                    if panel.shape!=(252,2240,3):
                        raise ValueError('Prior paired PNG has an unexpected shape')
                    for branch,index in (('motion',2),('static',4)):
                        pred=predictions[branch]
                        state=select_state(w['condition'].coordinates,pred['features'][h-1],
                                           pred['logits'][h-1],p['max_predicted_latent_nodes'])
                        images,gs,sh=render(bridge,state,record,views=(0,1))
                        now=preview.rgb_array(images[0]).astype(np.int16)
                        old=panel[:,index*448:(index+1)*448].astype(np.int16)
                        maxdiff=int(np.max(np.abs(now-old)))
                        if maxdiff>cfg['reference_png_max_channel_difference']:
                            raise RuntimeError('Recomputed prediction differs from reviewed PNG')
                        name=f's{start:03d}_h{h:02d}_{branch}'
                        packet=dict(schema='two_future_native_packet_v1',name=name,start=start,horizon=h,
                            branch=branch,source=source_binding,gs=cpu(gs),sh=cpu(sh),
                            cameras=cpu(record['cameras']),intrinsics=cpu(record['intrinsics']),
                            near=record['near'],far=record['far'],resolution_hw=[252,448])
                        rows.append({'name':name,'start':start,'horizon':h,'branch':branch,'gs_count':len(gs),
                            'packet':save(out/'packets'/(name+'.pt'),packet),
                            'reference':save(out/'references'/(name+'.pt'),cpu(images)),
                            'reviewed_png_sha256':sha(png),'reviewed_png_max_channel_difference':maxdiff})
        bridge.assert_frozen()
        if runtime.digest(decoder)!=cfg['decoder_digest'] or len(rows)!=36 or {r['name'] for r in rows}!=expected_names(cfg):
            raise RuntimeError('Decoder changed or packet coverage incomplete')
        write(out/'PACKETS.json',{'status':'complete','source':source_binding,'rows':rows,
              'producer_pid':os.getpid(),'decoder_future_geometry_input':False,'training_updates':0})
        write(out/'PRODUCED.json',{'status':'complete','packets':len(rows),'views':2*len(rows),
              'packets_sha256':sha(out/'PACKETS.json'),'decoder_and_codec_frozen':True,
              'reviewed_png_max_channel_difference':max(r['reviewed_png_max_channel_difference'] for r in rows),
              'elapsed_seconds':time.monotonic()-began,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        print('PASS: 36 exact paired GS packets and 72 references',flush=True)
    except BaseException as error:
        write(out/'PRODUCER_FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'training_updates':0,'no_automatic_retry':True})
        raise


if __name__=='__main__':
    run()

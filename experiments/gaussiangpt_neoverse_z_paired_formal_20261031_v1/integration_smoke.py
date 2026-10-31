"""Real-data integration smoke; one discarded update and two windows per mode."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch
from codec_binding import require,sha
from formal_common import verify_sources,gpu_setup,build,seed_common,model_digest,write_new
from paired_step import step_native
from runner_core import make_action_packet,write_action_packet,read_action_packet,select_control_codes
from evaluate import verify_final_latent_replay,render_and_packet
from native_packet import save_new


def main():
    root=Path(__file__).resolve().parent;p,source=verify_sources(root)
    smoke=root/'integration_smoke';smoke.mkdir(exist_ok=False)
    started=time.monotonic();context={'phase':'init'}
    try:
        gpu_setup(p)
        for mode in p['arms']:
            context.update(mode=mode,phase='build')
            out=smoke/mode;out.mkdir()
            (out/'native_packets').mkdir();(out/'native_refs').mkdir()
            write_new(out/'STARTED.json',dict(pid=os.getpid(),diagnostic_smoke=True,formal_updates=0))
            rt=build(p,source,mode,training=True)
            model,bridge,data=rt['model'],rt['bridge'],rt['data']
            optim=torch.optim.AdamW([{'params':[x for x in model.parameters() if x.requires_grad],'lr':p['motion_lr']},
                                    {'params':[x for x in bridge.parameters() if x.requires_grad],'lr':p['native_lr']}],
                                   betas=tuple(p['betas']),weight_decay=p['weight_decay'])
            bias=rt['null_bias'].freeze_null_key_bias(model,optim);seed_common(p['seed'])
            context['phase']='one_discarded_update'
            stats,codes,ids=step_native(model,bridge,optim,data,[116,77,61,94],make_condition=rt['api'].make_condition,
                temporal=rt['temporal'],teacher_loss=rt['teacher'].teacher_loss,aux_multiplier=.1,grad_clip=1.)
            require(stats['action_mode']==mode and tuple(codes.shape)==(4,9,16),'Smoke action shape')
            require((ids is None) if mode=='continuous16' else tuple(ids.shape)==(4,9),'Smoke ID semantics')
            rt['null_bias'].assert_null_key_bias_unchanged(model,optim,bias)
            bridge.assert_frozen();model.eval();bridge.eval()
            model_before,codec_before=model_digest(model),model_digest(bridge.codec)
            ck=out/'diagnostic_checkpoint.pt'
            save_new(ck,dict(schema='integration_smoke_checkpoint_v1',mode=mode,model=model.state_dict(),
                             codec=bridge.codec.state_dict(),optimizer=optim.state_dict(),formal_updates=0))
            saved=torch.load(ck,map_location='cpu',weights_only=True)
            require(saved['mode']==mode and saved['formal_updates']==0,'Smoke checkpoint identity')
            model.load_state_dict(saved['model'],strict=True);bridge.codec.load_state_dict(saved['codec'],strict=True);del saved
            require(model_digest(model)==model_before and model_digest(bridge.codec)==codec_before,'Checkpoint roundtrip differs')
            context['phase']='latent_and_action_roundtrip'
            with torch.no_grad():
                latent=verify_final_latent_replay(rt,p,out,frames=[44,132])
                require(latent['frames']==2 and latent['failed_frames']==0,'Smoke latent replay')
                recipients=[35,123]
                donors=sorted({0,*recipients,*[(s+offset)%168 for s in recipients for offset in (42,84,126)]})
                action=[];actual_ids=[]
                for donor in donors:
                    encoded=model.encode_windows([data.window(donor)])
                    action.append(encoded['action_codes'].cpu())
                    if mode=='lfq16':actual_ids.append(encoded['indices'].cpu())
                    else:require(encoded['indices'] is None,'Continuous ID error')
                action=torch.cat(action)
                ps=dict(model_checkpoint_sha256=sha(ck),codec_binding_sha256=sha(root/'codec_binding.py'),
                        state_manifest_sha256=source['state_manifest_sha256'],adapter_sha256=sha(root/'action_bottleneck.py'))
                packet=make_action_packet(action,mode=mode,starts=donors,s_ref=p['s_ref'],source=ps)
                info=write_action_packet(out/'actions.pt',packet)
                packet=read_action_packet(out/'actions.pt',expected_file_sha256=info['file_sha256'],expected_mode=mode,
                                          expected_source=ps,expected_s_ref=p['s_ref'],expected_starts=donors)
                control={name:select_control_codes(packet,name,recipients,expected_mode=mode,
                         expected_source=ps,expected_s_ref=p['s_ref'])['codes']
                         for name in ('correct','fixed_bundle','fixed_single','shuffle42','shuffle84','shuffle126')}
                rows=[]
                for i,start in enumerate(recipients):
                    condition=[rt['api'].make_condition(data.states[start])]
                    for label in control:
                        predicted=model.decode_action_codes(control[label][i:i+1].cuda(),condition)[0]
                        require(len(predicted)==10 and all(torch.equal(x['coordinates'],data.states[start]['coordinates'])
                                for x in predicted),'Smoke decode shape/support')
                        if label=='correct':
                            h=9;base=f's{start:03d}_h{h:02d}_{label}'
                            meta=dict(start=start,horizon=h,frame=start+h,control=label,mode=mode,
                                pts=float(data.states[start+h]['pts']),normalization=data.normalization,
                                model_checkpoint_sha256=sha(ck),final_codec_digest=codec_before,
                                target_state_reference=False,posterior_reconstruction=True)
                            _,record=render_and_packet(bridge,predicted[h],data.records[start+h],
                                out/'native_packets'/f'{base}.pt',out/'native_refs'/f'{base}.pt',meta)
                            rows.append(record)
                write_new(out/'native_manifest.json',dict(schema='z_native_packet_manifest_v1',mode=mode,
                    checkpoint_sha256=sha(ck),final_codec_digest=codec_before,packets=rows,diagnostic_smoke=True))
                write_new(out/'EVALUATION.json',dict(status='complete',mode=mode,diagnostic_smoke=True,
                    checkpoint_sha256=sha(ck),final_codec_digest=codec_before,
                    native_manifest_sha256=sha(out/'native_manifest.json'),latent_replay=latent,
                    donors=donors,recipient_windows=recipients,formal_updates=0))
            print('INTEGRATION_SMOKE',mode,'DONE',flush=True)
            del rt,model,bridge,data,optim,codes,ids,stats,condition,predicted,encoded
            gc.collect();torch.cuda.empty_cache()
        write_new(smoke/'PRODUCER_DONE.json',dict(status='complete',diagnostic_smoke=True,
                  discarded_updates_per_arm=1,formal_updates=0,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_new(smoke/'FAILED.json',dict(status='failed',context=context,error=repr(error),
                  traceback=traceback.format_exc(),formal_updates=0))
        raise


if __name__=='__main__':main()

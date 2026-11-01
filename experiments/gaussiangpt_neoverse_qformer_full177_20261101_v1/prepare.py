import os,time,traceback
from dataclasses import fields
import torch
from io_utils import ROOT,read,write,sha,verify
from full_model import build_models,frozen_digest
from runtime import save
from posterior import make_memories
from support_decoder import make_condition,target_labels
from data import tree


def main():
    p=verify();began=time.monotonic();out=ROOT/'cache';out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one preparation GPU')
        free,total=torch.cuda.mem_get_info()
        if free<p['minimum_free_gpu_gib']*2**30:raise RuntimeError('Preparation GPU memory unavailable')
        torch.cuda.set_per_process_memory_fraction(8*2**30/total)
        decoder,posterior=build_models(p,'cuda',train_decoder=False);del posterior
        from codec_binding import BaselineStates
        from teacher_binding import BoundTeachers
        binding=BaselineStates(ROOT.parent/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline')
        teachers=BoundTeachers(binding);states=[binding.load(f) for f in range(177)];samples=[]
        for frame in range(177):
            samples.append(teachers.load_record(frame))
            if sha(teachers.native_root/f'native_group_{frame:03d}.npz')!=teachers.assets['records'][frame]['source_sha256']:
                raise RuntimeError('Native input changed')
        std=states[0]['features'].std(0,unbiased=False)
        if not torch.allclose(std,decoder.base.output_std.cpu(),atol=1e-6,rtol=1e-6):raise RuntimeError('Latent units changed')
        frame_entry=save(out/'frames.pt',dict(samples=samples,std=std,state_sources=[row['state_sha256'] for row in binding.rows]))
        del samples
        rows={}
        for start in p['starts']:
            if time.monotonic()-began>p['preparation_max_seconds']:raise RuntimeError('Preparation deadline')
            seq=tree(states[start:start+10],'cuda')
            condition=make_condition(decoder.base,seq[0],p['candidate_radius'],p['max_candidates'])
            targets=target_labels(condition,seq[1:])
            memories=make_memories(decoder.base,seq)
            initial=make_memories(decoder.base,seq,True)
            packet=tree(dict(start=start,states=seq,targets=targets,
                condition={field.name:getattr(condition,field.name) for field in fields(condition)},
                memories=[dict(keys=m.keys,coordinates=m.coordinates) for m in memories],
                initial_memories=[dict(keys=m.keys,coordinates=m.coordinates) for m in initial]),'cpu')
            entry=save(out/f'window_{start:03d}.pt',packet)
            rows[str(start)]=dict(**entry,candidates=len(condition.coordinates),target_frames=list(range(start+1,start+10)),
                target_coverage_complete=True,state_sources=[binding.rows[f]['state_sha256'] for f in range(start,start+10)])
            if start%8==0 or start==167:
                write(ROOT/'PREPARE_STATUS.json',dict(status='running',completed_windows=start+1,total_windows=168,
                    elapsed_seconds=time.monotonic()-began),replace=True)
            del seq,condition,targets,memories,initial,packet
        if sha(ROOT/'COVERAGE_AUDIT.json')!=p['coverage_audit_sha256']:raise RuntimeError('Coverage audit changed')
        write(out/'MANIFEST.json',dict(status='complete',starts=p['starts'],frames=frame_entry,windows=rows,
            protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
            frozen_model_digest=frozen_digest(decoder),candidate_radius=p['candidate_radius'],
            all_source_frames_preserved=True,all_future_cells_covered=True,elapsed_seconds=time.monotonic()-began))
        write(ROOT/'PREPARE_STATUS.json',dict(status='complete',completed_windows=168,total_windows=168,
            cache_manifest_sha256=sha(out/'MANIFEST.json'),elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(ROOT/'PREPARE_FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()

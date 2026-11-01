"""Synthetic CPU test of atomic reviewed-teacher receipt construction."""
import json
from pathlib import Path
import tempfile
import types
import torch

import teacher_builder as builder
from teacher_receipt_v2 import sha,state_digest,validate

ROOT=Path(__file__).resolve().parent


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(2);torch.manual_seed(23)
    model=torch.nn.Linear(4,3).requires_grad_(False).eval()
    runtime=types.SimpleNamespace(digest=lambda m:state_digest(m.state_dict()))
    digest=runtime.digest(model);codec='a'*64;cache='b'*64
    with tempfile.TemporaryDirectory(prefix='codex_teacher_builder_') as folder:
        root=Path(folder);checkpoint=root/'checkpoint.pt';codes=root/'codes.pt'
        torch.save(dict(decoder=model.state_dict()),checkpoint)
        torch.save(dict(starts=list(range(168)),decoder_digest=digest,codes=torch.zeros(168,9,16)),codes)
        rows=[dict(start=s,horizon=h,metrics={'correct':{}},support={'correct':{}})
              for s in range(168) for h in range(1,10)]
        packet_sha='e'*64;evaluation=root/'evaluation.json'
        write(evaluation,dict(status='complete',window_predictions=1512,unique_future_frames=176,
          all_source_frames_preserved=True,decoder_future_geometry_input=False,packets_sha256=packet_sha,rows=rows,
          source=dict(decoder_digest=digest,codec_sha256=codec,cache_manifest_sha256=cache,merged_codes_sha256=sha(codes))))
        replay_rows=[dict(name=f's{s:03d}_h{h:02d}_correct',passed=True)
                     for s in (0,35,56,84,112,123,140,167) for h in range(10)]
        replay_rows += [dict(name=f's{s:03d}_h09_{name}',passed=True)
                        for s in (0,35,56,84,112,123,140,167)
                        for name in ('initial_only','swapped_window_codes','actual_target')]
        replay=root/'replay.json';replay_value=dict(status='passed',packets=104,views=208,packets_sha256=packet_sha,
          encoder_or_codec_loaded=False,rows=replay_rows);write(replay,replay_value)
        review=root/'review.json';write(review,dict(status='reviewed',major_motion_usable=True,
          reviewed_starts=[0,35,56,84,112,123,140,167],evaluation_sha256=sha(evaluation)))
        args=dict(decoder_checkpoint=checkpoint,codes=codes,evaluation=evaluation,native_replay=replay,
                  visual_review=review,codec_sha256=codec,cache_manifest_sha256=cache)
        output=root/'teacher.json';builder.create(model,runtime,output=output,**args)
        if not output.is_file() or output.with_suffix('.json.pending').exists():
            raise RuntimeError('Valid teacher receipt was not finalized atomically')
        validate(output,model,runtime,codec,cache)
        replay_value['rows'][0]['passed']=False;write(replay,replay_value)
        bad=root/'bad.json'
        try:builder.create(model,runtime,output=bad,**args)
        except RuntimeError:pass
        else:raise RuntimeError('Bad replay produced a finalized teacher receipt')
        if bad.exists() or not bad.with_suffix('.json.pending').exists():
            raise RuntimeError('Failed receipt was not preserved as pending evidence')
    result=dict(status='passed',builder_sha256=sha(ROOT/'teacher_builder.py'),test_sha256=sha(ROOT/'builder_tests.py'),
       validator_sha256=sha(ROOT/'teacher_receipt_v2.py'),valid_receipt_finalized=True,
       failed_replay_does_not_create_final_receipt=True,failed_pending_receipt_preserved=True,
       scope='Synthetic CPU provenance test only; no real teacher selected')
    with (ROOT/'BUILDER_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: reviewed teacher receipt construction and failed-replay preservation')


if __name__=='__main__':main()

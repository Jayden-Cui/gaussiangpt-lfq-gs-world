"""CPU tests for balanced schedules and strict teacher-pair provenance."""
import json
from pathlib import Path
import tempfile
import types
import torch

from c_schedule import schedule,audit
from teacher_receipt_v2 import read,sha,state_digest,validate

ROOT=Path(__file__).resolve().parent


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(2)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Stage-C preparation code changed')
    items=schedule(2,2,20261102);sampler=audit(items,2,2)
    if items!=schedule(2,2,20261102) or len(items)!=840:raise RuntimeError('Schedule is not reproducible')
    try:audit(items[:-1],2,2)
    except RuntimeError:pass
    else:raise RuntimeError('Incomplete joint coverage was not rejected')
    with tempfile.TemporaryDirectory(prefix='codex_teacher_receipt_') as folder:
        root=Path(folder);torch.manual_seed(12)
        model=torch.nn.Linear(4,3).requires_grad_(False).eval()
        runtime=types.SimpleNamespace(digest=lambda m:state_digest(m.state_dict()))
        digest=runtime.digest(model);codec='a'*64;cache='b'*64
        checkpoint=root/'checkpoint.pt';torch.save(dict(decoder=model.state_dict()),checkpoint)
        codes=root/'codes.pt';tensor=torch.randn(168,9,16)
        torch.save(dict(starts=list(range(168)),decoder_digest=digest,codes=tensor),codes)
        evaluation=root/'evaluation.json'
        replay_rows=[dict(name=f's{s:03d}_h{h:02d}_correct',passed=True)
                     for s in (0,35,56,84,112,123,140,167) for h in range(10)]
        replay_rows += [dict(name=f's{s:03d}_h09_{name}',passed=True)
                        for s in (0,35,56,84,112,123,140,167)
                        for name in ('initial_only','swapped_window_codes','actual_target')]
        packet_sha='e'*64;replay=root/'replay.json'
        replay_value=dict(status='passed',packets=104,views=208,packets_sha256=packet_sha,
                          encoder_or_codec_loaded=False,rows=replay_rows)
        write(replay,replay_value)
        rows=[dict(start=s,horizon=h,metrics={'correct':{}},support={'correct':{}})
              for s in range(168) for h in range(1,10)]
        write(evaluation,dict(status='complete',window_predictions=1512,unique_future_frames=176,
             all_source_frames_preserved=True,decoder_future_geometry_input=False,packets_sha256=packet_sha,rows=rows,
             source=dict(decoder_digest=digest,codec_sha256=codec,cache_manifest_sha256=cache,
                         merged_codes_sha256=sha(codes))))
        review=root/'visual_review.json'
        review_value=dict(status='reviewed',major_motion_usable=True,
                          reviewed_starts=[0,35,56,84,112,123,140,167],evaluation_sha256=sha(evaluation))
        write(review,review_value)
        receipt=root/'receipt.json'
        value=dict(schema='continuous_action_teacher_v1',status='reviewed',scope='single_train_video_168_windows',
          quantized=False,codec_sha256=codec,cache_manifest_sha256=cache,decoder_tensor_digest=digest,
          decoder_checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),state_key='decoder'),
          codes=dict(path=str(codes),sha256=sha(codes)),
          evaluation=dict(path=str(evaluation),sha256=sha(evaluation)),
          native_replay=dict(path=str(replay),sha256=sha(replay)),
          visual_review=dict(path=str(review),sha256=sha(review)))
        write(receipt,value)
        verified=validate(receipt,model,runtime,codec,cache)
        if not torch.equal(verified['codes'],tensor):raise RuntimeError('Teacher codes changed')
        bad=dict(value,quantized=True);write(receipt,bad)
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Quantized teacher was accepted')
        write(receipt,value);review_value['major_motion_usable']=False;write(review,review_value)
        modified=dict(value,visual_review=dict(path=str(review),sha256=sha(review)));write(receipt,modified)
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Unusable motion teacher was accepted')
        review_value['major_motion_usable']=True;write(review,review_value)
        replay_value['rows'][0]['passed']=False;write(replay,replay_value)
        changed_replay=dict(value,native_replay=dict(path=str(replay),sha256=sha(replay)));write(receipt,changed_replay)
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Failed native GS replay was accepted')
        replay_value['rows'][0]['passed']=True;write(replay,replay_value)
        evaluation_value=read(evaluation);evaluation_value['rows'].append(evaluation_value['rows'][0]);write(evaluation,evaluation_value)
        review_value['evaluation_sha256']=sha(evaluation);write(review,review_value)
        duplicate=dict(value,evaluation=dict(path=str(evaluation),sha256=sha(evaluation)),
                       visual_review=dict(path=str(review),sha256=sha(review)));write(receipt,duplicate)
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Duplicate teacher prediction was accepted')
        evaluation_value['rows'].pop();write(evaluation,evaluation_value)
        review_value['evaluation_sha256']=sha(evaluation);write(review,review_value);write(receipt,value)
        with codes.open('ab') as stream:stream.write(b'tamper')
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Modified teacher code file was accepted')
    result=dict(status='passed',validator_sha256=sha(ROOT/'teacher_receipt_v2.py'),test_sha256=sha(ROOT/'receipt_tests_v2.py'),
      core_manifest_sha256=sha(ROOT/'CODE.sha256'),sampler=sampler,
      deterministic_common_sampler=True,missing_window_horizon_rejected=True,
      teacher_checkpoint_tensor_and_code_pair_verified=True,
      quantized_unreviewed_tampered_and_failed_replay_teachers_rejected=True,
      complete_1512_prediction_rows_and_80_correct_native_packets_required=True,duplicate_prediction_rejected=True,
      scope='Synthetic teacher for provenance tests only; no formal teacher or GPU training')
    with (ROOT/'RECEIPT_TEST_V2.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: complete common C schedule and synthetic teacher provenance/rejection checks')


if __name__=='__main__':main()

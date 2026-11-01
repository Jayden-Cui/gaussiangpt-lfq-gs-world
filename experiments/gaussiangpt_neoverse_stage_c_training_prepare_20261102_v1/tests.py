"""CPU tests for balanced schedules and strict teacher-pair provenance."""
import json
from pathlib import Path
import tempfile
import types
import torch

from c_schedule import schedule,audit
from teacher_receipt import read,sha,state_digest,validate

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
        write(evaluation,dict(status='complete',window_predictions=1512,unique_future_frames=176,
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
        review_value['major_motion_usable']=True;write(review,review_value);write(receipt,value)
        with codes.open('ab') as stream:stream.write(b'tamper')
        try:validate(receipt,model,runtime,codec,cache)
        except RuntimeError:pass
        else:raise RuntimeError('Modified teacher code file was accepted')
    result=dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),sampler=sampler,
      deterministic_common_sampler=True,missing_window_horizon_rejected=True,
      teacher_checkpoint_tensor_and_code_pair_verified=True,
      quantized_unreviewed_and_tampered_teachers_rejected=True,
      scope='Synthetic teacher for provenance tests only; no formal teacher or GPU training')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: complete common C schedule and synthetic teacher provenance/rejection checks')


if __name__=='__main__':main()

"""Test conditional stage-B launch checks without starting a GPU job."""
import hashlib
import json
from pathlib import Path
import tempfile
import types

from b_worker_v2 import validate_config,validate_review

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    config=dict(schema='stage_b_shared_decoder_free_codes_v1',batch_size=4,starts=list(range(168)),
       quantized=False,anchors=[35,123],sweeps=20,decoder_lr=.0001,code_lr=.003,max_seconds=21600,
       progress_every=25,checkpoint_sweeps=[5,10,20],sampler_seed=20261102)
    items,sampler=validate_config(config)
    if len(items)!=7560 or sampler['visits_per_pair']!=20:raise RuntimeError('Full schedule differs')
    for patch in (dict(anchors=[35,124]),dict(decoder_lr=float('nan')),dict(checkpoint_sweeps=[5,10])):
        try:validate_config(dict(config,**patch))
        except ValueError:pass
        else:raise RuntimeError('Invalid configuration was accepted: '+str(patch))
    with tempfile.TemporaryDirectory(prefix='codex_stage_b_review_') as directory:
        root=Path(directory);evaluation=root/'evaluation.json'
        write(evaluation,dict(status='complete',window_predictions=1512,correction=dict(status='applied'),
                              source=dict(sweep=20)))
        review=root/'review.json';review_value=dict(status='reviewed',decision='stage_B',
          reviewed_starts=[0,35,56,84,112,123,140,167],evaluation=dict(path=str(evaluation),sha256=sha(evaluation)))
        write(review,review_value)
        c=dict(config,stage_a_review=dict(path=str(review),sha256=sha(review)))
        binding=types.SimpleNamespace(sha=sha,read=read)
        validate_review(c,binding)
        review_value['decision']='stage_C';write(review,review_value)
        c['stage_a_review']['sha256']=sha(review)
        try:validate_review(c,binding)
        except RuntimeError:pass
        else:raise RuntimeError('Opposite branch decision was accepted')
        review_value['decision']='stage_B';write(review,review_value)
        c['stage_a_review']['sha256']=sha(review)
        with evaluation.open('a') as stream:stream.write(' ')
        try:validate_review(c,binding)
        except RuntimeError:pass
        else:raise RuntimeError('Changed stage-A evaluation was accepted')
    result=dict(status='passed',worker_sha256=sha(ROOT/'b_worker_v2.py'),
      worker_test_sha256=sha(ROOT/'worker_tests_v2.py'),core_manifest_sha256=sha(ROOT/'CODE.sha256'),
      balanced_20_sweep_schedule=sampler,invalid_config_rejected=True,
      reviewed_branch_and_evaluation_hash_required=True,
      scope='CPU launch-gate test using synthetic review only; no GPU job or teacher result')
    with (ROOT/'WORKER_TEST_V2.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: fixed full-clip B4 schedule, branch review and evaluation binding')


if __name__=='__main__':main()

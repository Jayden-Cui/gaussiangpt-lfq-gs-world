"""Verify that the C worker uses replay-checked teachers and bound loss code."""
import json
from pathlib import Path

import c_worker_v2 as worker
import teacher_receipt_v2 as receipt
from c_schedule import schedule,audit

ROOT=Path(__file__).resolve().parent


def main():
    if worker.validate is not receipt.validate:raise RuntimeError('Worker uses an older teacher validator')
    core=worker.load_training_core()
    if not callable(core.step) or not callable(core.teacher_scale):raise RuntimeError('Bound training core missing')
    sampler=audit(schedule(1,1,20261102),1,1)
    result=dict(status='passed',worker_sha256=receipt.sha(ROOT/'c_worker_v2.py'),
      test_sha256=receipt.sha(ROOT/'worker_tests_v2.py'),core_manifest_sha256=receipt.sha(ROOT/'CODE.sha256'),
      requires_replay_checked_teacher_validator=True,bound_distillation_and_joint_loss_core_loaded=True,
      sampler=sampler,scope='CPU integration check only; no teacher training or GPU use')
    with (ROOT/'WORKER_TEST_V2.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: stage-C worker uses the replay-checked teacher validator and bound loss core')


if __name__=='__main__':main()

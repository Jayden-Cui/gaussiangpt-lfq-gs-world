"""Build a validated teacher receipt from completed A/B artifacts and review."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch

from teacher_receipt_v2 import sha,validate

ROOT=Path(__file__).resolve().parent
B_ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1')


def create(decoder,runtime,*,decoder_checkpoint,codes,evaluation,native_replay,visual_review,
           codec_sha256,cache_manifest_sha256,output,state_key='decoder'):
    output=Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite teacher receipt')
    files={name:Path(path).resolve(strict=True) for name,path in dict(decoder_checkpoint=decoder_checkpoint,
          codes=codes,evaluation=evaluation,native_replay=native_replay,visual_review=visual_review).items()}
    receipt=dict(schema='continuous_action_teacher_v1',status='reviewed',scope='single_train_video_168_windows',
        quantized=False,codec_sha256=codec_sha256,cache_manifest_sha256=cache_manifest_sha256,
        decoder_tensor_digest=runtime.digest(decoder))
    for name,path in files.items():receipt[name]=dict(path=str(path),sha256=sha(path))
    receipt['decoder_checkpoint']['state_key']=state_key
    temporary=output.with_suffix(output.suffix+'.pending')
    if temporary.exists():raise FileExistsError('Prior pending teacher receipt exists')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    # Validation enforces real review, code/decoder pair, full evaluation and
    # independent native GS replay. A failed pending receipt is preserved.
    validate(temporary,decoder,runtime,codec_sha256,cache_manifest_sha256)
    # A hard link is atomic and fails if output appeared concurrently.
    os.link(temporary,output)
    temporary.unlink()
    return receipt


def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    witness=json.loads((ROOT/'BUILDER_TEST.json').read_text())
    if (witness['status']!='passed' or witness['builder_sha256']!=sha(ROOT/'teacher_builder.py')
            or witness['test_sha256']!=sha(ROOT/'builder_tests.py')
            or witness['validator_sha256']!=sha(ROOT/'teacher_receipt_v2.py')):
        raise RuntimeError('Teacher builder test changed')
    receipt_test=json.loads((ROOT/'RECEIPT_TEST_V2.json').read_text())
    if (receipt_test['status']!='passed' or receipt_test['validator_sha256']!=sha(ROOT/'teacher_receipt_v2.py')
            or receipt_test['test_sha256']!=sha(ROOT/'receipt_tests_v2.py')
            or receipt_test['core_manifest_sha256']!=sha(ROOT/'CODE.sha256')):
        raise RuntimeError('Teacher validator test changed')
    for root in (ROOT,B_ROOT):
        prior=json.loads((root/'CPU_TESTS.json').read_text())
        if prior['status']!='passed' or prior['code_sha256']!=sha(root/'CODE.sha256'):
            raise RuntimeError('Teacher builder source manifest changed')
        for line in (root/'CODE.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            if Path(name).name!=name or sha(root/name)!=digest:raise RuntimeError('Teacher builder source changed: '+name)
    sys.path.insert(0,str(B_ROOT))
    from b_core import load_bound_sources
    bundle=load_bound_sources('cpu');decoder=bundle['decoder'];runtime=bundle['runtime']
    checkpoint=torch.load(args.decoder_checkpoint,map_location='cpu',weights_only=True)
    decoder.load_state_dict(checkpoint[args.state_key],strict=True);decoder.requires_grad_(False).eval()
    create(decoder,runtime,decoder_checkpoint=args.decoder_checkpoint,codes=args.codes,
        evaluation=args.evaluation,native_replay=args.native_replay,visual_review=args.visual_review,
        codec_sha256=bundle['p']['codec_checkpoint_sha256'],
        cache_manifest_sha256=sha(bundle['paths']['cache_manifest']),output=args.output,state_key=args.state_key)
    print('PASS: reviewed continuous teacher receipt created; no model training performed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('decoder-checkpoint','codes','evaluation','native-replay','visual-review','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--state-key',default='decoder')
    main(parser.parse_args())

"""Construct a reproducible C posterior and invoke the fixed-teacher worker."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import torch

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
VARIANTS=OUTPUTS/'gaussiangpt_neoverse_posterior_variants_20261102_v2'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def verify_manifest(root,expected):
    path=Path(root)/'CODE.sha256'
    if sha(path)!=expected:raise RuntimeError('Code manifest hash changed: '+str(root))
    for line in path.read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(Path(root)/name)!=digest:raise RuntimeError('Code file changed: '+name)


def build_variant(module,decoder,posterior,runtime,config):
    variant=config['variant'];include_time=config['include_time'];include_change=config['include_change']
    if (variant not in ('C0','C1','C2') or type(include_time)!=bool or type(include_change)!=bool
            or (variant!='C2' and (include_time or include_change))
            or (variant=='C2' and not include_time)):
        raise ValueError('Invalid C variant or time/change settings')
    if runtime.digest(posterior)!=config['shared_posterior_digest']:
        raise RuntimeError('Posterior initialization differs across variants')
    torch.manual_seed(config['model_seed'])
    model=module.ContinuousPosterior(decoder.base,posterior,variant,
          include_time=include_time,include_change=include_change,checkpoint_prefixes=True)
    if runtime.digest(model.posterior)!=config['shared_posterior_digest']:
        raise RuntimeError('Shared Q-Former initialization changed')
    if model.encoder is not None:
        expected=module.PosteriorEncoderCopy(decoder.base)
        if runtime.digest(model.encoder)!=runtime.digest(expected):
            raise RuntimeError('Posterior encoder copy differs from frozen initial encoder')
    return model


def main(config_path,run_root):
    # Set this exactly once and before any model work; GPU helper only checks it.
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    config_path=Path(config_path);config=read(config_path)
    if config.get('schema')!='stage_c_fixed_teacher_train_v1' or config.get('quantized') is not False:
        raise RuntimeError('Invalid continuous stage-C configuration')
    for root in (B_ROOT,ROOT,VARIANTS):
        witness=read(root/'CPU_TESTS.json')
        if witness['status']!='passed':raise RuntimeError('Required source witness failed')
        verify_manifest(root,witness['code_sha256'])
    driver_test=read(ROOT/'TRAIN_DRIVER_TEST_V2.json')
    if (driver_test['status']!='passed' or driver_test['driver_sha256']!=sha(ROOT/'c_train_driver_v2.py')
            or driver_test['test_sha256']!=sha(ROOT/'train_driver_tests_v2.py')
            or driver_test['core_manifest_sha256']!=sha(ROOT/'CODE.sha256')):
        raise RuntimeError('C training driver test changed')
    receipt_test=read(ROOT/'RECEIPT_TEST_V2.json')
    if (receipt_test['status']!='passed' or receipt_test['validator_sha256']!=sha(ROOT/'teacher_receipt_v2.py')
            or receipt_test['test_sha256']!=sha(ROOT/'receipt_tests_v2.py')
            or receipt_test['core_manifest_sha256']!=sha(ROOT/'CODE.sha256')):
        raise RuntimeError('Teacher replay and provenance validator changed')
    worker_test=read(ROOT/'WORKER_TEST_V2.json')
    if (worker_test['status']!='passed' or worker_test['worker_sha256']!=sha(ROOT/'c_worker_v2.py')
            or worker_test['test_sha256']!=sha(ROOT/'worker_tests_v2.py')
            or worker_test['core_manifest_sha256']!=sha(ROOT/'CODE.sha256')):
        raise RuntimeError('C training worker witness changed')
    gpu_test=read(B_ROOT/'GPU_RUNTIME_TEST.json')
    if (gpu_test['status']!='passed' or gpu_test['runtime_sha256']!=sha(B_ROOT/'gpu_runtime.py')
            or gpu_test['test_sha256']!=sha(B_ROOT/'gpu_runtime_tests.py')):
        raise RuntimeError('One-time GPU setup witness changed')
    sys.path.insert(0,str(B_ROOT))
    from b_core import load_bound_sources
    from gpu_runtime import setup_gpu_once
    from teacher_receipt_v2 import validate
    from c_worker_v2 import fit
    bundle=load_bound_sources('cpu');binding=bundle['binding'];runtime=bundle['runtime'];p=bundle['p']
    if (config['codec_sha256']!=p['codec_checkpoint_sha256']
            or config['cache_manifest_sha256']!=binding.sha(bundle['paths']['cache_manifest'])):
        raise RuntimeError('Stage-C codec or training cache differs')
    teacher_entry=config['teacher_receipt'];teacher_path=Path(teacher_entry['path'])
    if not teacher_path.is_absolute() or sha(teacher_path)!=teacher_entry['sha256']:
        raise RuntimeError('Reviewed teacher receipt changed')
    receipt=read(teacher_path);decoder=bundle['decoder']
    checkpoint=torch.load(receipt['decoder_checkpoint']['path'],map_location='cpu',weights_only=True)
    decoder.load_state_dict(checkpoint[receipt['decoder_checkpoint'].get('state_key','decoder')],strict=True)
    decoder.requires_grad_(False).eval()
    teacher=validate(teacher_path,decoder,runtime,config['codec_sha256'],config['cache_manifest_sha256'])
    initialization=config['posterior_initialization'];init_path=Path(initialization['path'])
    if not init_path.is_absolute() or sha(init_path)!=initialization['sha256']:
        raise RuntimeError('Shared Q-Former initialization checkpoint changed')
    posterior=bundle['posterior'];initial=torch.load(init_path,map_location='cpu',weights_only=True)
    posterior.load_state_dict(initial[initialization.get('state_key','posterior')],strict=True);posterior.eval()
    variants=binding.explicit('_stage_c_training_variants',VARIANTS/'posterior_variants.py')
    model=build_variant(variants,decoder,posterior,runtime,config)
    run_root=Path(run_root);run_root.mkdir();phase='gpu_preflight'
    try:
        write(run_root/'CONFIG.json',config)
        write(run_root/'BINDING.json',dict(source_config_sha256=sha(config_path),
             config_snapshot_sha256=sha(run_root/'CONFIG.json'),teacher_receipt_sha256=teacher['receipt_sha256'],
             decoder_digest=runtime.digest(decoder),shared_posterior_digest=runtime.digest(posterior),
             posterior_encoder_initial_digest=runtime.digest(model.encoder) if model.encoder is not None else None,
             variant=config['variant'],model_initial_digest=runtime.digest(model),
             driver_sha256=sha(ROOT/'c_train_driver_v2.py'),worker_sha256=sha(ROOT/'c_worker_v2.py'),
             teacher_validator_sha256=sha(ROOT/'teacher_receipt_v2.py'),quantized=False))
        setup_gpu_once(p);decoder=decoder.cuda();model=model.cuda()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        phase='training';fit(model,decoder,bridge,bundle['store'],p,runtime,teacher_path,config,run_root/'training')
        write(run_root/'DONE.json',dict(status='complete',training='complete',evaluation='pending',
              train_done_sha256=sha(run_root/'training/TRAIN_DONE.json')))
    except BaseException as error:
        write(run_root/'FAILED.json',dict(status='failed',phase=phase,error=repr(error),
              traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-root',required=True)
    args=parser.parse_args();main(args.config,args.run_root)

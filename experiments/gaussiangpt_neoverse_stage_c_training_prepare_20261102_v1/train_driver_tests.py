"""CPU check that C variants share the same real Q/encoder initialization."""
import json
from pathlib import Path
import sys
import torch

import c_train_driver as driver

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for root in (driver.B_ROOT,driver.ROOT,driver.VARIANTS):
        witness=driver.read(root/'CPU_TESTS.json');driver.verify_manifest(root,witness['code_sha256'])
    sys.path.insert(0,str(driver.B_ROOT))
    from b_core import load_bound_sources
    bundle=load_bound_sources('cpu');binding=bundle['binding'];runtime=bundle['runtime']
    decoder,posterior=bundle['decoder'],bundle['posterior']
    module=binding.explicit('_stage_c_driver_variant_test',driver.VARIANTS/'posterior_variants.py')
    shared=runtime.digest(posterior);common=dict(shared_posterior_digest=shared,model_seed=20261102)
    models={}
    for variant,time,change in [('C0',False,False),('C1',False,False),('C2',True,True)]:
        config=dict(common,variant=variant,include_time=time,include_change=change)
        models[variant]=driver.build_variant(module,decoder,posterior,runtime,config)
        if runtime.digest(models[variant].posterior)!=shared:raise RuntimeError('Q-Former initial weights differ')
    if models['C0'].encoder is not None or models['C1'].encoder is None or models['C2'].encoder is None:
        raise RuntimeError('Encoder variants differ')
    if runtime.digest(models['C1'].encoder)!=runtime.digest(models['C2'].encoder):
        raise RuntimeError('Trainable posterior encoders do not start identically')
    repeat=driver.build_variant(module,decoder,posterior,runtime,dict(common,variant='C2',include_time=True,include_change=True))
    if runtime.digest(repeat)!=runtime.digest(models['C2']):raise RuntimeError('C2 initialization is not reproducible')
    try:driver.build_variant(module,decoder,posterior,runtime,dict(common,variant='C1',include_time=True,include_change=False))
    except ValueError:pass
    else:raise RuntimeError('Invalid variant settings were accepted')
    result=dict(status='passed',driver_sha256=driver.sha(ROOT/'c_train_driver.py'),
      test_sha256=driver.sha(ROOT/'train_driver_tests.py'),core_manifest_sha256=driver.sha(ROOT/'CODE.sha256'),
      real_bound_qformer_digest=shared,all_variants_share_qformer_initialization=True,
      c1_c2_encoder_initialization_identical=True,c2_seed_deterministic=True,
      invalid_variant_settings_rejected=True,
      scope='CPU model-construction witness only; no reviewed teacher or GPU training')
    driver.write(ROOT/'TRAIN_DRIVER_TEST.json',result)
    print('PASS: actual C0/C1/C2 shared Q initialization, copied encoders and deterministic C2')


if __name__=='__main__':main()

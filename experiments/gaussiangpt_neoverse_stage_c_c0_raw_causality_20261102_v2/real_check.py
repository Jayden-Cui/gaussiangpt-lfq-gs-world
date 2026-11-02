"""Check the raw C0 path on two real cached windows on GPU, without training."""
import json
from pathlib import Path
import sys
import time
import torch

from guard import audit_raw_c0

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
VARIANTS = OUTPUTS / 'gaussiangpt_neoverse_posterior_variants_20261102_v2'


def main():
    import driver_v5
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or driver_v5.sha(ROOT / name) != digest:
            raise RuntimeError('Raw causality source changed: ' + name)
    sys.path.insert(0, str(B_ROOT))
    from b_core import load_bound_sources
    from gpu_runtime import setup_gpu_once
    bundle = load_bound_sources('cpu')
    decoder, posterior, runtime = bundle['decoder'], bundle['posterior'], bundle['runtime']
    setup_gpu_once(dict(bundle['p'], allocator_limit_gib=12, minimum_free_gpu_gib=32))
    decoder.requires_grad_(False).eval().cuda()
    variants = bundle['binding'].explicit('_raw_c0_check_variants', VARIANTS / 'posterior_variants.py')
    model = variants.ContinuousPosterior(decoder.base, posterior, 'C0').cuda().eval()
    make_memories = driver_v5.get_make_memories()
    start = time.monotonic()
    decoder_before, model_before = runtime.digest(decoder), runtime.digest(model)
    result = audit_raw_c0(model, decoder.base, bundle['store'], make_memories)
    if runtime.digest(decoder) != decoder_before or runtime.digest(model) != model_before:
        raise RuntimeError('Raw-state causality check changed a fixed model')
    result.update(code_sha256=driver_v5.sha(ROOT / 'CODE.sha256'),
                  source_cache_sha256=driver_v5.sha(bundle['paths']['cache_manifest']),
                  frozen_initial_context_digest=bundle['frozen_digest'],
                  decoder_digest=decoder_before, posterior_digest=model_before,
                  elapsed_seconds=time.monotonic()-start, optimizer_updates=0, device='cuda',
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  scope='Two real training windows and pretrained Q; no trained stage-C model')
    with (ROOT / 'REAL_SOURCE_CHECK.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()

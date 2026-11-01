"""Exercise GPU setup semantics without allocating a GPU."""
import hashlib
import json
from pathlib import Path
import types
import torch
import gpu_runtime

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_interop_threads(1)
    calls=[];old=gpu_runtime.torch.cuda
    class FakeCuda:
        free=70*2**30;count=1
        @staticmethod
        def is_available():return True
        @classmethod
        def device_count(cls):return cls.count
        @classmethod
        def mem_get_info(cls):return cls.free,80*2**30
        @staticmethod
        def set_per_process_memory_fraction(value):calls.append(value)
    gpu_runtime.torch.cuda=FakeCuda
    try:
        p=dict(minimum_free_gpu_gib=32,allocator_limit_gib=24)
        gpu_runtime.setup_gpu_once(p);gpu_runtime.setup_gpu_once(p)
        if calls!=[.3,.3]:raise RuntimeError('Allocator limit differs')
        FakeCuda.free=1
        try:gpu_runtime.setup_gpu_once(p)
        except RuntimeError:pass
        else:raise RuntimeError('Insufficient GPU memory was accepted')
        FakeCuda.free=70*2**30;FakeCuda.count=2
        try:gpu_runtime.setup_gpu_once(p)
        except RuntimeError:pass
        else:raise RuntimeError('Multiple exposed GPUs were accepted')
    finally:gpu_runtime.torch.cuda=old
    output=dict(status='passed',runtime_sha256=sha(ROOT/'gpu_runtime.py'),test_sha256=sha(ROOT/'gpu_runtime_tests.py'),
      repeat_setup_does_not_reset_interop_threads=True,insufficient_memory_and_multiple_gpus_rejected=True,
      scope='Mock GPU CPU test; no real GPU allocation or formal preflight')
    with (ROOT/'GPU_RUNTIME_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: GPU setup preserves one-time inter-op initialization and validates resources')


if __name__=='__main__':main()

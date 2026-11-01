"""GPU checks after the process has configured PyTorch inter-op threads once."""
import torch


def setup_gpu_once(p):
    # Calling set_num_interop_threads a second time raises on this environment.
    # The entrypoint sets it before model construction; never set it here again.
    if torch.get_num_interop_threads()!=1:
        raise RuntimeError('Entrypoint must set PyTorch inter-op threads to one before model work')
    torch.set_num_threads(6)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU')
    free,total=torch.cuda.mem_get_info()
    if free<p['minimum_free_gpu_gib']*2**30:raise RuntimeError('Insufficient free GPU memory')
    fraction=p['allocator_limit_gib']*2**30/total
    if not 0<fraction<=1:raise RuntimeError('Invalid allocator limit')
    torch.cuda.set_per_process_memory_fraction(fraction)

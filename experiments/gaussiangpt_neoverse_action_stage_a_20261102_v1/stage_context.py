"""Bind stage A to the exact frozen decoder, cached states and 9560-step codes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def explicit(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError('Unexpected source path: '+name)
    return module


def check_manifest(root, expected):
    root = Path(root)
    if sha(root/'CODE.sha256') != expected:
        raise RuntimeError('Code manifest changed: '+str(root))
    for line in (root/'CODE.sha256').read_text().splitlines():
        digest, name = line.split('  ',1)
        if Path(name).name != name or sha(root/name)!=digest:
            raise RuntimeError('Bound source changed: '+name)


def setup():
    check_manifest(ROOT, sha(ROOT/'CODE.sha256'))
    c = read(ROOT/'stage_a.json')
    fit, orig, resume = (Path(c[key]) for key in ['fit_root','data_root','resume_helper_root'])
    check_manifest(fit,c['fit_code_manifest_sha256'])
    check_manifest(resume,c['resume_code_manifest_sha256'])
    if sha(resume/'resume.py')!=c['resume_helper_sha256']:
        raise RuntimeError('Restore helper changed')
    helper = explicit('_stage_a_resume_helper',resume/'resume.py')
    parent_binding, bound_orig, p = helper.setup()
    if bound_orig.resolve()!=orig.resolve() or sha(orig/'protocol.json')!=c['original_protocol_sha256']:
        raise RuntimeError('Original data protocol changed')
    paths = {
        'checkpoint':fit/'posterior/step_09560.pt',
        'receipt':fit/'step_09560/posterior/TRAIN_DONE.json',
        'evaluation':fit/'step_09560/posterior/evaluation/EVALUATION.json',
        'codes':fit/'step_09560/posterior/evaluation/action_codes.pt',
        'cache_manifest':orig/'cache/MANIFEST.json',
    }
    for name,path in paths.items():
        if sha(path)!=c['bindings'][name]:raise RuntimeError('Bound artifact changed: '+name)
    receipt, evaluation = read(paths['receipt']), read(paths['evaluation'])
    if (receipt['status']!='complete' or receipt['steps']!=9560
            or receipt['checkpoint_sha256']!=c['bindings']['checkpoint']
            or receipt['decoder_digest']!=c['decoder_tensor_digest']
            or evaluation['status']!='complete' or evaluation['window_predictions']!=1512
            or evaluation['action_codes']['sha256']!=c['bindings']['codes']
            or evaluation['source']['trained_checkpoint_sha256']!=c['bindings']['checkpoint']):
        raise RuntimeError('9560-step result binding mismatch')
    if c['starts']!=p['starts'] or c['original_protocol_sha256']!=parent_binding['parent_protocol_sha256']:
        raise RuntimeError('Window scope or original protocol mismatch')
    return c, helper, orig, p, paths, receipt, evaluation


def load_codes(paths, evaluation):
    import torch
    payload = torch.load(paths['codes'],map_location='cpu',weights_only=True)
    if (payload['schema']!='full177_continuous_codes_v1' or payload['source']!=evaluation['source']
            or payload['starts']!=list(range(168)) or payload['shape']!=[168,9,16]
            or payload['quantized'] or set(payload['controls'])!={'correct','initial_only','swapped_window_codes'}):
        raise RuntimeError('Unexpected baseline action-code schema')
    for name,tensor in payload['controls'].items():
        if tensor.dtype!=torch.float32 or tuple(tensor.shape)!=(168,9,16) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError('Invalid baseline code: '+name)
    return payload['controls']


def load_decoder(c,helper,orig,p,paths,receipt,device):
    import torch
    modules = helper.model_modules(orig)
    fm, data, runtime = modules
    decoder, unused_posterior = fm.build_models(p,'cpu',train_decoder=False)
    checkpoint = torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    if (checkpoint['schema']!='full177_fit_budget_continuation_v1' or checkpoint['mode']!='posterior'
            or checkpoint['step']!=9560 or checkpoint['source']!=receipt['source']):
        raise RuntimeError('Wrong 9560-step checkpoint')
    decoder.load_state_dict(checkpoint['decoder'],strict=True)
    if runtime.digest(decoder)!=c['decoder_tensor_digest']:
        raise RuntimeError('Restored decoder tensor digest differs')
    decoder.requires_grad_(False).eval()
    del checkpoint, unused_posterior
    return decoder.to(device), modules


def source_record(c, paths, receipt, schedule=None):
    return dict(schema='full177_fixed_decoder_free_code_stage_a_v1',
        checkpoint_sha256=c['bindings']['checkpoint'],decoder_digest=c['decoder_tensor_digest'],
        baseline_codes_sha256=c['bindings']['codes'],baseline_evaluation_sha256=c['bindings']['evaluation'],
        cache_manifest_sha256=c['bindings']['cache_manifest'],codec_sha256=receipt['source']['codec_sha256'],
        protocol_sha256=c['original_protocol_sha256'],stage_a_config_sha256=sha(ROOT/'stage_a.json'),
        stage_a_code_sha256=sha(ROOT/'CODE.sha256'),sampler=schedule,
        scope='168 windows in the same TRAIN video; optimized per-window continuous codes; not deployable inference')


def assert_frozen(decoder, bridge, runtime, expected):
    bridge.assert_frozen()
    if runtime.digest(decoder)!=expected or any(p.requires_grad or p.grad is not None for p in decoder.parameters()):
        raise RuntimeError('Motion decoder changed or acquired gradients')


def setup_gpu(p):
    import torch
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU')
    free,total = torch.cuda.mem_get_info()
    if free < p['minimum_free_gpu_gib']*2**30:
        raise RuntimeError('Insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)


def dataset(modules,p,c):
    _, data, _ = modules
    store = data.WindowStore(p)
    if (store.manifest['status']!='complete' or store.manifest['starts']!=c['starts']
            or sha(store.root/'MANIFEST.json')!=c['bindings']['cache_manifest']):
        raise RuntimeError('Cached state source changed')
    return store

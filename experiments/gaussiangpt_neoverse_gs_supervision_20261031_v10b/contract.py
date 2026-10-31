"""Small immutable-source contract for the masked GS supervision experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ARMS = ('rgb_z', 'gs_z', 'rgb_constant', 'gs_constant')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition: raise RuntimeError(message)

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def verify(root, cpu_witness=True):
    root = Path(root).resolve()
    p = read(root/'protocol.json')
    require(p['schema'] == 'gs_supervision_v10' and tuple(p['arms']) == ARMS,
            'Unexpected scientific matrix')
    require(p['train']['source_step'] == 600 and p['train']['total_steps'] == 800
            and p['train']['additional_steps'] == 200, 'Fixed training budget changed')
    require(len(p['train_plan']) == 800 and all(len(x) == 4 and all(type(s) is int and 0 <= s < 168 for s in x)
            for x in p['train_plan']), 'Complete B4 schedule required')
    require(p['loss'] == {'delta_weight': 1.0, 'teacher_weights': [0.0, 0.25],
                         'alpha_threshold': 0.90, 'erosion_kernel': 3}, 'Loss/mask specification changed')
    require(p['resources']['peak_limit_gib'] == 16 and p['resources']['minimum_free_gib'] == 20,
            'Resource admission changed')
    for arm in ARMS:
        require(p['arms'][arm]['source_arm'] == ('spatial_constant' if arm.endswith('constant') else 'spatial_z'),
                'Source checkpoint pairing changed')
        require(p['arms'][arm]['teacher_weights'] == ([0., .25] if arm.startswith('gs_') else [0., 0.]),
                'Teacher treatment changed')
    require(not p['automatic_retry'] and not p['goal_complete'], 'Retry/success claim changed')
    manifest = read(root/'CODE_MANIFEST.json')
    for name, expected in manifest.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, 'Source changed: '+name)
    require({f.name for f in root.glob('*.py')} <= set(manifest), 'Unbound Python source')
    proof = dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'), protocol_sha256=sha(root/'protocol.json'))
    if cpu_witness:
        w = read(root/'CPU_VALIDATION.json')
        require(w['status'] == 'passed' and w['gpu_executed'] is False
                and all(w.get(k) == v for k,v in proof.items()), 'Fresh bound CPU witness required')
    return p, proof

def verify_sources(p):
    for rel, expected in p['source_hashes'].items():
        require(sha(Path(p['source_root'])/rel) == expected, 'Source v9 changed: '+rel)
    for key, item in p['external_sources'].items():
        require(sha(item['path']) == item['sha256'], 'Bound external source changed: '+key)


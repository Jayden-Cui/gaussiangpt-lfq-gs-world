from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream: stream.write(content)


def verify():
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT/name) != digest:
            raise RuntimeError('Diagnostic source changed: '+name)
    p = read(ROOT/'protocol.json')
    for name, digest in p['dependencies'].items():
        if sha(Path(p['formal_root'])/name) != digest:
            raise RuntimeError('Bound historical source changed: '+name)
    prior=p['previous_run'];previous=Path(prior['root'])
    for name,key in (('protocol.json','protocol_sha256'),('CODE.sha256','code_manifest_sha256'),
                     ('SUMMARY.json','summary_sha256'),('core.py','core_sha256'),
                     ('CPU_TESTS.json','cpu_core_receipt_sha256')):
        if sha(previous/name)!=prior[key]:
            raise RuntimeError('Previous diagnostic evidence changed: '+name)
    if sha(ROOT/'core.py')!=prior['core_sha256']:
        raise RuntimeError('Previously tested metric/loss implementation changed')
    old=read(previous/'protocol.json')
    same=('starts','horizon','modes','steps_per_case','learning_rate','final_learning_rate','schedule',
          'optimizer','betas','eps','weight_decay','gradient_clipping','seed','std_source_frame','std_unbiased',
          'native_codec','codec_checkpoint_sha256','decoder_precision','feature_dtype','resolution_hw',
          'loss','hand_roi_xyxy','motion_weight','secondary_view','snapshots','checkpoint_selection','object_proxies')
    if any(p[key]!=old[key] for key in same):
        raise RuntimeError('Matched optimizer/data/codec/loss contract changed')
    if read(previous/'SUMMARY.json')['engineering_passed'] is not True:
        raise RuntimeError('Previous comparison did not pass engineering checks')
    return p

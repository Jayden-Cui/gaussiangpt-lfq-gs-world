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
    return p

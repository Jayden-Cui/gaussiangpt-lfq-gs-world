"""Small source-integrity utilities shared by independent processes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, data):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def expected_names(p):
    return {f's{s:03d}_h{h:02d}_{b}' for s in p['windows'] for h in p['horizons'] for b in p['branches']}


def verify_self():
    witness = read(ROOT/'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT/'CODE.sha256'):
        raise ValueError('CPU provenance witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected,name = line.split('  ',1)
        if Path(name).name != name or sha(ROOT/name) != expected:
            raise ValueError('Replay source changed: '+name)


def validate_protocol(p):
    if (p.get('schema') != 'two_future_native_replay_v1' or p['windows'] != [35,123]
            or p['horizons'] != list(range(1,10)) or p['branches'] != ['motion','static']
            or p['packets'] != 36 or p['views'] != 72 or len(expected_names(p)) != 36
            or p['quantized'] is not False):
        raise ValueError('Replay protocol changed')

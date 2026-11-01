"""Bounded paired-Q protocol, schedules and integrity helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value,*,replace=False):
    path=Path(path)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(text,encoding='utf8')
        tmp.replace(path)
    else:
        with path.open('x',encoding='utf8',newline='\n') as f:
            f.write(text)


def verify_self():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('Paired-Q CPU witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=expected:
            raise ValueError('Paired-Q source changed: '+name)


def validate_protocol(p):
    if (p.get('schema')!='two_future_fixed_decoder_qformer_pilot_v1'
            or p['windows']!=[35,123] or p['horizons']!=list(range(1,10))
            or p['conditions']!=['motion','static'] or p['logical_batch']!=4
            or p['distill_updates']!=600 or p['joint_updates']!=594
            or p['checkpoint_steps']!=[0,600,900,1194] or p['quantized'] is not False):
        raise ValueError('Fixed paired-Q protocol differs')
    for name in ('teacher_checkpoint_sha256','teacher_decoder_digest','native_replay_sha256',
                 'native_packets_sha256','visual_review_sha256','cache_manifest_sha256','codec_sha256'):
        x=p[name]
        if not isinstance(x,str) or len(x)!=64:
            raise ValueError('Invalid bound SHA-256: '+name)
        try:bytes.fromhex(x)
        except ValueError as e:raise ValueError('Invalid bound SHA-256: '+name) from e


def schedule(p):
    validate_protocol(p)
    examples=[(s,c) for s in p['windows'] for c in p['conditions']]
    rows=[]
    for step in range(1,p['distill_updates']+p['joint_updates']+1):
        phase='distill' if step<=p['distill_updates'] else 'joint'
        horizon=None if phase=='distill' else 1+(step-p['distill_updates']-1)%9
        rows.append({'step':step,'phase':phase,'horizon':horizon,'examples':examples})
    return rows


def expected_packet_names(p):
    return {f's{s:03d}_h{h:02d}_{c}' for s in p['windows'] for h in p['horizons'] for c in p['conditions']}

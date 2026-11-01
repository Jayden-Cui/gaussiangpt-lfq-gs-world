import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value,replace=False):
    path=Path(path);content=json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as f:f.write(content)

def sources(p):
    formal=Path(p['formal_root'])
    for name,digest in p['dependencies'].items():
        if sha(formal/name)!=digest:raise RuntimeError('Source changed: '+name)
    if sha(p['common_checkpoint'])!=p['common_checkpoint_sha256']:raise RuntimeError('Common model init changed')
    previous=p['previous_future_support']
    if sha(Path(previous['root'])/'SUMMARY.json')!=previous['summary_sha256']:raise RuntimeError('Oracle diagnostic changed')

def verify():
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Own source changed: '+name)
    p=read(ROOT/'protocol.json');sources(p);return p

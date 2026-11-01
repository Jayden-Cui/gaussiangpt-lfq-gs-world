"""Validate chunk assembly and immutable initialization source binding."""
import json
from pathlib import Path
import torch

from prepare import ROOT,B_ROOT,assemble,sha


def main():
    torch.set_num_threads(2)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Preparation code changed')
    prior=json.loads((B_ROOT/'CPU_TESTS.json').read_text())
    if prior['status']!='passed' or prior['code_sha256']!=sha(B_ROOT/'CODE.sha256'):
        raise RuntimeError('Bound two-window initialization witness changed')
    values=torch.arange(168,dtype=torch.float32)[:,None,None].expand(168,9,16).clone()
    chunks=[(list(range(s,min(s+8,168))),values[s:min(s+8,168)]) for s in range(0,168,8)]
    if not torch.equal(assemble(list(reversed(chunks))),values):raise RuntimeError('Window ordering changed')
    try:assemble(chunks[:-1])
    except ValueError:pass
    else:raise RuntimeError('Missing windows were accepted')
    try:assemble(chunks+[chunks[0]])
    except ValueError:pass
    else:raise RuntimeError('Duplicate windows were accepted')
    write=dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),all_168_windows_ordered=True,
       missing_and_duplicate_windows_rejected=True,stage_b_model_witness_sha256=sha(B_ROOT/'CPU_TESTS.json'),
       scope='Synthetic chunk assembly; real two-window model already checked separately; no training')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(write,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: complete ordered initialization chunks and source witness')


if __name__=='__main__':main()

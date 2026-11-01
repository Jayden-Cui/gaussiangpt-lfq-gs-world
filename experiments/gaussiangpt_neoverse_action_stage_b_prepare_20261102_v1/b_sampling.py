"""Exactly balanced complete-clip sampling, independent of stage-A sharding."""
from collections import Counter
import hashlib
import json
import random


def schedule(sweeps,seed,batch_size=4):
    if type(sweeps)!=int or sweeps<=0 or batch_size!=4:
        raise ValueError('Positive sweeps and B4 required')
    rng=random.Random(seed);items=[]
    for sweep in range(1,sweeps+1):
        pairs=[(start,horizon) for start in range(168) for horizon in range(1,10)]
        rng.shuffle(pairs)
        for offset in range(0,len(pairs),4):
            items.append(dict(step=len(items)+1,sweep=sweep,
                              batch=[dict(start=s,horizon=h) for s,h in pairs[offset:offset+4]]))
    return items


def audit(items,sweeps):
    expected={(s,h) for s in range(168) for h in range(1,10)}
    for sweep in range(1,sweeps+1):
        rows=[r for r in items if r['sweep']==sweep]
        counts=Counter((m['start'],m['horizon']) for r in rows for m in r['batch'])
        if len(rows)!=378 or set(counts)!=expected or set(counts.values())!={1}:
            raise RuntimeError('Sweep did not visit each window/horizon once')
    if (len(items)!=378*sweeps or any(len(r['batch'])!=4 for r in items)
            or [r['step'] for r in items]!=list(range(1,len(items)+1))):
        raise RuntimeError('Update count or batch size differs')
    return dict(status='passed',windows=168,horizons=9,pairs=1512,sweeps=sweeps,
                updates=len(items),visits_per_pair=sweeps,
                sha256=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())

"""Shared, exhaustive schedules for fixed-teacher posterior comparisons."""
from collections import Counter
import hashlib
import json
import random


def schedule(distill_sweeps,joint_sweeps,seed):
    if any(type(v)!=int or v<=0 for v in (distill_sweeps,joint_sweeps)):
        raise ValueError('Both training phases require positive complete sweeps')
    rng=random.Random(seed);items=[]
    for phase,sweeps in (('distill',distill_sweeps),('joint',joint_sweeps)):
        for sweep in range(1,sweeps+1):
            values=(list(range(168)) if phase=='distill' else
                    [(s,h) for s in range(168) for h in range(1,10)])
            rng.shuffle(values)
            for offset in range(0,len(values),4):
                batch=values[offset:offset+4]
                # Distillation matches all nine codes at once; horizon is unused.
                rows=([dict(start=s,horizon=9) for s in batch] if phase=='distill' else
                      [dict(start=s,horizon=h) for s,h in batch])
                items.append(dict(step=len(items)+1,phase=phase,sweep=sweep,batch=rows))
    return items


def audit(items,distill_sweeps,joint_sweeps):
    for phase,sweeps in (('distill',distill_sweeps),('joint',joint_sweeps)):
        expected=set(range(168)) if phase=='distill' else {(s,h) for s in range(168) for h in range(1,10)}
        for sweep in range(1,sweeps+1):
            rows=[r for r in items if r['phase']==phase and r['sweep']==sweep]
            pairs=Counter((m['start'] if phase=='distill' else (m['start'],m['horizon']))
                          for r in rows for m in r['batch'])
            if set(pairs)!=expected or set(pairs.values())!={1}:
                raise RuntimeError('An example is missing or duplicated in '+phase)
    if (len(items)!=42*distill_sweeps+378*joint_sweeps
            or any(len(x['batch'])!=4 for x in items)
            or [x['step'] for x in items]!=list(range(1,len(items)+1))):
        raise RuntimeError('Unexpected number/order of B4 updates')
    return dict(status='passed',distill_sweeps=distill_sweeps,joint_sweeps=joint_sweeps,
        distill_updates=42*distill_sweeps,joint_updates=378*joint_sweeps,
        total_updates=len(items),windows=168,future_pairs=1512,
        sha256=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())

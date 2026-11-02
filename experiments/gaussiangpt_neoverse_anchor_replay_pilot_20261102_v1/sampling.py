"""Full 168-window coverage plus fixed, balanced replay of two anchors."""
from collections import Counter
import hashlib
import json
import random


STARTS = (35, 123)


def schedule(sweeps, base_seed, replay_seed):
    if type(sweeps) is not int or sweeps <= 0 or any(type(x) is not int for x in (base_seed, replay_seed)):
        raise ValueError('Expected positive sweeps and integer seeds')
    base_rng, replay_rng, items = random.Random(base_seed), random.Random(replay_seed), []
    for sweep in range(1, sweeps+1):
        base = [(s,h) for s in range(168) for h in range(1,10)]
        replay = [(s,h) for s in STARTS for h in range(1,10)] * 10
        base_rng.shuffle(base)
        replay_rng.shuffle(replay)
        cursor = 0
        def append(kind, pairs):
            items.append(dict(step=len(items)+1,sweep=sweep,kind=kind,
                              batch=[dict(start=s,horizon=h) for s,h in pairs]))
        for i in range(378):
            append('base',base[4*i:4*i+4])
            if (i+1)*45//378 > i*45//378:
                append('replay',replay[cursor:cursor+4]);cursor+=4
        if cursor != 180:
            raise RuntimeError('Replay interleaving differs')
    return items


def audit(items, sweeps):
    expected={(s,h) for s in range(168) for h in range(1,10)}
    anchors={(s,h) for s in STARTS for h in range(1,10)}
    if (len(items)!=423*sweeps or [r['step'] for r in items]!=list(range(1,len(items)+1))
            or any(len(r['batch'])!=4 or r['kind'] not in ('base','replay') for r in items)):
        raise RuntimeError('Unexpected replay schedule length or batch shape')
    for sweep in range(1,sweeps+1):
        rows=[r for r in items if r['sweep']==sweep]
        base=Counter((x['start'],x['horizon']) for r in rows if r['kind']=='base' for x in r['batch'])
        replay=Counter((x['start'],x['horizon']) for r in rows if r['kind']=='replay' for x in r['batch'])
        if (len(rows)!=423 or sum(r['kind']=='base' for r in rows)!=378
                or sum(r['kind']=='replay' for r in rows)!=45 or set(base)!=expected
                or set(base.values())!={1} or set(replay)!=anchors or set(replay.values())!={10}):
            raise RuntimeError('Incomplete base or unbalanced anchor replay')
    return dict(status='passed',windows=168,horizons=9,base_pairs=1512,sweeps=sweeps,
        updates=len(items),base_visits_per_pair=sweeps,replay_visits_per_anchor_pair=10*sweeps,
        total_visits_per_anchor_pair=11*sweeps,base_batches_per_sweep=378,replay_batches_per_sweep=45,
        sha256=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())

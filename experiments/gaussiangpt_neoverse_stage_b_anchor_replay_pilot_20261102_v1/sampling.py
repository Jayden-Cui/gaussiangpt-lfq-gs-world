"""Deterministic, balanced subset for a bounded matched replay pilot."""
from collections import Counter
import hashlib
import json
import random


def make_schedule(main_seed,anchor_seed):
    pairs=[(s,h) for s in range(168) for h in range(1,10) if (s+h)%2==0]
    random.Random(main_seed).shuffle(pairs)
    base=[(s,h) for s in (35,123) for h in range(1,10)]
    anchors=base*10+base[:9]
    random.Random(anchor_seed).shuffle(anchors)
    return [{'step':i+1,
             'batch':[{'start':s,'horizon':h} for s,h in pairs[i*4:i*4+4]],
             'anchor':{'start':anchors[i][0],'horizon':anchors[i][1]}}
            for i in range(189)]


def audit(items):
    if len(items)!=189 or [x['step'] for x in items]!=list(range(1,190)):
        raise ValueError('Expected 189 ordered updates')
    pairs=[(v['start'],v['horizon']) for x in items for v in x['batch']]
    expected={(s,h) for s in range(168) for h in range(1,10) if (s+h)%2==0}
    if any(len(x['batch'])!=4 for x in items) or len(pairs)!=756 or set(pairs)!=expected:
        raise ValueError('Main sample coverage differs or contains duplicates')
    per_start=Counter(s for s,h in pairs);per_h=Counter(h for s,h in pairs)
    if set(per_start)!=set(range(168)) or any(n not in (4,5) for n in per_start.values()):
        raise ValueError('Unbalanced window counts')
    if per_h!={h:84 for h in range(1,10)}:raise ValueError('Unbalanced horizons')
    anchors=Counter((x['anchor']['start'],x['anchor']['horizon']) for x in items)
    if (set(anchors)!={(s,h) for s in (35,123) for h in range(1,10)}
            or set(anchors.values())!={10,11}):raise ValueError('Unbalanced replay anchors')
    digest=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'status':'passed','updates':189,'main_pairs':756,'windows':168,'per_horizon':84,
            'anchor_examples':189,'anchor_pair_counts':[10,11],'sha256':digest}

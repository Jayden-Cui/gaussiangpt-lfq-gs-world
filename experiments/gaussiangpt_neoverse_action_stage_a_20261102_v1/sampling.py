"""Disjoint, exhaustive, reproducible RGB sampling for free-code fitting."""
from collections import Counter
import hashlib
import json
import random


def build(config, shard):
    starts = config['shards'][str(shard)]
    if len(starts) != 84 or config['batch_size'] != 4:
        raise ValueError('Expected 84 complete windows per shard and effective B4')
    randomizer = random.Random(config['sampler_seed'] + int(shard))
    items = []
    for sweep in range(1, config['sweeps'] + 1):
        pairs = [(start, h) for start in starts for h in range(1, 10)]
        randomizer.shuffle(pairs)
        for offset in range(0, len(pairs), 4):
            items.append(dict(step=len(items)+1, sweep=sweep,
                              batch=[dict(start=s, horizon=h) for s, h in pairs[offset:offset+4]]))
    return items


def audit(config):
    groups = [set(config['shards'][str(i)]) for i in (0, 1)]
    if groups[0] & groups[1] or groups[0] | groups[1] != set(range(168)):
        raise RuntimeError('Shards must cover all 168 windows exactly once')
    summary = {}
    for shard in (0, 1):
        items = build(config, shard)
        expected = {(s, h) for s in groups[shard] for h in range(1, 10)}
        for sweep in range(1, config['sweeps']+1):
            rows = [row for row in items if row['sweep']==sweep]
            pairs = Counter((m['start'], m['horizon']) for row in rows for m in row['batch'])
            if len(rows)!=189 or set(pairs)!=expected or set(pairs.values())!={1}:
                raise RuntimeError('A sweep missed or repeated a window/horizon pair')
        if len(items)!=config['steps_per_shard'] or any(len(row['batch'])!=4 for row in items):
            raise RuntimeError('Incorrect schedule budget')
        summary[str(shard)] = dict(starts=len(groups[shard]), pairs=len(expected), updates=len(items),
            direct_rgb_visits_per_pair=config['sweeps'],
            sha256=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return dict(status='passed', full_windows=168, future_pairs=1512, shards=summary)

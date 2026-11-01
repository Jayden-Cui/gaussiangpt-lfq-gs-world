"""Exact balanced visits to all TRAIN (window, future-step) pairs."""
from collections import Counter
import hashlib
import json
import random


def schedule(config):
    if config['starts'] != list(range(168)) or config['horizons'] != list(range(1, 10)):
        raise ValueError('Full adjacent T10 coverage required')
    if config['batch_size'] != 4:
        raise ValueError('Effective B4 required')
    rng = random.Random(config['sampler_seed'])
    result = []
    for sweep in range(1, config['sweeps'] + 1):
        pairs = [(start, h) for start in config['starts'] for h in config['horizons']]
        rng.shuffle(pairs)
        for offset in range(0, len(pairs), 4):
            result.append(dict(phase='render', sweep=sweep,
                               step=config['parent_step'] + len(result) + 1,
                               batch=[dict(start=start, horizon=h) for start, h in pairs[offset:offset+4]]))
    return result


def audit(items, config):
    expected = {(s, h) for s in config['starts'] for h in config['horizons']}
    all_counts = Counter()
    for sweep in range(1, config['sweeps'] + 1):
        rows = [row for row in items if row['sweep'] == sweep]
        pairs = Counter((m['start'], m['horizon']) for row in rows for m in row['batch'])
        if len(rows) != 378 or set(pairs) != expected or set(pairs.values()) != {1}:
            raise RuntimeError('Each sweep must render each of 1512 pairs exactly once')
        all_counts.update(pairs)
    if len(items) != config['additional_steps'] or set(all_counts.values()) != {config['sweeps']}:
        raise RuntimeError('Incorrect total render coverage')
    if any(len(row['batch']) != 4 or row['phase'] != 'render' for row in items):
        raise RuntimeError('Batch/phase changed')
    if [row['step'] for row in items] != list(range(config['parent_step']+1, config['final_step']+1)):
        raise RuntimeError('Global step sequence changed')
    return dict(status='passed', sweeps=config['sweeps'], unique_window_horizon_pairs=len(expected),
                direct_rgb_visits_per_pair=config['sweeps'], updates=len(items),
                effective_batch_size=4, window_visits=len(items)*4,
                sha256=hashlib.sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest())

"""Deterministic, source-balanced schedules for future multi-video GS training.

This module prepares data order only. It does not select a teacher, train a
model, or claim that the historically exposed evaluation video is pristine.
"""
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
import sys

SOURCE_ROOT = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_multisource_action_data_20261102_v1')
sys.path.insert(0, str(SOURCE_ROOT))
from source_store import TRAIN, EVAL

STARTS = tuple(range(168))
HORIZONS = tuple(range(1, 10))


def build_schedule(phase, sweeps, seed):
    """Each B4 update contains two windows from each of two TRAIN videos."""
    if phase not in ('distill', 'joint') or type(sweeps) != int or sweeps <= 0 or type(seed) != int:
        raise ValueError('Expected a phase, positive integer sweeps and integer seed')
    if len(TRAIN) != 2 or set(TRAIN) & set(EVAL):
        raise RuntimeError('Expected two disjoint training sources')
    generator = random.Random(seed)
    schedule = []
    for sweep in range(1, sweeps + 1):
        pools = {}
        for clip in TRAIN:
            examples = ([(s, None) for s in STARTS] if phase == 'distill'
                        else [(s, h) for s in STARTS for h in HORIZONS])
            generator.shuffle(examples)
            pools[clip] = examples
        for offset in range(0, len(pools[TRAIN[0]]), 2):
            batch = []
            for clip in TRAIN:
                for start, horizon in pools[clip][offset:offset + 2]:
                    row = {'clip': clip, 'start': start}
                    if horizon is not None:
                        row.update(horizon=horizon, frame=start + horizon)
                    batch.append(row)
            generator.shuffle(batch)
            schedule.append({'step': len(schedule) + 1, 'phase': phase, 'sweep': sweep, 'batch': batch})
    return schedule


def audit_schedule(schedule, phase, sweeps):
    if phase not in ('distill', 'joint') or type(sweeps) != int or sweeps <= 0:
        raise ValueError('Invalid schedule specification')
    per_source = 168 if phase == 'distill' else 1512
    if len(schedule) != (per_source // 2) * sweeps:
        raise RuntimeError('Unexpected number of source-balanced B4 updates')
    if [row.get('step') for row in schedule] != list(range(1, len(schedule) + 1)):
        raise RuntimeError('Schedule step numbers are not continuous')
    for sweep in range(1, sweeps + 1):
        rows = [row for row in schedule if row.get('sweep') == sweep]
        observed = Counter()
        for row in rows:
            batch = row.get('batch', [])
            if row.get('phase') != phase or len(batch) != 4:
                raise RuntimeError('Phase or logical batch size differs')
            if Counter(x.get('clip') for x in batch) != Counter({clip: 2 for clip in TRAIN}):
                raise RuntimeError('Batch is not balanced over training sources')
            for item in batch:
                clip, start = item['clip'], item['start']
                if clip in EVAL or clip not in TRAIN or type(start) != int or start not in STARTS:
                    raise RuntimeError('Evaluation source or invalid window entered training')
                if phase == 'distill':
                    if 'horizon' in item or 'frame' in item:
                        raise RuntimeError('Distillation examples represent all nine codes')
                    observed[(clip, start)] += 1
                else:
                    h = item.get('horizon')
                    if type(h) != int or h not in HORIZONS or item.get('frame') != start + h:
                        raise RuntimeError('Future frame or horizon differs')
                    observed[(clip, start, h)] += 1
        expected = ({(c, s) for c in TRAIN for s in STARTS} if phase == 'distill' else
                    {(c, s, h) for c in TRAIN for s in STARTS for h in HORIZONS})
        if set(observed) != expected or set(observed.values()) != {1}:
            raise RuntimeError('An example is missing, duplicated or comes from another source')
    digest = hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'status': 'passed', 'phase': phase, 'sweeps': sweeps, 'train_sources': list(TRAIN),
            'eval_sources_excluded': list(EVAL), 'batch_size': 4, 'per_source_per_batch': 2,
            'updates': len(schedule), 'examples_per_source_per_sweep': per_source,
            'frame_stride': 1, 'sha256': digest,
            'scope': 'Data-order validation only; no model or teacher has been trained'}


def evaluation_pairs(clip):
    """Evaluation uses an entire EVAL source; never a random window split."""
    if clip not in EVAL:
        raise ValueError('Expected an EVAL-only video')
    return [{'clip': clip, 'start': s, 'horizon': h, 'frame': s + h}
            for s in STARTS for h in HORIZONS]


def code_index(clip, start):
    """Stable index into a future two-source bank with shape [2,168,9,16]."""
    if clip not in TRAIN or type(start) != int or start not in STARTS:
        raise ValueError('Code bank accepts only valid TRAIN source/window keys')
    return TRAIN.index(clip), start

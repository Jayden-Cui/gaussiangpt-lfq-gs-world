"""Small deterministic balanced schedule, with every (window,horizon) per epoch."""
import random


def make_schedule(starts, horizons, epochs, seed):
    if len(set(starts)) != len(starts) or len(set(horizons)) != len(horizons):
        raise ValueError('Repeated window or horizon')
    if not starts or not horizons or epochs <= 0:
        raise ValueError('Empty training schedule')
    pairs = [(s, h) for s in starts for h in horizons]
    generator = random.Random(seed)
    out = []
    for _ in range(epochs):
        order = list(pairs)
        generator.shuffle(order)
        out.extend(order)
    return out

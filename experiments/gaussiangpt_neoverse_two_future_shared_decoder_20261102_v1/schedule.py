"""One balanced moving/static pair per update for every start and horizon."""
import random


def schedule(starts, horizons, epochs, seed):
    if not starts or not horizons or len(set(starts)) != len(starts) or len(set(horizons)) != len(horizons):
        raise ValueError('Missing or repeated windows/horizons')
    if type(epochs) is not int or epochs <= 0:
        raise ValueError('Invalid epoch count')
    pairs = [(s, h) for s in starts for h in horizons]
    generator = random.Random(seed)
    result = []
    for _ in range(epochs):
        order = list(pairs)
        generator.shuffle(order)
        result.extend(order)
    return result

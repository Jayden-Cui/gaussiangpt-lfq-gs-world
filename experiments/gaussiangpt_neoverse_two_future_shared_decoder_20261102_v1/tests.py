"""Check the fixed paired schedule before any GPU allocation."""
from collections import Counter
from schedule import schedule


starts, horizons = [35, 123], list(range(1, 10))
items = schedule(starts, horizons, 20, 20261107)
assert len(items) == 360
assert Counter(items) == {(s, h): 20 for s in starts for h in horizons}
assert all(len(set(items[i:i+18])) == 18 for i in range(0, 360, 18))
assert items == schedule(starts, horizons, 20, 20261107)
assert items != schedule(starts, horizons, 20, 20261108)
for bad in (([35,35], horizons, 1, 0), (starts, [], 1, 0), (starts, horizons, 0, 0)):
    try:
        schedule(*bad)
        raise AssertionError('Invalid paired schedule accepted')
    except ValueError:
        pass
print('PASS: 360 updates; 20 visits per same-S0 motion/static horizon pair; deterministic schedule')

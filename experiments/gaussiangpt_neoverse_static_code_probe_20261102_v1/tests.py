"""Verify balanced coverage and reproducibility of the declared training budget."""
from collections import Counter
from schedule import make_schedule


starts, horizons, epochs = [35, 123], list(range(1, 10)), 30
schedule = make_schedule(starts, horizons, epochs, 20261106)
assert len(schedule) == 540
assert Counter(schedule) == {(s, h): 30 for s in starts for h in horizons}
assert all(len(set(schedule[i:i+18])) == 18 for i in range(0, 540, 18))
assert schedule == make_schedule(starts, horizons, epochs, 20261106)
assert schedule != make_schedule(starts, horizons, epochs, 20261107)
try:
    make_schedule([35, 35], horizons, 1, 0)
    raise AssertionError('Duplicate window accepted')
except ValueError:
    pass
print('PASS: 540 updates; 30 visits for each of 18 window/horizon pairs; deterministic ordering')

"""Verify balanced coverage and reproducibility of the declared training budget."""
from collections import Counter
from schedule import make_schedule
from probe import predicted_node_limit


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
assert predicted_node_limit({'p': {'max_predicted_latent_nodes': 25000}}) == 25000
for bad in (0, -1, True, 1.5):
    try:
        predicted_node_limit({'p': {'max_predicted_latent_nodes': bad}})
        raise AssertionError('Invalid model node limit accepted')
    except ValueError:
        pass
print('PASS: 540 updates; 30 visits for each of 18 window/horizon pairs; deterministic ordering')
print('PASS: evaluation uses the bound model node limit and rejects invalid limits')

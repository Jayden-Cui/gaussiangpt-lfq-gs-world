"""Exercise raw-state, cache-corruption and future-leak detection on CPU."""
import copy
from dataclasses import dataclass
import types

import torch
from torch import nn
import driver_v5
from guard import audit_raw_c0


@dataclass
class Memory:
    keys: torch.Tensor
    coordinates: torch.Tensor


def memories(base, states, initial_only=False, *, leak=False):
    values = [dict(states[0], pts=s['pts']) for s in states] if initial_only else states
    result = []
    for h in range(1, 10):
        keys = torch.cat([v['features'] for v in values[:h+1]])
        coords = torch.cat([v['coordinates'] for v in values[:h+1]])
        if leak:
            keys = keys + values[-1]['features'].mean()
        result.append(Memory(keys, coords))
    return result


class Model(nn.Module):
    variant = 'C0'
    def forward_cached(self, memories):
        return torch.stack([m.keys.mean().expand(16) for m in memories])


def make_window():
    states = [dict(coordinates=torch.tensor([[i, 0, 0]], dtype=torch.int32),
                   features=torch.full((1, 64), float(i)), pts=i / 30) for i in range(10)]
    base = nn.Linear(1, 1).requires_grad_(False).eval()
    window = dict(states=states, memories=memories(base, states),
                  initial_memories=memories(base, states, initial_only=True))
    return base, window


class Store:
    def __init__(self, value): self.value = value
    def get(self, start): return self.value


def rejected(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError('Expected an invalid cache or future leak to be rejected')


def run():
    model = Model().eval()
    base, window = make_window()
    result = audit_raw_c0(model, base, Store(window), memories, starts=(35,), horizons=(1, 3, 7))
    assert result['status'] == 'passed' and len(result['raw_future_perturbations']) == 3
    assert all(r['later_memory_mutation_detected'] for r in result['raw_future_perturbations'])

    wrong = copy.deepcopy(window)
    wrong['memories'][2].keys[0, 0] += 1
    rejected(lambda: audit_raw_c0(model, base, Store(wrong), memories, starts=(35,), horizons=(3,)))
    leaking = lambda b, s, initial_only=False: memories(b, s, initial_only, leak=True)
    corrupted = dict(window, memories=leaking(base, window['states']),
                     initial_memories=leaking(base, window['states'], initial_only=True))
    rejected(lambda: audit_raw_c0(model, base, Store(corrupted), leaking, starts=(35,), horizons=(3,)))

    # The wrapper must invoke the new check for C0, leave C1 unchanged, and
    # restore the original evaluator functions even on failure.
    class Runtime:
        def digest(self, module): return 'unchanged'
    decoder = types.SimpleNamespace(base=base)
    store = Store(window)
    def load(*a, **kw): return ({'runtime': Runtime()}, None, None, decoder, model, None)
    def old_check(m, s): return {'status': 'passed', 'rows': ['existing']}
    fake = types.SimpleNamespace(load_trained=load, check_prefix_causality=old_check)
    def evaluate(*args):
        fake.load_trained()
        return fake.check_prefix_causality(model, store)
    fake.evaluate = evaluate
    previous_get_memories = driver_v5.get_make_memories
    driver_v5.get_make_memories = lambda: memories
    try:
        value = driver_v5.evaluate_with_guard(fake, None, 1, None, source_metadata={'test': True})
        assert value['raw_state_causality']['status'] == 'passed'
        assert value['wrapper'] == {'test': True} and value['rows'] == ['existing']
        assert fake.load_trained is load and fake.check_prefix_causality is old_check
        model.variant = 'C1'
        value = driver_v5.evaluate_with_guard(fake, None, 1, None, source_metadata={'test': True})
        assert value == {'status': 'passed', 'rows': ['existing']}
        model.variant = 'C0'
        store.value = wrong
        rejected(lambda: driver_v5.evaluate_with_guard(fake, None, 1, None, source_metadata={}))
        assert fake.load_trained is load and fake.check_prefix_causality is old_check
    finally:
        driver_v5.get_make_memories = previous_get_memories
    print('PASS: raw future perturbation, cache corruption, future leakage and evaluator wrapper')


if __name__ == '__main__': run()

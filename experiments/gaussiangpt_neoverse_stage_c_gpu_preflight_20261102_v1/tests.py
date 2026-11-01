"""CPU guards for stage-C engineering preflight; no real teacher is created."""
import math
from pathlib import Path
import torch
from torch import nn
import pilot
import coordinator


def expect_error(function):
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid pilot input was accepted')


def main():
    value = torch.zeros(168, 9, 16)
    assert pilot.check_codes(value).shape == (168, 9, 16)
    expect_error(lambda: pilot.check_codes(torch.zeros(168, 9, 15)))
    bad = value.clone()
    bad[0, 0, 0] = math.nan
    expect_error(lambda: pilot.check_codes(bad))
    class Posterior(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = nn.Linear(2, 2)
            self.project_in = nn.Linear(2, 2)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(2, 2)
            self.encoder_backbone = nn.Linear(2, 2)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.posterior = Posterior()
            self.encoder = Encoder()
            self.time_projection = nn.Linear(2, 2)
            self.delta_projection = nn.Linear(2, 2)
    model = Model()
    expect_error(lambda: pilot.required_gradients(model, 'C2'))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    assert set(pilot.required_gradients(model, 'C2')) == {
        'queries', 'projection', 'encoder_input', 'encoder_backbone', 'time_projection', 'delta_projection'}
    called = []
    coordinator.execute(called.append)
    assert called == ['C0', 'C1', 'C2']
    called.clear()
    def failed(variant):
        called.append(variant)
        if variant == 'C1':
            raise RuntimeError('synthetic failure')
    expect_error(lambda: coordinator.execute(failed))
    assert called == ['C0', 'C1']
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or pilot.sha(root / name) != digest:
            raise RuntimeError('Pilot source changed')
    pilot.write(root / 'CPU_TESTS.json', {'status': 'passed', 'code_sha256': pilot.sha(root / 'CODE.sha256'),
                'rejects_invalid_synthetic_code': True, 'requires_all_variant_gradients': True,
                'scope': 'CPU synthetic guards only; no model training or quality claim'})
    print('PASS: synthetic code scope and variant gradient guards')


if __name__ == '__main__':
    main()

"""Verify the target-support oracle selects predictions, never target features."""
from types import SimpleNamespace
import torch
from diagnose import oracle_support


def main():
    condition = SimpleNamespace(coordinates=torch.tensor([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    prediction = {'features': torch.arange(3 * 4 * 2).reshape(3, 4, 2)}
    indices = torch.tensor([2, 0], dtype=torch.int64)
    result = oracle_support(condition, prediction, 2, indices)
    assert torch.equal(result['coordinates'], condition.coordinates[indices])
    assert torch.equal(result['features'], prediction['features'][1, indices])
    for bad in (torch.tensor([], dtype=torch.int64), torch.tensor([1, 1]), torch.tensor([4]), torch.tensor([-1])):
        try:
            oracle_support(condition, prediction, 2, bad)
        except ValueError:
            pass
        else:
            raise RuntimeError('Invalid target support accepted')
    print('PASS: oracle selects only predicted features and rejects invalid target support')


if __name__ == '__main__':
    main()

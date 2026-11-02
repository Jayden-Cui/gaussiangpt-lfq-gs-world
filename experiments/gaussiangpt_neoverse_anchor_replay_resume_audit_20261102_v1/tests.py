"""Exercise exact optimizer-state comparison and Adam untouched-parameter behavior."""
import copy
import torch
from audit import equal


def main():
    assert equal({'a':[torch.tensor([1.,2.]),3]}, {'a':[torch.tensor([1.,2.]),3]})
    assert not equal({'a':torch.tensor([1.])},{'a':torch.tensor([2.])})
    assert not equal({'a':1},{'b':1})
    x=torch.nn.Parameter(torch.tensor([1.]));y=torch.nn.Parameter(torch.tensor([2.]))
    optimizer=torch.optim.Adam([x,y],lr=.01)
    (x.square()+y.square()).sum().backward();optimizer.step()
    saved=copy.deepcopy(optimizer.state_dict())
    x2=torch.nn.Parameter(x.detach().clone());y2=torch.nn.Parameter(y.detach().clone())
    restored=torch.optim.Adam([x2,y2],lr=.01);restored.load_state_dict(saved)
    assert equal(saved,restored.state_dict())
    before=y2.detach().clone();restored.zero_grad(set_to_none=True);x2.grad=torch.ones_like(x2);restored.step()
    assert torch.equal(y2,before)
    print('PASS: exact nested state comparison, Adam restore and untouched parameter')


if __name__=='__main__':main()

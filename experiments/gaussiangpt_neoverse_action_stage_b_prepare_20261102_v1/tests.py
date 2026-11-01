"""CPU witnesses on the exact successful two-window decoder and real data."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

from b_core import FreeCodes,initialize_codes,load_bound_sources,optimizer_for
from b_sampling import schedule,audit

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Preparation code changed')
    bundle=load_bound_sources('cpu')
    decoder,posterior,store,p=(bundle[k] for k in ('decoder','posterior','store','p'))
    fm,runtime=bundle['fm'],bundle['runtime']
    sampler=audit(schedule(20,20261102),20)
    starts=[0,35,123,167]
    codes=initialize_codes(posterior,decoder,store,starts,'cpu')
    if not torch.equal(codes[1],decoder.codes[0].detach()) or not torch.equal(codes[2],decoder.codes[1].detach()):
        raise RuntimeError('Anchor codes do not belong to the original decoder gauge')
    with torch.no_grad():
        window=store.get(0,'cpu');reference=posterior([window['memories']])['codes'][0]
    if not torch.equal(codes[0],reference):raise RuntimeError('Other windows do not use the matching posterior')
    if any(x.requires_grad for x in posterior.parameters()):raise RuntimeError('Initialization posterior must remain frozen')
    # An inactive code with existing Adam momentum must not change.
    toy=nn.Linear(1,1);bank=FreeCodes([0,1],torch.ones(2,9,16),'cpu')
    opt=optimizer_for(toy,bank,p,.0001,.003)
    bank.code(1).square().sum().backward();opt.step();before=bank.code(1).detach().clone()
    opt.zero_grad(set_to_none=True);bank.code(0).square().sum().backward();opt.step()
    if bank.code(1).grad is not None or not torch.equal(before,bank.code(1).detach()):
        raise RuntimeError('Unselected window moved under Adam momentum')
    # Actual cached spatial state: state loss must train both Z and shared decoder.
    from support_decoder import state_loss
    window=store.get(35,'cpu');z=nn.Parameter(codes[1].clone());decoder.train()
    prediction=decoder(window['condition'],z)
    loss,terms=state_loss(prediction,window['targets'],store.frames['std'])
    loss.backward()
    grads=dict(free_code=float(z.grad.norm()),decoder_features=float(decoder.base.features.weight.grad.norm()),
               occupancy=float(decoder.occupancy.weight.grad.norm()))
    if any(not torch.isfinite(torch.tensor(v)) or v<=0 for v in grads.values()):
        raise RuntimeError('Real state loss did not reach a required trainable module')
    if any(x.grad is not None for x in decoder.parameters() if not x.requires_grad):
        raise RuntimeError('Frozen initial encoder acquired gradients')
    if fm.frozen_digest(decoder)!=bundle['frozen_digest']:
        raise RuntimeError('Frozen initial decoder context changed')
    result=dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),sampler=sampler,
      exact_two_window_decoder_digest=runtime.digest(decoder),
      original_anchor_codes_preserved=True,remaining_codes_use_matching_posterior=True,
      anchors=[35,123],real_window=35,real_latent_to_code_and_decoder_gradients=grads,
      frozen_initial_context_unchanged=True,inactive_adam_code_unchanged=True,
      scope='CPU preparation only; no native RGB gradient, no stage-B optimization or scientific result')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: bound two-window decoder and anchors, full sampler, real-state gradients and inactive Adam isolation')


if __name__=='__main__':main()

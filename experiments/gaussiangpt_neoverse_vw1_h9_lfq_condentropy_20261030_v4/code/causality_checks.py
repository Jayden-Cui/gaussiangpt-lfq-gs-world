"""Prefix encoder plus stricter causal GS temporal decoder intervention tests."""
import torch

def close(a,b,tolerance=1e-5):
    error=float((a-b).abs().max());assert error<=tolerance,error
    return error

def check_causality(model,window,condition):
    prior_mode=model.training;reports=[]
    for training in [False,True]:
        model.train(training)
        for cutoff in [1,4,8]:
            changed=[]
            for i,s in enumerate(window):
                t=dict(s)
                if i>cutoff:
                    # New future-only coordinates, permutation and changed count.
                    t['features']=torch.cat([-7*s['features'].flip(0)+.3,s['features'][:1]],0)
                    t['coordinates']=torch.cat([s['coordinates'].flip(0)+1000,s['coordinates'][:1]+3000],0)
                    t['pts']=float(s['pts'])+2
                changed.append(t)
            with torch.no_grad():
                hs=list(range(1,cutoff+1))
                a=model.encode_windows([window],hs);b=model.encode_windows([changed],hs)
                c=model.encode_windows([window[:cutoff+1]],hs)
                perturb=close(a['z_prequant'],b['z_prequant']);truncate=close(a['z_prequant'],c['z_prequant'])
                assert torch.equal(a['indices'],b['indices']) and torch.equal(a['indices'],c['indices'])
            reports.append(dict(train=training,cutoff=cutoff,future_perturb_error=perturb,truncated_prefix_error=truncate,ids_equal=True))
    model.eval();leaves=[]
    for s in window:
        t=dict(s);t['features']=s['features'].detach().clone().requires_grad_(True);leaves.append(t)
    z=model.encode_windows([leaves],[6])
    scalar=(z['z_prequant']*torch.linspace(-1,1,64,device='cuda')).sum()
    grads=torch.autograd.grad(scalar,[s['features'] for s in leaves],allow_unused=True)
    future=max(0 if g is None else float(g.abs().max()) for g in grads[7:])
    history=sum(0 if g is None else float(g.square().sum()) for g in grads[:5])**.5
    assert future==0 and history>0
    # Locality makes this small coordinate subset sufficient for the temporal
    # dependency test; full supports are separately exercised by render witness.
    c={k:v[:128] if torch.is_tensor(v) and v.ndim>0 else v for k,v in condition.items()}
    code=torch.randn(1,9,64,device='cuda',requires_grad=True)
    preds=model.decode_latents(code,[c])[0]
    scalar=(preds[3]['features']*torch.linspace(-1,1,64,device='cuda')).sum()
    code_grad=torch.autograd.grad(scalar,code)[0]
    assert float(code_grad[:,4:].abs().max())==0
    assert bool((code_grad[:,:4].square().sum(-1)>0).all())
    with torch.no_grad():
        changed=code.detach().clone();changed[:,4:]=1000
        result=model.decode_latents(changed,[c])[0]
        decoder_error=max(close(preds[h]['features'],result[h]['features']) for h in range(4))
        changed_initial=dict(c);changed_initial['features']=c['features']*.7
        changed_initial['base_features']=c['base_features']*.7
        initial_changed=model.decode_latents(code.detach(),[changed_initial])[0]
        initial_effect=float((initial_changed[8]['features']-preds[8]['features']).abs().max())
        assert initial_effect>0
    model.train(prior_mode)
    return dict(passed=True,encoder_prefix=reports,future_feature_gradient_max_abs=future,
        history_gradient_norm=history,decoder_future_code_gradient_max_abs=0.,decoder_future_code_perturb_error=decoder_error,
        decoder_prefix_code_gradient_norms=code_grad[0,:4].norm(dim=-1).tolist(),initial_effect_at_h9=initial_effect,
        scope='Conditional GS-prefix causality. Decoder is deliberately stricter than official VideoWorld global normalization; NeoVerse offline extraction is outside this guarantee.')

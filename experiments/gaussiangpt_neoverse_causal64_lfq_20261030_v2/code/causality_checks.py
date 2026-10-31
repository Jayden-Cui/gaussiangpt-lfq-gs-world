"""Counterfactual future perturbation and truncated-prefix equivalence."""
import torch


def compare(a, b, tolerance=1e-5):
    error=float((a['z_prequant']-b['z_prequant']).abs().max())
    assert error<=tolerance, ('Future/prefix changed prior floating feature',error)
    assert torch.equal(a['indices'],b['indices']), 'Future/prefix changed prior integer IDs'
    return error


def check_causality(model, window):
    assert len(window)==10
    previous_mode=model.training
    reports=[]
    for train_mode in [False,True]:
        model.train(train_mode)
        for cutoff in [1,4,8]:
            changed=[]
            for j,state in enumerate(window):
                modified=dict(state)
                if j>cutoff:
                    modified['features']=-7*state['features']+.3
                    modified['coordinates']=state['coordinates']+state['coordinates'].new_tensor([1000,2000,3000])
                    modified['pts']=float(state['pts'])+2.
                changed.append(modified)
            with torch.no_grad():
                baseline=model.encode_prefix(window,cutoff)
                perturbed=model.encode_prefix(changed,cutoff)
                truncated=model.encode_prefix(window[:cutoff+1])
                pe=compare(baseline,perturbed);te=compare(baseline,truncated)
            reports.append({'mode':'train' if train_mode else 'eval','target_offset':cutoff,
                'changed_later_frames':len(window)-cutoff-1,'future_features_coordinates_pts_changed':True,
                'future_perturb_prequant_max_abs':pe,'truncated_prefix_prequant_max_abs':te,'ids_equal':True})
    model.eval()
    # Future state features and earlier history remain separate leaves.
    differentiable=[]
    for state in window:
        copy=dict(state);copy['features']=state['features'].detach().clone().requires_grad_(True)
        differentiable.append(copy)
    cutoff=6
    encoded=model.encode_prefix(differentiable,cutoff)
    weight=torch.linspace(-1,1,64,device=encoded['z_prequant'].device)
    scalar=(encoded['z_prequant']*weight).sum()
    leaves=[s['features'] for s in differentiable]
    grads=torch.autograd.grad(scalar,leaves,allow_unused=True)
    future_max=max(0. if g is None else float(g.abs().max()) for g in grads[cutoff+1:])
    assert future_max==0.,future_max
    # This excludes the two endpoint frames, so it checks actual older history.
    history_norm=sum(0. if g is None else float(g.square().sum()) for g in grads[:cutoff-1])**.5
    assert history_norm>0 and all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    model.train(previous_mode)
    return {'passed':True,'scope':'Conditional GS-input causality, not upstream NeoVerse causality',
        'future_and_prefix_tests':reports,'future_feature_gradient_max_abs':future_max,
        'older_history_feature_gradient_norm':history_norm}

"""Fix exactly the proven softmax-common key biases at their loaded values.

This policy is shared by the benchmark and actual training. It never removes
parameters from optimizer groups, clears moments, or changes forward values.
Call after every complete model/native/optimizer restore; requires_grad is not
stored in a state_dict. The constant arm may already omit all27 biases from its
optimizer, in which case their absent state stays absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

POLICY = 'qformer_softmax_key_bias_fixed_v1'
PARAMETER_NAMES = tuple(
    f'posteriors.{h}.{suffix}.k_proj.bias'
    for h in range(9)
    for suffix in ('layers.0.cross_attn_token_to_image',
                   'layers.1.cross_attn_token_to_image', 'final_attn_token_to_image')
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(('tensor:' + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
            # These small selected tensors also work for bfloat16/scalars and
            # require no NumPy dependency or checkpoint deserialization.
            digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update((type(item).__name__ + ':' + str(len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update((type(item).__name__ + ':' + repr(item)).encode())
        digest.update(b'\0')

    visit(value)
    return digest.hexdigest()


def _group_layout(optimizer):
    return tuple(tuple(id(parameter) for parameter in group['params']) for group in optimizer.param_groups)


def _selected(model):
    named = dict(model.named_parameters())
    present = {name for name in named if name.startswith('posteriors.') and name.endswith('.k_proj.bias')}
    _require(present == set(PARAMETER_NAMES), 'Expected exactly27 identified posterior key biases')
    selected = {name: named[name] for name in PARAMETER_NAMES}
    for name, parameter in selected.items():
        layer = model.get_submodule(name.rsplit('.', 1)[0])
        _require(isinstance(layer, torch.nn.Linear) and layer.bias is parameter
                 and layer.in_features == 64 and layer.out_features == 32
                 and tuple(parameter.shape) == (32,) and parameter.is_floating_point(),
                 'Unexpected key projection parameter geometry: '+name)
        _require(bool(torch.isfinite(parameter).all()), 'Nonfinite fixed key bias: '+name)
    return named, selected


def _state_hash(selected, optimizer, membership):
    # Do not index the optimizer defaultdict for an absent parameter: that
    # would silently create a new state entry in the constant arm.
    return _hash({name: dict(in_optimizer=id(parameter) in membership,
                            has_state=parameter in optimizer.state,
                            state=optimizer.state.get(parameter))
                  for name, parameter in selected.items()})


@dataclass(frozen=True)
class NullKeyBiasGuard:
    parameter_names: tuple[str, ...]
    parameter_ids: tuple[tuple[str, int], ...]
    other_requires_grad: tuple[tuple[str, bool], ...]
    optimizer_group_layout: tuple[tuple[int, ...], ...]
    values_sha256: str
    optimizer_state_sha256: str
    in_optimizer_count: int


def freeze_null_key_bias(model, optimizer) -> NullKeyBiasGuard:
    """Freeze only the exact27 loaded biases without touching optimizer state.

    The optimizer must already be constructed/restored. Posterior arms retain
    all27 existing group members; the constant arm retains zero members/state.
    Mixed membership, duplicated parameters or changed shapes are errors.
    """
    named, selected = _selected(model)
    layout = _group_layout(optimizer)
    entries = [identity for group in layout for identity in group]
    _require(len(entries) == len(set(entries)), 'Optimizer groups contain duplicate parameters')
    membership = set(entries)
    count = sum(id(parameter) in membership for parameter in selected.values())
    _require(count in (0, 27), 'All27 fixed biases must retain either all or no optimizer memberships')
    if count == 0:
        _require(all(not parameter.requires_grad and parameter not in optimizer.state
                     for parameter in selected.values()), 'An omitted key bias must already be frozen and stateless')
    guard = NullKeyBiasGuard(
        parameter_names=PARAMETER_NAMES,
        parameter_ids=tuple((name, id(parameter)) for name, parameter in named.items()),
        other_requires_grad=tuple((name, parameter.requires_grad) for name, parameter in named.items()
                                  if name not in selected),
        optimizer_group_layout=layout,
        values_sha256=_hash(selected),
        optimizer_state_sha256=_state_hash(selected, optimizer, membership),
        in_optimizer_count=count,
    )
    for parameter in selected.values():
        parameter.requires_grad_(False)
        parameter.grad = None  # A zero tensor would still permit Adam momentum updates.
    assert_null_key_bias_unchanged(model, optimizer, guard)
    return guard


def assert_null_key_bias_unchanged(model, optimizer, guard: NullKeyBiasGuard) -> dict:
    """Verify the policy after updates/before saves and return a JSON record."""
    _require(isinstance(guard, NullKeyBiasGuard), 'Expected NullKeyBiasGuard returned after restore')
    named, selected = _selected(model)
    _require(tuple((name, id(parameter)) for name, parameter in named.items()) == guard.parameter_ids,
             'Model parameter identities changed after policy application')
    _require(_group_layout(optimizer) == guard.optimizer_group_layout, 'Optimizer group membership/order changed')
    _require(all(named[name].requires_grad == flag for name, flag in guard.other_requires_grad),
             'A non-policy parameter changed its requires_grad flag')
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in selected.values()),
             'A fixed key bias became trainable or received a gradient')
    _require(_hash(selected) == guard.values_sha256, 'A fixed key bias value changed')
    membership = {identity for group in guard.optimizer_group_layout for identity in group}
    _require(_state_hash(selected, optimizer, membership) == guard.optimizer_state_sha256,
             'A fixed key bias optimizer state changed or was created')
    return dict(policy=POLICY, parameter_names=list(guard.parameter_names), count=27, scalar_count=864,
                values_sha256=guard.values_sha256, optimizer_state_sha256=guard.optimizer_state_sha256,
                in_optimizer_count=guard.in_optimizer_count, unchanged=True)

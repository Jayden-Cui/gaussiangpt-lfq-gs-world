"""Decoder-only precision adapter; importing this file does not import torch/ME.

The adapter changes only the three named decoder-side modules. CPU snapshots
bind parameters and all registered buffers, including nonpersistent buffers.
"""
from __future__ import annotations

import hashlib
import json


SELECTED_MODULES = ('ae.decoder', 'ae.feature_heads', 'ae.gaussian_multiplier')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def selected(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in SELECTED_MODULES)


def _live_state(model):
    tensors, kinds = {}, {}
    for kind, values in (('parameter', model.named_parameters(remove_duplicate=False)),
                         ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, tensor in values:
            require(name not in tensors, 'Duplicate model state name: ' + name)
            tensors[name], kinds[name] = tensor, kind
    return tensors, kinds


def _snapshot(model):
    tensors, kinds = _live_state(model)
    return {name: value.detach().cpu().clone() for name, value in tensors.items()}, kinds


def _tensor_digest(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    header = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_digest(state, kinds):
    entries = {name: {'kind': kinds[name], 'sha256': _tensor_digest(tensor)}
               for name, tensor in state.items()}
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _dtype_counts(state, kinds):
    result = {}
    for name, tensor in state.items():
        bucket = result.setdefault(str(tensor.dtype), {'tensors': 0, 'parameters': 0, 'buffers': 0,
                                                      'elements': 0, 'bytes': 0})
        bucket['tensors'] += 1
        bucket['parameters' if kinds[name] == 'parameter' else 'buffers'] += 1
        bucket['elements'] += tensor.numel()
        bucket['bytes'] += tensor.numel() * tensor.element_size()
    return result


class DecoderPrecision:
    def __init__(self, model, decoder_precision):
        import torch
        require(decoder_precision in ('float32', 'float64'), 'Decoder precision must be float32 or float64')
        self.model, self.decoder_precision = model, decoder_precision
        self.dtype = torch.float32 if decoder_precision == 'float32' else torch.float64
        modules = [model.get_submodule(name) for name in SELECTED_MODULES]
        require(all(isinstance(module, torch.nn.Module) for module in modules), 'Selected decoder components must be modules')
        live, _ = _live_state(model)
        require(bool(live), 'Model must expose parameters or buffers')
        for name, tensor in live.items():
            require(not tensor.is_complex(), 'Complex model state is unsupported: ' + name)
            if tensor.is_floating_point():
                require(tensor.dtype == torch.float32, 'Original floating model state must be float32: ' + name)
                require(bool(torch.isfinite(tensor).all()), 'Nonfinite original model state: ' + name)
        # Shared registered tensor objects across the boundary would also cast
        # encoder state. Reject this before changing any module.
        selected_ids = {id(tensor) for name, tensor in live.items() if selected(name)}
        require(not any(id(tensor) in selected_ids for name, tensor in live.items() if not selected(name)),
                'Selected decoder state aliases nonselected state')
        self.original_state, self.kinds = _snapshot(model)
        self.original_state_digest = _state_digest(self.original_state, self.kinds)
        for module in modules:
            module.to(dtype=self.dtype)
        self.mode_state, mode_kinds = _snapshot(model)
        require(set(self.mode_state) == set(self.original_state) and mode_kinds == self.kinds,
                'Precision conversion changed the registered state schema')
        for name, original in self.original_state.items():
            current = self.mode_state[name]
            if selected(name):
                if original.is_floating_point():
                    require(current.dtype == self.dtype, 'Selected state did not receive requested precision: ' + name)
                require(_tensor_digest(current.to(dtype=original.dtype)) == _tensor_digest(original),
                        'Selected state castback differs from checkpoint: ' + name)
            else:
                require(_tensor_digest(current) == _tensor_digest(original),
                        'Nonselected encoder/model state changed: ' + name)
        self.mode_state_digest = _state_digest(self.mode_state, self.kinds)
        self.report = {
            'schema': 'decoder_precision_adapter_v1', 'decoder_precision': self.decoder_precision,
            'selected_modules': list(SELECTED_MODULES),
            'original_state_digest': self.original_state_digest, 'mode_state_digest': self.mode_state_digest,
            'original_dtype_counts': _dtype_counts(self.original_state, self.kinds),
            'mode_dtype_counts': _dtype_counts(self.mode_state, self.kinds),
            'selected_mode_dtype_counts': _dtype_counts({n: t for n, t in self.mode_state.items() if selected(n)}, self.kinds),
            'original_floating_state_float32': True, 'castback_exact': True, 'encoder_exact': True,
            'nonselected_state_exact': True, 'state_scope': 'All named parameters and registered buffers, including nonpersistent buffers',
        }

    def assert_unchanged(self):
        current, kinds = _snapshot(self.model)
        digest = _state_digest(current, kinds)
        changed = sorted(name for name in set(current) | set(self.mode_state)
                         if name not in current or name not in self.mode_state
                         or kinds.get(name) != self.kinds.get(name)
                         or _tensor_digest(current[name]) != _tensor_digest(self.mode_state[name]))
        require(not changed, 'Mode model state changed: ' + ', '.join(changed[:20]))
        require(digest == self.mode_state_digest, 'Mode model state digest changed')
        return {'unchanged': True, 'state_digest': digest, 'mode_state_digest': self.mode_state_digest,
                'decoder_precision': self.decoder_precision, 'checked_tensors': len(current)}

    def prepare_sparse(self, z, ME):
        import torch
        features = z.F
        require(isinstance(features, torch.Tensor) and features.dtype == torch.float32
                and features.ndim == 2 and features.shape[1] == 64,
                'Original sparse features must be float32 Nx64')
        require(bool(torch.isfinite(features).all()), 'Nonfinite original sparse features')
        if self.decoder_precision == 'float32':
            return z
        before = _tensor_digest(features)
        map_key, manager = z.coordinate_map_key, z.coordinate_manager
        converted = features.double()
        prepared = ME.SparseTensor(features=converted, coordinate_map_key=map_key, coordinate_manager=manager)
        require(prepared.coordinate_map_key is map_key and prepared.coordinate_manager is manager,
                'Sparse precision conversion changed coordinate key or manager identity')
        require(z.F is features and features.dtype == torch.float32 and _tensor_digest(features) == before,
                'Sparse precision conversion changed original float32 features')
        require(prepared.F.dtype == torch.float64 and tuple(prepared.F.shape) == tuple(features.shape)
                and bool(torch.isfinite(prepared.F).all())
                and _tensor_digest(prepared.F) == _tensor_digest(converted),
                'Sparse precision conversion changed feature rows or values')
        return prepared

    def to_render_points(self, points):
        import torch
        require(isinstance(points, dict) and bool(points), 'Renderer points must be a nonempty tensor dictionary')
        result = {}
        for key, value in points.items():
            require(isinstance(value, torch.Tensor) and not value.is_complex(),
                    'Renderer packet contains an unsupported value: ' + repr(key))
            require(bool(torch.isfinite(value).all()), 'Nonfinite renderer packet value: ' + repr(key))
            rendered = value.float() if value.is_floating_point() else value
            require(bool(torch.isfinite(rendered).all()), 'Renderer float32 conversion overflow: ' + repr(key))
            result[key] = rendered
        return result

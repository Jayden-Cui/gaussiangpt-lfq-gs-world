"""Hash-bound, CPU-capable injection into the existing AnchoredLDM/v9 APIs.

No codec, GPU setup, training loop, dataset, installation or source modification.
Call load_pinned(experiments_parent), construct the returned AnchoredLDM, then
attach_adapters. Source modules are compiled from verified bytes without pycache.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True

import torch
from torch.nn.utils.rnn import pad_sequence
from action_bottleneck import ActionBottleneck, validate_quantizer, MODES
from initial_norm import replace_decoder_norms

LFQ_SHA = '9fb68bdcb3763ac580c2a0f931640a70a14a0fed65e0b29a8800ee015b4452eb'
V6 = 'gaussiangpt_neoverse_native_z_20261031_v6/code'
V9 = 'gaussiangpt_neoverse_spatial_action_20261031_v9/spatial_action.py'
PINNED = {
    'spatial64': 'c3a650ca24a41fecfb195ea0f1bf4971c036709a9eeeddb6cfe97cdaa404845f',
    'model_core': '5c6d898cc137c21af5c46482e3cc0f503f20ba70542966bc5b4de08153b22ce6',
    'upstream': '3b3fed071cd115a3bc26265677401859d93b44f987a5c31b176da7fd4b5d9bcf',
    'aligned_ldm': '171f776ce6212da04ec38319c0fcdc4d0a643386736c0f0b100bd9029d4ee454',
    'multiscale64': 'c884059fa2673024d0070879f147322d273061a7c68e9fa66eabd5d5e636a385',
    'anchored_ldm': '2fdc8fb4582c4b667c47b0c5c3d3efeac43409d3864dd0f4103c43c486c6f23f',
}
VENDOR = {
    'qformer.py': '220f24100c6449e5bd30e340b9ab0aeb0ba059c487eaf54f262d0aae19e678da',
    'finite_scalar_quantization.py': 'a5da889b1607c33688146b627975c86cc278a4cbf4e301e989bbfb09b1fc6e21',
    'calvin_ldm.py': '300dd9168bb14dffec9b054b2536e216b2b32c792c3ede767a7055b996b2e355',
    'provenance.json': '21f962642d991549f93b24305c4dc74e7e108eb8d79d2785ed9fd1db62f5776b',
}
SPATIAL_SHA = '16d7e315eea1ab4f818359c079017b1dc9f7319133d98dca0a847e0a3d82539f'


def verified_bytes(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Bound source hash mismatch: '+str(path))
    return data


def _module(name, path, expected):
    data = verified_bytes(path, expected)
    if name in sys.modules:
        module = sys.modules[name]
        if (getattr(module, '__adapter_source_sha256__', None) != expected
                or Path(module.__file__).resolve() != Path(path).resolve()):
            raise RuntimeError('Refusing unverified imported module collision: '+name)
        return module
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__adapter_source_sha256__ = expected
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_pinned(experiments_parent):
    parent = Path(experiments_parent).resolve()
    folder = parent / V6
    for name, digest in PINNED.items():
        verified_bytes(folder / (name+'.py'), digest)
    for name, digest in VENDOR.items():
        verified_bytes(folder/'vendor'/name, digest)
    verified_bytes(parent/V9, SPATIAL_SHA)
    package = importlib.util.find_spec('vector_quantize_pytorch')
    if package is None or not package.submodule_search_locations:
        raise ModuleNotFoundError('Real pinned LFQ is unavailable; do not install or substitute a mock')
    lfq_path = Path(next(iter(package.submodule_search_locations))) / 'lookup_free_quantization.py'
    verified_bytes(lfq_path, LFQ_SHA)
    modules = {name: _module(name, folder/(name+'.py'), digest) for name, digest in PINNED.items()}
    # Verify the actual class import as well, not only a neighboring disk file.
    from vector_quantize_pytorch import LFQ
    actual = sys.modules[LFQ.__module__]
    if Path(actual.__file__).resolve() != lfq_path.resolve():
        raise RuntimeError('LFQ import resolves to an unbound implementation')
    verified_bytes(actual.__file__, LFQ_SHA)
    spatial = _module('_z_adapter_pinned_spatial_v9', parent/V9, SPATIAL_SHA)
    return types.SimpleNamespace(AnchoredLDM=modules['anchored_ldm'].AnchoredLDM,
                                 make_condition=modules['anchored_ldm'].make_condition,
                                 attach_spatial_action=spatial.attach_spatial_action,
                                 LFQ=LFQ, source_parent=parent)


def _posterior_vectors(self, windows):
    if not windows or any(len(w) != 10 for w in windows):
        raise ValueError('Posterior requires full ten-state windows')
    from model_core import qformer_sequence
    prepared = [self.prepare(window) for window in windows]
    vectors = []
    for h in range(1, 10):
        keys = [p['hidden'][:h+1][p['present'][:h+1]] for p in prepared]
        positions = [self.position(p['coordinates'][None].expand(h+1, -1, -1)[p['present'][:h+1]]) for p in prepared]
        k, pos = pad_sequence(keys, batch_first=True), pad_sequence(positions, batch_first=True)
        lengths = torch.tensor([len(value) for value in keys], device=k.device)
        valid = torch.arange(k.shape[1], device=k.device)[None] < lengths[:, None]
        query = self.queries.weight[h-1:h][None].expand(len(windows), -1, -1)
        vectors.append(qformer_sequence(self.posteriors[h-1], k, pos, query, valid))
    return torch.cat(vectors, dim=1)


def _encode_windows(self, windows):
    from aligned_ldm import EncodedInitial
    z = self.posterior_vectors(windows)
    encoded = self.action_bottleneck(z, self.quantizer, inv_temperature=self.inv_temperature)
    contexts = []
    for window in windows:
        initial = window[0]
        contexts.append(EncodedInitial(self.encode_initial(initial), initial['coordinates'], initial['features'],
            (initial['coordinates']._version, initial['features']._version), float(initial['pts']),
            id(self), self.encoder_versions()))
    return dict(**encoded, z_prequant=z, z_quantized=encoded['action_latents'],
                initial_contexts=contexts, posterior_used=True)


def _decode_action_codes(self, codes, conditions, *, initial_contexts=None):
    latents = self.action_bottleneck.codes_to_latents(codes, self.quantizer)
    return self.decode_latents(latents, conditions, initial_contexts=initial_contexts)


def _decode_ids(self, ids, conditions):
    if self.action_bottleneck.mode != 'lfq16':
        raise ValueError('Continuous actions require actual float codes, not diagnostic IDs')
    if (ids.shape != (len(conditions), 9) or ids.dtype not in (torch.int32, torch.int64)
            or not bool(((ids >= 0) & (ids < 65536)).all())):
        raise ValueError('Expected LFQ IDs [B,9] in [0,65536)')
    codes = self.quantizer.indices_to_codes(ids, project_out=False)
    return self.decode_action_codes(codes, conditions)


@torch.no_grad()
def _calibrate_actions(self, batches):
    modes = [(module, module.training) for module in self.modules()]
    self.eval()
    try:
        return self.action_bottleneck.calibrate((self.posterior_vectors(w) for w in batches), self.quantizer)
    finally:
        # Preserve pre-existing mixed modes exactly without recursive resets.
        for module, mode in modes:
            module.training = mode


def attach_adapters(model, *, mode='lfq16', s_ref=None):
    """Inject into a verified newly constructed or loaded common model.

    Attach BEFORE creating optimizers. Existing Parameters retain their exact
    objects and names. No old checkpoint or CODE_MANIFEST is edited.
    """
    source = sys.modules.get(type(model).__module__)
    if (getattr(source, '__adapter_source_sha256__', None) != PINNED['anchored_ldm']
            or type(model).__name__ != 'AnchoredLDM' or model.code_mode != 'posterior'
            or hasattr(model, 'action_bottleneck')):
        raise ValueError('Expected an unadapted pinned posterior AnchoredLDM')
    if mode not in MODES:
        raise ValueError('Unknown action mode')
    validate_quantizer(model.quantizer)
    bottleneck = ActionBottleneck(mode, s_ref=s_ref).to(model.queries.weight.device)
    bottleneck.train(model.training)
    replacements = replace_decoder_norms(model.decoder_backbone)
    model.action_bottleneck = bottleneck
    for name, function in [('posterior_vectors', _posterior_vectors), ('encode_windows', _encode_windows),
                           ('decode_action_codes', _decode_action_codes), ('decode_ids', _decode_ids),
                           ('calibrate_actions', _calibrate_actions)]:
        setattr(model, name, types.MethodType(function, model))
    return dict(normalization='initial_reference_four_slots', replacements=replacements,
                action_mode=mode, lfq_source_sha256=LFQ_SHA, extra_trainable_parameters=0)


def clone_pair(common_model):
    if not hasattr(common_model, 'action_bottleneck') or not bool(common_model.action_bottleneck.scale_bound):
        raise ValueError('A common adapted model with bound TRAIN scale is required')
    pair = {mode: copy.deepcopy(common_model) for mode in MODES}
    for mode, model in pair.items():
        model.action_bottleneck.mode = mode
    return pair

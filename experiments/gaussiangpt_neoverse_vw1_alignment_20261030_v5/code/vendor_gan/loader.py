"""Verify unchanged upstream files, then isolate only their imports in memory.

No process-global sys.modules stubs, pip installs, downloads, or source edits.
The original class/function bodies and registration decorators are retained.
"""
import ast
from copy import deepcopy
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from . import compat

ROOT = Path(__file__).resolve().parent


def source_manifest():
    data = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    for entry in data['files']:
        actual = hashlib.sha256((ROOT / entry['file']).read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise RuntimeError(f"Modified upstream source: {entry['file']}")
    return data


def _execute(filename, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(ROOT / filename), 'exec'), namespace)


@lru_cache(maxsize=1)
def load_classes():
    source_manifest()
    namespace = dict(__name__='videoworld_calvin_isolated',
        Optional=Optional, deepcopy=deepcopy, partial=partial, np=np,
        torch=torch, nn=nn, F=F, Tensor=Tensor,
        autograd=torch.autograd, dist=torch.distributed,
        _calculate_correct_fan=nn.init._calculate_correct_fan,
        MODELS=compat.Registry(), BaseModule=compat.BaseModule,
        ConvModule=compat.ConvModule, normal_init=compat.normal_init,
        autocast=compat.autocast, get_dist_info=compat.get_dist_info,
        build_norm_layer=compat.build_norm_layer,
        upfirdn2d=compat.upfirdn2d,
        fused_bias_leakyrelu=compat.fused_bias_leakyrelu,
        FusedBiasLeakyReLU=compat.FusedBiasLeakyReLU,
        _load_checkpoint_with_prefix=compat.load_checkpoint_with_prefix)
    _execute('stylegan2_utils.py', namespace)
    _execute('stylegan2_discriminator.py', namespace)
    return namespace


def build_discriminator():
    """Original CALVIN 128px D, FP32, group_size=4, no ADA/pretrained override."""
    return load_classes()['StyleGAN2Discriminator'](in_size=128)

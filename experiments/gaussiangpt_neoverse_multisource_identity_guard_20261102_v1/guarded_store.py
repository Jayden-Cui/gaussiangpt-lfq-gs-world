"""Add bound source-video identity checks to the existing multi-source loader.

This preserves the existing split and cached state representation. It does not
claim that previously exposed evaluation videos are untouched holdouts.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'gaussiangpt_neoverse_multisource_action_data_20261102_v1'
UPSTREAM = ROOT.parent / 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
TRAIN = ('buildingblock_tower_17_176', 'buildingblock_tower_10_176')
EVAL = ('buildingblock_tower_14_0',)
# These are the source-video hashes already pinned in the historical sources.py.
SOURCE_SHA = {
    TRAIN[0]: '966d0fca220959e95608b383e808ecda897b4140c54fdeab93fb1a791cd24a90',
    TRAIN[1]: 'a8d76842e1d7de31cb8bedfe437264a28b3203c9f3bfbaebf2a6f0864a051dab',
    EVAL[0]: '456d178c177d6790196f2e27f3bf5e1f5eebbb9fdaecc716794be90fbbab57f8',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_original_code():
    witness = read(ORIGINAL / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ORIGINAL / 'CODE.sha256'):
        raise RuntimeError('Original multi-source loader witness differs')
    for line in (ORIGINAL / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ORIGINAL / name) != digest:
            raise RuntimeError('Original multi-source source changed: ' + name)
    adapter = read(ORIGINAL / 'ADAPTER_TEST.json')
    if (adapter.get('status') != 'passed'
            or adapter.get('adapter_sha256') != sha(ORIGINAL / 'window_adapter.py')
            or adapter.get('test_sha256') != sha(ORIGINAL / 'adapter_tests.py')
            or adapter.get('radius_audit_sha256') != sha(ORIGINAL / 'GLOBAL_RADIUS6_AUDIT.json')):
        raise RuntimeError('Original window-adapter witness differs')
    upstream_witness = read(UPSTREAM / 'CPU_VALIDATION.json')
    upstream_manifest = read(UPSTREAM / 'CODE_MANIFEST.json')
    if (upstream_witness.get('status') != 'passed'
            or upstream_witness.get('code_manifest_sha256') != sha(UPSTREAM / 'CODE_MANIFEST.json')
            or upstream_manifest.get('sources.py') != sha(UPSTREAM / 'sources.py')):
        raise RuntimeError('Historical source identity constants are not bound')
    spec = importlib.util.spec_from_file_location('_bound_multisource_identity_origin', UPSTREAM / 'sources.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    if tuple(upstream.TRAIN_CLIPS) != TRAIN or upstream.EVAL_CLIP != EVAL[0] or upstream.VIDEO_SHA != SOURCE_SHA:
        raise RuntimeError('Historical source identities or split differ')


def validate_mappings(roots, bindings):
    """Verify bound mapping bytes, expected source identities and group split."""
    if set(roots) != set(SOURCE_SHA):
        raise RuntimeError('Unexpected source clips')
    rows = []
    for clip in (*TRAIN, *EVAL):
        mapping_path = Path(roots[clip]) / 'mappings' / f'{clip}.json'
        expected = bindings.get(str(mapping_path))
        if expected is None:
            raise RuntimeError('Mapping is absent from the pinned data binding: ' + clip)
        raw = mapping_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected.get('bytes') != len(raw) or expected.get('sha256') != digest:
            raise RuntimeError('Mapping bytes differ from the pinned data binding: ' + clip)
        mapping = json.loads(raw)
        if mapping.get('source_sha256') != SOURCE_SHA[clip]:
            raise RuntimeError('Mapping names a different source video: ' + clip)
        if (mapping.get('source_frame_indices') != list(range(177))
                or len(mapping.get('source_pts_seconds', [])) != 177):
            raise RuntimeError('Mapping no longer describes all 177 consecutive frames: ' + clip)
        rows.append(dict(clip=clip, split='TRAIN' if clip in TRAIN else 'EVAL',
                         source_group=clip.rsplit('_', 1)[0], source_video_sha256=SOURCE_SHA[clip],
                         mapping_path=str(mapping_path), mapping_sha256=digest, mapping_bytes=len(raw)))
    if len({row['source_video_sha256'] for row in rows}) != len(rows):
        raise RuntimeError('A source-video identity appears in multiple clips')
    train_groups = {row['source_group'] for row in rows if row['split'] == 'TRAIN'}
    eval_groups = {row['source_group'] for row in rows if row['split'] == 'EVAL'}
    if len(train_groups) != len(TRAIN) or len(eval_groups) != len(EVAL) or train_groups & eval_groups:
        raise RuntimeError('Training and evaluation source groups overlap')
    return dict(status='passed', rows=rows, train_groups=sorted(train_groups), eval_groups=sorted(eval_groups),
                source_video_hashes_distinct=True, mapping_bytes_verified_against_data_binding=True,
                evaluation_source_previously_exposed=True, official_independent_holdout=False,
                scope='Bound metadata identity only; different hashes do not prove different underlying footage')


def _base_module():
    verify_original_code()
    expected = (ORIGINAL / 'source_store.py').resolve()
    module = sys.modules.get('source_store')
    if module is not None:
        if Path(module.__file__).resolve() != expected or sha(module.__file__) != sha(expected):
            raise RuntimeError('An unrelated source_store module was already loaded')
        return module
    spec = importlib.util.spec_from_file_location('source_store', expected)
    module = importlib.util.module_from_spec(spec)
    sys.modules['source_store'] = module
    spec.loader.exec_module(module)
    return module


BASE = _base_module()


class GuardedSourceStore(BASE.SourceStore):
    """Existing loader plus verification of source identity mapping bytes."""
    def __init__(self):
        super().__init__()
        self.identity_evidence = validate_mappings(self.roots, self.binding)


def make_window_store(*args, **kwargs):
    """Construct the existing adapter and replace its loader with the guard.

    The original adapter's initial constructor validates roles but reads no
    training window. It remains unchanged for the reproducibility of old runs.
    """
    verified = GuardedSourceStore()
    spec = importlib.util.spec_from_file_location('_identity_guarded_window_adapter', ORIGINAL / 'window_adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = module.MultiSourceWindowStore(*args, **kwargs)
    store.source = verified
    store.source.authorize(store.clip, store.purpose)
    return store

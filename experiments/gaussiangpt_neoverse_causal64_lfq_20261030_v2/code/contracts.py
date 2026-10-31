"""Immutable source bindings; never repair, modify, or recreate old data."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_sources(root, full_cache=False):
    binding = json.loads((root/'source_binding.json').read_text())
    for item in binding['dependencies']:
        assert sha(item['path']) == item['sha256'], 'Source changed: '+item['path']
    assert sha(root/'cache_manifest.json') == binding['cache_manifest_sha256']
    manifest = json.loads((root/'cache_manifest.json').read_text())
    for item in manifest:
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'], 'Cache changed: '+str(path)
        if full_cache: assert sha(path) == item['sha256'], 'Cache hash changed: '+str(path)
    code = json.loads((root/'code_manifest.json').read_text())
    for name, expected in code.items():
        assert sha(root/name) == expected, 'Submitted code changed: '+name
    return {'dependencies': len(binding['dependencies']), 'cache_files': len(manifest),
        'cache_hashes_checked': full_cache, 'submitted_files': len(code)}

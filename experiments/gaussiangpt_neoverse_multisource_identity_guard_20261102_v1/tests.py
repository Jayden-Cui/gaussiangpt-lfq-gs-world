"""Exercise real split contracts on synthetic mapping files, without data writes."""
import hashlib
import json
from pathlib import Path
import tempfile

from guarded_store import SOURCE_SHA, TRAIN, EVAL, validate_mappings


def rejected(call, message):
    try:
        call()
    except RuntimeError as error:
        if message not in str(error):
            raise
    else:
        raise AssertionError('Expected rejection: ' + message)


def main():
    with tempfile.TemporaryDirectory(prefix='gs_source_guard_') as temp:
        root = Path(temp).resolve()
        if root.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError('Unexpected temporary test directory')
        roots, bindings = {}, {}
        for clip in (*TRAIN, *EVAL):
            roots[clip] = root / clip
            folder = roots[clip] / 'mappings'
            folder.mkdir(parents=True)
            path = folder / f'{clip}.json'
            value = dict(source_sha256=SOURCE_SHA[clip], source_frame_indices=list(range(177)),
                         source_pts_seconds=[i / 29.97 for i in range(177)])
            raw = json.dumps(value).encode('utf-8')
            path.write_bytes(raw)
            bindings[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        result = validate_mappings(roots, bindings)
        assert result['source_video_hashes_distinct'] and result['evaluation_source_previously_exposed']
        target = roots[EVAL[0]] / 'mappings' / f'{EVAL[0]}.json'
        forged = json.loads(target.read_text(encoding='utf-8'))
        forged['source_sha256'] = SOURCE_SHA[TRAIN[0]]
        raw = json.dumps(forged).encode('utf-8')
        target.write_bytes(raw)
        rejected(lambda: validate_mappings(roots, bindings), 'Mapping bytes differ')
        bindings[str(target)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        rejected(lambda: validate_mappings(roots, bindings), 'different source video')
        bindings.pop(str(target))
        rejected(lambda: validate_mappings(roots, bindings), 'absent from the pinned data binding')
    print('PASS: pinned mapping change, forged source identity and missing binding rejected')


if __name__ == '__main__':
    main()

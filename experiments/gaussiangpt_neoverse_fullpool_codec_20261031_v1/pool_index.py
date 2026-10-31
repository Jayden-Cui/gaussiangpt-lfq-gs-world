"""Read-only full-pool metadata admission and deterministic complete-frame sampling.

No model construction, latent-state reads, GPU initialization or output writes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
from types import ModuleType, SimpleNamespace

POOL_RUN = 'gaussiangpt_neoverse_fulltrain_pool_20261031_v1'
LEGACY_RUN = 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
MANIFEST_SHA = '3d889cabcd5e9ef58f91466c888020b60d81fd9e36d24e5c91abca39b62182b0'
INPUT_POOL_SHA = '438d3113a246c214222db14805088773356c8ac251da2946ab27b75ef111f96c'
DONE_SHA = '920a7f7c030733b2ff6d1be62aafb243bf581ea0a9bcc53ba3fe775b3d0f694b'
COMPLETE_SHA = '8ffe7055cba065a49c233245b106631665dcd710b3ebee2fd08368ab543224df'
BINDINGS_SHA = '57d33d88f2eb6cf91edaebb9fad4f0bab980aeac91d608a0938d0d46cf9b2c5b'
LEGACY_SHA = '7e8eaa327d1c05e7d00fa07121832c1f66d617c6e76a016eb711f1453606b249'
PROCESS_HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
PREPARE_HELPER_SHA = '3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85'
GROUP_OFFSETS = {0:(0,176,352),1:(0,176),3:(0,176,352),4:(0,176,352),5:(0,176,352),
    6:(0,176,352),7:(0,176),8:(0,176,352),9:(0,176,352),10:(0,176),11:(0,176),
    12:(0,176),13:(0,176),15:(0,176),16:(0,176),17:(176,),18:(0,176,352,528),
    20:(0,176,352,528),21:(0,176,352,528)}
EXPECTED = {f'buildingblock_tower_{group}_{offset}': f'buildingblock_tower_{group}'
            for group, offsets in GROUP_OFFSETS.items() for offset in offsets}
REUSE = {'buildingblock_tower_10_176', 'buildingblock_tower_17_176'}
FRAMES, POINTS = 177, 112896
_MODULES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def exact(value, expected, label):
    require(type(value) is type(expected) and value == expected, label + ' differs')


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def signature(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def key(path):
    # Keep absolute POSIX receipt paths intact when CPU fixtures run on Windows.
    text = str(path)
    if PurePosixPath(text).is_absolute():
        require('..' not in PurePosixPath(text).parts, 'Receipt path traversal')
        return PurePosixPath(text).as_posix()
    p = Path(text)
    require(p.is_absolute() and '..' not in p.parts, 'Receipt path must be absolute')
    return str(p)


class Audit:
    def __init__(self):
        self.seen = {}

    def check(self, path, digest=None, size=None, *, json_data=False):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'Missing or symlink bound file: '+str(path))
        before = signature(path)
        require(size is None or before[2] == size, 'Bound size differs: '+str(path))
        if json_data:
            require(before[2] <= 128 << 20, 'Oversized metadata')
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
        else:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
            actual = h.hexdigest()
            raw = None
        require(signature(path) == before, 'File changed while reading: '+str(path))
        require(digest is None or (valid_sha(digest) and actual == digest), 'Bound SHA differs: '+str(path))
        previous = self.seen.get(path)
        require(previous is None or previous == (before, actual), 'Binding changed during index admission')
        self.seen[path] = before, actual
        return (parse(raw), actual) if json_data else actual

    def finish(self):
        for path, (before, _) in self.seen.items():
            require(not path.is_symlink() and signature(path) == before, 'Input changed during index admission')


def table(rows):
    require(isinstance(rows, list) and bool(rows), 'Missing bound file records')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and valid_sha(row.get('sha256'))
                and type(row.get('bytes')) is int and row['bytes'] >= 0, 'Invalid bound file record')
        name = key(row.get('path', ''))
        require(name not in result, 'Duplicate bound file path')
        result[name] = row
    return result


def record(path, records, other=None):
    name = key(path)
    require(name in records, 'Receipt omitted file: '+name)
    row = records[name]
    if other is not None:
        alternate = other[name] if isinstance(other, dict) and name in other else other
        require(isinstance(alternate, dict) and key(alternate.get('path', '')) == name
                and alternate.get('bytes') == row['bytes'] and alternate.get('sha256') == row['sha256'],
                'Independent file receipts disagree: '+name)
    return row


def bound_json(path, records, audit, other=None):
    row = record(path, records, other)
    return audit.check(path, row['sha256'], row['bytes'], json_data=True)[0]


def isolated(path, digest, label):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing pinned source: '+str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'Pinned legacy source changed')
    cache_key = (str(path.resolve()), digest)
    if cache_key not in _MODULES:
        module = ModuleType('_fullpool_'+label)
        module.__file__ = str(path)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        if label == 'sources':
            module.ALL_CLIPS = tuple(sorted(EXPECTED))
            module.TRAIN_CLIPS = module.ALL_CLIPS
            module.ROLES = {clip: 'TRAIN' for clip in EXPECTED}
        _MODULES[cache_key] = module
    return _MODULES[cache_key]


def layout(root, clip):
    if clip == 'buildingblock_tower_17_176':
        directory = root.parent/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
        teacher = root.parent/'neoverse_tower177_fps30_20261029_v1/native448'
        rows = directory/'prepare'/f'{clip}.json'
    elif clip == 'buildingblock_tower_10_176':
        directory = root.parent/'gaussiangpt_neoverse_tower10_replay_20261031_v1'
        teacher = directory/'teacher'/clip
        rows = directory/'quality'/f'{clip}.json'
    else:
        directory = root/'clips'/clip
        teacher = directory/'teacher'/clip
        rows = directory/'quality'/f'{clip}.json'
    return directory, teacher, directory/'cache'/clip, rows


@dataclass(frozen=True)
class FrameRef:
    clip: str
    source_group: str
    original_role: str
    current_role: str
    frame: int
    pts_seconds: float
    cache_path: str
    cache_bytes: int
    cache_sha256: str
    native_gs_sha256: str
    normalization_json: str
    legacy_source_path: str
    pool_complete_sha256: str

    @property
    def normalization(self):
        return parse(self.normalization_json)

    def legacy_row(self):
        return dict(path=self.cache_path, bytes=self.cache_bytes, sha256=self.cache_sha256,
                    frame=self.frame, pts_seconds=self.pts_seconds, native_gs_sha256=self.native_gs_sha256)


def load_pool_index(pool_root):
    """Admit exactly 8850 FrameRefs. Hash cache files sequentially; never torch.load here."""
    root = Path(pool_root).resolve(strict=True)
    require(root.name == POOL_RUN, 'Unexpected pool root')
    for path in (root/'FAILED.json', root/'queue/FAILED.json'):
        require(not path.exists() and not path.is_symlink(), 'Pool has a failure receipt')
    audit = Audit()
    done, _ = audit.check(root/'queue/DONE.json', DONE_SHA, json_data=True)
    for field, expected in dict(status='complete', clips=50, fresh_exports=48, reused_exports=2,
            clip_internal_frames=8850, cross_clip_physical_frame_unique_count=None,
            automatic_retry=False, training_executed=False, goal_complete=False).items():
        exact(done.get(field), expected, 'DONE.'+field)
    manifest, _ = audit.check(root/'CODE_MANIFEST.json', MANIFEST_SHA, json_data=True)
    require(isinstance(manifest, dict) and manifest, 'Invalid code manifest')
    allowed_external = {'../gaussiangpt_neoverse_tower10_replay_20261031_v1/input_binding.json',
                        '../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py'}
    for name, digest in manifest.items():
        p = PurePosixPath(name)
        require(not p.is_absolute() and '\\' not in name and ':' not in name and valid_sha(digest)
                and ('..' not in p.parts or name in allowed_external), 'Unsafe manifest entry')
        audit.check(root/name, digest)
    pool, _ = audit.check(root/'INPUT_POOL.json', INPUT_POOL_SHA, json_data=True)
    bindings = isolated(root/'bindings.py', BINDINGS_SHA, 'bindings')
    bindings.validate_pool(pool)
    exact(set(bindings.EXPECTED_CLIPS), set(EXPECTED), 'Authoritative membership')
    proof = done.get('admission', {})
    for name, expected in dict(code_manifest_sha256=MANIFEST_SHA, input_pool_sha256=INPUT_POOL_SHA,
            protocol_sha256=manifest['protocol.json'], process_helper_sha256=PROCESS_HELPER_SHA).items():
        exact(proof.get(name), expected, 'DONE.admission.'+name)
    require(valid_sha(proof.get('cpu_validation_sha256')), 'Missing CPU witness digest')
    cpu, _ = audit.check(root/'CPU_VALIDATION.json', proof['cpu_validation_sha256'], json_data=True)
    require(cpu.get('status') == 'passed' and cpu.get('gpu_executed') is False
            and type(cpu.get('actual_tests_executed')) is int and cpu['actual_tests_executed'] > 0
            and cpu.get('code_manifest_sha256') == MANIFEST_SHA
            and cpu.get('protocol_sha256') == manifest['protocol.json'], 'Missing bound CPU admission witness')
    outcomes = done.get('outcomes', [])
    jobs = {'reuse_admission', 'final_admission'} | {clip+'_'+stage for clip in EXPECTED if clip not in REUSE
                                                           for stage in ('binding', 'teacher', 'prepare')}
    require(len(outcomes) == 146 and {r.get('name') for r in outcomes} == jobs
            and all(r.get('status') == 'complete' and r.get('exit_code') == 0 and r.get('timed_out') is False
                    for r in outcomes), 'Incomplete or duplicate child outcomes')
    require(valid_sha(done.get('pool_complete_sha256')) and valid_sha(done.get('reuse_admission_sha256')),
            'Terminal omitted admission hashes')
    exact(done['pool_complete_sha256'], COMPLETE_SHA, 'Frozen terminal pool digest')
    complete, complete_sha = audit.check(root/'POOL_COMPLETE.json', done['pool_complete_sha256'], json_data=True)
    reused, _ = audit.check(root/'REUSE_ADMISSION.json', done['reuse_admission_sha256'], json_data=True)
    base = dict(status='complete', code_manifest_sha256=MANIFEST_SHA, input_pool_sha256=INPUT_POOL_SHA,
                training_executed=False, training_performed=False, goal_complete=False,
                official_independent_holdout=False, cross_clip_physical_frame_unique_count=None)
    for receipt in (complete, reused):
        for name, expected in base.items(): exact(receipt.get(name), expected, 'admission.'+name)
    for name, expected in dict(schema='gs_fulltrain_pool_complete_v1', all_clips_complete=True,
            planned_clip_internal_frames_total=8850, actual_clip_internal_frames_total=8850,
            actual_adjacent_pairs_total=8800, fresh_clips=48, reused_clips=2, source_groups=19,
            native_width=448, native_height=252, gs_per_frame=POINTS, frame_stride=1,
            state_encodings_reused_for_new_codec=False).items(): exact(complete.get(name), expected, name)
    exact(reused.get('schema'), 'gs_fulltrain_reuse_admission_v1', 'reuse schema')
    exact(reused.get('actual_clip_internal_frames_total'), 354, 'reuse frames')
    records, reuse_records = table(complete.get('files')), table(reused.get('files'))
    for name, row in reuse_records.items(): record(name, records, row)
    require(isinstance(complete.get('reuse_admission'), dict), 'Missing reuse admission file binding')
    bound_json(root/'REUSE_ADMISSION.json', records, audit, complete['reuse_admission'])
    for filename, digest in [('CODE_MANIFEST.json', MANIFEST_SHA), ('INPUT_POOL.json', INPUT_POOL_SHA)]:
        row = record(root/filename, records)
        exact(row['sha256'], digest, filename+' file record')
        audit.check(root/filename, row['sha256'], row['bytes'])
    clips = complete.get('clips', [])
    require(len(clips) == 50 and {c.get('clip_id') for c in clips} == set(EXPECTED), 'Complete clip membership differs')
    clips = {c['clip_id']: c for c in clips}
    reuse_clips = reused.get('clips', [])
    require(len(reuse_clips) == 2 and {c.get('clip_id') for c in reuse_clips} == REUSE, 'Reuse clip membership differs')
    require(all(clips[c['clip_id']] == c for c in reuse_clips), 'Reuse admission differs from pool admission')
    legacy_path = root.parent/LEGACY_RUN/'sources.py'
    audit.check(legacy_path, LEGACY_SHA)
    legacy = isolated(legacy_path, LEGACY_SHA, 'sources')
    frames = []
    for item in pool['clips']:
        clip = item['clip_id']; accepted = clips[clip]
        video = record(item['video_path'], records)
        require(video['sha256'] == item['video_sha256'] and video['bytes'] == item['video_bytes'],
                'Admission source video identity differs from INPUT_POOL')
        for name, expected in dict(status='complete', source_group=item['source_group'], original_role=item['original_role'],
                current_role='TRAIN', source_video_sha256=item['video_sha256'], actual_frames=177,
                actual_adjacent_pairs=176, actual_source_points_per_frame=POINTS,
                actual_cache_files=177, actual_native_gs_files=177, state_encodings_reused_for_new_codec=False).items():
            exact(accepted.get(name), expected, clip+'.'+name)
        directory, teacher, cache, rows_path = layout(root, clip)
        exact(accepted.get('paths'), dict(teacher=str(teacher), cache=str(cache)), 'Declared teacher/cache paths')
        for failure in ('FAILED.json', 'ADAPTER_FAILED.json'):
            require(not (root/'clips'/clip/failure).exists(), 'Clip acquired a failure receipt')
        extra = reuse_records
        if clip in REUSE:
            exact(accepted.get('origin'), 'admitted_existing_full177_sequence', 'Reuse origin')
            exact(accepted.get('native_arrays_and_cache_tensors_checked'), True, 'Reuse tensor audit')
            exact(accepted.get('role_basis'), item['role_basis'], 'Reuse authorization basis')
            reference = bound_json(root/'clips'/clip/'reuse_reference.json', records, audit, reuse_records)
            for name, expected in dict(schema='gs_fulltrain_reuse_reference_v1', clip_id=clip,
                    status='reference_only_requires_admission', current_role='TRAIN', original_role=item['original_role'],
                    source_group=item['source_group'], paths_requiring_admission=accepted['paths']).items():
                exact(reference.get(name), expected, 'Reuse reference.'+name)
        else:
            exact(accepted.get('origin'), 'new_full177_export', 'Fresh origin')
            receipt = bound_json(directory/'PREPARATION_RECEIPT.json', records, audit, accepted.get('preparation_receipt'))
            for name, expected in dict(schema='gs_fulltrain_clip_preparation_receipt_v1', status='complete', clip=clip,
                    source_group=item['source_group'], original_role=item['original_role'], current_role='TRAIN',
                    data_origin='new_full177_export', training_performed=False, goal_complete=False,
                    helper_sha256=PREPARE_HELPER_SHA, code_manifest_sha256=MANIFEST_SHA,
                    input_pool_sha256=INPUT_POOL_SHA, actual_frames=177, actual_cache_files=177,
                    actual_native_gs_files=177, actual_source_points_per_frame=POINTS, actual_adjacent_pairs=176).items():
                exact(receipt.get(name), expected, 'Preparation receipt.'+name)
            for filename, field in [('input_binding.json','input_binding_sha256'), ('SUMMARY.json','summary_sha256'),
                                    ('HELPER_SUMMARY.json','helper_summary_sha256')]:
                bound_json(directory/filename, records, audit)
                exact(record(directory/filename, records)['sha256'], receipt.get(field), 'Preparation '+field)
            extra = table(receipt.get('output_files'))
            for name, row in extra.items(): record(name, records, row)
            inputs = bound_json(directory/'inputs_hashes.json', records, audit, extra)
            require(inputs.get('schema') == 'gs_transfer_inputs_v1'
                    and inputs.get('teacher_hashes_computed_after_export') is True,
                    'Preparation input receipt differs')
            record(directory/'input_binding.json', records, inputs.get('binding'))
            for name, row in table(inputs.get('teacher_and_source_files')).items(): record(name, records, row)
            queue_clip, _ = audit.check(root/'queue'/f'{clip}_COMPLETE.json', json_data=True)
            require(queue_clip.get('status') == 'complete' and queue_clip.get('clip') == clip
                    and queue_clip.get('goal_complete') is False
                    and queue_clip.get('preparation_receipt_sha256') == accepted['preparation_receipt']['sha256'],
                    'Queue clip completion differs from admitted preparation')
        normal = bound_json(directory/'normalization.json', records, audit, extra)
        mapping = bound_json(directory/'mappings'/f'{clip}.json', records, audit, extra)
        row_container = bound_json(rows_path, records, audit, extra)
        rows = row_container['rows' if clip == 'buildingblock_tower_17_176' else 'frames']
        times = legacy.validate_index(clip, rows, mapping, normal)
        require(normal.get('uses_future_frames', False) is False
                and normal.get('changes_codec_output_std', False) is False, 'Clip-local normalization policy changed')
        require(mapping['source_sha256'] == item['video_sha256'], 'Mapping names a different source video')
        if clip in REUSE: exact(normal, accepted.get('normalization'), 'Reuse normalization')
        if 'adjacent_pairs' in mapping:
            exact(mapping['adjacent_pairs'], [[i,i+1] for i in range(176)], 'Adjacent frame pairs')
        normal_json = json.dumps(normal, sort_keys=True, separators=(',', ':'))
        for frame, row in enumerate(rows):
            require(abs(times[frame]-item['probe']['pts_seconds'][frame]) <= 1e-6, 'Measured source PTS differs')
            path = cache/f'{frame:03d}.pt'
            cache_record = record(path, records, extra)
            require(cache_record['bytes'] == row['cache_bytes'] and cache_record['sha256'] == row['cache_sha256'],
                    'Cache row/receipt bytes or SHA differ')
            native = record(teacher/'gaussians'/f'native_group_{frame:03d}.npz', records)
            exact(native['sha256'], row['native_gs_sha256'], 'Native GS provenance')
            audit.check(path, cache_record['sha256'], cache_record['bytes'])
            frames.append(FrameRef(clip, item['source_group'], item['original_role'], 'TRAIN', frame, times[frame],
                str(path), cache_record['bytes'], cache_record['sha256'], row['native_gs_sha256'], normal_json,
                str(legacy_path), complete_sha))
    require(len(frames) == 8850 and len({(r.clip,r.frame) for r in frames}) == 8850, 'Incomplete full-pool frame coverage')
    audit.finish()
    return tuple(frames)


def load_frame(ref, device='cpu'):
    """Original same-bytes hash -> BytesIO -> weights_only Torch path; one full frame."""
    require(isinstance(ref, FrameRef) and ref.clip in EXPECTED and ref.source_group == EXPECTED[ref.clip]
            and ref.current_role == 'TRAIN' and type(ref.frame) is int and 0 <= ref.frame < FRAMES
            and ref.original_role == ('DEV' if ref.clip == 'buildingblock_tower_17_176' else 'TRAIN'),
            'Invalid admitted FrameRef')
    legacy = isolated(ref.legacy_source_path, LEGACY_SHA, 'sources')
    source = SimpleNamespace(clip=ref.clip, role=ref.current_role, normalization=ref.normalization,
                             frames={ref.frame: ref.legacy_row()})
    return legacy.load_frame(source, ref.frame, device=device)


def assert_coverage(index, batches):
    expected = {(r.clip,r.frame): r for r in index}
    require(len(expected) == len(index) and bool(index), 'Index contains duplicate/empty frame identities')
    seen = Counter()
    for batch in batches:
        require(bool(batch), 'Empty batch')
        for ref in batch:
            identity = (ref.clip,ref.frame)
            require(identity in expected and ref == expected[identity], 'Plan contains foreign or mutated frame')
            seen[identity] += 1
    require(seen == Counter({identity:1 for identity in expected}), 'Coverage missing/duplicated frames')
    return True


def balanced_plan(index, seed, batch_size):
    """One complete pass: shuffled per-clip frames, cyclic equal clip exposure, no padding/drop."""
    require(type(seed) is int and type(batch_size) is int and batch_size > 0, 'Invalid sampler parameters')
    by_clip = defaultdict(list)
    for ref in index: by_clip[ref.clip].append(ref)
    require(bool(by_clip), 'Empty index')
    clips = sorted(by_clip)
    random.Random(seed).shuffle(clips)
    for clip in clips:
        by_clip[clip].sort(key=lambda r:r.frame)
        random.Random(hashlib.sha256(f'{seed}:{clip}'.encode()).digest()).shuffle(by_clip[clip])
    flat = tuple(by_clip[clip][round_] for round_ in range(max(map(len,by_clip.values())))
                 for clip in clips if round_ < len(by_clip[clip]))
    batches = tuple(flat[i:i+batch_size] for i in range(0,len(flat),batch_size))
    assert_coverage(index, batches)
    return batches


def plan_summary(index, batches):
    assert_coverage(index, batches)
    flat = [r for batch in batches for r in batch]
    identities = [(r.clip,r.frame) for r in flat]
    paths = {r.cache_path:r.cache_bytes for r in index}
    return dict(frames=len(flat), updates=len(batches), actual_batch_sizes=[len(b) for b in batches],
                clip_exposure=dict(sorted(Counter(r.clip for r in flat).items())),
                source_group_exposure=dict(sorted(Counter(r.source_group for r in flat).items())),
                plan_sha256=hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest(),
                cache_file_count=len(paths), unique_path_cache_bytes=sum(paths.values()),
                maximum_cache_file_bytes=max(paths.values()), cross_clip_physical_frame_unique_count=None,
                historical_states_reused=False)

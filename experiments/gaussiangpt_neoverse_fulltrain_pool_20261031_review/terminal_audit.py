"""One CPU terminal/size audit; payload content hashes remain upstream evidence."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

ROOT = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fulltrain_pool_20261031_v1')
OUT = ROOT.parent/'gaussiangpt_neoverse_fulltrain_pool_20261031_review'
PINS = {
    'queue/DONE.json':'920a7f7c030733b2ff6d1be62aafb243bf581ea0a9bcc53ba3fe775b3d0f694b',
    'POOL_COMPLETE.json':'8ffe7055cba065a49c233245b106631665dcd710b3ebee2fd08368ab543224df',
    'CODE_MANIFEST.json':'3d889cabcd5e9ef58f91466c888020b60d81fd9e36d24e5c91abca39b62182b0',
    'INPUT_POOL.json':'438d3113a246c214222db14805088773356c8ac251da2946ab27b75ef111f96c',
}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name, pinned in PINS.items():
    assert sha(ROOT/name) == pinned, name
assert not (ROOT/'queue/FAILED.json').exists()
done = read(ROOT/'queue/DONE.json')
pool = read(ROOT/'POOL_COMPLETE.json')
inputs = read(ROOT/'INPUT_POOL.json')
assert done['status'] == pool['status'] == 'complete'
assert done['pool_complete_sha256'] == PINS['POOL_COMPLETE.json']
assert done['reuse_admission_sha256'] == sha(ROOT/'REUSE_ADMISSION.json')
assert pool['code_manifest_sha256'] == PINS['CODE_MANIFEST.json']
assert pool['input_pool_sha256'] == PINS['INPUT_POOL.json']
for name, pinned in read(ROOT/'CODE_MANIFEST.json').items():
    assert sha(ROOT/name) == pinned, name
assert len(done['outcomes']) == 146
assert all(r['status'] == 'complete' and r['exit_code'] == 0 and r['timed_out'] is False for r in done['outcomes'])
declared = {r['clip_id']:r for r in inputs['clips']}
clips = {r['clip_id']:r for r in pool['clips']}
assert len(clips) == len(pool['clips']) == len(declared) == 50 and set(clips) == set(declared)
assert pool['actual_clip_internal_frames_total'] == 8850
assert pool['actual_adjacent_pairs_total'] == 8800
assert pool['native_width'] == 448 and pool['native_height'] == 252
assert pool['gs_per_frame'] == 112896 and pool['frame_stride'] == 1
assert pool['fresh_clips'] == 48 and pool['reused_clips'] == 2
assert pool['cross_clip_physical_frame_unique_count'] is None
records = {str(Path(r['path']).resolve()):r for r in pool['files']}
assert len(records) == len(pool['files'])
for path, row in records.items():
    assert Path(path).is_file() and Path(path).stat().st_size == row['bytes'], path

per_clip = []
groups = collections.Counter()
categories = {'cache':set(),'teacher':set(),'old_state':set()}
for clip, row in clips.items():
    assert row['status'] == 'complete' and row['current_role'] == 'TRAIN'
    assert row['actual_frames'] == row['actual_cache_files'] == row['actual_native_gs_files'] == 177
    assert row['actual_adjacent_pairs'] == 176 and row['actual_source_points_per_frame'] == 112896
    assert row['source_group'] == declared[clip]['source_group']
    groups[row['source_group']] += 1
    cache = Path(row['paths']['cache']).resolve()
    teacher = Path(row['paths']['teacher']).resolve()
    cache_paths = {str(cache/f'{i:03d}.pt') for i in range(177)}
    gs_paths = {str(teacher/'gaussians'/f'native_group_{i:03d}.npz') for i in range(177)}
    assert cache_paths <= set(records) and gs_paths <= set(records)
    assert set(cache.glob('*.pt')) == {Path(p) for p in cache_paths}
    assert set((teacher/'gaussians').glob('*.npz')) == {Path(p) for p in gs_paths}
    teacher_paths = {p for p in records if p.startswith(str(teacher)+'/')}
    state_paths = {p for p in records if '/states/'+clip+'/' in p}
    categories['cache'].update(cache_paths)
    categories['teacher'].update(teacher_paths)
    categories['old_state'].update(state_paths)
    per_clip.append(dict(clip_id=clip,source_group=row['source_group'],origin=row['origin'],
        cache_files=len(cache_paths),cache_bytes=sum(records[p]['bytes'] for p in cache_paths),
        max_cache_frame_bytes=max(records[p]['bytes'] for p in cache_paths),
        native_gs_files=len(gs_paths),teacher_bytes=sum(records[p]['bytes'] for p in teacher_paths),
        old_state_files=len(state_paths),old_state_bytes=sum(records[p]['bytes'] for p in state_paths)))
assert len(groups) == 19 and len(categories['cache']) == 8850
totals = {k:dict(unique_paths=len(paths),bytes=sum(records[p]['bytes'] for p in paths)) for k,paths in categories.items()}
result = dict(status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    terminal_finished_at=done['finished_at'],bindings=PINS,child_processes_complete=146,
    clips=50,source_groups=dict(sorted(groups.items())),clip_internal_frames=8850,adjacent_pairs=8800,
    cross_clip_physical_frame_unique_count=None,all_manifest_sources_hashed=True,
    payload_content_rehashed_this_audit=False,upstream_payload_hash_verification='Pinned terminal admission receipt; every declared file size rechecked now',
    stat_files_checked=len(records),totals=totals,per_clip=per_clip,
    max_cache_frame_bytes=max(r['max_cache_frame_bytes'] for r in per_clip),
    state_encodings_reused_for_new_codec=False,gpu_executed=False,training_executed=False,goal_complete=False)
OUT.mkdir(exist_ok=True)
with (OUT/'TERMINAL_AUDIT.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('per_clip','source_groups','bindings')}))
print(json.dumps({'terminal_audit_sha256':sha(OUT/'TERMINAL_AUDIT.json')}))

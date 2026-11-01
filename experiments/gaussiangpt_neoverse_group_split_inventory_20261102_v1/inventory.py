"""Inventory tower clips without assigning new data or claiming unseen holdouts."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
GUARD = ROOT.parent / 'gaussiangpt_neoverse_multisource_identity_guard_20261102_v1'
PATTERN = re.compile(r'^(buildingblock_tower_\d+)_(\d+)\.mp4$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_name(filename):
    match = PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError('Unexpected tower clip filename')
    return match.group(1), int(match.group(2))


def validate_partition(rows, assignments):
    """Allow multiple clips in one split, reject group or byte-identical leakage."""
    by_clip = {}
    for row in rows:
        clip = row['clip']
        group, start = parse_name(clip + '.mp4')
        if clip in by_clip or row['source_group'] != group or row['encoded_start_index'] != start:
            raise ValueError('Duplicate clip or inconsistent group identity')
        digest = row['source_video_sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid source-video SHA-256')
        by_clip[clip] = row
    if not set(assignments).issubset(by_clip) or any(role not in ('TRAIN', 'EVAL') for role in assignments.values()):
        raise ValueError('Invalid assigned clip or role')
    groups, hashes = {}, {}
    for clip, role in assignments.items():
        group, digest = by_clip[clip]['source_group'], by_clip[clip]['source_video_sha256']
        groups.setdefault(group, set()).add(role)
        hashes.setdefault(digest, set()).add(role)
    if any(len(roles) > 1 for roles in groups.values()):
        raise ValueError('The same source group appears in TRAIN and EVAL')
    if any(len(roles) > 1 for roles in hashes.values()):
        raise ValueError('Byte-identical source video appears in TRAIN and EVAL')
    return {group: next(iter(roles)) for group, roles in groups.items()}


def probe(path, timeout):
    process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True, timeout=timeout)
    streams = json.loads(process.stdout)['streams']
    if len(streams) != 1:
        raise RuntimeError('Expected exactly one selected video stream')
    info = streams[0]
    fps = Fraction(info['r_frame_rate'])
    return dict(width=int(info['width']), height=int(info['height']), fps_numerator=fps.numerator,
                fps_denominator=fps.denominator,
                frames=int(info['nb_frames']) if info.get('nb_frames', '').isdigit() else None,
                duration_seconds=float(info['duration']) if info.get('duration') not in (None, 'N/A') else None)


def main(output):
    config = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    guard = json.loads((GUARD / 'protocol.json').read_text(encoding='utf-8'))
    if (config.get('schema') != 'tower_group_inventory_v1' or config.get('gpu_used') is not False
            or config.get('training_performed') is not False
            or config['identity_guard_code_sha256'] != sha(GUARD / 'CODE.sha256')
            or config['identity_guard_protocol_sha256'] != sha(GUARD / 'protocol.json')):
        raise RuntimeError('Inventory protocol or source-identity guard changed')
    assignments = config['existing_assignments']
    if (set(guard['train_clips']) != {k for k, v in assignments.items() if v == 'TRAIN'}
            or set(guard['eval_only_clips']) != {k for k, v in assignments.items() if v == 'EVAL'}):
        raise RuntimeError('Inventory must preserve the previously fixed assignments')
    source = Path(config['dataset_root']).resolve(strict=True)
    files = sorted((p for p in source.iterdir() if p.is_file() and PATTERN.fullmatch(p.name)), key=lambda p: p.name)
    if len(files) != config['expected_tower_clip_count']:
        raise RuntimeError('Tower clip count differs from the fixed inventory')
    began = time.monotonic()
    rows = []
    for path in files:
        if time.monotonic() - began > config['max_seconds']:
            raise TimeoutError('Inventory exceeded the fixed CPU time budget')
        if path.resolve().parent != source:
            raise RuntimeError('Clip path leaves the expected dataset directory')
        group, start = parse_name(path.name)
        row = dict(clip=path.stem, source_group=group, encoded_start_index=start,
                   source_video_sha256=sha(path), bytes=path.stat().st_size,
                   enrolled_role=assignments.get(path.stem, 'NOT_ENROLLED'),
                   exposure_history='known previously exposed' if path.stem in assignments else 'not established',
                   **probe(path, config['ffprobe_timeout_seconds']))
        if path.stem in assignments and row['source_video_sha256'] != guard['source_video_sha256'][path.stem]:
            raise RuntimeError('An already bound raw source video changed')
        rows.append(row)
    group_roles = validate_partition(rows, assignments)
    groups = {row['source_group'] for row in rows}
    if len(groups) != config['expected_tower_group_count']:
        raise RuntimeError('Tower source-group count differs from the fixed inventory')
    for row in rows:
        row['reserved_group_role'] = group_roles.get(row['source_group'], 'UNASSIGNED')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='complete', dataset_root=str(source), clips=len(rows), source_groups=len(groups),
        existing_assignments=assignments, reserved_group_roles=group_roles, rows=rows,
        unique_video_hashes=len({row['source_video_sha256'] for row in rows}),
        code_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'protocol.json'),
        identity_guard_protocol_sha256=sha(GUARD / 'protocol.json'),
        raw_video_hashes_recomputed=True, video_pixels_decoded=False, gpu_used=False, training_performed=False,
        clean_holdout_established=False, max_seconds=config['max_seconds'], elapsed_seconds=time.monotonic() - began,
        scope='Filename-derived source groups; assignments unchanged; no GS state cache or novel holdout implied')
    with (output / 'INVENTORY.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'clips': len(rows), 'groups': len(groups),
                      'unique_video_hashes': result['unique_video_hashes']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New, non-existing output directory')
    args = parser.parse_args()
    main(args.output)

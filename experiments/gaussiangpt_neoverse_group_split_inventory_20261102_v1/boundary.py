"""Compare three already-exposed neighbouring clip boundaries, CPU only."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from inventory import ROOT, sha


def decode_two(path, width, height, lo, hi, timeout):
    slash = chr(92)
    select = f'select=between(n{slash},{lo}{slash},{hi})'
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-vf', select, '-vsync', '0',
           '-frames:v', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[:1000])
    frames = np.frombuffer(result.stdout, dtype=np.uint8)
    if frames.size != 2 * width * height * 3:
        raise RuntimeError('Unexpected decoded frame dimensions or count')
    return frames.reshape(2, height, width, 3).astype(np.int16)


def compare(a, b):
    delta = np.abs(a - b)
    return dict(mae_8bit=float(delta.mean()), exact_channel_fraction=float((delta == 0).mean()),
                max_channel_error=int(delta.max()))


def main(inventory_path, expected_sha, output):
    inventory_path = Path(inventory_path)
    if len(expected_sha) != 64 or sha(inventory_path) != expected_sha:
        raise RuntimeError('Bound inventory hash differs')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    if (inventory.get('status') != 'complete' or inventory.get('clips') != 60
            or inventory.get('protocol_sha256') != sha(ROOT / 'protocol.json')):
        raise RuntimeError('Unexpected inventory')
    root = Path(inventory['dataset_root'])
    rows = {row['clip']: row for row in inventory['rows']}
    began = time.monotonic()
    results = []
    for group in config['boundary_groups']:
        if time.monotonic() - began > config['boundary_max_seconds']:
            raise TimeoutError('Boundary analysis exceeded the fixed CPU time budget')
        first = f'buildingblock_tower_{group}_0'
        second = f'buildingblock_tower_{group}_176'
        a, b = rows[first], rows[second]
        if (a['frames'] != 177 or b['frames'] != 177 or a['width'] != b['width'] or a['height'] != b['height']
                or a['fps_numerator'] != b['fps_numerator'] or a['fps_denominator'] != b['fps_denominator']):
            raise RuntimeError('Boundary clip frame shape, count or FPS differs')
        pa, pb = root / (first + '.mp4'), root / (second + '.mp4')
        if sha(pa) != a['source_video_sha256'] or sha(pb) != b['source_video_sha256']:
            raise RuntimeError('Boundary source video changed after inventory')
        previous = decode_two(pa, a['width'], a['height'], 175, 176, config['ffmpeg_timeout_seconds'])
        following = decode_two(pb, b['width'], b['height'], 0, 1, config['ffmpeg_timeout_seconds'])
        results.append(dict(source_group=a['source_group'], first_clip=first, next_clip=second,
            first_sha256=a['source_video_sha256'], next_sha256=b['source_video_sha256'],
            last_vs_next_first=compare(previous[1], following[0]),
            penultimate_vs_next_first=compare(previous[0], following[0]),
            last_vs_next_second=compare(previous[1], following[1])))
    result = dict(status='complete', inventory_sha256=sha(inventory_path), code_sha256=sha(__file__), rows=results,
        gpu_used=False, training_performed=False,
        scope='Three previously exposed naming groups, decoded RGB boundary comparison',
        limitation='Similar boundary images support grouping; lossy clips alone do not prove identical timestamps')
    with Path(output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'source_groups': len(results)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--expected-inventory-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.inventory, args.expected_inventory_sha256, args.output)

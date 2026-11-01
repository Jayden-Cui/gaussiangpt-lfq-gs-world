"""Build fixed contact sheets for a completed stage-B visual review."""
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def create(evaluation_dir, out):
    evaluation_dir = Path(evaluation_dir)
    out = Path(out)
    evaluation_path = evaluation_dir / 'EVALUATION.json'
    replay_path = evaluation_dir / 'REPLAY.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
            or evaluation.get('source', {}).get('sweep') != 20 or replay.get('status') != 'passed'
            or replay.get('packets') != 104 or replay.get('views') != 208
            or replay.get('packets_sha256') != evaluation.get('packets_sha256')):
        raise RuntimeError('Complete final stage-B evaluation and native replay required')
    starts = (0, 35, 56, 84, 112, 123, 140, 167)
    horizons = (0, 3, 6, 9)
    out.mkdir()
    rows = []
    for start in starts:
        canvas = Image.new('RGB', (1792, 504))
        inputs = []
        for i, h in enumerate(horizons):
            path = evaluation_dir / 'snapshots' / f's{start:03d}' / f'frame_{h:02d}.png'
            with Image.open(path) as image:
                if image.size != (1792, 276):
                    raise RuntimeError('Unexpected diagnostic panel size')
                canvas.paste(image.crop((0, 24, 896, 276)), ((i % 2) * 896, (i // 2) * 252))
            inputs.append({'horizon': h, 'path': str(path), 'sha256': sha(path)})
        target = out / f's{start:03d}_contact.png'
        canvas.save(target)
        rows.append({'start': start, 'file': target.name, 'sha256': sha(target), 'frames': inputs})
    result = {'status': 'complete', 'evaluation_sha256': sha(evaluation_path),
              'native_replay_sha256': sha(replay_path), 'starts': list(starts),
              'horizons': list(horizons), 'rows': rows, 'labels_in_image': False,
              'interpretation': 'Each cell: source left, free-code reconstruction right; h0 is codec initial state',
              'visual_verdict': 'not yet reviewed'}
    with (out / 'CONTACTS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result

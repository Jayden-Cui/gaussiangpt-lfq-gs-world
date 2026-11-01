"""Build eight unlabelled sweep-10 contact sheets from valid panel columns."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
EVAL = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1/sweep_10/free_z/evaluation')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    result = json.loads((ROOT / 'INTERIM_RENDER.json').read_text(encoding='utf-8'))
    if (result.get('status') != 'intermediate_complete'
            or result['sweeps']['10']['source']['evaluation_sha256'] != sha(EVAL / 'EVALUATION.json')):
        raise RuntimeError('Sweep-10 evaluation binding differs')
    out = ROOT / 'CONTACTS_10'
    out.mkdir()
    rows = []
    for start in (0, 35, 56, 84, 112, 123, 140, 167):
        canvas = Image.new('RGB', (1792, 504))
        inputs = []
        for index, horizon in enumerate((0, 3, 6, 9)):
            path = EVAL / 'snapshots' / f's{start:03d}' / f'frame_{horizon:02d}.png'
            with Image.open(path) as image:
                if image.size != (1792, 276):
                    raise RuntimeError('Unexpected panel size')
                canvas.paste(image.crop((0, 24, 896, 276)), ((index % 2) * 896, (index // 2) * 252))
            inputs.append({'horizon': horizon, 'sha256': sha(path)})
        target = out / f's{start:03d}_contact.png'
        canvas.save(target)
        rows.append({'start': start, 'file': target.name, 'sha256': sha(target), 'inputs': inputs})
    metadata = {'status': 'complete', 'sweep': 10, 'evaluation_sha256': sha(EVAL / 'EVALUATION.json'),
                'collector_sha256': sha(ROOT / 'collect_contacts.py'), 'rows': rows,
                'layout': 'Per cell: source left and optimized free-code output right; h0, h3, h6, h9',
                'visual_verdict': 'not yet reviewed', 'not_final_stage_a_result': True}
    with (out / 'CONTACTS.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('PASS: eight bound contact sheets for the completed sweep-10 evaluation')


if __name__ == '__main__':
    main()

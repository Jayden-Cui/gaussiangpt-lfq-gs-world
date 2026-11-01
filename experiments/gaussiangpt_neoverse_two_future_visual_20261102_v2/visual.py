"""CPU-only contact sheets and source/motion/static videos for a completed probe."""
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def valid_digest(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_self():
    witness = read(ROOT/'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT/'CODE.sha256'):
        raise ValueError('Visual CPU witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT/name) != expected:
            raise ValueError('Visual source changed: ' + name)


def panels(image):
    image = image.convert('RGB')
    if image.size != (2240, 252):
        raise ValueError('Expected five ordered 448x252 panels')
    return [image.crop((448*i, 0, 448*(i+1), 252)) for i in range(5)]


def arrange(parts, indices):
    out = Image.new('RGB', (448*len(indices), 252))
    for offset, index in enumerate(indices):
        out.paste(parts[index], (448*offset, 0))
    return out


def create(protocol):
    if (protocol.get('schema') != 'two_future_visual_review_assets_v1'
            or protocol['windows'] != [35, 123] or protocol['horizons'] != list(range(1, 10))
            or protocol['video_fps'] != 5 or protocol['gpu_used'] is not False
            or any(not valid_digest(protocol.get(k)) for k in
                   ('launch_sha256','source_code_sha256','source_protocol_sha256'))):
        raise ValueError('Visual protocol changed')
    source = Path(protocol['source_directory'])
    if (sha(source/'LAUNCHED.json') != protocol['launch_sha256']
            or sha(source/'CODE.sha256') != protocol['source_code_sha256']
            or sha(source/'protocol.json') != protocol['source_protocol_sha256']):
        raise ValueError('Paired diagnostic provenance changed')
    done, summary_path = read(source/'DONE.json'), source/'result/SUMMARY.json'
    if done.get('status') != 'complete' or done['summary_sha256'] != sha(summary_path):
        raise ValueError('Paired training did not finish or summary changed')
    summary = read(summary_path)
    evaluation_path = source/'result/step_00360/EVALUATION.json'
    evaluation = read(evaluation_path)
    pairs = {(r['start'], r['horizon']) for r in evaluation['rows']}
    if (summary.get('status') != 'complete' or summary.get('updates') != 360
            or evaluation.get('status') != 'complete' or evaluation.get('step') != 360
            or len(evaluation['rows']) != 18 or pairs != {(s, h) for s in (35, 123) for h in range(1, 10)}
            or summary['summaries']['360'] != evaluation['summary']
            or evaluation.get('same_decoder_initial_condition_and_camera') is not True):
        raise ValueError('Incomplete or mismatched paired evaluation')
    out = ROOT/'result'
    out.mkdir()
    frames = []
    for start in protocol['windows']:
        folder = out/f's{start:03d}'
        folder.mkdir()
        motion_sheet = Image.new('RGB', (896*3, (252+24)*3), 'white')
        static_sheet = Image.new('RGB', (896*3, (252+24)*3), 'white')
        for h in protocol['horizons']:
            original = source/f'result/step_00360/s{start:03d}/frame_{h:02d}.png'
            parts = panels(Image.open(original))
            video_frame = arrange(parts, (0, 2, 4))
            video_path = folder/f'frame_{h:02d}.png'
            video_frame.save(video_path)
            x, y = ((h-1)%3)*896, ((h-1)//3)*(252+24)
            for sheet, selection in ((motion_sheet, (0, 2)), (static_sheet, (3, 4))):
                ImageDraw.Draw(sheet).text((x+5, y+5), f'h{h}', fill='black')
                sheet.paste(arrange(parts, selection), (x, y+24))
            frames.append({'start': start, 'horizon': h, 'source_sha256': sha(original),
                           'video_frame_sha256': sha(video_path)})
        motion_sheet.save(out/f's{start:03d}_motion_contact.png')
        static_sheet.save(out/f's{start:03d}_static_contact.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-start_number', '1',
            '-i', str(folder/'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(out/f's{start:03d}_source_motion_static_5fps.mp4')], check=True, timeout=90)
    manifest = {'status': 'complete', 'source_summary_sha256': sha(summary_path),
        'source_evaluation_sha256': sha(evaluation_path), 'source_code_sha256': protocol['source_code_sha256'],
        'video_columns': protocol['video_columns'], 'motion_sheet_columns': protocol['motion_sheet_columns'],
        'static_sheet_columns': protocol['static_sheet_columns'], 'video_fps': protocol['video_fps'],
        'frames': frames, 'visual_review': 'pending', 'scope': protocol['limits']}
    write(out/'MANIFEST.json', manifest)
    return manifest

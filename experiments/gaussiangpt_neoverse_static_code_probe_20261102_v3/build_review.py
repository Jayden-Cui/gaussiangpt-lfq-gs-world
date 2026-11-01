"""Create static-code comparison videos and review sheets from saved frames."""
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'result/step_00540'
OUT = ROOT / 'result/visual_review'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = json.loads((SOURCE/'EVALUATION.json').read_text(encoding='utf8'))
    if result['status'] != 'complete' or result['step'] != 540 or len(result['rows']) != 18:
        raise ValueError('Expected the completed 540-step two-window result')
    OUT.mkdir(exist_ok=False)
    artifacts = []
    for start in (35, 123):
        folder = OUT/f's{start:03d}'
        folder.mkdir()
        # Review sheet: static codec reference, old repeated-S0 Q and optimized code.
        sheet = Image.new('RGB', (2*1344, 5*(252+24)), 'white')
        for h in range(1, 10):
            path = SOURCE/f's{start:03d}'/f'frame_{h:02d}.png'
            image = Image.open(path).convert('RGB')
            if image.size != (2240, 252):
                raise ValueError('Expected five 448x252 image panels')
            pieces = [image.crop((448*i, 0, 448*(i+1), 252)) for i in range(5)]
            three = Image.new('RGB', (1344, 252))
            # User-facing video: source, same-camera static codec, optimized code.
            for i, part in enumerate((pieces[0], pieces[2], pieces[4])):
                three.paste(part, (448*i, 0))
            three.save(folder/f'frame_{h:02d}.png')
            x, y = ((h-1)%2)*1344, ((h-1)//2)*(252+24)
            ImageDraw.Draw(sheet).text((x+4, y+4), f'h{h}', fill='black')
            for i, part in enumerate((pieces[2], pieces[3], pieces[4])):
                sheet.paste(part, (x+448*i, y+24))
            artifacts.append({'start': start, 'horizon': h, 'source_sha256': sha(path),
                              'three_column_sha256': sha(folder/f'frame_{h:02d}.png')})
        sheet.save(OUT/f's{start:03d}_static_old_optimized_contact.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1',
            '-i',str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
            str(OUT/f's{start:03d}_source_static_optimized_5fps.mp4')], check=True, timeout=90)
    manifest = {'status': 'complete', 'step': 540, 'source_evaluation_sha256': sha(SOURCE/'EVALUATION.json'),
        'video_columns': ['source video', 'static initial state with matching camera', 'optimized static code'],
        'contact_columns': ['static initial state with matching camera', 'repeated-S0 old Q code', 'optimized static code'],
        'frames': artifacts, 'video_fps': 5, 'visual_review': 'pending'}
    with (OUT/'MANIFEST.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('PASS: two 9-frame comparison videos and complete review sheets')


if __name__ == '__main__':
    main()

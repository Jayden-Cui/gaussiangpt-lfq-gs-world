"""Make unlabelled source/Q-motion videos from the verified Q final render."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
Q = ROOT.parent / 'gaussiangpt_neoverse_two_future_qformer_20261102_v1'
SOURCE = Q / 'result/step_01194'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    export_path = Q / 'result/FINAL_EXPORT.json'
    export = json.loads(export_path.read_text(encoding='utf-8'))
    expected = {x['path']: x['sha256'] for x in export['files']}
    if export['status'] != 'complete' or len(expected) != 26:
        raise RuntimeError('Final Q export is incomplete')
    rows = []
    for start in (35, 123):
        source = SOURCE / f's{start:03d}_source_q_motion_q_static_5fps.mp4'
        key = source.relative_to(Q).as_posix()
        if sha(source) != expected[key]:
            raise RuntimeError('Source video hash differs')
        destination = output / f's{start:03d}_source_q_motion_5fps.mp4'
        subprocess.run(['ffmpeg', '-n', '-v', 'error', '-i', str(source), '-vf', 'crop=896:252:0:0',
                        '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(destination)], check=True, timeout=45)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(destination)],
                               capture_output=True, text=True, check=True, timeout=30)
        stream = json.loads(probe.stdout)['streams'][0]
        if (stream['width'], stream['height'], stream['nb_frames'], stream['r_frame_rate']) != (896, 252, '9', '5/1'):
            raise RuntimeError('Derived video layout or timing differs')
        rows.append(dict(start=start, input_sha256=sha(source), output_sha256=sha(destination),
                         output=destination.name, layout='source, Q motion; no text overlay', **stream))
    result = dict(status='complete', rows=rows, fps=5, source_export_sha256=sha(export_path),
                  generator_code_sha256=sha(__file__),
                  derivation='Crop first two unlabelled columns of the bound final Q render')
    with (output / 'VIDEO_MANIFEST.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: two unlabelled source/Q videos; each has 9 frames at 5 fps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New, non-existing output directory')
    args = parser.parse_args()
    main(args.output)

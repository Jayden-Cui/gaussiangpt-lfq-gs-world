"""One CPU-only document witness; inherited native environment is unchanged."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from postfit_queue import run_child, sha, verify


def main(root):
    begun = time.monotonic()
    verify(root)
    start_hashes = {name:sha(root/name) for name in ('POSTFIT_MANIFEST.json','CODE_MANIFEST.json')}
    stages = []
    for script in ('future_information_probe.py', 'render_comparison.py', 'postfit_queue.py', 'validate_postfit.py'):
        ast.parse((root / 'analysis_code' / script).read_text(encoding='utf-8'), filename=script)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for script in ('future_information_probe.py', 'render_comparison.py'):
        result = subprocess.run([sys.executable, '-B', str(root/'analysis_code'/script), '--self-test'],
                                env=env, capture_output=True, text=True, timeout=30)
        stages.append(dict(name=script, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError('CPU self-test failed: '+script+' '+result.stderr)
    for command in (['ffmpeg', '-version'], ['ffprobe', '-version']):
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError('Missing media tool: '+command[0])
        stages.append(dict(name=command[0], exit_code=0, version=result.stdout.splitlines()[0]))
    # Actual bounded child timeout using a separately created process group.
    # The child sleeps without importing Torch, allocating GPU or touching data.
    with tempfile.TemporaryDirectory(prefix='gs_native_postfit_cpu_') as directory:
        media_path = Path(directory) / 'four_frames.mp4'
        media = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', '16x16', '-framerate', '2997/100', '-i', 'pipe:0',
                                '-an', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(media_path)],
                               input=bytes(16*16*3*4), capture_output=True, timeout=20)
        if media.returncode:
            raise RuntimeError('CPU video encoder witness failed: '+media.stderr.decode(errors='replace'))
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
                                '-of', 'json', str(media_path)], capture_output=True, text=True, timeout=10)
        if probe.returncode:
            raise RuntimeError('Video decode/count witness failed: '+probe.stderr)
        stream = json.loads(probe.stdout)['streams'][0]
        if (stream['width'],stream['height'],int(stream['nb_read_frames']),stream['avg_frame_rate']) != (16,16,4,'2997/100'):
            raise RuntimeError('CPU video witness changed frame count, resolution or source fps')
        stages.append(dict(name='cpu_h264_video_roundtrip', exit_code=0, stream=stream, synthetic_frames=4))
        child_ready = Path(directory) / 'child_ready_pid.txt'
        grandchild = ('import os,signal,sys,time,pathlib; '
                      'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      'pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)')
        parent = ('import subprocess,sys,time; '
                  'subprocess.Popen([sys.executable,"-B","-c",sys.argv[1],sys.argv[2]]); time.sleep(30)')
        outcome = run_child([sys.executable, '-B', '-c', parent, grandchild, str(child_ready)],
                            env, Path(directory)/'harmless_timeout.log', 1.)
        if not outcome['timed_out'] or outcome['exit_code'] == 0:
            raise RuntimeError('Own-child timeout did not stop the test process')
        if Path('/proc', str(outcome['pid'])).exists():
            raise RuntimeError('Timed-out test process was not reaped')
        grandchild_pid = int(child_ready.read_text())
        # SIGKILL may leave a brief orphan zombie; it must have no live work.
        deadline = time.monotonic()+2
        while True:
            stat = Path('/proc', str(grandchild_pid), 'stat')
            try:
                value=stat.read_text(); dead=value[value.rfind(')')+2:].split()[0]=='Z'
            except FileNotFoundError:
                dead=True
            if dead:break
            if time.monotonic()>=deadline:raise RuntimeError('SIGTERM-ignoring descendant survived own-group timeout')
            time.sleep(.01)
        stages.append(dict(name='own_child_process_group_timeout_with_sigterm_ignoring_descendant',
                           descendant_pid=grandchild_pid, descendant_stopped=True, **outcome))
    verify(root)
    if any(sha(root/name)!=expected for name,expected in start_hashes.items()):
        raise RuntimeError('Code or manifests changed during CPU document witness')
    result = dict(status='passed', stages=stages, elapsed_seconds=time.monotonic()-begun,
                  postfit_manifest_sha256=start_hashes['POSTFIT_MANIFEST.json'],
                  main_manifest_sha256=start_hashes['CODE_MANIFEST.json'],
                  gpu_executed=False, training_queries=0,
                  scope='CPU algorithms, output timeline, manifest, four-frame H264 encode/decode and own-child timeout only; native GPU witness inherited unchanged')
    with (root/'POSTFIT_VALIDATION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve())

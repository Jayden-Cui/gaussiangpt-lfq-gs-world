"""Build unlabelled source/refinement videos from verified saved images."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'result'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    transfer=read(RESULT/'TRANSFER_SHA.json');verified=read(RESULT/'VERIFIED_RESULT.json')
    if (transfer.get('schema')!='anchor_code_refinement_transfer_v1' or verified.get('status')!='passed'
            or verified.get('transfer_sha256')!=sha(RESULT/'TRANSFER_SHA.json')
            or verified.get('training_updates')!=360):
        raise RuntimeError('Verified refinement result is missing or changed')
    for start in (35,123):
        names=['initial/s%03d_frame_00.png'%start]+['step_00360/s%03d_h%02d.png'%(start,h) for h in range(1,10)]
        for name in names:
            p=RESULT/name;entry=transfer['files'][name]
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
                raise RuntimeError('Source image changed: '+name)
    out=RESULT/'videos';out.mkdir();reports=[]
    for start in (35,123):
        initial=RESULT/('initial/s%03d_frame_00.png'%start)
        with Image.open(initial) as source:
            if source.size!=(1792,276):raise RuntimeError('Unexpected initial diagnostic image')
            first=out/('s%03d_frame_00.png'%start)
            source.convert('RGB').crop((0,24,896,276)).save(first)
        inputs=[first]+[RESULT/('step_00360/s%03d_h%02d.png'%(start,h)) for h in range(1,10)]
        for p in inputs:
            with Image.open(p) as image:
                if image.size!=(896,252):raise RuntimeError('Unexpected source/prediction frame size')
        playlist=out/('s%03d_concat.txt'%start)
        with playlist.open('x',encoding='utf-8',newline='\n') as stream:
            for p in inputs:
                relative=p.relative_to(RESULT).as_posix()
                target=p.name if p.parent==out else '../'+relative
                stream.write("file '%s'\nduration 0.2\n"%target)
            stream.write("file '../step_00360/s%03d_h09.png'\n"%start)
        video=out/('s%03d_source_vs_refined_5fps.mp4'%start)
        subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),
            '-r','5','-frames:v','10','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
            '-movflags','+faststart','-n',str(video)],check=True,timeout=120)
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames','-of','json',str(video)],
            check=True,capture_output=True,text=True,timeout=30)
        info=json.loads(probe.stdout)['streams'][0]
        if (info.get('width')!=896 or info.get('height')!=252 or info.get('avg_frame_rate')!='5/1'
                or int(info.get('nb_frames',0))!=10):raise RuntimeError('Video format or frame count differs')
        decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-f','rawvideo',
            '-pix_fmt','rgb24','-'],check=True,capture_output=True,timeout=60).stdout
        frame_bytes=896*252*3
        if len(decoded)!=10*frame_bytes:raise RuntimeError('Decoded video frame count differs')
        frames=np.frombuffer(decoded,dtype=np.uint8).reshape((10,252,896,3));errors=[]
        for i,p in enumerate(inputs):
            expected=np.asarray(Image.open(p).convert('RGB'),dtype=np.int16)
            errors.append(float(np.abs(frames[i].astype(np.int16)-expected).mean()/255))
        if max(errors)>5/255:raise RuntimeError('Video/image frame correspondence differs')
        reports.append(dict(start=start,video=video.name,video_sha256=sha(video),format=info,
            input_frames=[dict(path=p.relative_to(RESULT).as_posix(),sha256=sha(p)) for p in inputs],
            decoded_rgb_mae_by_frame=errors))
    result=dict(status='complete',videos=reports,source_verified_result_sha256=sha(RESULT/'VERIFIED_RESULT.json'),
        source_transfer_sha256=sha(RESULT/'TRANSFER_SHA.json'),labels_in_image=False,
        layout='source left; refined continuous free-code reconstruction right',
        first_frame='initial GS codec reconstruction; h1-h9 use fixed B decoder and refined codes',
        playback='5 fps slow playback; source frames were sampled adjacent at approximately 30 fps',
        scope='Two training windows; not a Q-Former output, independent GS replay, or generalization result')
    with (out/'VIDEO_MANIFEST.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'complete','videos':[x['video'] for x in reports],
                      'max_decoded_frame_mae':max(max(x['decoded_rgb_mae_by_frame']) for x in reports)}))


if __name__=='__main__':main()

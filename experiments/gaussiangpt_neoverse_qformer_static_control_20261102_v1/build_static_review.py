"""Make an unlabelled source / static reference / static-Q comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image


ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    output=Path(output);output.mkdir()
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg required')
    source=ROOT/'result';records=[]
    for start in (35,123):
        folder=output/f's{start:03d}';folder.mkdir();rows=[]
        for h in range(1,10):
            original=source/f's{start:03d}'/f'frame_{h:02d}.png'
            img=Image.open(original).convert('RGB')
            if img.size!=(2240,252):raise ValueError('Unexpected five-column input')
            tile=Image.new('RGB',(1344,252))
            for i,column in enumerate((0,3,4)):
                tile.paste(img.crop((column*448,0,(column+1)*448,252)),(i*448,0))
            tile.save(folder/f'frame_{h:02d}.png')
            rows.append({'horizon':h,'input_sha256':sha(original),'output_sha256':sha(folder/f'frame_{h:02d}.png')})
        video=output/f's{start:03d}_source_static_reference_static_Q_5fps.mp4'
        subprocess.run([ffmpeg,'-y','-loglevel','error','-framerate','5','-start_number','1',
                        '-i',str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(video)],
                       check=True,timeout=120)
        records.append({'start':start,'video':video.name,'sha256':sha(video),'frames':rows})
    (output/'MANIFEST.json').write_text(json.dumps({'status':'complete','columns':['source','same-camera static S0 codec',
        'Q repeated-S0 output'],'fps':5,'frames':9,'rows':records},indent=2)+'\n',encoding='utf8')
    print('PASS: two source/static-reference/static-Q videos')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'result/static_review')
    main(parser.parse_args().output)

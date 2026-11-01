"""Create unlabelled, aligned 3-column videos and all-future comparison sheets."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

from PIL import Image, ImageChops


ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()


def main(out):
    out=Path(out)
    out.mkdir()
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg is required')
    records=[]
    for start in (35,123):
        folder=out/f's{start:03d}'
        folder.mkdir()
        tiles=[];sources=[]
        for h in range(10):
            a_path=ROOT/'warm_start'/f's{start:03d}'/f'frame_{h:02d}.png'
            b_path=ROOT/'sweep_05'/f's{start:03d}'/f'frame_{h:02d}.png'
            a=Image.open(a_path).convert('RGB');b=Image.open(b_path).convert('RGB')
            if a.size!=(896,252) or b.size!=a.size:
                raise ValueError('Unexpected two-column input shape')
            if ImageChops.difference(a.crop((0,0,448,252)),b.crop((0,0,448,252))).getbbox() is not None:
                raise ValueError('Source images differ across model versions')
            tile=Image.new('RGB',(1344,252))
            tile.paste(a.crop((0,0,448,252)),(0,0))
            tile.paste(a.crop((448,0,896,252)),(448,0))
            tile.paste(b.crop((448,0,896,252)),(896,0))
            tile.save(folder/f'frame_{h:02d}.png')
            tiles.append(tile)
            sources.append({'horizon':h,'warm_png_sha256':sha(a_path),'sweep05_png_sha256':sha(b_path),
                            'source_pixels_exact_match':True})
        canvas=Image.new('RGB',(2688,1260),(0,0,0))
        for h in range(1,10):
            canvas.paste(tiles[h],(((h-1)%2)*1344,((h-1)//2)*252))
        montage=out/f's{start:03d}_all_future.png';canvas.save(montage)
        video=out/f's{start:03d}_source_warm_sweep05_5fps.mp4'
        subprocess.run([ffmpeg,'-y','-loglevel','error','-framerate','5','-i',str(folder/'frame_%02d.png'),
                        '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True,timeout=120)
        records.append({'start':start,'sources':sources,'montage':montage.name,'montage_sha256':sha(montage),
                        'video':video.name,'video_sha256':sha(video)})
    manifest={'status':'complete','scope':'Two known windows, 9 future frames; no model update',
              'columns':['source','warm decoder + warm free codes','sweep05 decoder + sweep05 free codes'],
              'video_fps':5,'video_frames':10,'rows':records}
    (out/'VISUALS.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print('PASS: aligned source images, 2 videos, 2 nine-future montages')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'review_output',
                        help='New output directory; never overwrites an existing review')
    main(parser.parse_args().output)

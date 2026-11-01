"""Generate unlabelled montage sheets from completed matched-pilot panels."""
from pathlib import Path
import argparse
import hashlib
import json
from PIL import Image


ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    out=Path(out);out.mkdir()
    source=ROOT/'result/visuals';records=[]
    for start in (35,123):
        canvas=Image.new('RGB',(1792,2268));inputs=[]
        for h in range(1,10):
            p=source/f's{start:03d}'/f'frame_{h:02d}.png'
            img=Image.open(p).convert('RGB')
            if img.size!=(1792,252):raise ValueError('Unexpected anchor panel size')
            canvas.paste(img,(0,(h-1)*252));inputs.append({'horizon':h,'sha256':sha(p)})
        path=out/f's{start:03d}_all_future.png';canvas.save(path)
        records.append({'start':start,'file':path.name,'sha256':sha(path),'inputs':inputs})
    starts=[0,35,56,84,112,123,140,167]
    canvas=Image.new('RGB',(1792,2016));inputs=[]
    for i,start in enumerate(starts):
        p=source/f's{start:03d}_h09.png';img=Image.open(p).convert('RGB')
        if img.size!=(1792,252):raise ValueError('Unexpected h9 panel size')
        canvas.paste(img,(0,i*252));inputs.append({'start':start,'sha256':sha(p)})
    path=out/'h09_eight_windows.png';canvas.save(path)
    records.append({'starts':starts,'file':path.name,'sha256':sha(path),'inputs':inputs})
    data={'status':'complete','rows':records,'columns':['source','sweep05','control','anchor_replay'],
          'scope':'No new model inference; source panels from matched pilot'}
    (out/'MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print('PASS: two 9-frame anchor sheets and eight-window h9 sheet')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'result/visual_review')
    main(parser.parse_args().output)

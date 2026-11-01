"""Prepare unlabelled contact sheets for manual dynamic review after stage A."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(evaluation_dir,out,starts=(0,35,56,84,112,123,140,167),horizons=(0,3,6,9)):
    evaluation_dir=Path(evaluation_dir);out=Path(out)
    evaluation_path=evaluation_dir/'EVALUATION.json'
    evaluation=json.loads(evaluation_path.read_text(encoding='utf-8'))
    if (evaluation.get('status')!='complete' or evaluation.get('window_predictions')!=1512
            or evaluation.get('correction',{}).get('status')!='applied'
            or evaluation.get('source',{}).get('sweep')!=20):
        raise RuntimeError('Final corrected stage-A evaluation is required')
    if len(starts)!=8 or not {35,123}.issubset(set(starts)) or horizons!=(0,3,6,9):
        raise RuntimeError('Fixed visual review coverage changed')
    out.mkdir();rows=[]
    for start in starts:
        canvas=Image.new('RGB',(1792,504));inputs=[]
        for i,h in enumerate(horizons):
            path=evaluation_dir/'snapshots'/f's{start:03d}'/f'frame_{h:02d}.png'
            with Image.open(path) as image:
                if image.size!=(1792,276):raise RuntimeError('Unexpected diagnostic panel shape')
                # Each cell is source (left) and predicted free-code GS (right),
                # without the panel's original top text band.
                canvas.paste(image.crop((0,24,896,276)),((i%2)*896,(i//2)*252))
            inputs.append(dict(horizon=h,path=str(path),sha256=sha(path)))
        target=out/f's{start:03d}_contact.png';canvas.save(target)
        rows.append(dict(start=start,file=target.name,sha256=sha(target),frames=inputs))
    result=dict(status='complete',evaluation_sha256=sha(evaluation_path),starts=list(starts),
        horizons=list(horizons),rows=rows,labels_in_image=False,
        interpretation='Each cell: source left, free-code reconstruction right; h0 is codec initial state',
        visual_verdict='not yet reviewed')
    with (out/'CONTACTS.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation-dir',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();create(args.evaluation_dir,args.out)

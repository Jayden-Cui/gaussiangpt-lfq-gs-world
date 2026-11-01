"""Check contact-sheet order and crop without claiming a visual result."""
import hashlib
import json
from pathlib import Path
import tempfile
from PIL import Image

from review_assets import create

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    starts=(0,35,56,84,112,123,140,167);horizons=(0,3,6,9)
    with tempfile.TemporaryDirectory(prefix='codex_review_assets_') as folder:
        root=Path(folder);evaluation=root/'evaluation';evaluation.mkdir()
        (evaluation/'EVALUATION.json').write_text(json.dumps(dict(status='complete',window_predictions=1512,
           correction=dict(status='applied'),source=dict(sweep=20))))
        for start in starts:
            path=evaluation/'snapshots'/f's{start:03d}';path.mkdir(parents=True)
            for h in horizons:
                image=Image.new('RGB',(1792,276),(0,0,0))
                Image.Image.paste(image,Image.new('RGB',(448,252),(h,10,20)),(0,24))
                Image.Image.paste(image,Image.new('RGB',(448,252),(h,30,40)),(448,24))
                image.save(path/f'frame_{h:02d}.png')
        result=create(evaluation,root/'contacts')
        if len(result['rows'])!=8 or result['visual_verdict']!='not yet reviewed':
            raise RuntimeError('Review coverage or verdict changed')
        with Image.open(root/'contacts/s035_contact.png') as image:
            if image.size!=(1792,504) or image.getpixel((0,0))!=(0,10,20) or image.getpixel((448,0))!=(0,30,40):
                raise RuntimeError('Source/reconstruction crop or order differs')
            if image.getpixel((896,0))!=(3,10,20) or image.getpixel((0,252))!=(6,10,20):
                raise RuntimeError('Horizon order differs')
    output=dict(status='passed',script_sha256=sha(ROOT/'review_assets.py'),test_sha256=sha(ROOT/'review_assets_tests.py'),
      source_prediction_crop_and_horizon_order=True,fixed_eight_window_coverage=True,
      no_automatic_scientific_verdict=True,scope='Synthetic CPU image test; no stage-A results reviewed')
    with (ROOT/'REVIEW_ASSETS_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: unlabelled source/reconstruction contact sheet coverage and ordering')


if __name__=='__main__':main()

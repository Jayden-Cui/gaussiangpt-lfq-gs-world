"""Check real-source view metrics without fabricating a missing mask."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

from source_store import SourceStore,TRAIN,sha

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('_cross_metric_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    c,helper,orig,p,paths,receipt,evaluation=binding.setup();helper.model_modules(orig)
    from cross_source_metrics import image_metrics_unmasked_teacher
    store=SourceStore();first=store.load_frame(TRAIN[1],26,purpose='train');later=store.load_frame(TRAIN[1],35,purpose='train')
    sample=later['sample'];sample['motion']=((sample['record']['real']-first['sample']['record']['real']).abs().mean(-1)/.1).clamp(0,1)
    roi=torch.zeros(252,448,dtype=torch.bool)
    for x0,y0,x1,y1 in p['hand_roi_xyxy']:roi[y0:y1,x0:x1]=True
    images=torch.stack([sample['record']['real'],sample['teacher_targets'][1]])
    result=image_metrics_unmasked_teacher(images,dict(roi=roi),sample)
    if (result['metrics']['mae']!=0 or result['metrics']['teacher_view1_full_mae']!=0
            or 'teacher_view1_masked_mae' in result['metrics'] or 'mask' in sample):
        raise RuntimeError('Unmasked teacher metric semantics differ')
    modified=images.clone();modified[1]=0
    changed=image_metrics_unmasked_teacher(modified,dict(roi=roi),sample)
    if not changed['metrics']['teacher_view1_full_mae']>0:
        raise RuntimeError('Full-view teacher mismatch was not detected')
    output=dict(status='passed',metrics_sha256=sha(ROOT/'cross_source_metrics.py'),
      test_sha256=sha(ROOT/'metric_tests.py'),source_clip=TRAIN[1],frames=[26,35],
      exact_target_zero_error=True,view1_mismatch_detected=True,
      missing_teacher_mask_not_fabricated=True,full_view_metric_distinguished_from_old_masked_metric=True,
      scope='CPU metric plumbing with real source images; no model reconstruction or cross-video performance')
    with (ROOT/'METRIC_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: real-source image metrics and explicitly unmasked second-view reference')


if __name__=='__main__':main()

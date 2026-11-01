"""Real-data CPU check for radius-6 initial-only cross-source conditioning."""
import json
from pathlib import Path
import sys
import torch

from source_store import TRAIN,EVAL,sha
from window_adapter import MultiSourceWindowStore

ROOT=Path(__file__).resolve().parent
B_ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1')


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    evidence=json.loads((ROOT/'GLOBAL_RADIUS6_AUDIT.json').read_text())
    if (evidence['status']!='complete' or evidence['code_sha256']!=sha(ROOT/'global_radius6_audit.py')
            or any(v['windows_above_25000'] for v in evidence['sources'].values())):
        raise RuntimeError('Radius-6 geometry feasibility evidence differs')
    sys.path.insert(0,str(B_ROOT))
    from b_core import load_bound_sources
    bundle=load_bound_sources('cpu');decoder=bundle['decoder'].eval();p=bundle['p']
    from support_decoder import make_condition,target_labels
    adapter=MultiSourceWindowStore(TRAIN[1],'train',decoder.base,bundle['store'].frames['std'],
         p['hand_roi_xyxy'],radius=6,max_candidates=25000,device='cpu')
    original=adapter.source.load_window(TRAIN[1],26,purpose='train',device='cpu')
    condition4=make_condition(decoder.base,original['states'][0],4,25000)
    try:target_labels(condition4,original['states'][1:])
    except ValueError as error:
        if 'outside initial-only candidate set' not in str(error):raise
    else:raise RuntimeError('Known radius-4 missing target was not rejected')
    window=adapter.get(26);sample=adapter.sample(26,9)
    if (len(window['states'])!=10 or len(window['targets']['indices'])!=9
            or window['source_frames']!=list(range(26,36)) or 'mask' in sample
            or tuple(sample['record']['real'].shape)!=(252,448,3)
            or tuple(sample['record']['cameras'].shape)!=(2,4,4)):
        raise RuntimeError('Cross-source training window or missing-mask semantics differ')
    if len(window['condition'].coordinates)!=evidence['sources'][TRAIN[1]]['per_start_counts'][26]:
        raise RuntimeError('Real candidate count differs from prior exact audit')
    original_loader=adapter.source
    class ChangedFuture:
        def load_window(self,*args,**kwargs):
            data=original_loader.load_window(*args,**kwargs);states=list(data['states'])
            states[5]=dict(states[5],features=states[5]['features']+1.)
            return dict(data,states=states)
        def __getattr__(self,name):return getattr(original_loader,name)
    adapter.source=ChangedFuture();modified=adapter.get(26)
    for name in ('coordinates','features','initial_present','geometry','context'):
        if not torch.equal(getattr(window['condition'],name),getattr(modified['condition'],name)):
            raise RuntimeError('Future state changed initial decoder condition: '+name)
    if torch.equal(window['targets']['features'][4],modified['targets']['features'][4]):
        raise RuntimeError('Future perturbation failed to reach its supervision label')
    try:MultiSourceWindowStore(EVAL[0],'train',decoder.base,bundle['store'].frames['std'],p['hand_roi_xyxy'],device='cpu')
    except PermissionError:pass
    else:raise RuntimeError('Evaluation source was accepted for training')
    result=dict(status='passed',adapter_sha256=sha(ROOT/'window_adapter.py'),test_sha256=sha(ROOT/'adapter_tests.py'),
      radius_audit_sha256=sha(ROOT/'GLOBAL_RADIUS6_AUDIT.json'),clip=TRAIN[1],start=26,
      radius4_missing_target_rejected=True,radius6_full_target_coverage=True,
      candidate_nodes=len(window['condition'].coordinates),future_feature_change_does_not_change_initial_condition=True,
      future_feature_change_reaches_supervision=True,eval_source_training_rejected=True,
      per_view_teacher_mask_not_invented=True,
      scope='Real-data CPU conditioning test only; no GPU, new training or cross-source action result')
    with (ROOT/'ADAPTER_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: real missing radius-4 target, radius-6 coverage and initial-only condition isolation')


if __name__=='__main__':main()

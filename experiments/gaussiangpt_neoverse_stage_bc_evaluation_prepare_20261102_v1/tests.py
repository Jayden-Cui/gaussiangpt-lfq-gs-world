"""Regression checks for evaluator controls, teacher provenance and causality."""
from collections import namedtuple
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch

from posterior_eval_core import extract_controls,check_prefix_causality,teacher_comparison,static_motion_controls

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('_evaluation_test_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        sha,name=line.split('  ',1)
        if Path(name).name!=name or binding.sha(ROOT/name)!=sha:raise RuntimeError('Evaluation code changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    fm,data,runtime=helper.model_modules(orig)
    spec=importlib.util.spec_from_file_location('_new_evaluator_witness',ROOT/'evaluate_shared.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    correct=torch.ones(168,9,16);initial=torch.full((168,9,16),2.);teacher=torch.full((168,9,16),3.)
    controls=module.build_controls(correct,initial,teacher,list(range(168)))
    decoded=module.build_predictions(lambda condition,code:code.clone(),None,controls,0,'cpu')
    if not torch.equal(decoded['correct'],correct[0]) or not torch.equal(decoded['initial_only'],initial[0]):
        raise RuntimeError('Correct and repeated-initial controls were aliased')
    if not torch.equal(decoded['swapped_window_codes'],correct[1]):raise RuntimeError('Swap changed')
    with tempfile.TemporaryDirectory(prefix='codex_eval_teacher_') as folder:
        root=Path(folder);model=torch.nn.Linear(3,2).requires_grad_(False).eval();digest=runtime.digest(model)
        codes=root/'codes.pt';torch.save(dict(starts=list(range(168)),decoder_digest=digest,codes=teacher),codes)
        source=dict(decoder_digest=digest,codec_sha256='a'*64,cache_manifest_sha256='b'*64,
                    teacher_codes_sha256=binding.sha(codes))
        ref=dict(status='complete',window_predictions=1512,source=dict(decoder_digest=digest,codec_sha256='a'*64,
          cache_manifest_sha256='b'*64,merged_codes_sha256=binding.sha(codes)),
          rows=[dict(start=s,horizon=h) for s in range(168) for h in range(1,10)])
        loaded,rows=module.validate_teacher_reference(source,ref,codes,model)
        if not torch.equal(loaded,teacher) or len(rows)!=1512:raise RuntimeError('Teacher binding changed')
        bad=dict(source,decoder_digest='c'*64)
        try:module.validate_teacher_reference(bad,ref,codes,model)
        except RuntimeError:pass
        else:raise RuntimeError('Mismatched teacher decoder was accepted')
        duplicate=dict(ref,rows=ref['rows']+[ref['rows'][0]])
        try:module.validate_teacher_reference(source,duplicate,codes,model)
        except RuntimeError:pass
        else:raise RuntimeError('Duplicate teacher row was accepted')
    memory_type=namedtuple('Memory','keys coordinates')
    class FakeStore:
        def get(self,start):
            states=[dict(coordinates=torch.tensor([[i,0,0]],dtype=torch.int32),
                         features=torch.full((1,64),float(i)),pts=i/29.97) for i in range(10)]
            memories=[memory_type(keys=torch.tensor([[float(i)]]),coordinates=torch.tensor([[i,0,0]])) for i in range(9)]
            return dict(start=start,states=states,memories=memories)
        def sample(self,start,h):return dict(start=start,h=h)
    class FakeModel:
        def __init__(self,variant,leak=False):self.variant=variant;self.leak=leak
        def eval(self):return self
        def forward_cached(self,memories):return torch.stack([m.keys.mean().expand(16) for m in memories])
        def code_for_h(self,states,h):
            chosen=states if self.leak else states[:h+1]
            return sum(s['features'].sum() for s in chosen).expand(16)
    fake=FakeModel('C1');store=FakeStore()
    def infer(model,window,static=False):return torch.full((9,16),float(window['start'])+(1. if static else 0.))
    a,b=extract_controls(fake,store,infer,list(range(168)))
    if a.shape!=(168,9,16) or not torch.equal(b,a+1):raise RuntimeError('Posterior controls changed')
    if check_prefix_causality(fake,store)['status']!='passed' or check_prefix_causality(FakeModel('C0'),store)['status']!='passed':
        raise RuntimeError('Valid prefix rejected')
    try:check_prefix_causality(FakeModel('C1',leak=True),store)
    except RuntimeError:pass
    else:raise RuntimeError('Future-leaking posterior was accepted')
    class Dummy:
        def eval(self):return self
    core=type('Core',(),{'same_initial_control':staticmethod(lambda *args:dict(same_initial_coordinates_and_features=True))})
    static=static_motion_controls(fake,Dummy(),Dummy(),store,core,{},[35,123],horizons=(1,9))
    if len(static['rows'])!=4 or not all(x['same_initial_coordinates_and_features'] for x in static['rows']):
        raise RuntimeError('Static control scope changed')
    future=[dict(start=s,horizon=h,metrics={'correct':{'metrics':{'motion_mae':.1}},
             'teacher_free':{'metrics':{'motion_mae':.1}}}) for s in range(168) for h in range(1,10)]
    result=teacher_comparison(dict(status='complete',window_predictions=1512,rows=future,
       metrics=dict(unique_frame_equal={'correct':{'motion_mae':.1},'teacher_free':{'motion_mae':.1}})))
    if not result['numerical_110pct_gate'] or result['h9_ratio']!=1.:raise RuntimeError('Teacher comparison differs')
    output=dict(status='passed',code_sha256=binding.sha(ROOT/'CODE.sha256'),
      actual_9560_source_binding_loaded=True,distinct_controls_decoded_independently=True,
      teacher_decoder_code_and_coverage_mismatch_rejected=True,
      future_prefix_leak_rejected=True,static_control_dispatch_passed=True,
      overall_and_h9_teacher_comparison_passed=True,
      scope='CPU logic tests with synthetic codes/models; no trained posterior or GPU evaluation')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: independent controls, bound teacher, causal perturbation and static-control dispatch')


if __name__=='__main__':main()

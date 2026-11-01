"""CPU checks for the conditional full-clip stage-B evaluator."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import torch

from b_core import STAGE_A
from b_evaluate import anchor_metrics,warm_initial_codes

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('_stage_b_eval_test_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    fm,data,runtime=helper.model_modules(orig)
    spec=importlib.util.spec_from_file_location('_stage_b_eval_test',ROOT/'evaluate_b.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    free=torch.ones(168,9,16);blind=torch.full((168,9,16),2.)
    controls=module.build_controls(free,blind,list(range(168)))
    predictions=module.build_predictions(lambda condition,z:z.clone(),None,controls,0,'cpu')
    if not torch.equal(predictions['correct'],free[0]) or not torch.equal(predictions['initial_only'],blind[0]):
        raise RuntimeError('Free and repeated-S0 controls were aliased')
    with tempfile.TemporaryDirectory(prefix='codex_stage_b_eval_') as directory:
        path=Path(directory)/'checkpoint.pt'
        model=torch.nn.Linear(4,2).requires_grad_(False).eval();digest=runtime.digest(model)
        torch.save(dict(schema='stage_b_shared_decoder_free_codes_v1',starts=list(range(168)),
                        decoder_digest=digest,codes=free),path)
        source=dict(decoder_digest=digest,codec_sha256='a'*64,cache_manifest_sha256='b'*64,
                    historical_checkpoint_sha256='c'*64,codes_sha256=binding.sha(path))
        historical=dict(status='complete',window_predictions=1512,
          source=dict(codec_sha256='a'*64,cache_manifest_sha256='b'*64,trained_checkpoint_sha256='c'*64),
          rows=[dict(start=s,horizon=h) for s in range(168) for h in range(1,10)])
        codes,rows=module.validate_reference(source,historical,path,model)
        if not torch.equal(codes,free) or len(rows)!=1512:raise RuntimeError('Reference binding differs')
        try:module.validate_reference(dict(source,decoder_digest='d'*64),historical,path,model)
        except RuntimeError:pass
        else:raise RuntimeError('Mismatched trained decoder was accepted')
        try:module.validate_reference(source,dict(historical,rows=historical['rows']+[historical['rows'][0]]),path,model)
        except RuntimeError:pass
        else:raise RuntimeError('Duplicate historical row was accepted')
    class FakePosterior:
        def eval(self):return self
        def __call__(self,windows):return dict(codes=torch.full((1,9,16),float(windows[0])))
    class FakeStore:
        def get(self,start):return dict(initial_memories=start)
    codes=warm_initial_codes(FakePosterior(),FakeStore(),list(range(168)))
    if codes.shape!=(168,9,16) or float(codes[123,0,0])!=123.:
        raise RuntimeError('Warm Q repeated-initial extraction differs')
    samples={};rows=[]
    for start,color in ((35,(1.,1.,0.)),(123,(0.,1.,.3))):
        image=torch.zeros(252,448,3);image[30:80,220:270]=torch.tensor(color)
        proxy=dict(target=dict(valid=True,centroid_xy=[245.,55.]),prediction=dict(valid=True,centroid_xy=[245.,55.]),
                   centroid_error_px=0.,axis_error_degrees=0.,iou=1.)
        for h in range(1,10):
            samples[start+h]=dict(record=dict(real=image))
            rows.append(dict(start=start,horizon=h,frame=start+h,metrics={'correct':{'object_proxy':proxy}}))
    result=anchor_metrics(rows,types.SimpleNamespace(frames=dict(samples=samples)),p)
    if any(x['retention_fraction']!=1 or x['center_within_10pct_fraction']!=1
           or x['median_axis_error_degrees']!=0 for x in result.values()):
        raise RuntimeError('Anchor regression proxy calculation differs')
    output=dict(status='passed',driver_sha256=binding.sha(ROOT/'b_evaluate.py'),
      evaluator_sha256=binding.sha(ROOT/'evaluate_b.py'),test_sha256=binding.sha(ROOT/'evaluation_tests.py'),
      core_manifest_sha256=binding.sha(ROOT/'CODE.sha256'),separate_control_decoding=True,
      historical_reference_explicitly_different_decoder=True,checkpoint_and_coverage_checked=True,
      warm_initial_only_extraction_and_anchor_proxies_passed=True,
      scope='CPU logic tests only; no stage-B GPU evaluation or native replay')
    with (ROOT/'EVAL_TEST.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: independent B controls, bound historical reference, warm Q and anchor proxy metrics')


if __name__=='__main__':main()

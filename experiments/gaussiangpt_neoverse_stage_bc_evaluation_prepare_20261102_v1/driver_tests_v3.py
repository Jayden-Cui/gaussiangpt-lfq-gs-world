"""CPU logic tests for cached same-initial controls and replay import isolation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import torch

import c_evaluate_v3 as c_evaluate

ROOT=Path(__file__).resolve().parent


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='codex_manifest_witness_') as folder:
        root=Path(folder);source=root/'module.py';source.write_text('x=1\n')
        manifest=root/'CODE.sha256';manifest.write_text(sha(source)+'  module.py\n')
        expected=sha(manifest);c_evaluate.verify_manifest(root,expected)
        source.write_text('x=2\n')
        try:c_evaluate.verify_manifest(root,expected)
        except RuntimeError:pass
        else:raise RuntimeError('Changed source file was accepted by its old manifest')
    runtime=types.ModuleType('runtime');support=types.ModuleType('support_decoder');panels=[];seen=[]
    def select_state(coords,features,logits,max_nodes):return dict(coordinates=coords,features=features)
    def render(bridge,state,record,views=(0,1)):
        seen.append(id(record));value=float(state['features'].reshape(-1)[0])
        return torch.full((2,2,2,3),value),None,None
    def panel(path,images):panels.append((str(path),len(images)))
    runtime.render=render;runtime.panel=panel;runtime.cpu=lambda x:x.detach().cpu()
    support.select_state=select_state
    originals={k:sys.modules.get(k) for k in ('runtime','support_decoder')}
    sys.modules['runtime']=runtime;sys.modules['support_decoder']=support
    class FakeDecoder:
        def __call__(self,condition,codes):
            return dict(features=codes[:,0,None,None],logits=torch.ones(9,1))
    record={'camera':'same-object'}
    class FakeStore:
        def get(self,start):
            return dict(condition=types.SimpleNamespace(coordinates=torch.tensor([[0,0,0]])),
                        states=[dict(coordinates=torch.tensor([[0,0,0]]),features=torch.tensor([[float(i>0)]])) for i in range(10)])
        def sample(self,start,h):return dict(record=record)
    correct=torch.ones(168,9,16);static=torch.zeros(168,9,16)
    try:
        with tempfile.TemporaryDirectory(prefix='codex_static_control_') as folder:
            result=c_evaluate.static_motion_from_codes(FakeDecoder(),object(),FakeStore(),
                {'max_predicted_latent_nodes':10},correct,static,[35],Path(folder))
        if len(result['rows'])!=4 or len(panels)!=4 or set(seen)!={id(record)}:
            raise RuntimeError('Static controls did not use one camera or cover four horizons')
        for row in result['rows']:
            if (row['motion_code_to_motion_reference']!=0 or row['static_code_to_static_reference']!=0
                    or row['motion_code_to_static_reference']!=1 or row['static_code_to_motion_reference']!=1
                    or row['horizon_code_rms_difference']!=1):
                raise RuntimeError('Motion/static matching metrics differ')
    finally:
        for name,value in originals.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    source='''import importlib.util,sys
from pathlib import Path
root=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
s=importlib.util.spec_from_file_location('_independent_replay_context',root/'stage_context.py')
c=importlib.util.module_from_spec(s);sys.modules[s.name]=c;s.loader.exec_module(c)
x,h,o,p,paths,r,e=c.setup();c.explicit('_independent_replay_test',o/'replay.py')
assert 'codec' not in sys.modules and 'MinkowskiEngine' not in sys.modules
'''
    subprocess.run([sys.executable,'-B','-c',source],check=True,timeout=30)
    output=dict(status='passed',driver_sha256=sha(ROOT/'c_evaluate_v3.py'),test_sha256=sha(ROOT/'driver_tests_v3.py'),
      core_manifest_sha256=sha(ROOT/'CODE.sha256'),same_camera_static_and_motion_metrics=True,
      distinct_motion_and_static_code_effects=True,independent_native_replay_import_isolation=True,source_manifest_tampering_rejected=True,
      scope='Synthetic CPU renderer logic; no real posterior or GPU evaluation')
    with (ROOT/'DRIVER_TEST_V3.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: cached motion/static control metrics, shared camera and isolated replay imports')


if __name__=='__main__':main()

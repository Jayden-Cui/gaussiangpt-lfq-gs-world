import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
PARENT = OUTPUTS/'gaussiangpt_neoverse_anchor_replay_pilot_20261102_v1'
B_PREP = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
FULL = OUTPUTS/'gaussiangpt_neoverse_qformer_full177_20261101_v1'
CHECKPOINT = PARENT/'run/training/sweep_05.pt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value,replace=False):
    path=Path(path);content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content,encoding='utf-8');temp.replace(path)
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(content)


def require(ok,message):
    if not ok:raise RuntimeError(message)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def tree_digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def verify(need_masks=False,need_tests=False):
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        require(Path(name).name==name and sha(ROOT/name)==digest,'Trial source changed: '+name)
    p=read(ROOT/'protocol.json')
    require(p['additional_updates_per_arm']==846 and p['final_sweep']==7 and p['parent_sweep']==5
        and p['batch_size']==4 and p['code_shape']==[168,9,16] and p['quantized'] is False,'Protocol scope differs')
    require(sha(CHECKPOINT)==p['parent_checkpoint_sha256'],'Parent checkpoint changed')
    require(sha(PARENT/'run/SUMMARY.json')==p['parent_summary_sha256'],'Parent results changed')
    require(sha(OUTPUTS/'gaussiangpt_neoverse_green_beam_data_audit_20261102_v2/result/AUDIT.json')==p['dataset_audit_sha256'],'Dataset diagnosis changed')
    done=read(PARENT/'run/training/TRAIN_DONE.json');summary=read(PARENT/'run/SUMMARY.json')
    require(done['status']=='complete' and done['updates']==2115 and done['final_checkpoint_sha256']==p['parent_checkpoint_sha256']
        and sha(PARENT/'run/training/TRAIN_DONE.json')==summary['train_done_sha256'],'Parent training incomplete')
    require(sha(FULL/'cache/MANIFEST.json')==done['source']['cache_manifest_sha256'],'Frame/window cache binding changed')
    require(sha(PARENT/'CODE.sha256')==done['source']['code_sha256'],'Parent source manifest changed')
    for line in (PARENT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);require(sha(PARENT/name)==digest,'Parent source changed: '+name)
    required={'b_core.py':'9f980425ea5019e5733adcf7a40fdd3cbbb93290258abbe8b77561a0299fc07d',
        'b_evaluate_v2.py':'d7c0d9571f5376aa1f835a8384b8316d03ca4efbdcfc404eb31578245a3dbe6b',
        'evaluate_b.py':'26a09159d7fe6b6986ce28b0b1383aa52c177dbcd352c2037a57134ca83418ad'}
    for name,digest in required.items():require(sha(B_PREP/name)==digest,'Shared implementation changed: '+name)
    if need_tests:
        tests=read(ROOT/'CPU_TESTS.json')
        require(tests['status']=='passed' and tests['code_sha256']==sha(ROOT/'CODE.sha256'),'CPU tests not bound to current code')
    if need_masks:
        m=read(ROOT/'masks/MASKS.json');review=read(ROOT/'MASK_REVIEW.json')
        require(m['status']=='complete' and m['protocol_sha256']==sha(ROOT/'protocol.json')
            and m['code_sha256']==sha(ROOT/'CODE.sha256'),'Mask provenance differs')
        require(sha(ROOT/'masks/masks.npz')==m['masks_sha256'],'Mask bytes changed')
        require(review['status']=='approved_for_training_proxy' and review['mask_receipt_sha256']==sha(ROOT/'masks/MASKS.json'),'Masks need visual review')
    return p


def schedule(p):
    sampling=module('_object_trial_sampling',PARENT/'sampling.py')
    old=sampling.schedule(5,p['base_sampler_seed'],p['replay_sampler_seed'])
    full=sampling.schedule(7,p['base_sampler_seed'],p['replay_sampler_seed'])
    require(full[:2115]==old and len(full)==2961,'Continuation changed earlier sampling')
    old_audit=sampling.audit(old,5)
    require(old_audit==read(PARENT/'run/training/TRAIN_DONE.json')['source']['sampler'],'Parent executed schedule differs')
    remainder=[dict(x,local_step=i+1) for i,x in enumerate(full[2115:])]
    require(len(remainder)==846 and {x['sweep'] for x in remainder}=={6,7},'Wrong continuation budget')
    return remainder,dict(updates=846,cumulative_updates=2961,sha256=tree_digest(remainder),
        base_visits_per_pair=2,replay_visits_per_anchor_pair=20,full_schedule=sampling.audit(full,7))

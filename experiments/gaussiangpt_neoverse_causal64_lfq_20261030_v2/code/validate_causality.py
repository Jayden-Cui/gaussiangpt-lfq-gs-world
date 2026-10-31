"""Repeat structural causal boundary check on final learned weights once."""
import argparse
from pathlib import Path
import torch
from motion_common import CLIPS, read, write, sha
from causal_data import load_prefix
from qmotion import build_model
from causality_checks import check_causality
from contracts import verify_sources


def main(root,arm):
    verify_sources(root)
    p=read(root/'protocol.json');config=next(c for c in p['arms'] if c['name']==arm)
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    checkpoint=root/arm/'motion_final.pt';training=read(root/arm/'train_done.json')
    assert sha(checkpoint)==training['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['protocol_sha256']==sha(root/'protocol.json')
    model=build_model(config).cuda().eval();model.load_state_dict(saved['model']);del saved
    checks=[]
    for end in [49,109]:
        window=load_prefix(root,CLIPS[0],end,10)
        checks.append({'start':end-9,'end':end,**check_causality(model,window)})
    write(root/arm/'causality.json',{'status':'passed','checkpoint_sha256':sha(checkpoint),'checks':checks,
        'conditional_gs_causality_only':True})
    print('TRAINED_CAUSALITY_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();main(a.root,a.arm)

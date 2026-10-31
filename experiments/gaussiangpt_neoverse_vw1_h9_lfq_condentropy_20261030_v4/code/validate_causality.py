import argparse
from pathlib import Path
import torch
from contracts import verify_sources
from motion_common import read,write,sha,CLIPS
from qmotion import build_model
from sequence_data import load_record
from causality_checks import check_causality

def run(root,arm):
    verify_sources(root);p=read(root/'protocol.json')
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.60)
    folder=root/arm;checkpoint=folder/'motion_final.pt'
    assert sha(checkpoint)==read(folder/'train_done.json')['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['protocol_sha256']==sha(root/'protocol.json')
    model=build_model(p['arms'][0]).cuda();model.load_state_dict(saved['model'])
    results=[]
    for start in [40,100]:
        record=load_record(root,CLIPS[0],start,None)
        results.append(dict(start=start,**check_causality(model,record['window'],record['condition'])))
    write(folder/'causality.json',dict(status='passed',checkpoint_sha256=sha(checkpoint),windows=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();run(a.root,a.arm)

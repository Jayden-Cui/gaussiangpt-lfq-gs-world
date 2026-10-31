"""Stride1 ten-frame windows. Last-frame replication only at the clip boundary."""
from pathlib import Path
import torch
from motion_common import load_state,InitialState,make_condition,alignment_indices,CLIPS

DT=0.033366700033366704

def window_ids():
    return [(clip,start) for clip in CLIPS for start in range(176)]

def load_window(root,clip,start,device='cuda'):
    assert 0<=start<176
    states=[]
    for h in range(10):
        frame=min(start+h,176)
        state=load_state(root,clip,frame,device)
        state['source_pts']=state['pts']
        state['pts']=float(state['pts'])+max(0,start+h-176)*DT
        state['frame_index']=frame;state['is_padding']=start+h>176
        states.append(state)
    return states

def load_record(root,clip,start,codec,device='cuda'):
    window=load_window(root,clip,start,device)
    s=window[0]
    initial=InitialState(s['coordinates'],s['features'],window[1]['pts']-s['pts'])
    geometry=torch.load(root/'geometry'/clip/f'{start:03d}.pt',map_location='cpu',weights_only=True)
    condition=make_condition(codec,initial,geometry)
    labels=[alignment_indices(condition['coordinates'],t['coordinates']).to(device) for t in window[1:]]
    return dict(window=window,initial=initial,condition=condition,labels=labels,
        frames=[t['frame_index'] for t in window],valid=[not t['is_padding'] for t in window[1:]])

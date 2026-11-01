from pathlib import Path
import torch
from io_utils import ROOT,read,sha
from support_decoder import Condition
from posterior import PrefixMemory
from core import image_metrics,object_proxy
from runtime import cpu


def tree(value,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device).contiguous()
    if isinstance(value,dict):return {k:tree(v,device) for k,v in value.items()}
    if isinstance(value,list):return [tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(tree(v,device) for v in value)
    return value


class WindowStore:
    def __init__(self,p):
        self.p=p;self.root=ROOT/'cache';self.manifest=read(self.root/'MANIFEST.json')
        if self.manifest['status']!='complete' or self.manifest['starts']!=p['starts']:
            raise RuntimeError('Window cache incomplete')
        if self.manifest['protocol_sha256']!=sha(ROOT/'protocol.json'):
            raise RuntimeError('Cache belongs to another protocol')
        self.cached={}
        frame=self.manifest['frames']
        if sha(self.root/frame['file'])!=frame['sha256']:raise RuntimeError('Frame cache changed')
        self.frames=torch.load(self.root/frame['file'],map_location='cpu',weights_only=True)
        self.roi=torch.zeros((252,448),dtype=torch.bool)
        for x0,y0,x1,y1 in p['hand_roi_xyxy']:self.roi[y0:y1,x0:x1]=True

    def get(self,start,device='cuda'):
        if start not in self.p['starts']:raise ValueError('Window outside full training source')
        if start not in self.cached:
            row=self.manifest['windows'][str(start)];path=self.root/row['file']
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise RuntimeError('Window cache changed')
            self.cached[start]=torch.load(path,map_location='cpu',weights_only=True)
        packet=tree(self.cached[start],device)
        packet['condition']=Condition(**packet['condition'])
        for key in ['memories','initial_memories']:
            packet[key]=[PrefixMemory(**m) for m in packet[key]]
        packet['roi']=self.roi.to(device)
        return packet

    def sample(self,start,h,device='cuda'):
        sample=tree(self.frames['samples'][start+h],device)
        initial=self.frames['samples'][start]['record']['real'].to(device)
        sample['motion']=((sample['record']['real']-initial).abs().mean(-1)/.1).clamp(0,1)
        return sample


def evaluate_metrics(images,w,sample,p):
    m=image_metrics(images[0],sample['record']['real'],w['roi'],sample['motion'])
    m['teacher_view1_masked_mae']=float((images[1]-sample['teacher_targets'][1]).abs().mean(-1)[sample['mask']].mean())
    result=dict(metrics=m)
    if str(w['start']) in p['object_proxies']:
        result['object_proxy']=object_proxy(cpu(images[0]).numpy(),cpu(sample['record']['real']).numpy(),p['object_proxies'][str(w['start'])])
    return result

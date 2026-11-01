"""Initial-only candidates and aligned future supervision for another source."""
import torch

from source_store import SourceStore


class MultiSourceWindowStore:
    def __init__(self,clip,purpose,base,codec_std,hand_roi_xyxy,*,radius=6,max_candidates=25000,
                 need_cached_memories=False,device='cuda'):
        self.source=SourceStore();self.source.authorize(clip,purpose)
        if radius!=6 or max_candidates!=25000:
            raise ValueError('Prepared multi-source feasibility policy is radius 6 and max 25000')
        if tuple(codec_std.shape)!=(64,) or not bool(torch.isfinite(codec_std).all()):
            raise ValueError('Expected the frozen training codec feature scale')
        self.clip=clip;self.purpose=purpose;self.base=base;self.radius=radius;self.max_candidates=max_candidates
        self.need_cached_memories=need_cached_memories;self.device=device
        self.frames=dict(std=codec_std.detach().cpu().clone())
        self.roi=torch.zeros((252,448),dtype=torch.bool)
        for x0,y0,x1,y1 in hand_roi_xyxy:self.roi[y0:y1,x0:x1]=True
        if not bool(self.roi.any()):raise ValueError('Empty fixed hand ROI')

    def get(self,start):
        from support_decoder import make_condition,target_labels
        data=self.source.load_window(self.clip,start,purpose=self.purpose,device=self.device)
        states=data['states']
        # Only this first state enters initial encoder/candidate construction.
        condition=make_condition(self.base,states[0],self.radius,self.max_candidates)
        # Future states are labels only; target_labels raises on any uncovered node.
        targets=target_labels(condition,states[1:])
        result=dict(start=start,clip=self.clip,states=states,condition=condition,targets=targets,
                    roi=self.roi.to(self.device),source_frames=data['source_frames'],
                    candidate_policy='S0 only; fixed radius 6, no truncation',
                    reference_mask='unavailable; no original per-view teacher mask supplied')
        if self.need_cached_memories:
            from posterior import make_memories
            result['memories']=make_memories(self.base,states,initial_only=False)
            result['initial_memories']=make_memories(self.base,states,initial_only=True)
        return result

    def sample(self,start,h):
        if not 0<=h<=9:raise ValueError('Expected horizon 0..9')
        sample=self.source.load_frame(self.clip,start+h,purpose=self.purpose,device=self.device)['sample']
        initial=self.source.load_frame(self.clip,start,purpose=self.purpose,device=self.device)['sample']['record']['real']
        sample['motion']=((sample['record']['real']-initial).abs().mean(-1)/.1).clamp(0,1)
        return sample

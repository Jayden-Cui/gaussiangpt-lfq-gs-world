"""Full-resolution data for the fixed baseline, with real-source admission."""
from __future__ import annotations

import torch
from codec_binding import BaselineStates, verify_native_replay, require
from teacher_binding import BoundTeachers, STATE_ROOT

PROTOCOL_SHA = '24add801f064c181658b43da98e1b9a42779c2b876105231ded2ef044b1205da'
REGION_BOXES = ((0,35,180,252), (265,25,448,252))


class BoundData:
    """Load all 177 verified records; only state windows enter the model."""
    def __init__(self, arm_root, native_results_root, device='cuda'):
        import json
        from codec_binding import checked_bytes
        self.binding = BaselineStates(arm_root)
        self.native_evidence = verify_native_replay(self.binding, native_results_root)
        self.state_evidence = self.binding.verify_all()
        self.teachers = BoundTeachers(self.binding)
        self.teacher_evidence = self.teachers.verify_all()
        config = json.loads(checked_bytes(STATE_ROOT/'protocol.json', PROTOCOL_SHA))
        boxes = config['evaluation']['region_boxes']['buildingblock_tower_17_176']['hand_activity']
        require(tuple(tuple(b) for b in boxes) == REGION_BOXES, 'Fixed activity ROI changed')
        device = torch.device(device)
        self.roi = torch.zeros((252,448), dtype=torch.bool, device=device)
        for x0,y0,x1,y1 in REGION_BOXES:
            self.roi[y0:y1,x0:x1] = True
        self.states = [self.binding.load(frame, device=device) for frame in range(177)]
        samples = [self.teachers.load_record(frame, device=device) for frame in range(177)]
        self.records = [sample['record'] for sample in samples]
        self.teacher_targets = [sample['teacher_targets'] for sample in samples]
        self.teacher_masks = [sample['mask'] for sample in samples]
        self.normalization = self.binding.rows[0]['normalization']
        require(len(self.states) == len(self.records) == len(self.teacher_targets) == len(self.teacher_masks) == 177,
                'Full 177-frame data required')

    def window(self, start):
        require(type(start) is int and 0 <= start < 168, 'Expected adjacent T10 window')
        result = self.states[start:start+10]
        require(len(result) == 10 and [s['frame_index'] for s in result] == list(range(start,start+10)),
                'Incomplete state window')
        return result

    def motion_weight(self, start, horizon):
        if horizon == 0:
            return torch.zeros_like(self.roi, dtype=torch.float32)
        return ((self.records[start+horizon]['real']-self.records[start]['real']).abs().mean(-1)/.1).clamp(0,1).detach()

    def loss(self, image, start, horizon):
        error = (image-self.records[start+horizon]['real']).abs().mean(-1)
        full, roi = error.mean(), error[self.roi].mean()
        weight = self.motion_weight(start, horizon)
        motion = (weight*error).sum()/weight.sum().clamp_min(1e-6)
        return full/10+roi/10+(motion/9 if horizon else 0), dict(
            full_l1=float(full.detach()), roi_l1=float(roi.detach()), motion_l1=float(motion.detach()))

    def metrics(self, image, start, horizon):
        error = image-self.records[start+horizon]['real']
        mse, mae = error.square(), error.abs().mean(-1)
        weight = self.motion_weight(start, horizon)
        return dict(psnr=float(-10*torch.log10(mse.mean().clamp_min(1e-12))),
                    hand_activity_psnr=float(-10*torch.log10(mse[self.roi].mean().clamp_min(1e-12))),
                    mae=float(mae.mean()), hand_activity_mae=float(mae[self.roi].mean()),
                    motion_mae=float((mae*weight).sum()/weight.sum().clamp_min(1e-6)))

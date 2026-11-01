"""Assemble two independent code shards without missing or reordering windows."""
import torch


def merge_parts(parts):
    result=torch.empty((168,9,16),dtype=torch.float32)
    seen=set()
    for starts,codes in parts:
        if len(starts)!=len(set(starts)) or tuple(codes.shape)!=(len(starts),9,16):
            raise ValueError('Bad shard shape or duplicate starts')
        for position,start in enumerate(starts):
            if start in seen or not 0<=start<168:raise ValueError('Overlapping or invalid shard')
            result[start]=codes[position];seen.add(start)
    if seen!=set(range(168)) or not bool(torch.isfinite(result).all()):
        raise ValueError('Incomplete or nonfinite assembled codes')
    return result

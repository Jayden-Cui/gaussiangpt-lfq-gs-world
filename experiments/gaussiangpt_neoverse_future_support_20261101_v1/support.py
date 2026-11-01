"""Deterministic spatial transfer using initial features and coordinates only."""
from __future__ import annotations
import hashlib
import numpy as np
import torch


def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256()
    h.update(str(tuple(value.shape)).encode())
    h.update(str(value.dtype).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def transfer_initial_features(initial_coordinates, initial_features, future_coordinates, *, cell_size=.032):
    """Exact-overlap copy, otherwise nearest cell; ties use lexicographic XYZ.

    These are fixed world-grid cells, not physical particles or motion matches.
    The interface intentionally never accepts future features. All destination
    coordinates survive and retain their input order. Distances are computed
    with integer CPU arithmetic to avoid CPU/GPU roundoff-dependent tie breaks.
    """
    for coords in (initial_coordinates,future_coordinates):
        if coords.ndim!=2 or coords.shape[1]!=3 or not len(coords) or coords.dtype not in (torch.int32,torch.int64):
            raise ValueError('Expected nonempty integer coordinates [M,3]')
        if len(torch.unique(coords,dim=0))!=len(coords):
            raise ValueError('Duplicate coordinates')
        if int(coords.abs().max())>1000000:
            raise ValueError('Coordinates exceed exact squared-distance budget')
    if (initial_features.ndim!=2 or len(initial_features)!=len(initial_coordinates)
            or not initial_features.is_floating_point() or not bool(torch.isfinite(initial_features).all())):
        raise ValueError('Invalid initial features')
    a=initial_coordinates.detach().cpu().numpy().astype(np.int64)
    b=future_coordinates.detach().cpu().numpy().astype(np.int64)
    order=np.lexsort((a[:,2],a[:,1],a[:,0]))
    chosen=[];distances=[]
    for begin in range(0,len(b),256):
        delta=b[begin:begin+256,None,:]-a[order][None,:,:]
        squared=(delta*delta).sum(-1)
        nearest=squared.argmin(1)
        chosen.append(order[nearest])
        distances.append(squared[np.arange(len(nearest)),nearest])
    indices=torch.from_numpy(np.concatenate(chosen)).long().to(initial_features.device)
    distance_squared=np.concatenate(distances)
    mapped=initial_features.detach().index_select(0,indices).clone()
    distance=np.sqrt(distance_squared)
    report=dict(method='exact coordinate overlap; otherwise nearest initial spatial cell',
                tie_break='lexicographic XYZ, invariant to source row order',
                initial_nodes=len(a),future_nodes=len(b),exact_overlap_nodes=int((distance_squared==0).sum()),
                nearest_filled_nodes=int((distance_squared>0).sum()),
                nearest_distance_cells={'mean':float(distance.mean()),'median':float(np.median(distance)),
                                        'p95':float(np.quantile(distance,.95)),'max':float(distance.max())},
                cell_size_in_normalized_scene_units=cell_size,
                max_distance_in_normalized_scene_units=float(distance.max()*cell_size),
                original_initial_features_sha256=tensor_sha(initial_features),
                transferred_features_sha256=tensor_sha(mapped),
                original_initial_coordinates_sha256=tensor_sha(initial_coordinates),
                future_coordinates_sha256=tensor_sha(future_coordinates),
                future_feature_values_used=False,physical_point_correspondence=False)
    return mapped,indices,report

"""CPU-only latent-support/residual-capacity audit of the immutable v6 inputs.

Full cache bytes are hashed; only coordinates/features are analyzed through
mmap. Embedded reconstruction tensors and raw RGB caches are never accessed.
Teacher latent support is not the support of the generated Gaussian scene.
No training, checkpoint loading, quality evaluation, or GPU initialization.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
FRAME_COUNT, WINDOW_COUNT, HORIZON, WIDTH = 177, 168, 9, 64
LATENT_STRIDE, GRID = 4, .008
UNIT = LATENT_STRIDE * GRID


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def bound_paths(root):
    """Extract known literal roots without importing GPU-backed main modules."""
    tree = ast.parse((root/'code/gs_bridge.py').read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('STATE_ROOT','V4_ROOT')):
            value=node.value
            if not (isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='Path'
                    and len(value.args)==1 and isinstance(value.args[0],ast.Constant)
                    and isinstance(value.args[0].value,str)):
                raise ValueError('Native source roots no longer use the admitted literal path contract')
            values[node.targets[0].id]=Path(value.args[0].value)
    if set(values)!=set(('STATE_ROOT','V4_ROOT')):
        raise ValueError('Missing bound native source roots')
    values['MOTION_ROOT']=values['STATE_ROOT'].parent/'motion'
    return values


def verify_sources(root):
    manifest=read(root/'CODE_MANIFEST.json')
    if not isinstance(manifest,dict) or 'protocol.json' not in manifest:
        raise ValueError('Main manifest does not bind protocol.json')
    for relative,expected in manifest.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha(path)!=expected:
            raise ValueError('Main immutable source mismatch: '+str(relative))
    cfg=read(root/'protocol.json')
    if (cfg['data']['frames']!=177 or cfg['data']['window_starts']!=168
            or cfg['data']['frames_per_window']!=10 or cfg['data']['frame_gap']!=1
            or cfg['model']['width']!=64 or cfg['model']['residual_scale']!=3.):
        raise ValueError('Unexpected v6 support/residual contract')
    paths=bound_paths(root)
    binding=read(paths['V4_ROOT']/'source_binding.json')
    cache_path=paths['V4_ROOT']/'cache_manifest.json'
    if sha(cache_path)!=binding['cache_manifest_sha256']:
        raise ValueError('V4 cache manifest hash mismatch')
    dependencies={row['path']:row for row in binding['dependencies']}
    codec_path=paths['STATE_ROOT']/'code/codec.py'
    selection_path=paths['STATE_ROOT']/'selection.json'
    state_protocol_path=paths['STATE_ROOT']/'protocol.json'
    for path in (codec_path,state_protocol_path):
        if str(path) not in dependencies or sha(path)!=dependencies[str(path)]['sha256']:
            raise ValueError('Native codec/protocol source changed: '+str(path))
    if read(selection_path)['grid']!=GRID or read(state_protocol_path)['grid_candidates']!=[GRID]:
        raise ValueError('Native grid is no longer .008')
    # The released v4 dependency list omits selection.json. Do not invent a
    # historical binding: its grid is checked against the hash-bound state
    # protocol and its present bytes are recorded/rechecked in this audit.
    selection_bound=str(selection_path) in dependencies
    if selection_bound and sha(selection_path)!=dependencies[str(selection_path)]['sha256']:
        raise ValueError('Native selection source changed')
    # Inspect the actual method in the source bound above. It is not imported.
    codec_tree=ast.parse(codec_path.read_text(encoding='utf-8'))
    methods={node.name:node for cls in codec_tree.body if isinstance(cls,ast.ClassDef) and cls.name=='StateCodec'
             for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ('_world','_sparse')}
    if set(methods)!=set(('_world','_sparse')):
        raise ValueError('Missing native coordinate conversion methods')
    rows=read(cache_path)
    if not isinstance(rows,list) or len({row['path'] for row in rows})!=len(rows):
        raise ValueError('Malformed/duplicate cache manifest paths')
    hashes={str(root/'CODE_MANIFEST.json'):sha(root/'CODE_MANIFEST.json'),
            str(root/'protocol.json'):sha(root/'protocol.json'),
            str(paths['V4_ROOT']/'source_binding.json'):sha(paths['V4_ROOT']/'source_binding.json'),
            str(cache_path):sha(cache_path),str(codec_path):sha(codec_path),str(selection_path):sha(selection_path),
            str(state_protocol_path):sha(state_protocol_path)}
    units=dict(grid=GRID,latent_stride=LATENT_STRIDE,latent_cell_normalized_coordinate_units=UNIT,
               codec_source=str(codec_path),
               world_method_line=methods['_world'].lineno,sparse_method_line=methods['_sparse'].lineno,
               world_formula='(decoded voxel coordinates + coordinate offsets) * grid',
               latent_formula='integer latent coordinates * stride before sparse decoding',
               selection_historically_hash_bound=selection_bound,
               selection_evidence='Current selection SHA recorded; grid agrees with the historical hash-bound state protocol',
               interpretation='Latent-grid Linf distance and .032 normalized-scene coordinate units; not meters or final GS displacement')
    return cfg,paths,{row['path']:row for row in rows},hashes,units


def coordinate_intersection(initial,future,np):
    """Exact integer row identity; never nearest-neighbor feature matching."""
    dtype=np.dtype([('x','<i8'),('y','<i8'),('z','<i8')])
    left=np.ascontiguousarray(initial,dtype=np.int64).view(dtype).reshape(-1)
    right=np.ascontiguousarray(future,dtype=np.int64).view(dtype).reshape(-1)
    _,li,ri=np.intersect1d(left,right,assume_unique=True,return_indices=True)
    return li,ri


def quantiles(values,np,levels=(.5,.9,.99)):
    if values.size==0:return {f'q{int(level*100)}':None for level in levels}
    answer=np.quantile(values,levels)
    return {f'q{int(level*100)}':float(value) for level,value in zip(levels,answer)}


def feature_capacity(initial,future,std,scale,np):
    """Analyze matched coordinate rows; exact zero std gets no epsilon freedom."""
    if initial.shape!=future.shape or initial.ndim!=2 or initial.shape[1]!=len(std):
        raise ValueError('Matched feature arrays have incompatible shapes')
    if bool((std<0).any()) or not bool(np.isfinite(std).all()):
        raise ValueError('Invalid fixed channel std')
    positive=std>0;zero=~positive
    delta=np.abs(future-initial)
    ratio=delta[:,positive]/std[positive] if bool(positive.any()) else np.empty((len(initial),0),dtype=np.float32)
    if not bool(np.isfinite(ratio).all()):
        raise ValueError('Nonfinite normalized feature delta; no epsilon replacement allowed')
    excess=ratio>scale
    excess_rows=excess.any(axis=1)
    changed_zero=(future[:,zero]!=initial[:,zero])
    changed_zero_rows=changed_zero.any(axis=1)
    common=len(initial);entries=int(ratio.size);zero_entries=int(changed_zero.size)
    info=dict(common_coordinate_rows=common,positive_std_channels=int(positive.sum()),zero_std_channels=int(zero.sum()),
              normalized_feature_entries=entries,over_scale_entries=int(excess.sum()),
              over_scale_rows=int(excess_rows.sum()),zero_std_entries=zero_entries,
              zero_std_changed_entries=int(changed_zero.sum()),zero_std_changed_rows=int(changed_zero_rows.sum()),
              bound_violating_rows=int((excess_rows|changed_zero_rows).sum()),
              abs_delta_over_std=quantiles(ratio.reshape(-1),np),
              over_scale_entry_fraction=float(excess.sum()/entries) if entries else None,
              over_scale_row_fraction=float(excess_rows.mean()) if common else None,
              zero_std_changed_entry_fraction=float(changed_zero.sum()/zero_entries) if zero_entries else None,
              zero_std_changed_row_fraction=float(changed_zero_rows.mean()) if common else None,
              bound_violating_row_fraction=float((excess_rows|changed_zero_rows).mean()) if common else None,
              zero_std_changed_entries_per_channel=[int(value) for value in changed_zero.sum(axis=0)])
    return info,ratio.reshape(-1).copy()


def histogram_quantiles(histogram,levels=(.5,.9)):
    count=sum(histogram.values())
    if not count:return {f'q{int(q*100)}':None for q in levels}
    ordered=sorted(histogram.items())
    def at_rank(rank):
        cumulative=0
        for value,number in ordered:
            cumulative+=number
            if rank<cumulative:return value
        raise AssertionError('Histogram rank out of range')
    result={}
    for q in levels:
        rank=(count-1)*q;lower=math.floor(rank);upper=math.ceil(rank)
        result[f'q{int(q*100)}']=float(at_rank(lower)+(rank-lower)*(at_rank(upper)-at_rank(lower)))
    return result


def pair_scalars(row):
    feature=row['feature_capacity'];distance=row['future_to_initial_linf_grid']
    return dict(initial_count=row['initial_count'],future_count=row['future_count'],intersection_count=row['intersection_count'],
                future_new_fraction=row['future_new_fraction'],initial_disappeared_fraction=row['initial_disappeared_fraction'],
                distance_q50=distance['q50'],distance_q90=distance['q90'],distance_max=distance['max'],
                feature_ratio_q50=feature['abs_delta_over_std']['q50'],
                feature_ratio_q90=feature['abs_delta_over_std']['q90'],feature_ratio_q99=feature['abs_delta_over_std']['q99'],
                over_scale_entry_fraction=feature['over_scale_entry_fraction'],over_scale_row_fraction=feature['over_scale_row_fraction'],
                zero_std_changed_row_fraction=feature['zero_std_changed_row_fraction'],
                bound_violating_row_fraction=feature['bound_violating_row_fraction'])


def mean_scalars(values):
    result,valid={},{}
    for key in values[0]:
        present=[row[key] for row in values if row[key] is not None]
        result[key]=statistics.fmean(present) if present else None
        valid[key]=len(present)
    return dict(mean=result,valid_observations=valid)


def weighted_summary(rows,arrays,histograms,np):
    """Pooled point/feature-entry exposure statistics, distinct from mean quantiles."""
    totals={key:sum(row[key] for row in rows) for key in
            ('initial_count','future_count','intersection_count','future_new_count','initial_disappeared_count')}
    feature_keys=('normalized_feature_entries','over_scale_entries','over_scale_rows','zero_std_entries',
                  'zero_std_changed_entries','zero_std_changed_rows','bound_violating_rows')
    counts={key:sum(row['feature_capacity'][key] for row in rows) for key in feature_keys}
    histogram={}
    for source in histograms:
        for value,count in source.items():histogram[value]=histogram.get(value,0)+count
    distance=histogram_quantiles(histogram)
    distance['max']=float(max(histogram)) if histogram else None
    distance['mean']=sum(value*count for value,count in histogram.items())/sum(histogram.values()) if histogram else None
    present=[array for array in arrays if array.size]
    # All entries are retained. In-place quantile partition acts only on this
    # newly concatenated temporary, never on the retained pair observations.
    if present:
        pooled=np.concatenate(present)
        q=np.quantile(pooled,[.5,.9,.99],overwrite_input=True)
        ratio_q=dict(zip(('q50','q90','q99'),map(float,q)))
        del pooled
    else:ratio_q=dict(q50=None,q90=None,q99=None)
    def divide(a,b):return a/b if b else None
    return dict(point_exposure_counts=totals,feature_entry_exposure_counts=counts,
                future_new_fraction=divide(totals['future_new_count'],totals['future_count']),
                initial_disappeared_fraction=divide(totals['initial_disappeared_count'],totals['initial_count']),
                future_point_weighted_linf_grid=distance,
                future_point_weighted_linf_normalized_units={key:(value*UNIT if value is not None else None) for key,value in distance.items()},
                pooled_feature_entry_weighted_abs_delta_over_std=ratio_q,
                over_scale_entry_fraction=divide(counts['over_scale_entries'],counts['normalized_feature_entries']),
                over_scale_common_row_fraction=divide(counts['over_scale_rows'],totals['intersection_count']),
                zero_std_changed_entry_fraction=divide(counts['zero_std_changed_entries'],counts['zero_std_entries']),
                zero_std_changed_common_row_fraction=divide(counts['zero_std_changed_rows'],totals['intersection_count']),
                bound_violating_common_row_fraction=divide(counts['bound_violating_rows'],totals['intersection_count']),
                warning='Repeated point/feature exposures across overlapping windows; not unique physical points or independent samples')


def report(summary):
    pair=summary['window_pair_uniform']['mean'];frame=summary['physical_target_frame_uniform']['mean'];point=summary['point_weighted']
    def fmt(value):return '—' if value is None else f'{value:.6f}'
    lines=['# 初态 latent 支持与残差容量诊断','',
           '**仅表示机制诊断，不测重建质量，不证明方案优劣或当前GS表示不可达。**',
           '完整缓存字节校验、仅latent张量分析；没有访问reconstruction或原始RGB缓存。','',
           f"177帧，168完整窗×9未来={summary['pairs']}对；固定frame0 population std，residual_scale=3，无epsilon替代。",'',
           '| 量 | 窗口对等权 | 物理目标帧等权 | 点／feature-entry曝光加权 |',
           '|---|---:|---:|---:|']
    for label,key,pvalue in [
        ('未来新增latent坐标比例','future_new_fraction',point['future_new_fraction']),
        ('初态消失latent坐标比例','initial_disappeared_fraction',point['initial_disappeared_fraction']),
        ('非零std频道超3 entry比例','over_scale_entry_fraction',point['over_scale_entry_fraction']),
        ('含任一超3频道的共同坐标行比例','over_scale_row_fraction',point['over_scale_common_row_fraction']),
        ('含zero-std变化或超3的共同坐标行比例','bound_violating_row_fraction',point['bound_violating_common_row_fraction'])]:
        lines.append(f'| {label} | {fmt(pair[key])} | {fmt(frame[key])} | {fmt(pvalue)} |')
    lines+=['','等权列中的q50/q90/q99是“每对分位数的平均”，不是混合分布的分位数。'
            'point_weighted 则合并全部距离／feature-entry曝光再计算分位数，重复曝光明确保留。','',
            '未来→初态最近L∞距离（合并未来点曝光）：'+json.dumps(point['future_point_weighted_linf_grid'],ensure_ascii=False)+' latent格点。',
            '乘0.032的归一化场景坐标单位：'+json.dumps(point['future_point_weighted_linf_normalized_units'],ensure_ascii=False)+'；不称米。',
            '共同坐标非零std频道 |Δfeature|/std 的合并分位数：'+json.dumps(point['pooled_feature_entry_weighted_abs_delta_over_std'])+'。','',
            f"frame0零std频道：{summary['calibration']['zero_std_channels']}。其残差自由度严格为零；"
            f"后续任一帧出现数值变化的频道：{summary['frame_std_analysis']['zero_std_future_changed_channels']}。",'',
            '| h | 新增比例：窗等权 | 最近距离q90：每窗均值 | feature q99：每窗均值 | 超界共同坐标行比例：点加权 |',
            '|---|---:|---:|---:|---:|']
    for h in range(1,10):
        value=summary['per_horizon'][str(h)];m=value['window_pair_uniform']['mean']
        lines.append(f"| {h} | {fmt(m['future_new_fraction'])} | {fmt(m['distance_q90'])} | "
                     f"{fmt(m['feature_ratio_q99'])} | {fmt(value['point_weighted']['bound_violating_common_row_fraction'])} |")
    lines+=['','所有177帧各频道std、对frame0比值、zero-std频道激活/常量漂移及每目标帧等权明细见SUMMARY.json。'
            '无共同坐标时feature统计为null，不能当作零变化。','',
            'teacher latent坐标新增/消失不等于native decoder生成的GS支持新增/消失。'
            '独立源码审计确认GS中心offset来自未clamp的Parameter与Linear、OffsetRepresentation为identity，'
            '_world=(coords+offset)*.008，因此架构没有硬位移半径。生成式稀疏上采样、预测剪枝及属性头也影响最终场景。'
            '新增latent cells比例不能推导GS中心不可达或需要扩域；本层确有的限制是固定frame0 std0和tanh×3残差幅度。'
            'feature超界也只能说明该teacher latent不能在同一anchor坐标逐项复现，不能单独证明RGB/GS不可达。'
            '此处未测decoder映射、最终GS位置、优化状态或质量。'
            '源码证据及动态核验边界见[原生支持语义审计](../../mechanism_audit/NATIVE_SUPPORT_SEMANTICS.md)。','',
            f"CPU耗时 {summary['elapsed_seconds']:.2f}s；线程 {summary['threads']}；无训练/GPU。",'']
    return '\n'.join(lines)


def run(root,output,threads,max_seconds):
    begun=time.monotonic();stage='verify_main_sources';completed=0
    output.mkdir(parents=True,exist_ok=False)
    previous_alarm=None
    def check_budget():
        if time.monotonic()-begun>=max_seconds:raise TimeoutError('CPU support-capacity audit budget exhausted')
    def deadline(signum,frame):raise TimeoutError('CPU support-capacity audit internal deadline expired')
    try:
        if hasattr(signal,'setitimer'):
            previous_alarm=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max_seconds)
        cfg,paths,manifest,input_hashes,units=verify_sources(root)
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]=str(threads)
        try:
            import numpy as np
            import scipy
            from scipy.spatial import cKDTree
            import torch
        except ImportError as error:
            raise RuntimeError('Require existing numpy/scipy.cKDTree/torch CPU packages; no installation or fallback') from error
        torch.set_num_threads(threads);torch.set_num_interop_threads(threads)
        if torch.cuda.is_initialized():raise RuntimeError('GPU was initialized unexpectedly; CPU audit refuses to continue')
        clip=cfg['data']['clip'];states=[];source_rows=[];normalization=None
        stage='verify_and_mmap177_latent_caches'
        for frame in range(177):
            check_budget()
            path=paths['MOTION_ROOT']/'states'/clip/f'{frame:03d}.pt'
            entry=manifest.get(str(path))
            if entry is None or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError('Latent cache bytes/SHA mismatch: '+str(path))
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            # Do NOT iterate/convert other tensors: reconstruction remains an
            # untouched mmap storage. Hashing above intentionally covers bytes.
            coordinates=payload['coordinates'];features=payload['features']
            if (coordinates.device.type!='cpu' or features.device.type!='cpu'
                    or coordinates.dtype not in (torch.int32,torch.int64)
                    or coordinates.ndim!=2 or coordinates.shape[1]!=3 or len(coordinates)==0
                    or features.shape!=(len(coordinates),64) or features.dtype!=torch.float32
                    or not bool(torch.isfinite(features).all())):
                raise ValueError('Invalid native64 CPU latent cache')
            c=coordinates.numpy().astype(np.int64,copy=True);f=features.numpy().copy()
            if len(np.unique(c,axis=0))!=len(c):raise ValueError('Duplicate latent coordinates')
            if payload.get('frame')!=frame or payload.get('clip')!=clip:
                raise ValueError('Latent source frame/clip mismatch')
            norm=json.dumps(payload['normalization'],sort_keys=True,separators=(',',':'))
            if normalization is None:normalization=norm
            if norm!=normalization:raise ValueError('Cross-frame normalized coordinate system changed')
            std=features.float().std(0,unbiased=False).numpy().copy()
            if not bool(np.isfinite(std).all()):raise ValueError('Nonfinite channel population std')
            states.append(dict(coordinates=c,features=f,std=std,pts=float(payload['pts'])))
            source_rows.append(dict(frame=frame,path=str(path),bytes=entry['bytes'],sha256=entry['sha256']))
            del payload,coordinates,features
        if any(b['pts']<=a['pts'] for a,b in zip(states,states[1:])):raise ValueError('Nonincreasing source PTS')
        std0=states[0]['std'];zero=std0==0;positive=~zero;zero_indices=np.flatnonzero(zero).tolist()
        if not bool(positive.any()):raise ValueError('All calibrated channels have zero std')
        calibration=dict(source=source_rows[0],population_std=std0.astype(float).tolist(),
                         zero_std_channels=zero_indices,positive_std_channels=np.flatnonzero(positive).tolist(),
                         formula='torch.float32.std(dim=0,unbiased=False), exactly the v6 calibration operation on CPU',
                         epsilon=None,residual_scale=3.,arithmetic='Cached FP32 feature differences and fixed FP32 std; CPU reduction may differ from GPU at rounding precision')
        frame_stats=[];zero_future_changed=set();zero_future_variable=set()
        reference0=states[0]['features'][0]
        for frame,state in enumerate(states):
            check_budget()
            changed=state['features'][:,zero]!=reference0[zero]
            changed_counts=changed.sum(0)
            variable=state['std'][zero]>0
            if frame:
                zero_future_changed.update(ch for ch,count in zip(zero_indices,changed_counts) if count)
                zero_future_variable.update(ch for ch,flag in zip(zero_indices,variable) if flag)
            ratios=np.divide(state['std'],std0,out=np.zeros_like(std0),where=positive)
            frame_stats.append(dict(frame=frame,latent_rows=len(state['coordinates']),
                                    population_std=state['std'].astype(float).tolist(),
                                    std_over_frame0=[float(ratios[ch]) if positive[ch] else None for ch in range(64)],
                                    zero_std_changed_values_per_channel=[int(value) for value in changed_counts],
                                    zero_std_future_variable_channels=[ch for ch,flag in zip(zero_indices,variable) if flag]))
        channel_summary=[]
        for ch in range(64):
            values=np.asarray([state['std'][ch] for state in states])
            item=dict(channel=ch,frame0_std=float(std0[ch]),all177_std_min=float(values.min()),all177_std_max=float(values.max()))
            item['std_ratio_quantiles']=quantiles(values/std0[ch],np) if positive[ch] else None
            item['std_ratio_max']=float((values/std0[ch]).max()) if positive[ch] else None
            channel_summary.append(item)
        stage='analyze1512_coordinate_and_feature_pairs'
        rows=[];ratio_arrays=[];histograms=[]
        trees=[cKDTree(state['coordinates']) for state in states[:168]]
        with (output/'pairs.jsonl').open('x',encoding='utf-8') as stream:
            for start in range(168):
                initial=states[start]
                for h in range(1,10):
                    check_budget();future=states[start+h]
                    li,ri=coordinate_intersection(initial['coordinates'],future['coordinates'],np)
                    common=len(li);ni=len(initial['coordinates']);nf=len(future['coordinates'])
                    distance,_=trees[start].query(future['coordinates'],k=1,p=float('inf'),workers=threads)
                    if not bool(np.isfinite(distance).all()):raise ValueError('Nonfinite nearest L-infinity distance')
                    values,counts=np.unique(distance,return_counts=True)
                    histogram={float(value):int(count) for value,count in zip(values,counts)}
                    distance_q=quantiles(distance,np,levels=(.5,.9));distance_q['max']=float(distance.max())
                    feature,ratios=feature_capacity(initial['features'][li],future['features'][ri],std0,3.,np)
                    row=dict(start=start,end=start+h,horizon=h,initial_count=ni,future_count=nf,intersection_count=common,
                             future_new_count=nf-common,initial_disappeared_count=ni-common,
                             future_new_fraction=(nf-common)/nf,initial_disappeared_fraction=(ni-common)/ni,
                             future_to_initial_linf_grid=distance_q,
                             future_to_initial_linf_normalized_units={key:value*UNIT for key,value in distance_q.items()},
                             feature_capacity=feature)
                    rows.append(row);ratio_arrays.append(ratios);histograms.append(histogram);completed+=1
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                if start%21==0:print('SUPPORT_CAPACITY',start+1,'/168',flush=True)
        stage='aggregate_explicit_weightings';check_budget()
        if len(rows)!=1512 or {(r['start'],r['horizon']) for r in rows}!={(s,h) for s in range(168) for h in range(1,10)}:
            raise RuntimeError('Incomplete pair coverage')
        per_target={}
        for target in range(1,177):
            selected=[row for row in rows if row['end']==target]
            per_target[str(target)]=dict(contributing_windows=len(selected),**mean_scalars([pair_scalars(row) for row in selected]))
        per_h={}
        for h in range(1,10):
            check_budget();indices=[index for index,row in enumerate(rows) if row['horizon']==h]
            selected=[rows[index] for index in indices]
            per_h[str(h)]=dict(window_pair_uniform=mean_scalars([pair_scalars(row) for row in selected]),
                              point_weighted=weighted_summary(selected,[ratio_arrays[index] for index in indices],
                                                              [histograms[index] for index in indices],np))
        pooled=weighted_summary(rows,ratio_arrays,histograms,np);check_budget()
        # Revalidate every main manifest entry, not only the manifest file.
        verify_sources(root)
        for path,digest in input_hashes.items():
            if sha(path)!=digest:raise RuntimeError('Bound source changed while auditing: '+path)
        # Each cache file was SHA-checked immediately before its read. This is
        # a read-only snapshot; no second full-cache byte scan is claimed.
        if torch.cuda.is_initialized():raise RuntimeError('GPU was initialized unexpectedly')
        summary=dict(status='complete',scope='完整缓存字节校验、仅latent张量分析；不访问reconstruction或STATE_ROOT/cache真实RGB',
                     training_updates=0,gpu_executed=False,goal_complete=False,pairs=1512,windows=168,source_frames=177,
                     physical_target_frames=176,threads=threads,max_seconds=max_seconds,
                     versions=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__),
                     calibration=calibration,units=units,input_hashes=input_hashes,latent_sources=source_rows,
                     native_support_semantics_reference='mechanism_audit/NATIVE_SUPPORT_SEMANTICS.md (independent explanatory source audit, not a runtime computation dependency)',
                     normalization_metadata_sha256=hashlib.sha256(normalization.encode()).hexdigest(),
                     frame_std_analysis=dict(frames=frame_stats,per_channel=channel_summary,
                         zero_std_future_changed_channels=sorted(zero_future_changed),
                         zero_std_future_variable_channels=sorted(zero_future_variable),
                         definition='Changed means any cached future value differs from frame0 first-row value; '
                                    'variable means future within-frame population std is positive; constant mean shifts are reported too'),
                     window_pair_uniform=mean_scalars([pair_scalars(row) for row in rows]),
                     physical_target_frame_uniform=mean_scalars([value['mean'] for value in per_target.values()]),
                     per_target_frame=per_target,per_horizon=per_h,point_weighted=pooled,
                     pair_quantile_aggregation='Equal-weight summaries average per-pair quantiles; pooled point/entry quantiles are separate and exact over all retained FP32 ratios',
                     pairs_sha256=sha(output/'pairs.jsonl'),script_sha256=sha(Path(__file__)),
                     elapsed_seconds=time.monotonic()-begun,
                     limitations=['Teacher latent support differs from generated GS support; new latent coordinates do not alone imply unreachable GS/RGB',
                                  'Native GS offsets have no architecture-imposed displacement radius; teacher cell novelty does not itself justify support expansion',
                                  'Feature capacity applies only at common integer coordinates and fixed latent anchor; decoder parameter changes can alter scene mapping',
                                  'No common-coordinate rows are invented for new/disappeared teacher coordinates; empty intersections have null feature statistics',
                                  'Grid units are normalized scene coordinates, not meters or final-Gaussian displacement',
                                  'No quality/semantic/AR evaluation, no ranking of proposed model changes',
                                  'All points/feature entries retained; overlapping window exposures are not independent samples'])
        check_budget();write_new(output/'SUMMARY.json',summary)
        with (output/'SUMMARY.md').open('x',encoding='utf-8') as stream:stream.write(report(summary))
        print(json.dumps(dict(status='complete',pairs=1512,gpu_executed=False,output=str(output))),flush=True)
    except BaseException as error:
        if previous_alarm is not None:signal.setitimer(signal.ITIMER_REAL,0)
        write_new(output/'FAILED.json',dict(status='failed',stage=stage,error=repr(error),traceback=traceback.format_exc(),
                  completed_pairs=completed,elapsed_seconds=time.monotonic()-begun,gpu_executed=False,training_updates=0))
        raise
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)


def self_test():
    import numpy as np
    left=np.asarray([[0,0,0],[2,0,0],[1,0,0]],dtype=np.int64)
    right=np.asarray([[1,0,0],[3,0,0]],dtype=np.int64)
    li,ri=coordinate_intersection(left,right,np)
    assert left[li].tolist()==right[ri].tolist()==[[1,0,0]]
    a=np.asarray([[2.,5.,1.],[4.,5.,2.]],dtype=np.float32)
    b=np.asarray([[3.,5.,5.],[4.,6.,2.]],dtype=np.float32)
    std=np.asarray([1.,0.,1.],dtype=np.float32)
    info,ratio=feature_capacity(a,b,std,3.,np)
    assert info['zero_std_channels']==1 and info['zero_std_changed_entries']==1
    assert info['over_scale_entries']==1 and info['over_scale_rows']==1
    assert info['zero_std_changed_rows']==1 and info['bound_violating_rows']==2
    assert ratio.size==4 and info['zero_std_changed_entries_per_channel']==[1]
    empty,_=feature_capacity(a[:0],b[:0],std,3.,np)
    assert empty['abs_delta_over_std']['q99'] is None and empty['over_scale_row_fraction'] is None
    hist={0.:2,1.:1,4.:1}
    q=histogram_quantiles(hist)
    expected=np.quantile(np.asarray([0.,0.,1.,4.]),[.5,.9])
    assert math.isclose(q['q50'],expected[0]) and math.isclose(q['q90'],expected[1])
    # A zero-std channel may shift as a constant without gaining within-frame std.
    shifted=np.asarray([6.,6.],dtype=np.float32)
    assert float(shifted.std())==0 and bool((shifted!=5.).any())
    assert 'torch' not in sys.modules and 'scipy' not in sys.modules
    print('CPU_SELF_TEST_PASSED: exact coordinate intersection, zero-std locked channels, common-row bounds, '
          'empty intersections, histogram quantiles, constant shifts; numpy only, no files/GPU')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--threads',type=int,choices=(1,2),default=2)
    parser.add_argument('--max-seconds',type=float,default=180.)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.root is None:parser.error('--root is required')
    root=args.root.resolve();expected=(root/'diagnostics/support_capacity_v1').resolve()
    output=args.output.resolve() if args.output else expected
    if not root.is_dir() or output!=expected:parser.error('Output must be ROOT/diagnostics/support_capacity_v1')
    if output.exists():parser.error('Output exists; no overwrite/retry/resume')
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=180:parser.error('Budget must be positive and at most180 seconds')
    run(root,output,args.threads,args.max_seconds)


if __name__=='__main__':main()

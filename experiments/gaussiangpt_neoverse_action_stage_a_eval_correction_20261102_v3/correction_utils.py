"""Pure operations for correcting the repeated-initial evaluation control."""
from collections import defaultdict
import copy
import math


CONTROLS=('correct','initial_only','swapped_window_codes','actual_target','baseline_qformer')
SUPPORT_CONTROLS=('correct','initial_only','swapped_window_codes','baseline_qformer')


def by_pair(rows,expected):
    result={}
    for row in rows:
        if row['horizon']<=0:continue
        key=(row['start'],row['horizon'])
        if key in result:raise RuntimeError('Duplicate window/horizon')
        result[key]=row
    if set(result)!=expected:raise RuntimeError('Window/horizon coverage differs')
    return result


def corrected_rows(stage_rows,reference_rows,expected,reference_tolerance=5e-5):
    stage=by_pair(stage_rows,expected)
    reference=by_pair(reference_rows,expected)
    rows=copy.deepcopy(stage_rows)
    worst=0.
    for row in rows:
        if row['horizon']<=0:continue
        key=(row['start'],row['horizon']);ref=reference[key]
        if row['frame']!=ref['frame']:raise RuntimeError('Mismatched physical target frame')
        # The target state, native codec, target image and camera are the same.
        # Check their independently rendered metric before substituting control.
        for metric in ('mae','roi_mae','motion_mae','object_box_mae','teacher_view1_masked_mae'):
            a=float(row['metrics']['actual_target']['metrics'][metric])
            b=float(ref['metrics']['actual_target']['metrics'][metric])
            delta=abs(a-b)
            if not math.isfinite(delta) or delta>reference_tolerance:
                raise RuntimeError('Actual-target reference mismatch for '+str((key,metric,delta)))
            worst=max(worst,delta)
        row['metrics']['initial_only']=copy.deepcopy(ref['metrics']['initial_only'])
        row['support']['initial_only']=copy.deepcopy(ref['support']['initial_only'])
    return rows,worst


def aggregate(rows,controls):
    if not rows:raise RuntimeError('Empty metric aggregation')
    metrics={};window_metrics={}
    for control in controls:
        names=list(rows[0]['metrics'][control]['metrics'])
        groups=defaultdict(list)
        for row in rows:groups[row['frame']].append(row['metrics'][control]['metrics'])
        metrics[control]={key:sum(sum(r[key] for r in group)/len(group)
                            for group in groups.values())/len(groups) for key in names}
        window_metrics[control]={key:sum(row['metrics'][control]['metrics'][key] for row in rows)/len(rows)
                                 for key in names}
    return dict(unique_frame_equal=metrics,window_prediction_equal=window_metrics,
                unique_frames=len(groups),window_predictions=len(rows))


def corrected_packet_rows(stage,baseline,diagnostic_starts,stage_root,baseline_root):
    original={row['name']:row for row in stage}
    reference={row['name']:row for row in baseline}
    if len(original)!=104 or len(reference)!=104:raise RuntimeError('Native packet count differs')
    expected={f's{s:03d}_h{h:02d}_correct' for s in diagnostic_starts for h in range(10)}
    expected|={f's{s:03d}_h09_{name}' for s in diagnostic_starts
               for name in ('initial_only','swapped_window_codes','actual_target')}
    if set(original)!=expected or set(reference)!=expected:raise RuntimeError('Native packet coverage differs')
    rows=[]
    for name in sorted(expected):
        use_baseline=name.endswith('_initial_only')
        source=reference if use_baseline else original
        rows.append(dict(name=name,origin='bound_9560_baseline' if use_baseline else 'stage_a_original',
                         evaluation_root=str(baseline_root if use_baseline else stage_root),
                         packet=source[name]['packet'],reference=source[name]['reference'],
                         gs_count=source[name]['gs_count']))
    return rows

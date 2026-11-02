"""Summarize existing per-frame codec metrics without loading image/state tensors."""
import math
import statistics


def clip_summary(rows, clip, group, *, historical=False):
    if len(rows) != 177 or {r.get('frame') for r in rows} != set(range(177)):
        raise ValueError('Missing, repeated, or extra frame: ' + clip)
    if any(r.get('clip') != clip for r in rows):
        raise ValueError('Wrong clip in frame record')
    if historical and any(r.get('split') != 'TRAIN' for r in rows):
        raise ValueError('Historical non-training frame entered the analysis')
    categories = ('codec_real', 'teacher_real', 'codec_teacher')
    result = dict(clip=clip, source_group=group, frames=177)
    for category in categories:
        metrics = [r[category] for r in rows]
        mae = [float(x['mae']) for x in metrics]
        psnr = [float(x['psnr']) for x in metrics]
        if not all(math.isfinite(x) and x >= 0 for x in mae + psnr):
            raise ValueError('Nonfinite or negative image metric')
        result[category] = dict(mae_mean=statistics.fmean(mae),
                                psnr_mean_db=statistics.fmean(psnr))
        if category == 'codec_real':
            result[category].update(mae_min=min(mae), mae_max=max(mae),
                                    frames_mae_above_005=sum(x > .05 for x in mae))
    return result


def summarize(items):
    if not items or len({r['clip'] for r in items}) != len(items):
        raise ValueError('Duplicate or empty clip summaries')
    groups = sorted({r['source_group'] for r in items})
    by_group = {}
    for group in groups:
        values = [r['codec_real']['mae_mean'] for r in items if r['source_group'] == group]
        by_group[group] = dict(clips=len(values), codec_real_mae_mean=statistics.fmean(values))
    values = [r['codec_real']['mae_mean'] for r in items]
    return dict(clips=len(items), source_groups=len(groups), frames=sum(r['frames'] for r in items),
                clip_equal_codec_real_mae_mean=statistics.fmean(values),
                clip_equal_codec_real_mae_median=statistics.median(values),
                group_equal_codec_real_mae_mean=statistics.fmean(x['codec_real_mae_mean'] for x in by_group.values()),
                lowest_clip_mean=min(values), highest_clip_mean=max(values),
                clips_mean_mae_above_005=sum(x > .05 for x in values),
                codec_real_frames_above_005=sum(r['codec_real']['frames_mae_above_005'] for r in items),
                groups=by_group)

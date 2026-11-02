"""Check frame coverage, training-role guard and aggregation weights."""
import copy
import analysis_core as core


def rows(clip, value):
    return [dict(clip=clip, frame=i, split='TRAIN',
                 codec_real={'mae':value,'psnr':20.},
                 teacher_real={'mae':.01,'psnr':30.},
                 codec_teacher={'mae':value-.001,'psnr':20.}) for i in range(177)]


def rejected(call):
    try:call()
    except (ValueError, KeyError):return
    raise AssertionError('Invalid metric records were accepted')


a=rows('a',.02);b=rows('b',.08);c=rows('c',.08)
ar=core.clip_summary(a,'a','one',historical=True)
br=core.clip_summary(b,'b','two',historical=True)
cr=core.clip_summary(c,'c','two',historical=True)
summary=core.summarize([ar,br,cr])
assert summary['frames']==531 and summary['source_groups']==2
assert abs(summary['clip_equal_codec_real_mae_mean']-.06)<1e-12
assert abs(summary['group_equal_codec_real_mae_mean']-.05)<1e-12
assert summary['codec_real_frames_above_005']==354
rejected(lambda:core.clip_summary(a[:-1]+[a[0]],'a','one'))
bad=copy.deepcopy(a);bad[0]['split']='EVAL'
rejected(lambda:core.clip_summary(bad,'a','one',historical=True))
bad=copy.deepcopy(a);bad[0]['codec_real']['mae']=float('nan')
rejected(lambda:core.clip_summary(bad,'a','one'))
rejected(lambda:core.summarize([ar,ar]))
print('PASS: exact frame coverage, training-only role and clip/group weighted summaries')

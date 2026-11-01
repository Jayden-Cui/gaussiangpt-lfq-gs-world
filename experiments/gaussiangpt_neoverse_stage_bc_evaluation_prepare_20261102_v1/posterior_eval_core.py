"""Continuous posterior controls, causal checks and fixed-initial diagnostics."""
import math
import torch


@torch.no_grad()
def extract_controls(model,store,infer_codes,starts):
    if starts!=list(range(168)):raise ValueError('Full adjacent training video required')
    model.eval();correct=[];static=[]
    for start in starts:
        window=store.get(start)
        a=infer_codes(model,window,static=False)
        b=infer_codes(model,window,static=True)
        if (a.shape!=(9,16) or b.shape!=(9,16) or a.dtype!=torch.float32 or b.dtype!=torch.float32
                or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
            raise RuntimeError('Invalid posterior control codes')
        correct.append(a.detach().cpu());static.append(b.detach().cpu())
    return torch.stack(correct),torch.stack(static)


@torch.no_grad()
def check_prefix_causality(model,store,starts=(35,123),horizons=(1,3,7)):
    model.eval();rows=[]
    for start in starts:
        window=store.get(start)
        for h in horizons:
            if model.variant=='C0':
                memory=window['memories'];changed=list(memory)
                for i in range(h,len(changed)):
                    item=changed[i]
                    changed[i]=type(item)(keys=item.keys+3.,coordinates=item.coordinates+100)
                before=model.forward_cached(memory)[:h]
                after=model.forward_cached(changed)[:h]
                method='perturb later cached prefixes; cache-construction causality is separately required'
            else:
                states=window['states'];changed=[]
                for i,state in enumerate(states):
                    if i<=h:changed.append(state);continue
                    coord=state['coordinates'];extra=coord.max(dim=0).values[None]+100
                    changed.append(dict(state,coordinates=torch.cat((coord,extra),0),
                        features=torch.cat((state['features']+3.,torch.zeros_like(state['features'][:1])),0)))
                before=model.code_for_h(states,h);after=model.code_for_h(changed,h)
                method='perturb only states and world coordinates after the requested prefix'
            error=float((before-after).abs().max())
            if not math.isfinite(error) or error>2e-5:
                raise RuntimeError('Prefix causality failed: '+str((start,h,error)))
            rows.append(dict(start=start,horizon=h,max_abs_error=error,method=method))
    return dict(status='passed',rows=rows,tolerance=2e-5,
                scope='Prefix perturbation on the trained model; not an online prediction or generalization result')


@torch.no_grad()
def static_motion_controls(model,decoder,bridge,store,training_core,p,starts,horizons=(1,3,6,9)):
    model.eval();decoder.eval();bridge.eval();rows=[]
    for start in starts:
        window=store.get(start)
        for h in horizons:
            sample=store.sample(start,h)
            result=training_core.same_initial_control(model,decoder,bridge,window,sample,h,p)
            result['start']=start;rows.append(result)
    return dict(status='complete',rows=rows,source_starts=list(starts),horizons=list(horizons),
        scope='Same S0, timestamps and camera; moving and static targets use the same frozen GS codec',
        action_success='requires dynamic and visual interpretation')


def teacher_comparison(evaluation):
    if evaluation['status']!='complete' or evaluation['window_predictions']!=1512:
        raise RuntimeError('Incomplete posterior evaluation')
    future=[x for x in evaluation['rows'] if x['horizon']>0]
    h9=[x for x in future if x['horizon']==9]
    pairs={(x['start'],x['horizon']) for x in future}
    if (len(future)!=1512 or len(h9)!=168 or pairs!={(s,h) for s in range(168) for h in range(1,10)}):
        raise RuntimeError('Incomplete future-step coverage')
    mean=lambda rows,name:sum(x['metrics'][name]['metrics']['motion_mae'] for x in rows)/len(rows)
    values=evaluation['metrics']['unique_frame_equal']
    overall_student=values['correct']['motion_mae'];overall_teacher=values['teacher_free']['motion_mae']
    h9_student=mean(h9,'correct');h9_teacher=mean(h9,'teacher_free')
    if min(overall_teacher,h9_teacher)<=0:raise RuntimeError('Invalid teacher metric')
    return dict(overall_motion_mae=overall_student,teacher_overall_motion_mae=overall_teacher,
        overall_ratio=overall_student/overall_teacher,h9_motion_mae=h9_student,
        teacher_h9_motion_mae=h9_teacher,h9_ratio=h9_student/h9_teacher,
        numerical_110pct_gate=overall_student<=1.1*overall_teacher and h9_student<=1.1*h9_teacher,
        visual_dynamic_review_required=True,scope='Matched decoder and single training video; ratios do not imply transfer')

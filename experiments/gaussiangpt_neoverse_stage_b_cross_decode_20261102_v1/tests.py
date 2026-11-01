"""CPU coverage for 2x2 provenance, matched preview metrics and row integrity."""
import copy
import cross_decode as c


def raises(fn):
    try:fn()
    except (KeyError,ValueError):return
    raise AssertionError('Bad matrix or mismatched prior preview was accepted')


protocol={'schema':'stage_b_cross_decode_diagnostic_v1','conditions':[n for n,_,_ in c.PLAN],
 'windows':[35,123],'future_horizons':list(range(1,10)),'updates':0,'cuda_visible_device':1}
c.verify_plan(protocol)
bad=copy.deepcopy(protocol);bad['conditions'][0]=bad['conditions'][1]
raises(lambda:c.verify_plan(bad))
rows=[]
for i,(name,d,z) in enumerate(c.PLAN):
 for s in (35,123):
  for h in range(1,10):
   v=float(i+h)/100
   rows.append({'condition':name,'decoder':d,'codes':z,'start':s,'horizon':h,'frame':s+h,
                'metrics':{'metrics':{'mae':v,'motion_mae':v}}})
result=c.aggregate(rows)
assert abs(result[c.PLAN[0][0]]['motion_mae']-.05)<1e-12
raises(lambda:c.aggregate(rows[:-1]+[rows[0]]))
bad=copy.deepcopy(rows);bad[0]['codes']='sweep05'
raises(lambda:c.aggregate(bad))
bad=copy.deepcopy(rows);bad[0]['metrics']['metrics']['mae']=float('nan')
raises(lambda:c.aggregate(bad))
reference={'rows':[{'start':r['start'],'horizon':r['horizon'],
                    'metrics':{'correct':r['metrics']}} for r in rows if r['condition']==c.PLAN[0][0]]}
assert c.verify_preview(rows,c.PLAN[0][0],reference)['max_abs_metric_difference']==0
bad=copy.deepcopy(reference);bad['rows'][0]['metrics']['correct']['metrics']['mae']+=1
raises(lambda:c.verify_preview(rows,c.PLAN[0][0],bad))
print('PASS: four distinct conditions, 72 pairs, finite metrics and exact prior-preview controls')

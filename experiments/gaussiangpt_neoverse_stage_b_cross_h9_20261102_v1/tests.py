"""CPU guards for all-window h9 coverage and group accounting."""
import copy
import cross_h9 as c


def raises(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Bad comparison accepted')


groups={n:list(range(i*42,(i+1)*42)) for i,n in enumerate(c.GROUP_NAMES)}
assert c.validate_groups({'groups':groups})==groups
bad=copy.deepcopy(groups);bad['highest'][-1]=bad['lowest'][0]
raises(lambda:c.validate_groups({'groups':bad}))
rows=[]
for i,(name,d,z) in enumerate(c.PLAN):
 for s in range(168):
  rows.append({'condition':name,'decoder':d,'codes':z,'start':s,'horizon':9,'frame':s+9,
               'metrics':{'metrics':{'mae':float(i+s)/1000,'motion_mae':float(i+s)/1000}}})
summary=c.summarize(rows,groups)
assert abs(summary[c.PLAN[0][0]]['overall']['mae']-.0835)<1e-12
assert abs(summary[c.PLAN[0][0]]['by_source_change_quartile']['highest']['mae']-.1465)<1e-12
raises(lambda:c.summarize(rows[:-1]+[rows[0]],groups))
bad=copy.deepcopy(rows);bad[0]['horizon']=8
raises(lambda:c.summarize(bad,groups))
earlier={'rows':[dict(r) for r in rows if r['start'] in (35,123)]}
assert c.validate_anchors(rows,earlier)['max_abs_metric_difference']==0
bad=copy.deepcopy(earlier);bad['rows'][0]['metrics']['metrics']['mae']+=1
raises(lambda:c.validate_anchors(rows,bad))
print('PASS: 672 unique h9 rows, factors, source groups and prior-anchor metrics')

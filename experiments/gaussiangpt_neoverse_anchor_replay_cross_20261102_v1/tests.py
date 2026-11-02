"""CPU checks of the actual four-way diagnostic and preview guard."""
import copy
import importlib.util
from pathlib import Path

import cross


def rejected(fn):
    try:
        fn()
    except (KeyError, ValueError, RuntimeError):
        return
    raise AssertionError('Invalid protocol, rows or preview accepted')


p=cross.read(cross.ROOT/'protocol.json')
cross.validate_plan(p)
bad=copy.deepcopy(p);bad['gpu']=0
rejected(lambda:cross.validate_plan(bad))
bad=copy.deepcopy(p);bad['conditions'][0]=bad['conditions'][1]
rejected(lambda:cross.validate_plan(bad))
old_path=cross.OLD/'cross_decode.py'
spec=importlib.util.spec_from_file_location('_previous_cross_test',old_path)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.PLAN=cross.PLAN
rows=[]
for i,(name,decoder,code) in enumerate(cross.PLAN):
    for start in (35,123):
        for h in range(1,10):
            val=(i+h)/100.0
            rows.append(dict(condition=name,decoder=decoder,codes=code,start=start,horizon=h,
                             frame=start+h,metrics={'metrics':{'mae':val,'motion_mae':val}}))
result=old.aggregate(rows)
assert abs(result[cross.PLAN[0][0]]['motion_mae']-.05)<1e-12
rejected(lambda:old.aggregate(rows[:-1]+[rows[0]]))
bad=copy.deepcopy(rows);bad[0]['codes']='replay'
rejected(lambda:old.aggregate(bad))
bad=copy.deepcopy(rows);bad[0]['frame']+=1
rejected(lambda:old.aggregate(bad))
bad=copy.deepcopy(rows);bad[0]['metrics']['metrics']['mae']=float('nan')
rejected(lambda:old.aggregate(bad))
preview={'rows':[dict(start=r['start'],horizon=r['horizon'],metrics={'correct':r['metrics']})
                 for r in rows if r['condition']==cross.PLAN[0][0]]}
assert old.verify_preview(rows,cross.PLAN[0][0],preview)['pairs']==18
bad=copy.deepcopy(preview);bad['rows'][0]['metrics']['correct']['metrics']['mae']+=1
rejected(lambda:old.verify_preview(rows,cross.PLAN[0][0],bad))
print('PASS: fixed protocol, complete 2x2 factors, 72 unique pairs and matched-preview checks')

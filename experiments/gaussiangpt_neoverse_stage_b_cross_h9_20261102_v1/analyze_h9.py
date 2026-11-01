"""Count paired h9 wins; descriptions are not independent-sample statistics."""
from pathlib import Path
import json
import statistics
import cross_h9


ROOT=Path(__file__).resolve().parent


def main():
    result_path=ROOT/'result/CROSS_H9.json'
    result=cross_h9.read(result_path)
    if result['status']!='complete' or result['window_predictions']!=672:
        raise ValueError('Full h9 cross-decoding must be complete')
    groups_path=ROOT.parent/'gaussiangpt_neoverse_motion_groups_20261102_v1/MOTION_GROUPS.json'
    if cross_h9.sha(groups_path)!=result['source_groups_sha256']:
        raise ValueError('Frozen grouping changed')
    groups=cross_h9.validate_groups(cross_h9.read(groups_path))
    summary=cross_h9.summarize(result['rows'],groups)
    if summary!=result['summary']:
        raise ValueError('Saved summary does not match rows')
    index={(r['condition'],r['start']):r for r in result['rows']}
    comparison={}
    for code in ('warm','sweep05'):
        a=f'warm_decoder_{code}_codes';b=f'sweep05_decoder_{code}_codes'
        delta={s:index[(a,s)]['metrics']['metrics']['motion_mae']-
                 index[(b,s)]['metrics']['metrics']['motion_mae'] for s in range(168)}
        comparison[code]={'sweep05_decoder_lower_error':sum(x>0 for x in delta.values()),
          'warm_decoder_lower_error':sum(x<0 for x in delta.values()),'equal_error':sum(x==0 for x in delta.values()),
          'starts_where_warm_decoder_is_lower':[s for s,x in delta.items() if x<0],
          'median_warm_minus_sweep05':statistics.median(delta.values()),
          'scope':'Paired h9 results from highly overlapping windows in one training video; not independent statistical trials'}
    data={'status':'complete','result_sha256':cross_h9.sha(result_path),'source_groups_sha256':cross_h9.sha(groups_path),
          'paired_comparisons':comparison}
    output=ROOT/'result/PAIRED_ANALYSIS.json'
    with output.open('x',encoding='utf8',newline='\n') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print('PASS: paired counts from 672 bound rows')


if __name__=='__main__':main()

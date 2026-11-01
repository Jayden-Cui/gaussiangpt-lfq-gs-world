"""Summarize completed target-support diagnosis without new model execution."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
DIAGNOSTIC=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1/diagnostics/DIAGNOSTIC.json')
EXPECTED_SHA='2439c556fb09e58fb8de5ac8c2b7fbfa5401f0c65bd26de4b424043d47ff30fd'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(DIAGNOSTIC)!=EXPECTED_SHA:raise RuntimeError('Bound diagnostic bytes changed')
    d=json.loads(DIAGNOSTIC.read_text());rows=d['rows']
    if d['status']!='complete' or len(rows)!=1512 or d['oracle_used_for_training']:
        raise RuntimeError('Full offline diagnostic scope differs')
    anchors={}
    for start in (35,123):
        row=next(r for r in rows if r['start']==start and r['horizon']==9)
        x=row['reachability'];values={}
        for name,result in row['metrics'].items():
            values[name]=dict(motion_mae=result['metrics']['motion_mae'],
                              color_proxy_iou=result['object_proxy']['iou'],
                              predicted_color_pixels=result['object_proxy']['prediction']['pixels'])
        anchors[str(start)]=dict(frame=row['frame'],target_feature_elements=x['target_feature_elements'],
          out_of_range_elements=x['out_of_range_elements'],out_of_range_fraction=x['out_of_range_elements']/x['target_feature_elements'],
          target_cells=x['target_cells'],cells_with_any_out_of_range=x['cells_with_any_out_of_range'],
          cells_with_any_out_of_range_fraction=x['cells_with_any_out_of_range']/x['target_cells'],
          results=values)
    diffs=[r['metrics']['oracle_target_support']['metrics']['motion_mae']-
           r['metrics']['predicted_support']['metrics']['motion_mae'] for r in rows]
    summary=d['metrics']['unique_frame_equal'];base=summary['predicted_support']['motion_mae']
    oracle=summary['oracle_target_support']['motion_mae'];reference=summary['actual_target']['motion_mae']
    output=dict(status='complete',source_sha256=EXPECTED_SHA,script_sha256=sha(ROOT/'anchor_analysis.py'),
      anchors=anchors,prediction_records=len(diffs),oracle_lower_motion_error_count=sum(x<0 for x in diffs),
      oracle_higher_motion_error_count=sum(x>0 for x in diffs),oracle_equal_motion_error_count=sum(x==0 for x in diffs),
      median_prediction_delta=statistics.median(diffs),
      unique_frame_equal=dict(predicted_support=base,oracle_support=oracle,actual_codec=reference,
          oracle_relative_improvement=1-oracle/base),
      interpretation='Overlapping window predictions are not independent samples. Target support is an offline oracle; feature out-of-range does not prove equivalent rendering is impossible.')
    with (ROOT/'ANCHOR_ANALYSIS.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: localized residual-range and target-support analysis from bound diagnostics')


if __name__=='__main__':main()

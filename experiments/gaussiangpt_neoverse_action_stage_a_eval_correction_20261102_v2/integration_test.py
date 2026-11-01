"""End-to-end correction on a disposable, explicitly synthetic result fixture."""
import copy
import json
import tempfile
from pathlib import Path
import torch

import correction
from correction_utils import by_pair


def main():
    c,training,baseline,old,old_packets,replay=correction.setup()
    real_root=correction.ROOT
    baseline_codes=torch.load(baseline/'action_codes.pt',map_location='cpu',weights_only=True)
    with tempfile.TemporaryDirectory(prefix='codex_control_correction_') as directory:
        temp=Path(directory);fixture=temp/'training';output=temp/'correction';output.mkdir()
        (output/'CODE.sha256').symlink_to(real_root/'CODE.sha256')
        sweep_root=fixture/'sweep_05';stage=sweep_root/'free_z/evaluation';stage.mkdir(parents=True)
        merged=sweep_root/'merged_codes.pt'
        torch.save(dict(codes=baseline_codes['controls']['correct']),merged)
        correction.write(sweep_root/'MERGED.json',dict(sha256=correction.sha(merged)))
        source=dict(checkpoint_sha256=c['checkpoint_sha256'],decoder_digest=c['decoder_tensor_digest'],
                    cache_manifest_sha256=c['cache_manifest_sha256'],stage_a_code_sha256=c['training_code_sha256'],
                    merged_codes_sha256=correction.sha(merged),sweep=5)
        fake_codes=copy.deepcopy(baseline_codes);fake_codes['source']=source
        torch.save(fake_codes,stage/'action_codes.pt')
        fake_packets=copy.deepcopy(old_packets);fake_packets['mode']='free_z';fake_packets['source']=source
        correction.write(stage/'PACKETS.json',fake_packets)
        correction.write(stage/'REPLAY.json',dict(status='passed',packets=104,views=208,
                         packets_sha256=correction.sha(stage/'PACKETS.json')))
        fake=copy.deepcopy(old);fake['mode']='free_z';fake['source']=source
        fake['packets_sha256']=correction.sha(stage/'PACKETS.json')
        fake['action_codes']['sha256']=correction.sha(stage/'action_codes.pt')
        for row in fake['rows']:
            if row['horizon']<=0:continue
            row['metrics']['baseline_qformer']=copy.deepcopy(row['metrics']['correct'])
            row['support']['baseline_qformer']=copy.deepcopy(row['support']['correct'])
            row['metrics']['initial_only']=copy.deepcopy(row['metrics']['correct'])
            row['support']['initial_only']=copy.deepcopy(row['support']['correct'])
        correction.write(stage/'EVALUATION.json',fake)
        correction.write(sweep_root/'free_z/STAGE_METRICS.json',dict(status='complete'))
        (stage/'snapshots').symlink_to(baseline/'snapshots',target_is_directory=True)
        for start in c['diagnostic_starts']:
            for view in (0,1):
                name=f's{start:03d}_comparison_v{view}.png'
                (stage/name).symlink_to(baseline/name)
        try:
            correction.ROOT=output
            result=correction.correct_one(5,c,fixture,baseline,old,old_packets,baseline_codes)
            actual=correction.read(output/'sweep_05/EVALUATION.json')
            expected={(s,h) for s in range(168) for h in range(1,10)}
            rows=by_pair(actual['rows'],expected);references=by_pair(old['rows'],expected)
            if any(rows[key]['metrics']['initial_only']!=references[key]['metrics']['initial_only'] for key in expected):
                raise RuntimeError('End-to-end control correction differs')
            if actual['correction']['max_actual_target_mae_difference']!=0 or actual['packets_sha256'] is not None:
                raise RuntimeError('Correction provenance or invalid old packet marker differs')
            manifest=correction.read(output/'sweep_05/CORRECTED_PACKETS.json')
            if len(manifest['rows'])!=104 or sum(x['origin']=='bound_9560_baseline' for x in manifest['rows'])!=8:
                raise RuntimeError('Native packet references differ')
            if not (output/'sweep_05/snapshots/s123/frame_09.png').is_file():
                raise RuntimeError('Corrected visual diagnostics missing')
            if result['metrics']['window_predictions']!=1512:raise RuntimeError('Corrected scope differs')
        finally:
            correction.ROOT=real_root
    correction.write(real_root/'INTEGRATION_TEST.json',dict(status='passed',
        code_sha256=correction.sha(real_root/'CODE.sha256'),scope='disposable synthetic stage result with real bound baseline',
        complete_metric_pipeline=True,native_packet_references=True,all_corrected_panels_written=True,
        real_training_or_results_modified=False))
    print('PASS: end-to-end correction, source binding, all metrics, native references and diagnostic panels')


if __name__=='__main__':main()

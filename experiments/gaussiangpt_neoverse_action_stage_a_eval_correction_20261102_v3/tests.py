"""CPU checks for full-coverage, control identity, and native-panel provenance."""
import copy
import os
import select
import subprocess
import sys
import torch
import numpy as np
from PIL import Image

from correction import ROOT,setup,sha,write
from correction_utils import CONTROLS,aggregate,by_pair,corrected_packet_rows,corrected_rows
from process_handle import pidfd_open


def main():
    c,training,baseline,old,packets,replay=setup()
    expected={(s,h) for s in range(168) for h in range(1,10)}
    reference=by_pair(old['rows'],expected)
    fake=copy.deepcopy(old['rows'])
    for row in fake:
        if row['horizon']<=0:continue
        row['metrics']['baseline_qformer']=copy.deepcopy(row['metrics']['correct'])
        row['support']['baseline_qformer']=copy.deepcopy(row['support']['correct'])
        # Reproduce the real defect: initial-only metrics were aliased to correct.
        row['metrics']['initial_only']=copy.deepcopy(row['metrics']['correct'])
        row['support']['initial_only']=copy.deepcopy(row['support']['correct'])
    corrected,worst=corrected_rows(fake,old['rows'],expected)
    fixed=by_pair(corrected,expected)
    if worst!=0 or any(fixed[key]['metrics']['initial_only']!=reference[key]['metrics']['initial_only']
                       or fixed[key]['support']['initial_only']!=reference[key]['support']['initial_only']
                       or fixed[key]['metrics']['correct']!=reference[key]['metrics']['correct'] for key in expected):
        raise RuntimeError('Full control substitution changed the wrong prediction')
    different=sum(reference[key]['metrics']['correct']!=reference[key]['metrics']['initial_only'] for key in expected)
    if not different:raise RuntimeError('Baseline does not distinguish real controls')
    summary=aggregate([r for r in corrected if r['horizon']>0],CONTROLS)
    if summary['unique_frames']!=176 or summary['window_predictions']!=1512:
        raise RuntimeError('Aggregation scope changed')
    bad=copy.deepcopy(fake);first=next(r for r in bad if r['horizon']>0)
    first['metrics']['actual_target']['metrics']['mae']+=.1
    try:corrected_rows(bad,old['rows'],expected)
    except RuntimeError:pass
    else:raise RuntimeError('Mismatched target provenance was not rejected')
    missing_index=next(i for i,row in enumerate(fake) if row['horizon']>0)
    try:by_pair(fake[:missing_index]+fake[missing_index+1:],expected)
    except RuntimeError:pass
    else:raise RuntimeError('Incomplete evaluation coverage was not rejected')
    references=corrected_packet_rows(packets['rows'],packets['rows'],c['diagnostic_starts'],baseline,baseline)
    if sum(r['origin']=='bound_9560_baseline' for r in references)!=8:
        raise RuntimeError('Wrong number of replaced native endpoint packets')
    record=next(r for r in packets['rows'] if r['name']=='s035_h09_initial_only')
    image=torch.load(baseline/'references'/record['reference']['file'],map_location='cpu',weights_only=True)
    with Image.open(baseline/'snapshots/s035/frame_09.png') as panel:
        pixels=np.asarray(panel.crop((896,24,1344,276)))
    expected_pixels=(image[0].numpy().clip(0,1)*255).round().astype(np.uint8)
    if not np.array_equal(pixels,expected_pixels):raise RuntimeError('Baseline panel does not match independently replayed reference')
    # The running experiment is never signalled: verify pidfd exit-event wait on a disposable child.
    child=subprocess.Popen([sys.executable,'-c','pass']);fd=pidfd_open(child.pid)
    try:
        poll=select.poll();poll.register(fd,select.POLLIN)
        if not poll.poll(5000) or child.wait(timeout=2)!=0:raise RuntimeError('pidfd exit event failed')
    finally:
        os.close(fd)
        if child.poll() is None:child.kill();child.wait()
    write(ROOT/'CPU_TESTS.json',dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),
       repaired_pairs=len(expected),baseline_control_distinct_pairs=different,
       preserves_correct_predictions=True,missing_pairs_and_target_mismatch_rejected=True,
       native_baseline_panel_matches_saved_replayed_reference=True,
       packet_references_replace_only_eight_initial_endpoints=True,pidfd_exit_event_passed=True,
       no_signal_sent_to_running_experiment=True))
    print('PASS: all 1512 control pairs, target binding, native panel and event-driven process wait')


if __name__=='__main__':main()

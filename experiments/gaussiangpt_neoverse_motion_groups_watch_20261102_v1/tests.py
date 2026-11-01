"""CPU tests for event parsing and completed evaluation/replay provenance."""
import json
from pathlib import Path
import struct
import tempfile
import watcher as w


def raises(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid input accepted')


name = b'SUMMARY.json\0'
assert w.event_names(struct.pack('iIII', 1, 8, 0, len(name))+name) == ['SUMMARY.json']
raises(lambda: w.event_names(b'\0'*10))
raises(lambda: w.event_names(struct.pack('iIII', 1, 8, 0, 100)+name))

with tempfile.TemporaryDirectory() as td:
    run = Path(td)
    (run/'training').mkdir()
    (run/'REVIEW_ASSETS').mkdir()
    (run/'training/TRAIN_DONE.json').write_text('{}')
    (run/'REVIEW_ASSETS/CONTACTS.json').write_text('{}')
    rows = []
    for sweep in (5,10,20):
        d = run/'training'/f'sweep_{sweep:02d}'/'free_z'/'evaluation'
        d.mkdir(parents=True)
        (d/'PACKETS.json').write_text('{}')
        psha = w.sha(d/'PACKETS.json')
        (d/'EVALUATION.json').write_text(json.dumps({'status':'complete','window_predictions':1512,
            'source':{'sweep':sweep}, 'packets_sha256':psha}))
        (d/'REPLAY.json').write_text(json.dumps({'status':'passed','packets':104,'views':208,
            'encoder_or_codec_loaded':False,'packets_sha256':psha}))
        rows.append({'sweep':sweep,'evaluation_path':str(d/'EVALUATION.json'),
            'evaluation_sha256':w.sha(d/'EVALUATION.json'),'replay':{'sha256':w.sha(d/'REPLAY.json')}})
    summary = {'status':'complete','stage':'B','action_codec_complete':False,'manual_dynamic_review_required':True,
        'training_sha256':w.sha(run/'training/TRAIN_DONE.json'),
        'visual_review_assets_sha256':w.sha(run/'REVIEW_ASSETS/CONTACTS.json'),'results':rows}
    assert set(w.validate_summary(summary,run)) == {5,10,20}
    rows[0]['replay']['sha256'] = 'incorrect'
    raises(lambda: w.validate_summary(summary,run))
    rows[0]['replay']['sha256'] = w.sha(run/'training/sweep_05/free_z/evaluation/REPLAY.json')
    summary['action_codec_complete'] = True
    raises(lambda: w.validate_summary(summary,run))
print('PASS: inotify event parsing, bound evaluations and independent replay, manual review guard')

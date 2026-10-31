"""Gate formal runs on the exact integration smoke and its independent replays."""
import json
from pathlib import Path
from codec_binding import require,sha
from formal_common import verify_sources,write_new


def main():
    root=Path(__file__).resolve().parent;p,source=verify_sources(root)
    smoke=root/'integration_smoke'
    producer=json.loads((smoke/'PRODUCER_DONE.json').read_text())
    require(producer['status']=='complete' and producer['formal_updates']==0
            and producer['discarded_updates_per_arm']==1,'Smoke producer incomplete')
    result={}
    for mode in p['arms']:
        out=smoke/mode
        ev=json.loads((out/'EVALUATION.json').read_text());r=json.loads((out/'NATIVE_REPLAY.json').read_text())
        require(ev['status']=='complete' and ev['mode']==mode and ev['diagnostic_smoke'] is True
                and ev['formal_updates']==0 and ev['latent_replay']['frames']==2
                and ev['latent_replay']['failed_frames']==0 and r['status']=='complete'
                and r['diagnostic_smoke'] is True and r['packets']==2 and r['views']==4
                and r['encoder_or_codec_loaded'] is False and r['replay_passed_all'] is True
                and r['producer_pid']!=r['replay_pid'] and r['evaluation_sha256']==sha(out/'EVALUATION.json'),
                'Integration smoke or independent native replay failed: '+mode)
        result[mode]=dict(evaluation_sha256=sha(out/'EVALUATION.json'),native_replay_sha256=sha(out/'NATIVE_REPLAY.json'),
                          checkpoint_sha256=ev['checkpoint_sha256'])
    write_new(root/'SMOKE_DONE.json',dict(status='passed',source=source,arms=result,formal_updates=0,
                                          diagnostic_scope='one discarded update and two windows per arm'))


if __name__=='__main__':main()

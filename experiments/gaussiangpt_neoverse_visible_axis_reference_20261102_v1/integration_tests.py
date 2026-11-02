"""Exercise completed stage-B input binding with a synthetic evaluation bundle.

The snapshots use existing real two-window images; the surrounding 1512-row
evaluation record is synthetic. This is not a stage-B training result.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import axis_audit as audit


def must_fail(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError('Invalid stage-B bundle unexpectedly accepted')


def main():
    reference = audit.read(audit.REFERENCE)
    with tempfile.TemporaryDirectory(prefix='gs_axis_synthetic_b_') as directory:
        root = Path(directory)
        target = {(row['start'], row['horizon']): row for row in reference['rows']}
        records = []
        for start in range(168):
            for horizon in range(1, 10):
                metrics = {arm: {'metrics': {'mae': 0.}} for arm in ('correct', 'actual_target')}
                row = target.get((start, horizon))
                if row is not None:
                    path = audit.TWO_Q / ('s%03d' % start) / ('frame_%02d.png' % horizon)
                    images = audit.three_panels(path)
                    audit.assert_source(images[0], row)
                    canvas = Image.new('RGB', (1792, 276), 'white')
                    for i, image in enumerate((images[0], images[1], images[2], images[0])):
                        canvas.paste(Image.fromarray(image), (i * 448, 24))
                    out = root / 'snapshots' / ('s%03d' % start) / ('frame_%02d.png' % horizon)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    canvas.save(out)
                    metrics['correct']['metrics']['mae'] = float(np.abs(
                        images[1].astype(np.float64) - images[0].astype(np.float64)).mean() / 255)
                records.append(dict(start=start, horizon=horizon, metrics=metrics))
        source = dict(schema='stage_b_shared_decoder_evaluation_v1', quantized=False,
            codec_sha256='a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917',
            cache_manifest_sha256='cb5c2b30f1b4fdac4db642046a46773cc0b96a4dabdab4917e0d3c79ff6eaa4a',
            test_fixture=True)
        path = root / 'EVALUATION.json'
        path.write_text(json.dumps(dict(status='complete', mode='free_z', window_predictions=1512,
                                        source=source, rows=records)), encoding='utf-8')
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        data, indexed = audit.stage_b_data(root, expected)
        must_fail(lambda: audit.stage_b_data(root, '0' * 64))
        for row in reference['rows']:
            panels, _ = audit.stage_b_images(root, indexed, row)
            assert audit.rgb_sha(panels['source']) == row['source_target_rgb_sha256']
        result_dir = root / 'result'
        audit.main(result_dir, root, expected)
        result = audit.read(result_dir / 'AXIS_DIAGNOSTIC.json')
        previous = audit.read(audit.ROOT / 'result_two_window/AXIS_DIAGNOSTIC.json')
        for start in ('35', '123'):
            assert (result['summaries'][start]['stage_b_free_motion']
                    == previous['summaries'][start]['two_window_q_motion'])
        snapshot = root / 'snapshots/s035/frame_01.png'
        image = Image.open(snapshot).convert('RGB')
        image.putpixel((0, 24), tuple(255 - x for x in image.getpixel((0, 24))))
        image.save(snapshot)
        must_fail(lambda: audit.stage_b_images(root, indexed, target[(35, 1)]))
        records.pop()
        path.write_text(json.dumps(dict(status='complete', mode='free_z', window_predictions=1512,
                                        source=source, rows=records)), encoding='utf-8')
        must_fail(lambda: audit.stage_b_data(root, hashlib.sha256(path.read_bytes()).hexdigest()))
    print('PASS: simulated complete stage-B bundle, row coverage, source integrity and correct panel mapping')


if __name__ == '__main__':
    main()

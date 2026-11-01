"""CPU tests for training-only codec audit coverage and quantile semantics."""
from pathlib import Path
import math
import analyze


def expect_error(function):
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Invalid codec audit was accepted')


def main():
    if not math.isclose(analyze.quantile([0, 10, 20], .25), 5):
        raise RuntimeError('Linear quantile differs')
    expect_error(lambda: analyze.quantile([], .1))
    clip = analyze.TRAIN[0]
    rows = [{'clip': clip, 'frame': frame, 'split': 'TRAIN', 'checkpoint_sha256': analyze.CODEC_SHA,
             'codec_real': {'mae': frame / 1000, 'psnr': 20.0}, 'teacher_real': {'mae': .01}}
            for frame in range(177)]
    result = analyze.summarize(rows, clip)
    assert result['frames'] == 177 and math.isclose(result['codec_real_mae_mean'], .088, abs_tol=1e-12)
    assert sum(x['frames'] for x in result['segments'].values()) == 177
    expect_error(lambda: analyze.summarize(rows[:-1], clip))
    duplicate = list(rows)
    duplicate[-1] = duplicate[0]
    expect_error(lambda: analyze.summarize(duplicate, clip))
    wrong = [dict(row, split='EVAL') if row['frame'] == 3 else row for row in rows]
    expect_error(lambda: analyze.summarize(wrong, clip))
    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or analyze.sha(root / name) != digest:
            raise RuntimeError('Codec audit code changed')
    import json
    with (root / 'CPU_TESTS.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'passed', 'code_sha256': analyze.sha(root / 'CODE.sha256'),
                   'rejects_missing_duplicate_and_nontrain_frames': True,
                   'quantile_and_frame_means_checked': True,
                   'scope': 'Synthetic CPU aggregation tests; no new model training'}, stream, indent=2)
        stream.write('\n')
    print('PASS: training-only frame coverage, quantiles and frame means')


if __name__ == '__main__':
    main()

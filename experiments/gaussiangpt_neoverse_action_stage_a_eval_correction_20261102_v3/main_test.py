"""Exercise correction.main with all three checkpoints on disposable fixtures."""
import json
from pathlib import Path
import tempfile
import torch
import correction


def main():
    real_root = correction.ROOT
    original_setup, original_correct, original_load = correction.setup, correction.correct_one, torch.load
    with tempfile.TemporaryDirectory(prefix='stage_a_correction_main_') as folder:
        root = Path(folder)
        output = root / 'output'
        output.mkdir()
        (output / 'CODE.sha256').symlink_to(real_root / 'CODE.sha256')
        training = root / 'training'
        training.mkdir()
        correction.write(training / 'SUMMARY.json', {'status': 'complete', 'code_sha256': 'bound-code'})
        correction.write(training / 'STATUS.json', {'status': 'complete'})
        correction.write(output / 'CPU_TESTS.json', {'status': 'passed',
                         'code_sha256': correction.sha(output / 'CODE.sha256')})
        config = {'sweeps': [5, 10, 20], 'training_code_sha256': 'bound-code'}
        called = []
        def fake_setup():
            return config, training, root, {}, {}, {}
        def fake_correct(sweep, c, t, b, old, packets, codes):
            called.append(sweep)
            assert c is config and t == training and codes['shape'] == [168, 9, 16]
            controls = {key: {'motion_mae': value} for key, value in
                        [('correct', 0.1), ('baseline_qformer', 0.2), ('initial_only', 0.3)]}
            return {'metrics': {'unique_frame_equal': controls}, 'preliminary_known_anchor_proxy_pass': False}
        try:
            correction.ROOT = output
            correction.setup = fake_setup
            correction.correct_one = fake_correct
            torch.load = lambda *args, **kwargs: {'starts': list(range(168)), 'shape': [168, 9, 16]}
            correction.main()
            summary = correction.read(output / 'SUMMARY.json')
            if (called != [5, 10, 20] or summary['status'] != 'complete'
                    or summary['final_sweep'] != 20 or list(summary['stages']) != ['5', '10', '20']):
                raise RuntimeError('Correction main selected the wrong final checkpoint')
            if '| 20 |' not in (output / 'SUMMARY.md').read_text(encoding='utf-8'):
                raise RuntimeError('Final checkpoint is absent from summary')
        finally:
            correction.ROOT = real_root
            correction.setup, correction.correct_one, torch.load = original_setup, original_correct, original_load
    correction.write(real_root / 'MAIN_TEST.json', {'status': 'passed',
                     'code_sha256': correction.sha(real_root / 'CODE.sha256'),
                     'all_three_sweeps_summarized': True, 'final_sweep': 20,
                     'scope': 'Synthetic orchestration test; no real result changed'})
    print('PASS: main summarizes all three checkpoints and selects final sweep 20')


if __name__ == '__main__':
    main()

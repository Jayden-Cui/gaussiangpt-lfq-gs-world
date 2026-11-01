"""CPU checks for the conditional B coordinator; never launch GPU workers."""
import copy
import hashlib
import json
import threading
from pathlib import Path
import tempfile

import coordinator


def expect_error(function):
    try:
        function()
    except (RuntimeError, OSError, KeyError, ValueError):
        return
    raise AssertionError('Invalid input unexpectedly passed')


def main():
    draft = {'batch_size': 4, 'sweeps': 20, 'stage_a_review': {'path': '/example', 'sha256': None},
             'status': 'draft_not_launchable_until_stage_a_review'}
    expect_error(lambda: coordinator.validate_config(draft, draft))
    ready = copy.deepcopy(draft)
    ready['stage_a_review']['sha256'] = 'a' * 64
    ready['status'] = 'ready_after_stage_a_review'
    assert coordinator.validate_config(ready, draft) == 'a' * 64
    for key, value in [('sweeps', 21), ('batch_size', 1)]:
        changed = copy.deepcopy(ready)
        changed[key] = value
        expect_error(lambda: coordinator.validate_config(changed, draft))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        evaluation = root / 'evaluation.json'
        evaluation.write_text(json.dumps({'status': 'complete', 'window_predictions': 1512,
            'correction': {'status': 'applied'}, 'source': {'sweep': 20}}), encoding='utf-8')
        review = root / 'review.json'
        review.write_text(json.dumps({'status': 'reviewed', 'decision': 'stage_B',
            'reviewed_starts': [0, 10, 20, 35, 60, 90, 123, 160],
            'evaluation': {'path': str(evaluation), 'sha256': coordinator.sha(evaluation)}}), encoding='utf-8')
        config = {'stage_a_review': {'path': str(review), 'sha256': coordinator.sha(review)}}
        assert coordinator.validate_review(config)['review_sha256'] == coordinator.sha(review)
        evaluation.write_text('{}', encoding='utf-8')
        expect_error(lambda: coordinator.validate_review(config))

    events = []
    lock = threading.Lock()
    def train():
        with lock:
            events.append(('train', None))
    def evaluate(sweep, gpu):
        with lock:
            events.append((sweep, gpu))
    coordinator.execute_plan(train, evaluate)
    assert events[0] == ('train', None) and set(events[1:3]) == {(5, 0), (10, 1)}
    assert events[3] == (20, 0)
    events.clear()
    def failure(sweep, gpu):
        evaluate(sweep, gpu)
        if sweep == 5:
            raise RuntimeError('simulated failure')
    expect_error(lambda: coordinator.execute_plan(train, failure))
    assert (20, 0) not in events and set(events[1:]) == {(5, 0), (10, 1)}
    events.clear()
    def train_failure():
        raise RuntimeError('simulated training failure')
    expect_error(lambda: coordinator.execute_plan(train_failure, evaluate))
    assert not events
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training = root / 'training'
        training.mkdir()
        config = {'stage_a_review': {'sha256': 'a' * 64}}
        (root / 'CONFIG.json').write_text(json.dumps(config), encoding='utf-8')
        for sweep in (5, 10, 20):
            checkpoint = training / f'sweep_{sweep:02d}.pt'
            checkpoint.write_bytes(str(sweep).encode())
            (training / f'sweep_{sweep:02d}.json').write_text(json.dumps({
                'status': 'complete', 'sweep': sweep, 'sha256': coordinator.sha(checkpoint)}), encoding='utf-8')
        done = {'status': 'complete', 'updates': 7560, 'sweeps': 20, 'code_shape': [168, 9, 16],
            'codec_and_initial_encoder_frozen': True,
            'source': {'config_sha256': coordinator.sha(root / 'CONFIG.json'),
                       'stage_a_review': {'review_sha256': 'a' * 64}}}
        (training / 'TRAIN_DONE.json').write_text(json.dumps(done), encoding='utf-8')
        assert coordinator.verify_training(training, config)['updates'] == 7560
        (root / 'CONFIG.json').write_text('{}', encoding='utf-8')
        expect_error(lambda: coordinator.verify_training(training, config))
    with tempfile.TemporaryDirectory() as directory:
        training = Path(directory)
        output = training / 'sweep_05' / 'free_z'
        evaluation = output / 'evaluation'
        evaluation.mkdir(parents=True)
        def save(path, value):
            path.write_text(json.dumps(value), encoding='utf-8')
        save(output / 'STATUS.json', {'status': 'complete'})
        save(output / 'STAGE_METRICS.json', {'status': 'complete', 'sweep': 5,
            'metrics': {'unique_frame_equal': {'correct': {'motion_mae': 0.1}}},
            'anchor_motion_mae': 0.2, 'original_two_window_anchor_motion_mae': 0.03,
            'preliminary_known_anchor_proxy_pass': False})
        save(evaluation / 'PACKETS.json', {'rows': []})
        packets_sha = coordinator.sha(evaluation / 'PACKETS.json')
        save(evaluation / 'EVALUATION.json', {'status': 'complete', 'window_predictions': 1512,
            'unique_future_frames': 176, 'source': {'sweep': 5},
            'decoder_future_geometry_input': False, 'packets_sha256': packets_sha})
        save(evaluation / 'REPLAY.json', {'status': 'passed', 'packets': 104, 'views': 208,
            'encoder_or_codec_loaded': False, 'packets_sha256': packets_sha, 'max_mae': 0.0})
        assert coordinator.verify_result(training, 5, True)['replay']['views'] == 208
        save(evaluation / 'REPLAY.json', {'status': 'passed', 'packets': 104, 'views': 1})
        expect_error(lambda: coordinator.verify_result(training, 5, True))

    root = Path(__file__).resolve().parent
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or coordinator.sha(root / name) != digest:
            raise RuntimeError('Coordinator source manifest changed')
    witness = {'status': 'passed', 'code_sha256': coordinator.sha(root / 'CODE.sha256'),
               'fixed_budget_and_review_gates': True, 'failure_stops_downstream': True,
               'evaluation_and_native_replay_gate': True, 'gpu_training_started': False,
               'scope': 'CPU mock tests; no real stage-B model, visual review or training'}
    with (root / 'CPU_TESTS.json').open('x', encoding='utf-8') as stream:
        json.dump(witness, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('PASS: fixed budget, reviewed input, execution order and failure gates')


if __name__ == '__main__':
    main()

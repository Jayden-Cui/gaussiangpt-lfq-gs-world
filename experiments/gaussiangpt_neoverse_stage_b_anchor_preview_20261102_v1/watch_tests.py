"""Verify the checkpoint event watcher without touching the trainer."""
import os
from pathlib import Path
import select
import tempfile
import watcher
import preview


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='b_inotify_test_') as directory:
        fd = watcher.open_watch(directory)
        try:
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            (Path(directory) / 'sweep_05.json').write_text('{}')
            if not poll.poll(5000):
                raise RuntimeError('Checkpoint-close event was not delivered')
            names = watcher.event_names(os.read(fd, 65536))
            if 'sweep_05.json' not in names:
                raise RuntimeError('Checkpoint event filename differs')
        finally:
            os.close(fd)
    preview.write(root / 'WATCH_TEST.json', {'status': 'passed',
                  'code_sha256': preview.sha(root / 'CODE.sha256'),
                  'checkpoint_close_event_received': True, 'training_signalled': False,
                  'scope': 'Disposable CPU file event; no GPU or training touched'})
    print('PASS: checkpoint close event without polling or training signals')


if __name__ == '__main__':
    main()

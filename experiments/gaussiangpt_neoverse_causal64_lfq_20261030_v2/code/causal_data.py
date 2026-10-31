"""All adjacent targets retain source fps; context is a consecutive past prefix."""
from motion_common import load_state, load_pair


def load_prefix(root, clip, end, max_frames=10):
    assert 1 <= end <= 176 and max_frames >= 2
    begin = max(0, end-max_frames+1)
    return [load_state(root, clip, frame) for frame in range(begin, end+1)]


def load_prefix_pair(root, clip, start, end, codec, max_frames=10):
    assert end == start+1
    s, t, initial, condition, labels = load_pair(root, clip, start, end, codec)
    begin = max(0, end-max_frames+1)
    history = [load_state(root, clip, frame) for frame in range(begin, start)]
    window = history + [s, t]
    return s, t, initial, condition, labels, window

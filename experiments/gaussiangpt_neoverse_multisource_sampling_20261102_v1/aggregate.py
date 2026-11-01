"""Explicitly separate source-, physical-frame- and window-level averages."""
from collections import defaultdict
import math


def physical_frame_macro(rows, expected_sources, metric):
    """Within each video, average overlapping contexts before averaging frames.

    Then average videos equally. The caller must supply and check the desired
    TRAIN or EVAL source set; this function never infers a split from data.
    """
    expected_sources = tuple(expected_sources)
    if not expected_sources or len(set(expected_sources)) != len(expected_sources):
        raise ValueError('Expected a nonempty unique source list')
    groups = defaultdict(list)
    pairs = set()
    for row in rows:
        clip = row['clip']
        start, horizon, frame = row['start'], row['horizon'], row['frame']
        value = float(row['metrics'][metric])
        if (clip not in expected_sources or type(start) != int or not 0 <= start <= 167
                or type(horizon) != int or not 1 <= horizon <= 9 or frame != start + horizon
                or not math.isfinite(value)):
            raise ValueError('Invalid source/window/horizon/metric')
        key = (clip, start, horizon)
        if key in pairs:
            raise ValueError('Duplicate window/horizon prediction')
        pairs.add(key)
        groups[(clip, frame)].append(value)
    expected = {(clip, start, horizon) for clip in expected_sources
                for start in range(168) for horizon in range(1, 10)}
    if pairs != expected:
        raise ValueError('Full 168-window coverage is required for every source')
    per_source = {}
    for clip in expected_sources:
        frames = [sum(groups[(clip, frame)]) / len(groups[(clip, frame)]) for frame in range(1, 177)]
        per_source[clip] = {'metric': sum(frames) / len(frames), 'unique_future_frames': len(frames),
                            'window_predictions': 1512}
    return {'metric_name': metric, 'per_source': per_source,
            'source_macro': sum(x['metric'] for x in per_source.values()) / len(per_source),
            'aggregation': 'average repeated contexts per physical frame, average 176 frames per source, then average sources equally',
            'samples_independent': False}

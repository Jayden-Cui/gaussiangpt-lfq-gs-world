"""Check C0 cached prefixes against independently recomputed raw-state prefixes."""
import math
import torch


def _difference(a, b):
    if a.shape != b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise RuntimeError('Nonfinite value or shape mismatch')
    return float((a - b).abs().max()) if a.numel() else 0.0


def _compare_memories(reference, actual, tolerance):
    if len(reference) != 9 or len(actual) != 9:
        raise RuntimeError('Expected nine cached prefixes')
    errors = []
    for expected, found in zip(reference, actual):
        if not torch.equal(expected.coordinates, found.coordinates):
            raise RuntimeError('Cached and recomputed world coordinates differ')
        diff = _difference(expected.keys, found.keys)
        if not math.isfinite(diff) or diff > tolerance:
            raise RuntimeError('Cached and recomputed prefix features differ: ' + str(diff))
        errors.append(diff)
    return errors


def _change_after(states, h):
    changed = []
    for index, state in enumerate(states):
        if index <= h:
            changed.append(state)
            continue
        coordinates = state['coordinates']
        if coordinates.ndim != 2 or len(coordinates) == 0:
            raise RuntimeError('Empty future state')
        extra = coordinates.max(dim=0).values[None] + 100
        changed.append(dict(state, coordinates=torch.cat((coordinates, extra), dim=0),
                            features=torch.cat((state['features'] + 3,
                                               torch.zeros_like(state['features'][:1])), dim=0)))
    return changed


@torch.no_grad()
def check_raw_c0(model, base, store, make_memories, starts=(35, 123), horizons=(1, 3, 7),
                 memory_tolerance=5e-5, code_tolerance=2e-5):
    """Rebuild two memory types and perturb future raw positions and features.

    This only claims the checked in-sample windows and horizons. The existing
    evaluator also checks attention against later cached memories separately.
    """
    if model.variant != 'C0' or model.training or base.training:
        raise RuntimeError('Expected C0 and frozen encoder in evaluation mode')
    if any(p.requires_grad for p in base.parameters()):
        raise RuntimeError('Frozen base is required')
    if not starts or not horizons or any(not 1 <= h <= 8 for h in horizons):
        raise ValueError('Expected nonempty windows and horizons with a future suffix')
    rows = []
    for start in starts:
        window = store.get(start)
        states = window['states']
        if len(states) != 10:
            raise RuntimeError('Expected ten original states')
        motion = make_memories(base, states)
        static = make_memories(base, states, initial_only=True)
        motion_errors = _compare_memories(window['memories'], motion, memory_tolerance)
        static_errors = _compare_memories(window['initial_memories'], static, memory_tolerance)
        base_code = model.forward_cached(motion)
        cached_code = model.forward_cached(window['memories'])
        cached_static_code = model.forward_cached(window['initial_memories'])
        static_code = model.forward_cached(static)
        motion_code_error = _difference(base_code, cached_code)
        static_code_error = _difference(static_code, cached_static_code)
        if max(motion_code_error, static_code_error) > code_tolerance:
            raise RuntimeError('Cached and recomputed action codes differ')
        for h in horizons:
            changed = make_memories(base, _change_after(states, h))
            # The perturbation must actually enter later memory, or this would
            # be a vacuous test even if the earlier output stayed unchanged.
            if torch.equal(motion[-1].coordinates, changed[-1].coordinates):
                raise RuntimeError('Raw future mutation did not reach the final prefix')
            changed_code = model.forward_cached(changed)
            error = _difference(base_code[:h], changed_code[:h])
            if not math.isfinite(error) or error > code_tolerance:
                raise RuntimeError('C0 raw-state causality failed: ' + str((start, h, error)))
            rows.append(dict(start=start, horizon=h, earlier_code_max_abs_error=error,
                             later_memory_mutation_detected=True))
        rows_for_start = dict(start=start, motion_memory_max_abs_error=max(motion_errors),
                              static_memory_max_abs_error=max(static_errors),
                              motion_code_max_abs_error=motion_code_error,
                              static_code_max_abs_error=static_code_error)
        yield rows_for_start, rows[-len(horizons):]


def audit_raw_c0(*args, **kwargs):
    windows, perturbations = [], []
    for window, rows in check_raw_c0(*args, **kwargs):
        windows.append(window)
        perturbations.extend(rows)
    return dict(status='passed', windows=windows, raw_future_perturbations=perturbations,
                memory_tolerance=kwargs.get('memory_tolerance', 5e-5),
                code_tolerance=kwargs.get('code_tolerance', 2e-5),
                scope='Two known training windows, three horizons; no generalization claim')

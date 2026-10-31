"""One bounded diagnostic for the existing real-GS gate coverage failure.

Hypotheses (not conclusions): the requested transpose map may differ from the
actual generated coordinates, or the ME map/coordinate indexing may be faulty.
The diagnostic does not repair either case. It records exact evidence first.

Budget: one GPU forward, seed 20261030, TRAIN window 40..49, first prediction
only, at most 48 GiB allocated, at least 55 GiB free at entry, six CPU threads.
Invoke under timeout --signal=TERM --kill-after=10s 180. Stop immediately on
the first missing map destination, or after the final transpose if none is
missing. No renderer, backward, optimizer, target support, or candidate removal.

Parent candidates are enumerated from offsets observed with the INSTALLED ME
CPU generative transpose, with identical geometry and tensor stride. We never
assign a unique floor parent, pad a missing gate, or infer nearest neighbors.
All artifacts go into a new --output directory, never previous witness files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch


class DiagnosticStop(RuntimeError):
    pass


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def map_degrees(kernel_map, source_count, output_count, device):
    degree = torch.zeros(output_count, device=device, dtype=torch.long)
    entries = []
    for kernel, pair in kernel_map.items():
        src, dst = (v.to(device=device, dtype=torch.long) for v in pair)
        valid = (src >= 0) & (src < source_count) & (dst >= 0) & (dst < output_count)
        if not bool(valid.all()):
            raise RuntimeError('Out-of-range actual kernel map indices')
        degree.index_add_(0, dst, torch.ones_like(dst))
        entries.append(dict(kernel=int(kernel), count=len(src)))
    return degree, entries


def cpu_observed_offsets(geometry, tensor_stride):
    """Probe only one CPU input coordinate using the actual installed operator."""
    import MinkowskiEngine as ME
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        op = ME.MinkowskiGenerativeConvolutionTranspose(
            1, 1, kernel_size=list(geometry.kernel_size), stride=list(geometry.stride),
            dilation=list(geometry.dilation), bias=False, dimension=3).cpu()
        z = ME.SparseTensor(features=torch.ones(1, 1),
                            coordinates=torch.zeros(1, 4, dtype=torch.int32),
                            tensor_stride=list(tensor_stride), device='cpu')
        out = op(z)
        return out.C.clone(), list(out.tensor_stride)


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / 'code'))
    from gs_bridge import NativeGSBridge, STATE_ROOT, MOTION_ROOT, V4_ROOT, verify_window_sources, sha
    from sampling import Segment, Window
    from online_state import load_online_window
    from aligned_ldm import make_condition
    from native_aligned_ldm import NativeAlignedLDM
    from soft_occupancy import SoftOccupancyTracker
    import MinkowskiEngine as ME

    started = time.monotonic()
    report = dict(status='started', scope='one_forward_to_first_gate_coverage_failure',
                  seed=20261030, frames=list(range(40, 50)), optimizer_updates=0,
                  renderer_calls=0, backward_calls=0, events=[],
                  budget=dict(timeout_seconds=180, max_allocated_gib=48, min_free_gib=55,
                              cpu_threads=6, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  hypotheses=['transpose map differs from generated coordinate relation',
                              'coordinate/index map inconsistency; mechanism not assumed'],
                  torch_version=torch.__version__, minkowski_version=ME.__version__,
                  code_sha256={str(p.relative_to(root)): file_sha(p)
                               for p in [Path(__file__), root/'code/soft_occupancy.py',
                                         root/'code/aligned_ldm.py', root/'code/native_aligned_ldm.py',
                                         root/'code/multiscale64.py', root/'code/gs_bridge.py']})

    def memory():
        torch.cuda.synchronize()
        return dict(allocated_gib=torch.cuda.memory_allocated()/2**30,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.memory_reserved()/2**30)

    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        if torch.cuda.is_initialized():
            report['memory'] = memory()
        write_json(output/'result.json', report)

    def terminate(signum, frame):
        report['status'] = 'timeout_signal'
        persist()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    persist()

    class DiagnosticTracker(SoftOccupancyTracker):
        def _transpose_hook(self, index, geometry, module, args, out):
            source = args[0]
            cm = source.coordinate_manager
            km = cm.kernel_map(source.coordinate_map_key, out.coordinate_map_key,
                               stride=list(geometry.stride), kernel_size=list(geometry.kernel_size),
                               dilation=list(geometry.dilation), is_transpose=True)
            degree, counts = map_degrees(km, len(source), len(out), source.F.device)
            missing = (degree == 0).nonzero().flatten()
            event = dict(kind='transpose', stage=index, module_path=geometry.module_path,
                         input_count=len(source), output_count=len(out),
                         input_stride=list(source.tensor_stride), output_stride=list(out.tensor_stride),
                         input_map_key=str(source.coordinate_map_key), output_map_key=str(out.coordinate_map_key),
                         geometry=dict(kernel=list(geometry.kernel_size), stride=list(geometry.stride),
                                       dilation=list(geometry.dilation),
                                       region_type=str(module.kernel_generator.region_type),
                                       expand_coordinates=module.kernel_generator.expand_coordinates),
                         map_entries=counts, map_edge_count=sum(v['count'] for v in counts),
                         covered_rows=int((degree > 0).sum()), missing_count=len(missing),
                         degree_min=int(degree.min()), degree_max=int(degree.max()), **memory())
            report['events'].append(event)
            print('TRANSPOSE', json.dumps(event), flush=True)
            persist()
            if len(missing):
                src_c, out_c = source.C.detach().cpu(), out.C.detach().cpu()
                missing_cpu = missing.cpu()
                offsets, cpu_out_stride = cpu_observed_offsets(geometry, source.tensor_stride)
                if cpu_out_stride != list(out.tensor_stride):
                    raise RuntimeError('CPU observed output stride differs from actual GPU stride')
                source_rows = {tuple(c): i for i, c in enumerate(src_c.tolist())}
                evidence = []
                for row in missing_cpu[:64].tolist():
                    coordinate = out_c[row]
                    parents = []
                    candidates = []
                    for offset in offsets:
                        c = coordinate.clone()
                        c[1:] -= offset[1:]
                        key = tuple(c.tolist())
                        found = source_rows.get(key)
                        entry = dict(coordinate=list(key), exists=found is not None,
                                     source_row=found, generated_offset=offset[1:].tolist())
                        candidates.append(entry)
                        if found is not None:
                            parents.append(entry)
                    evidence.append(dict(output_row=row, output_coordinate=coordinate.tolist(),
                                         exact_parent_count=len(parents), exact_parents=parents,
                                         all_exact_rule_candidates=candidates,
                                         output_feature_finite=bool(torch.isfinite(out.F[row]).all()),
                                         output_feature_norm=float(out.F[row].detach().norm())))
                event['missing_evidence'] = evidence
                event['cpu_observed_offsets'] = offsets.tolist()
                event['input_coordinate_unique_count'] = len(source_rows)
                event['output_coordinate_unique_count'] = len(torch.unique(out_c, dim=0))
                event['input_coordinate_range'] = [src_c.min(0).values.tolist(), src_c.max(0).values.tolist()]
                event['output_coordinate_range'] = [out_c.min(0).values.tolist(), out_c.max(0).values.tolist()]
                reverse = cm.kernel_map(out.coordinate_map_key, source.coordinate_map_key,
                                        stride=list(geometry.stride), kernel_size=list(geometry.kernel_size),
                                        dilation=list(geometry.dilation), is_transpose=False)
                # Reverse ordinary map has [fine,out-row ; coarse,input-row].
                reverse_degree, reverse_counts = map_degrees(
                    {k: (v[1], v[0]) for k, v in reverse.items()}, len(source), len(out), source.F.device)
                event['reverse_ordinary_map'] = dict(
                    edges=sum(v['count'] for v in reverse_counts),
                    missing_count=int((reverse_degree == 0).sum()),
                    degree_at_missing=reverse_degree.index_select(0, missing).cpu().tolist(),
                    agrees_with_requested_transpose_degree=bool(torch.equal(reverse_degree, degree)))
                identity = cm.kernel_map(out.coordinate_map_key, out.coordinate_map_key,
                                         stride=1, kernel_size=1, dilation=1, is_transpose=False)
                identity_degree, _ = map_degrees(identity, len(out), len(out), source.F.device)
                event['output_identity_map'] = dict(
                    missing_count=int((identity_degree == 0).sum()),
                    degree_at_missing=identity_degree.index_select(0, missing).cpu().tolist())
                torch.save(dict(source_coordinates=src_c, output_coordinates=out_c,
                                degree=degree.cpu(), missing_rows=missing_cpu,
                                cpu_observed_offsets=offsets,
                                source_tensor_stride=list(source.tensor_stride),
                                output_tensor_stride=list(out.tensor_stride)), output/'failed_stage_coordinates.pt')
                report['status'] = 'diagnostic_stop_missing_map'
                report['tracker_events_before_failure'] = self.events
                persist()
                raise DiagnosticStop('Captured first missing map row; no correction, renderer or backward')
            super()._transpose_hook(index, geometry, module, args, out)
            if index == len(self.stages)-1:
                report['status'] = 'diagnostic_stop_no_missing_map'
                report['tracker_events_before_stop'] = self.events
                persist()
                raise DiagnosticStop('All transpose maps covered; stop before remaining AE/output/render')

        def _stage_hook(self, index, module, args, out):
            super()._stage_hook(index, module, args, out)
            report['events'].append(dict(kind='stage_end', stage=index, output_count=len(out), **memory()))
            persist()

        def _prune_hook(self, module, args, out):
            super()._prune_hook(module, args, out)
            report['events'].append(dict(kind='prune', input_count=len(args[0]), output_count=len(out), **memory()))
            persist()

    try:
        torch.set_num_threads(6)
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info()
        report['gpu'] = dict(name=torch.cuda.get_device_name(), free_gib=free/2**30, total_gib=total/2**30)
        if free < 55*2**30:
            raise RuntimeError('Less than 55 GiB free; do not disturb shared jobs')
        torch.cuda.set_per_process_memory_fraction(min(.60, 48*2**30/total))
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(20261030)
        bridge = NativeGSBridge().train()
        spec = Window(Segment('buildingblock_tower_17_176', 0, 176), 40)
        report['source_files'] = verify_window_sources(spec)
        calibration = MOTION_ROOT/'states'/spec.clip/'000.pt'
        row = next(v for v in json.loads((V4_ROOT/'cache_manifest.json').read_text()) if v['path'] == str(calibration))
        if sha(calibration) != row['sha256']:
            raise ValueError('TRAIN calibration source mismatch')
        reference = torch.load(calibration, map_location='cpu', weights_only=True)['features']
        model = NativeAlignedLDM(coordinate_scale=bridge.codec.grid*bridge.codec.ae.stride,
                                 checkpoint_blocks=True).cuda().train()
        model.initialize_output_units(reference.cuda())
        report['stage'] = 'online_state_encoder'; persist()
        record = load_online_window(STATE_ROOT, MOTION_ROOT, spec, bridge.codec, 'cuda')
        states = record['window']
        report['stage'] = 'native64_ldm'; persist()
        condition = make_condition(states[0], radius=6)
        candidates = {tuple(c) for c in condition.coordinates.detach().cpu().tolist()}
        for state in states:
            if any(tuple(c) not in candidates for c in state['coordinates'].detach().cpu().tolist()):
                raise RuntimeError('Future outside initial candidate domain; no target expansion')
        z = model.encode_windows([states])
        predictions = model.decode_latents(z['z_quantized'], [condition], initial_contexts=z['initial_contexts'])[0]
        prediction = predictions[0]
        report['stage'] = 'first_prediction_ae_decode'
        report['initial_candidate_count'] = len(prediction['coordinates'])
        report['latent_state_counts'] = [len(s['coordinates']) for s in states]
        report['predicted_feature_summary'] = dict(mean=float(prediction['features'].detach().mean()),
                                                 std=float(prediction['features'].detach().std()),
                                                 minimum=float(prediction['features'].detach().min()),
                                                 maximum=float(prediction['features'].detach().max()))
        persist()
        latent = bridge.codec._sparse(prediction['coordinates'], prediction['features'], bridge.codec.ae.stride)
        input_coordinates = torch.cat([
            torch.zeros((len(prediction['coordinates']), 1), device='cuda', dtype=torch.int32),
            prediction['coordinates'].to(device='cuda', dtype=torch.int32)*int(bridge.codec.ae.stride)], 1)
        with DiagnosticTracker(bridge.codec.ae.decoder, latent, prediction['occupancy_logits'],
                               input_coordinates=input_coordinates):
            bridge.codec.ae.decode(latent, target_key=None)
        raise RuntimeError('Diagnostic decoder unexpectedly completed')
    except DiagnosticStop as error:
        report['stop_reason'] = str(error)
        persist()
        print('DIAGNOSTIC_STOP', report['status'], flush=True)
    except BaseException as error:
        report['status'] = 'failed_before_diagnostic_stop'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

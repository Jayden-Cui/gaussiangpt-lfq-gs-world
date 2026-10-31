"""Small CPU integration contracts; these fixtures are not GPU evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
import inference_profile as ip
import compare as compare


def profile():
    return {**ip.FIXED, 'diagnostic_result_sha256': 'a' * 64}


def conversion():
    counts = {'tensors': 2, 'elements': 8}
    return dict(schema='decoder_precision_adapter_v1', decoder_precision='float64', selected_modules=ip.SELECTED,
        original_floating_state_float32=True, castback_exact=True, encoder_exact=True, nonselected_state_exact=True,
        original_state_digest='b' * 64, mode_state_digest='c' * 64,
        original_dtype_counts={'torch.float32': counts}, selected_mode_dtype_counts={'torch.float64': counts},
        mode_dtype_counts={'torch.float32': counts, 'torch.float64': counts})


def verification():
    return dict(unchanged=True, decoder_precision='float64', state_digest='c' * 64,
                mode_state_digest='c' * 64, checked_tensors=4)


class FakeME:
    SparseTensorOperationMode = SimpleNamespace(SEPARATE_COORDINATE_MANAGER='separate')
    def __init__(self): self.manager = None; self.flag = False; self.mode = 'separate'
    def global_coordinate_manager(self): return self.manager
    def sparse_tensor_operation_mode(self): return self.mode
    def is_deterministic(self): return self.flag
    # There are deliberately no set/clear methods: integration must never mutate global state.


class Manager:
    pass


def diagnostic_result(probe):
    cases = [dict(clip=clip, frame=frame, replay_summaries=dict(fresh_same_state={'passed': True},
                historical_state_comparison={'passed': False})) for clip, frame in sorted(ip.CASES)]
    chosen = dict(decoder_precision='float64', checkpoint_sha256=probe.CHECKPOINT_SHA,
        checkpoint_model_state_sha256='d' * 64, conversion=conversion(), state_verification=verification(),
        inference_runtime_before=ip.RUNTIME, inference_runtime_after=ip.RUNTIME,
        replay_summaries={'fresh_same_state': {'passed': True}}, cases=cases)
    return dict(status='completed', completed=True, gpu_executed=True, source={'source': 'fixture'},
        formal_updates=0, checkpoint_sha256=probe.CHECKPOINT_SHA, original_checkpoint_model_state_sha256='d' * 64,
        thresholds={'mae_max': 1e-5, 'psnr_min': 60}, original_deterministic_flag_restore={'restored': True},
        inference_runtime=ip.RUNTIME, decoder_precision_results=[{'decoder_precision': 'float32',
            'replay_summaries': {'fresh_same_state': {'passed': False}}}, chosen],
        installed_minkowski_sources={n: {'sha256': d} for n, d in probe.MINKOWSKI_SOURCES.items()})


class InferenceProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent / Path(ip.DIAGNOSTIC_ROOT).name
        cls.probe = ip.import_bound(root / 'probe.py', '_cpu_test_v7_probe', ip.PROBE_SHA)
        cls.precision = ip.import_bound(root / 'precision.py', '_cpu_test_v7_precision', ip.PRECISION_SHA)

    def test_profile_and_pinned_helper_match_exact_scope(self):
        ip.validate_profile(profile())
        self.assertEqual(list(self.precision.SELECTED_MODULES), ip.SELECTED)
        for method in ('prepare_sparse', 'to_render_points', 'assert_unchanged'):
            self.assertTrue(callable(getattr(self.precision.DecoderPrecision, method)))
        for change in ({'diagnostic_result_sha256': None}, {'diagnostic_result_sha256': '0' * 64},
                       {'minkowski_deterministic': True}, {'decoder_precision': 'float32'},
                       {'pruning_capture_hooks': True}, {'cold_manager_replay_required': False}):
            with self.assertRaises(RuntimeError): ip.validate_profile({**profile(), **change})

    def test_default_manager_lifecycle_releases_old_identity_before_replay_decode(self):
        engine = FakeME(); record = {}; events = []; references = []
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: events.append('synchronize')))
        packet = {'metadata': {'inference_profile': profile(), 'precision_conversion': conversion()}}
        saved = []
        def direct(register):
            ip.entry_state(engine)
            manager = Manager(); register(manager); references.append(weakref.ref(manager))
            events.append('fresh_encode_direct'); return packet
        def save_load(value):
            ip.entry_state(engine); self.assertIsNotNone(references[0]())
            events.append('save_load'); saved.append(deepcopy(value)); return deepcopy(value)
        def replay(value, release):
            ip.entry_state(engine)
            manager = Manager(); self.assertIsNot(manager, references[0]())
            release(manager); self.assertIsNone(references[0]())
            events.append('decode_new_manager'); return 'dense_replay'
        result = ip.independent_replay(torch, engine, direct, replay, save_load, record)
        self.assertEqual(result, (packet, 'dense_replay'))
        self.assertEqual(events, ['fresh_encode_direct', 'synchronize', 'save_load', 'decode_new_manager', 'synchronize'])
        self.assertEqual(saved[0], packet); self.assertNotIn('inference_scopes', packet['metadata'])
        ip.entry_state(engine); ip.validate_frame_scopes(record)
        broken = deepcopy(record); broken['replay']['manager_python_id'] = broken['direct']['manager_python_id']
        with self.assertRaisesRegex(RuntimeError, 'identities must differ'): ip.validate_frame_scopes(broken)

    def test_phase_failure_does_not_toggle_global_state_or_claim_replay_complete(self):
        for failing in ('direct', 'replay'):
            engine = FakeME(); record = {}
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None))
            def direct(register):
                register(Manager())
                if failing == 'direct': raise RuntimeError('direct failure')
                return 'packet'
            def replay(value, release):
                manager = Manager(); release(manager)
                raise RuntimeError('replay failure')
            with self.assertRaisesRegex(RuntimeError, failing + ' failure'):
                ip.independent_replay(torch, engine, direct, replay, lambda v: v, record)
            ip.entry_state(engine)
            self.assertFalse(record[failing]['body_completed'])
            self.assertEqual(record[failing]['after'], ip.RUNTIME)
            self.assertNotIn('cold_manager_replay_verified', record)

    def test_foreign_global_state_is_rejected_without_mutation(self):
        for change in ('manager', 'flag', 'mode'):
            engine = FakeME(); setattr(engine, change, Manager() if change == 'manager' else True if change == 'flag' else 'share')
            before = (engine.manager, engine.flag, engine.mode)
            with self.assertRaises(RuntimeError): ip.entry_state(engine)
            self.assertEqual(before, (engine.manager, engine.flag, engine.mode))

    def test_only_float64_fresh_must_pass_with_exact_conversion_and_unchanged_state(self):
        result = diagnostic_result(self.probe)
        ip.validate_diagnostic(result, {'source': 'fixture'}, self.probe)
        for category, field, value in [('conversion', 'castback_exact', False), ('conversion', 'encoder_exact', False),
                                       ('state_verification', 'unchanged', False), ('state_verification', 'state_digest', 'e' * 64)]:
            bad = deepcopy(result); bad['decoder_precision_results'][1][category][field] = value
            with self.assertRaises(RuntimeError): ip.validate_diagnostic(bad, {'source': 'fixture'}, self.probe)
        bad = deepcopy(result); bad['decoder_precision_results'][1]['replay_summaries']['fresh_same_state']['passed'] = False
        with self.assertRaisesRegex(RuntimeError, 'fresh replay'): ip.validate_diagnostic(bad, {'source': 'fixture'}, self.probe)
        bad = deepcopy(result); bad['thresholds']['mae_max'] = 2e-5
        with self.assertRaisesRegex(RuntimeError, 'thresholds'): ip.validate_diagnostic(bad, {'source': 'fixture'}, self.probe)

    def test_baseline_uses_preconversion_legacy_digest_and_separate_adapter_digest(self):
        evidence = dict(checkpoint_model_state_sha256='d' * 64, checkpoint_decoder_sha256='e' * 64,
            model_state_sha256='f' * 64, precision_conversion=conversion(), encoder_state_dtype='float32',
            decoder_precision='float64', renderer_gs_dtype='float32')
        evaluation = {**deepcopy(evidence), 'precision_state_verification': verification()}
        compare.precision_fields(evaluation, evidence, baseline_initial='d' * 64)
        with self.assertRaisesRegex(RuntimeError, 'Original baseline'):
            compare.precision_fields(evaluation, evidence, baseline_initial='f' * 64)
        bad = deepcopy(evaluation); bad['precision_state_verification']['state_digest'] = 'f' * 64
        with self.assertRaisesRegex(RuntimeError, 'state_digest'):
            compare.precision_fields(bad, evidence, baseline_initial='d' * 64)


if __name__ == '__main__':
    unittest.main()

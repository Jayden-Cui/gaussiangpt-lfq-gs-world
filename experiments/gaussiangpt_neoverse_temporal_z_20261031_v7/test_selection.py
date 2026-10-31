"""CPU selection failure cases; fixtures exist only inside a temporary directory."""
import json
from pathlib import Path
import tempfile
import unittest
from pipeline import choose_common, sha


class SelectionTest(unittest.TestCase):
    def fixture(self, path):
        arms = ['z_rgb','constant_rgb','z_delta','constant_delta']
        cfg = dict(arms={a:{} for a in arms},train=dict(save_steps=[200,300,400]))
        (path/'CODE_MANIFEST.json').write_text('{}')
        (path/'protocol.json').write_text(json.dumps(cfg))
        for index, arm in enumerate(arms):
            out=path/arm;out.mkdir()
            (out/'PLAN.json').write_text(json.dumps(dict(starts=[[0,42,84,126]]*400)))
            for step in [200,300]:
                (out/f'step_{step:04d}.pt').write_bytes(b'selection-fixture-only')
            (out/'final.pt').write_bytes(b'never-use-this-in-selection')
            (out/'TRAIN_DONE.json').write_text(json.dumps(dict(status='complete',steps=350+index,
                checkpoint_sha256={str(s):sha(out/f'step_{s:04d}.pt') for s in [200,300]})))
        return cfg

    def test_largest_common_not_final(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);cfg=self.fixture(path)
            self.assertEqual(choose_common(path,cfg),300)
            result=json.loads((path/'SELECTION.json').read_text())
            self.assertEqual(result['original_completed_steps']['constant_delta'],353)
            self.assertTrue(all(x['path'].endswith('step_0300.pt') for x in result['checkpoints'].values()))

    def test_mismatched_plan_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);cfg=self.fixture(path)
            (path/'z_delta/PLAN.json').write_text(json.dumps(dict(starts=[[1,42,84,126]]*400)))
            with self.assertRaisesRegex(RuntimeError,'sampling plans differ'):
                choose_common(path,cfg)

    def test_no_common_saved_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);cfg=self.fixture(path)
            for step in [200,300]:
                (path/'z_delta'/f'step_{step:04d}.pt').unlink()
            with self.assertRaisesRegex(RuntimeError,'No common'):
                choose_common(path,cfg)

    def test_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);cfg=self.fixture(path)
            (path/'z_delta/step_0300.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'differs from training receipt'):
                choose_common(path,cfg)


if __name__=='__main__':
    unittest.main(verbosity=2)

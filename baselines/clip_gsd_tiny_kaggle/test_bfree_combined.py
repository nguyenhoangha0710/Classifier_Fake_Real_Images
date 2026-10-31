"""Torch-free checks for the combined test notebook's bounded locator and resume patch."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

import build_bfree_samecat_origbg_notebooks as b


class LocatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / 'input'
        self.root = root
        ck = root / 'datasets/hoang/check-point/gsd_bfree_samecat_origbg_v1'
        ck.mkdir(parents=True)
        (ck / 'best.pt').write_bytes(b'x' * 10)
        (ck / 'checkpoint_ve1.pt').write_bytes(b'y' * 10)
        (ck / 'run_state.json').write_text('{}')
        stray = root / 'datasets/other/best_elsewhere'
        stray.mkdir(parents=True)
        (stray / 'best.pt').write_bytes(b'z')                       # no run_state.json -> ignored
        bf = root / 'datasets/baopv051/b-free-training-data/COCO_real_512/COCO_real_512'
        bf.mkdir(parents=True)
        (bf.parent.parent / 'valid_list.csv').write_text('id\n')
        man = root / 'datasets/hadinho0710/mainfest-train-genimage/seed42'
        man.mkdir(parents=True)
        pd.DataFrame(dict(relative_path=['sd15/val/ai/a.png'])).to_csv(man / 'test_manifest.csv', index=False)
        tiny = root / 'datasets/yangsangtai/tiny-genimage/sd15/val/ai'
        tiny.mkdir(parents=True)
        (tiny / 'a.png').write_bytes(b'p')
        prev = root / 'notebooks/me/test-run/gsd_bfree_samecat_origbg_best_tiny9_v1'
        prev.mkdir(parents=True)
        (prev / 'evaluation_state.json').write_text(json.dumps(dict(evaluator='EV', fingerprint='fp', saved_rows={'native': 80})))
        calls = []
        self.calls = calls
        ns = dict(time=time, json=json, Path=Path, pd=pd, UPLOAD_EVALUATOR='EV',
                  restore_uploaded_progress=lambda out, options, fp: calls.append(options['resume_from']) or 'restored')
        exec(b.LOCATOR_CELL, ns)
        self.ns = ns

    def tearDown(self):
        self.tmp.cleanup()

    def test_locates_all_inputs(self):
        r = str(self.root)
        self.assertEqual(self.ns['locate_checkpoint']('best.pt', r).parent.name, 'gsd_bfree_samecat_origbg_v1')
        self.assertEqual(self.ns['locate_checkpoint']('checkpoint_ve1.pt', r).name, 'checkpoint_ve1.pt')
        manifest = self.ns['locate_test_manifest'](r)
        self.assertEqual(manifest.parent.name, 'seed42')
        self.assertEqual(self.ns['locate_tiny_root'](manifest, r).name, 'tiny-genimage')

    def test_skips_image_folders(self):
        visited = {d.name for d in self.ns['bounded_dirs'](str(self.root))}
        self.assertNotIn('COCO_real_512', visited)
        self.assertNotIn('ai', visited)

    def test_bounded_resume_matches_fingerprint_only(self):
        out = Path(self.tmp.name) / 'working/out'
        out.mkdir(parents=True)
        opts = dict(resume_from='auto', input_root=str(self.root))
        self.assertEqual(self.ns['restore_uploaded_progress'](out, opts, 'fp'), 'restored')
        self.assertTrue(self.calls[-1].endswith('gsd_bfree_samecat_origbg_best_tiny9_v1'))
        self.assertIsNone(self.ns['restore_uploaded_progress'](out, opts, 'other'))
        self.ns['restore_uploaded_progress'](out, dict(opts, resume_from=None), 'fp')
        self.assertIsNone(self.calls[-1])

    def test_notebook_builds_and_reuses_runtime(self):
        nb = b.build_combined()
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        orig = json.loads((b.HERE / 'test_gsd_uploaded_best_bfree_valid_kaggle.ipynb').read_text(encoding='utf-8'))
        runtime = max((''.join(c['source']) for c in orig['cells'] if c['cell_type'] == 'code'), key=len)
        runtime = runtime.strip('\n')
        self.assertIn(runtime, code)                                   # byte-identical apart from edge newlines
        self.assertLess(code.index(runtime), code.index(b.COMPAT_CELL.strip('\n')))


if __name__ == '__main__':
    unittest.main(verbosity=2)

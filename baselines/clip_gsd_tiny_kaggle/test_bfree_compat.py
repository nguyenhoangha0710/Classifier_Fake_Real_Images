"""Torch-free check: a B-Free checkpoint config passes the ORIGINAL evaluator validation
(evaluate_uploaded_best.validate_uploaded_payload + runtime_raw_all7.validate_raw_config)
through the compat cell inserted in the evaluation notebooks."""
import ast
import json
import math
import unittest
from pathlib import Path

import build_bfree_samecat_origbg_notebooks as builder

HERE = Path(__file__).resolve().parent


def run_statements(ns, file, names):
    """Exec only the named top-level assignments/functions of a runtime file, in file order."""
    tree = ast.parse((HERE / file).read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), file, 'exec'), ns)


def safe_types(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and safe_types(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(safe_types(v) for v in value)
    return value is None or isinstance(value, (str, bool, int, float))


class CompatTests(unittest.TestCase):
    def setUp(self):
        ns = dict(math=math)
        run_statements(ns, 'runtime.py', {'DEFAULT_CONFIG'})
        raw = {}
        run_statements(raw, 'runtime_raw_all7.py', {'GENS7'})
        ns['GENS7'] = raw['GENS7']
        run_statements(ns, 'runtime_raw_all7.py', {'DEFAULT_CONFIG', 'validate_raw_config'})
        ns['_raw_validate'] = ns['validate_raw_config']
        run_statements(ns, 'runtime_aug_all7.py', {'VALIDATION_MONITOR', 'DEFAULT_CONFIG', 'validate_raw_config'})
        ns['BFREE_REAL'] = 'COCO_real_512'
        ns['BFREE_ALL_BRANCHES'] = ('SD2.1_selfconditioned', 'SD2.1_selfconditioned_origBG',
                                    'SD2.1_inpainted_samecat', 'SD2.1_inpainted_samecat_origBG',
                                    'SD2.1_inpainted_diffcat', 'SD2.1_inpainted_diffcat_origBG')
        run_statements(ns, 'bfree_train.py', {'BFREE_TRAIN_PROTOCOL', 'INFERENCE_KEYS', 'BFREE_TRAIN_DEFAULTS',
                                              'bfree_smoke_overrides', 'validate_bfree_config',
                                              'eval_compat_config', 'check_evaluator_compatibility'})
        self.ns = ns
        # The original evaluator check, executed against the RAW validator like in the eval notebooks.
        ev = dict(validate_raw_config=ns['_raw_validate'])
        run_statements(ev, 'evaluate_uploaded_best.py', {'validate_uploaded_payload'})
        exec(builder.COMPAT_CELL, ev)
        self.ev = ev

    def payload(self, **cfg_updates):
        cfg = dict(self.ns['BFREE_TRAIN_DEFAULTS'], **cfg_updates)
        self.ns['validate_bfree_config'](cfg)
        config = dict(cfg, eval_compat_config=self.ns['eval_compat_config'](cfg))
        return dict(model={'w': 1}, epoch=0, signature='abc', config=config, protocol='x',
                    validation={}, val_loss=0.1, images_seen=25200)

    def test_default_config_is_tiny_recipe(self):
        cfg = self.ns['BFREE_TRAIN_DEFAULTS']
        tiny_nb = json.loads((HERE / 'train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb').read_text(encoding='utf-8'))
        tiny_cfg_cell = ''.join(tiny_nb['cells'][4]['source'])
        tiny = dict(self.ns['DEFAULT_CONFIG'])
        exec(tiny_cfg_cell.replace('print(json.dumps(CONFIG, indent=2))', ''), dict(DEFAULT_CONFIG=tiny, dict=dict),
             loc := {})
        tiny = loc['CONFIG']
        for key in ('model_id', 'input_mode', 'crop_max_side', 'train_batch', 'test_batch', 'micro_batch',
                    'anchor_micro_batch', 'lr', 'head_lr', 'modulation_lr', 'weight_decay', 'betas', 'eps',
                    'k', 'lambda_max', 'a_init', 'b_init', 'precision', 'gradient_checkpointing', 'attention',
                    'basis_method', 'pooling', 'threshold', 'jpeg_probability', 'jpeg_train_quality',
                    'jpeg_subsampling', 'blur_probability', 'blur_sigma', 'horizontal_flip_probability',
                    'validation_processing_seed', 'validation_monitor', 'min_delta'):
            self.assertEqual(cfg[key], tiny[key], key)

    def test_payload_accepted_by_original_evaluator_via_compat(self):
        payload = self.payload()
        self.assertTrue(safe_types(payload['config']))
        self.assertTrue(self.ns['check_evaluator_compatibility'](payload, payload['config']))
        opts = dict(expected_signature=None, expected_epoch=None)
        cfg = self.ev['validate_uploaded_payload'](payload, opts)
        self.assertEqual(cfg['seeds'], [42])
        self.assertEqual(cfg['test_batch'], 8)
        self.assertIsNone(cfg['manifest_root'])
        self.assertEqual(set(cfg['train_generators']), set(self.ns['GENS7']))
        for key in self.ns['INFERENCE_KEYS']:
            self.assertEqual(cfg[key], payload['config'][key])

    def test_compat_rejects_other_checkpoints_and_smoke(self):
        opts = dict(expected_signature=None, expected_epoch=None)
        tiny_like = dict(model={'w': 1}, epoch=0, signature='s', config=dict(self.ns['DEFAULT_CONFIG']))
        with self.assertRaises(ValueError):
            self.ev['validate_uploaded_payload'](tiny_like, opts)
        smoke = self.payload(smoke=True)
        with self.assertRaises(ValueError):
            self.ev['validate_uploaded_payload'](smoke, opts)
        tampered = self.payload()
        tampered['config']['eval_compat_config']['k'] = 32
        with self.assertRaises(ValueError):
            self.ev['validate_uploaded_payload'](tampered, opts)

    def test_config_guards(self):
        with self.assertRaises(ValueError):
            self.payload(horizontal_flip_probability=0.5)
        with self.assertRaises(ValueError):
            self.payload(train_branch='SD2.1_unknown')
        with self.assertRaises(ValueError):
            self.payload(val_ids=2001)
        with self.assertRaises(ValueError):
            self.ns['validate_bfree_config'](dict(self.ns['BFREE_TRAIN_DEFAULTS'], typo_key=1))
        smoke = self.ns['bfree_smoke_overrides'](dict(self.ns['BFREE_TRAIN_DEFAULTS']))
        self.ns['validate_bfree_config'](smoke)
        self.assertTrue(smoke['output_root'].endswith('_smoke'))

    def test_eval_notebooks_keep_original_runtime_byte_identical(self):
        for derived, original in ((builder.TINY_NB, 'test_gsd_uploaded_best_variants_kaggle.ipynb'),
                                  (builder.BFREE_NB, 'test_gsd_uploaded_best_bfree_valid_kaggle.ipynb')):
            new = json.loads((HERE / derived).read_text(encoding='utf-8'))['cells']
            old = json.loads((HERE / original).read_text(encoding='utf-8'))['cells']
            old_code = [''.join(c['source']) for c in old if c['cell_type'] == 'code']
            new_code = [''.join(c['source']) for c in new if c['cell_type'] == 'code']
            self.assertIn(max(old_code, key=len), new_code)          # runtime untouched
            for cell in old_code:
                if 'CONFIG.update(' in cell:
                    continue
                self.assertIn(cell, new_code)                         # run/report cells untouched
            config = next(c for c in new_code if 'CONFIG.update(' in c)
            self.assertIn('expected_signature=None', config)
            self.assertIn('gsd_bfree_samecat_origbg', config)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Offline checks for the torch-free B-Free paired data layer (synthetic dataset)."""
import ast
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import bfree_data as d

HERE = Path(__file__).resolve().parent
SIDE = 56


def extract_function(file, name, extra_globals=None):
    """Exec ONE top-level function from an existing runtime file (no torch import)."""
    tree = ast.parse((HERE / file).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(extra_globals or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), ns)
    return ns[name]


class BFreeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / 'input' / 'datasets' / 'owner' / 'b-free-training-data'
        cls.root = root
        rng = np.random.default_rng(0)
        cls.train_ids = [f'{100000000000 + i:012d}' for i in range(120)]
        cls.valid_ids = cls.train_ids[:20]
        (root).mkdir(parents=True)
        pd.DataFrame(dict(id=cls.train_ids, same_category='x', diff_category='y')).to_csv(root / 'train_list.csv', index=False)
        pd.DataFrame(dict(id=cls.valid_ids, same_category='x', diff_category='y')).to_csv(root / 'valid_list.csv', index=False)
        real_dir = root / d.BFREE_REAL / d.BFREE_REAL          # double nested
        fake_dir = root / 'SD2.1_inpainted_samecat_origBG'      # single nested
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        cls.expected = {}
        for k, i in enumerate(cls.train_ids):
            real = rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8)
            fake = real.copy()
            side = 2 + (k % 30)                                   # varied inpaint area
            if k == 25:
                side = 0                                          # identical pair -> excluded
            if side:
                fake[3:3 + side, 5:5 + side] = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
                fake[3:3 + side, 5:5 + side, 0] = (real[3:3 + side, 5:5 + side, 0].astype(int) + 100) % 256
            Image.fromarray(real).save(real_dir / f'{i}.png')
            if k != 40:                                           # missing fake -> unreadable
                Image.fromarray(fake).save(fake_dir / f'{i}.png')
            cls.expected[i] = side * side / (SIDE * SIDE)
        cls.cfg = dict(input_root=str(Path(cls.tmp.name) / 'input'), bfree_root=None, image_side=SIDE,
                       mask_diff_threshold=8, min_mask_fraction=0.001, max_mask_fraction=0.95, audit_workers=4,
                       val_ids=12, ve_ids=24, max_virtual_epochs=3, pairs_per_batch=4,
                       blur_probability=0.5, jpeg_probability=0.5, blur_sigma=[0.0, 2.0],
                       jpeg_train_quality=[75, 100], jpeg_subsampling=0, validation_processing_seed=42,
                       crop_max_side=560)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def audit(self):
        root = d.find_bfree_root(self.cfg)
        train, _ = d.read_bfree_list(root / 'train_list.csv', None, None, check_md5=False)
        valid, _ = d.read_bfree_list(root / 'valid_list.csv', None, None, check_md5=False)
        ids = [i for i in train.id if i not in set(valid.id)]
        real = d.bfree_folder(root, d.BFREE_REAL, ids[:20])
        fake = d.bfree_folder(root, 'SD2.1_inpainted_samecat_origBG', ids[:20])
        return d.mark_eligibility(d.audit_pairs(ids, real, fake, self.cfg, 'train', log_every=50), self.cfg)

    def test_root_layout_and_lists(self):
        root = d.find_bfree_root(self.cfg)
        self.assertEqual(root, self.root)
        self.assertEqual(d.bfree_folder(root, d.BFREE_REAL, self.train_ids[:5]), root / d.BFREE_REAL / d.BFREE_REAL)
        with self.assertRaises(ValueError):
            d.read_bfree_list(root / 'train_list.csv', d.BFREE_TRAIN_LIST_MD5, None, check_md5=True)

    def test_audit_mask_fraction_and_exclusions(self):
        audit = self.audit().set_index('coco_id')
        self.assertEqual(len(audit), 100)
        for i, row in audit.iterrows():
            k = self.train_ids.index(i)
            if k == 40:
                self.assertFalse(row.ok)
                self.assertEqual(row.exclusion_reason, 'unreadable_or_missing')
            elif k == 25:
                self.assertTrue(row.identical)
                self.assertEqual(row.exclusion_reason, 'mask_too_small_or_identical')
            else:
                self.assertAlmostEqual(row.mask_fraction, self.expected[i], places=6)
                self.assertTrue(row.eligible)
        self.assertEqual(int(audit.eligible.sum()), 98)

    def test_splits_stratified_disjoint_deterministic(self):
        audit = self.audit()
        val, pool, summary = d.make_bfree_splits(audit, self.cfg, 42)
        val2, pool2, _ = d.make_bfree_splits(audit.sample(frac=1, random_state=3), self.cfg, 42)
        self.assertEqual(val.coco_id.tolist(), val2.coco_id.tolist())
        self.assertEqual(pool.coco_id.tolist(), pool2.coco_id.tolist())
        self.assertEqual(len(val), 12)
        self.assertEqual(val.area_group.value_counts().to_dict(), dict(small=4, medium=4, large=4))
        self.assertFalse(set(val.coco_id) & set(pool.coco_id))
        self.assertEqual(summary['virtual_epochs'], 3)
        self.assertEqual(pool.virtual_epoch.value_counts().sort_index().to_dict(), {-1: 14, 0: 24, 1: 24, 2: 24})
        self.assertTrue(set(audit[~audit.eligible].coco_id).isdisjoint(set(pool.coco_id) | set(val.coco_id)))
        other, _, _ = d.make_bfree_splits(audit, self.cfg, 7)
        self.assertNotEqual(other.coco_id.tolist(), val.coco_id.tolist())

    def test_audit_csv_round_trip_gives_identical_splits(self):
        audit = self.audit()
        path = Path(self.tmp.name) / 'audit_train.csv'
        d.atomic_csv(path, audit)
        back = d.restore_audit_types(pd.read_csv(path, dtype={'coco_id': str, 'exclusion_reason': str},
                                                 keep_default_na=False))
        self.assertTrue(np.array_equal(back.mask_fraction.to_numpy()[back.ok], audit.mask_fraction.to_numpy()[audit.ok]))
        self.assertEqual(back.eligible.tolist(), audit.eligible.tolist())
        self.assertEqual(int(back.identical.sum()), 1)
        _, _, s1 = d.make_bfree_splits(audit, self.cfg, 42)
        _, _, s2 = d.make_bfree_splits(back, self.cfg, 42)
        self.assertEqual(s1, s2)

    def test_pair_rows_and_batches(self):
        frame = pd.DataFrame(dict(coco_id=['a', 'b', 'c', 'e'], real_path=list('1234'), fake_path=list('5678')))
        rows = d.pair_rows(frame)
        self.assertEqual([r['label'] for r in rows], [0, 1] * 4)
        self.assertEqual([r['coco_id'] for r in rows][0::2], [r['coco_id'] for r in rows][1::2])
        self.assertEqual(d.pair_batches(8, 4), [list(range(8)), list(range(8, 16))])
        with self.assertRaises(ValueError):
            d.pair_batches(6, 4)

    def test_pair_processing_shared_local_and_epoch_aware(self):
        random.seed(5)
        state = random.getstate()
        a = [d.pair_processing_parameters(str(i), self.cfg, 42, True, 0) for i in range(200)]
        self.assertEqual(state, random.getstate())
        self.assertEqual(a, [d.pair_processing_parameters(str(i), self.cfg, 42, True, 0) for i in range(200)])
        self.assertNotEqual(a, [d.pair_processing_parameters(str(i), self.cfg, 42, True, 1) for i in range(200)])
        v = d.pair_processing_parameters('x', self.cfg, 42, False, 0)
        self.assertEqual(v, d.pair_processing_parameters('x', self.cfg, 42, False, 2))
        blur = np.mean([p['blur_applied'] for p in a]); jpg = np.mean([p['jpeg_applied'] for p in a])
        self.assertTrue(0.35 < blur < 0.65 and 0.35 < jpg < 0.65)
        q = [p['jpeg_quality'] for p in a if p['jpeg_applied']]
        self.assertTrue(min(q) >= 75 and max(q) <= 100)

    def test_identical_images_get_identical_processing(self):
        img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8))
        params = dict(blur_applied=True, blur_sigma=1.3, jpeg_applied=True, jpeg_quality=81)
        x = np.asarray(d.apply_pair_processing(img.copy(), params, self.cfg))
        y = np.asarray(d.apply_pair_processing(img.copy(), params, self.cfg))
        self.assertTrue(np.array_equal(x, y))
        self.assertFalse(np.array_equal(x, np.asarray(img)))

    def test_geometry_and_jpeg_match_existing_runtime(self):
        geometry = extract_function('runtime_raw_all7.py', 'input_geometry')
        cfg = dict(input_mode='center_crop_max_edge_pad14', crop_max_side=560)
        for w, h in [(512, 512), (500, 375), (1024, 768), (64, 64), (561, 300)]:
            old = geometry(w, h, cfg)
            new = d.crop_geometry(w, h, 560)
            self.assertEqual({k: old[k] for k in new}, new)
        self.assertEqual(d.crop_geometry(512, 512, 560)['tokens'], 1370)
        jpeg = extract_function('runtime.py', 'jpeg', dict(io=io, Image=Image))
        img = Image.fromarray(np.random.default_rng(2).integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8))
        self.assertTrue(np.array_equal(np.asarray(jpeg(img, 83, 0)), np.asarray(d.jpeg_roundtrip(img, 83, 0))))
        aug = extract_function('runtime_aug_all7.py', 'apply_processing',
                               dict(ImageFilter=__import__('PIL.ImageFilter', fromlist=['x']), jpeg=jpeg))
        params = dict(blur_applied=True, blur_sigma=0.7, jpeg_applied=True, jpeg_quality=90)
        self.assertTrue(np.array_equal(np.asarray(aug(img, params, dict(jpeg_subsampling=0))),
                                       np.asarray(d.apply_pair_processing(img, params, self.cfg))))

    def test_load_processed_crop_keeps_native_512(self):
        path = Path(self.tmp.name) / 'big.png'
        Image.fromarray(np.zeros((512, 512, 3), np.uint8)).save(path)
        self.assertEqual(d.load_processed_crop(path, self.cfg).size, (512, 512))

    def test_run_hash_and_previous_output_discovery(self):
        cfg = dict(self.cfg, output_root='/a', time_budget_hours=11, seed=42)
        h = d.run_identity_hash(cfg)
        self.assertEqual(h, d.run_identity_hash(dict(cfg, output_root='/b', time_budget_hours=3)))
        self.assertNotEqual(h, d.run_identity_hash(dict(cfg, seed=1)))
        prev = Path(self.cfg['input_root']) / 'notebooks' / 'me' / 'run' / 'gsd_bfree_samecat_origbg_v1'
        prev.mkdir(parents=True)
        d.atomic_json(prev / 'run_state.json', dict(run_hash=h, images_seen=1000))
        other = Path(self.cfg['input_root']) / 'old_output'
        other.mkdir()
        d.atomic_json(other / 'run_state.json', dict(run_hash='different', images_seen=5000))
        found = d.find_previous_outputs(cfg, h)
        self.assertEqual([m[2] for m in found], [prev])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Metadata-only manifest reuse; reject bad input instead of silently scanning images."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import runtime_aug_all7 as r
from package_all7_manifests import package_manifests
import zipfile


class ManifestReuseTests(unittest.TestCase):
    def make_fixture(self, folder):
        root = folder / 'input'
        dataset = root / 'images' / 'tiny-genimage'
        manifests = root / 'previous' / 'old_run'
        seed_folder = manifests / 'seed42'
        seed_folder.mkdir(parents=True)
        frames = {}
        for part in r.MANIFEST_PARTITIONS:
            rows = []
            source = 'val' if part == 'test' else 'train'
            for gen in r.GENS7:
                for label in (0, 1):
                    for index in range(2):
                        rel = f'{gen}/{source}/{"ai" if label else "nature"}/{part}{index}.png'
                        path = dataset / rel
                        path.parent.mkdir(parents=True, exist_ok=True)
                        data = np.random.default_rng(len(rows)).integers(0, 256, (29, 43, 3), dtype=np.uint8)
                        Image.fromarray(data).save(path)
                        # Distinct saved hashes emulate a prior audit. Fixture construction alone
                        # reads images; the loader must never recompute these content hashes.
                        rows.append(dict(sample_id=hashlib.sha256(rel.encode()).hexdigest(),
                            relative_path=rel, path='/stale/kaggle/mount/' + rel, generator=gen,
                            split='test' if part == 'test' else 'train', label=label, width=43, height=29,
                            format='PNG', file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                            pixel_hash=hashlib.sha256(('pixels:' + rel).encode()).hexdigest(),
                            file_bytes=path.stat().st_size))
            # Deliberately not label/gen sorted: exact saved row order must survive.
            frame = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
            frame.to_csv(seed_folder / f'{part}_manifest.csv', index=False)
            frames[part] = frame
        summary = {part: len(frame) for part, frame in frames.items()}
        summary.update({part + '_hash': r.digest(frame.sample_id.tolist()) for part, frame in frames.items()})
        r.dump(seed_folder / 'split_summary.json', summary)
        r.dump(manifests / 'config.json', dict(train_generators=r.GENS7, smoke=False, validation_fraction=.1))
        output = folder / 'output'
        output.mkdir()
        cfg = dict(r.DEFAULT_CONFIG, input_root=str(root), dataset_root=None, manifest_root=None)
        return cfg, output, dataset, manifests, frames

    def load_without_image_reads(self, cfg, output):
        original_read = Path.read_bytes

        def only_metadata(path):
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                raise AssertionError(f'Unexpected image-content read: {path}')
            return original_read(path)

        with patch.object(r.Image, 'open', side_effect=AssertionError('RGB decoding forbidden')), \
                patch.object(Path, 'read_bytes', only_metadata), \
                patch.object(r, '_full_scan', side_effect=AssertionError('Full scan forbidden')):
            return r.scan_dataset(cfg, output)

    def test_reuses_paths_and_cohorts_without_decoding_hashing_or_resplitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, dataset, source, frames = self.make_fixture(Path(tmp))
            index = self.load_without_image_reads(cfg, output)
            self.assertEqual(len(index), 84)
            self.assertEqual([Path(p).relative_to(dataset).as_posix() for p in index.path],
                             index.relative_path.tolist())
            seed_out = output / 'seed42'
            seed_out.mkdir()
            with patch.object(r, '_raw_splits', side_effect=AssertionError('Re-split forbidden')):
                reused = r.make_splits(index, cfg, 42, seed_out)
            for part, frame in zip(r.MANIFEST_PARTITIONS, reused):
                self.assertEqual(frame.sample_id.tolist(), frames[part].sample_id.tolist())
                self.assertEqual(frame.file_hash.tolist(), frames[part].file_hash.tolist())
                self.assertEqual(frame.label.tolist(), frames[part].label.tolist())
            plan = pd.read_csv(seed_out / 'validation_processing_plan.csv')
            self.assertEqual(plan.sample_id.tolist(), frames['validation'].sample_id.tolist())
            provenance = json.loads((output / 'manifest_reuse.json').read_text())
            self.assertEqual(provenance['manifest_root'], str(source.resolve()))
            self.assertIn('image_decode', provenance['skipped'])
            self.assertTrue((output / 'reused_manifest_index.csv').is_file())
            again = self.load_without_image_reads(cfg, output)
            self.assertEqual(again.sample_id.tolist(), index.sample_id.tolist())

    def test_explicit_seed_folder_and_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, dataset, source, _ = self.make_fixture(Path(tmp))
            cfg.update(manifest_root=str(source / 'seed42'), dataset_root=str(dataset))
            self.assertEqual(len(self.load_without_image_reads(cfg, output)), 84)

    def test_missing_input_and_ambiguous_roots_never_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, source, _ = self.make_fixture(Path(tmp))
            cfg['manifest_root'] = str(source / 'does-not-exist')
            with self.assertRaisesRegex(FileNotFoundError, 'Missing saved'):
                self.load_without_image_reads(cfg, output)
            cfg['manifest_root'] = None
            duplicate = Path(cfg['input_root']) / 'duplicate' / 'seed42'
            duplicate.mkdir(parents=True)
            for part in r.MANIFEST_PARTITIONS:
                (duplicate / f'{part}_manifest.csv').write_bytes((source / 'seed42' / f'{part}_manifest.csv').read_bytes())
            with self.assertRaisesRegex(ValueError, 'manifest_root explicitly'):
                self.load_without_image_reads(cfg, output)

    def test_explicit_root_wins_when_manifest_and_checkpoint_are_both_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, source, _ = self.make_fixture(Path(tmp))
            dedicated = Path(cfg['input_root']) / 'datasets/hadinho0710/mainfest-train-genimage'
            seed_folder = dedicated / 'seed42'
            seed_folder.mkdir(parents=True)
            for part in r.MANIFEST_PARTITIONS:
                name = f'{part}_manifest.csv'
                (seed_folder / name).write_bytes((source / 'seed42' / name).read_bytes())
            cfg['manifest_root'] = str(dedicated)
            # The checkpoint/old_run remains attached with all three manifests too.
            index = self.load_without_image_reads(cfg, output)
            self.assertEqual(len(index), 84)
            provenance = json.loads((output / 'manifest_reuse.json').read_text())
            self.assertEqual(provenance['manifest_root'], str(dedicated.resolve()))
            self.assertTrue(all(str(dedicated.resolve()) in p['path']
                                for p in provenance['sources']['42'].values()))

    def test_missing_or_changed_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, dataset, _, frames = self.make_fixture(Path(tmp))
            target = dataset / frames['train'].relative_path.iloc[0]
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Missing or changed file'):
                self.load_without_image_reads(cfg, output)

    def test_same_size_change_is_explicitly_not_claimed_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, dataset, _, frames = self.make_fixture(Path(tmp))
            target = dataset / frames['train'].relative_path.iloc[0]
            target.write_bytes(b'X' * target.stat().st_size)
            self.assertEqual(len(self.load_without_image_reads(cfg, output)), 84)
            provenance = json.loads((output / 'manifest_reuse.json').read_text())
            self.assertIn('Same-size content changes are NOT detected', provenance['trust'])

    def test_manifest_path_label_schema_and_summary_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _, _, source, frames = self.make_fixture(Path(tmp))
            frame = frames['train']
            variants = [frame.drop(columns='pixel_hash'), frame.assign(label=2),
                        frame.assign(relative_path='../outside.png'), frame.assign(width=0),
                        frame.assign(file_bytes=1.5), frame.assign(split='test'),
                        frame.assign(sample_id='bad'), frame.assign(label=1-frame.label)]
            for bad in variants:
                with self.subTest(columns=list(bad)), self.assertRaises(ValueError):
                    r.validate_manifest_frame(bad, cfg, 'train')
            frame.iloc[::-1].to_csv(source / 'seed42' / 'train_manifest.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'split_summary'):
                r.read_manifest_cohorts(source, cfg, 42)

    def test_saved_pixel_overlap_rejected_without_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, source, frames = self.make_fixture(Path(tmp))
            frame = frames['validation'].copy()
            # Same label preserves the within-cohort checks, but still leaks train pixels.
            target = frame.index[frame.label == frames['train'].label.iloc[0]][0]
            frame.loc[target, 'pixel_hash'] = frames['train'].pixel_hash.iloc[0]
            frame.to_csv(source / 'seed42' / 'validation_manifest.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'pixel hashes overlap'):
                self.load_without_image_reads(cfg, output)

    def test_smoke_subsets_existing_cohorts_without_new_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, _, frames = self.make_fixture(Path(tmp))
            cfg.update(smoke=True, smoke_train_per_class=1, smoke_eval_per_class=1)
            index = self.load_without_image_reads(cfg, output)
            seed_out = output / 'seed42'
            seed_out.mkdir()
            reused = r.make_splits(index, cfg, 42, seed_out)
            for part, frame in zip(r.MANIFEST_PARTITIONS, reused):
                self.assertEqual(len(frame), 14)
                self.assertTrue(set(frame.sample_id) <= set(frames[part].sample_id))
                self.assertTrue((frame.groupby(['generator', 'label']).size() == 1).all())

    def test_manifest_edit_on_resume_requires_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, source, frames = self.make_fixture(Path(tmp))
            self.load_without_image_reads(cfg, output)
            changed = frames['train'].assign(width=44)
            changed.to_csv(source / 'seed42' / 'train_manifest.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                self.load_without_image_reads(cfg, output)

    def test_full_scan_is_only_used_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, _, _ = self.make_fixture(Path(tmp))
            with patch.object(r, '_full_scan', return_value='full') as full:
                self.assertEqual(r.scan_dataset(dict(cfg, data_mode='full_scan'), output), 'full')
                full.assert_called_once()

    def test_multiple_seed_ordering_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, source, frames = self.make_fixture(Path(tmp))
            second = source / 'seed43'
            second.mkdir()
            summary = {}
            for part, frame in frames.items():
                reverse = frame.iloc[::-1]
                reverse.to_csv(second / f'{part}_manifest.csv', index=False)
                summary[part] = len(reverse)
                summary[part + '_hash'] = r.digest(reverse.sample_id.tolist())
            r.dump(second / 'split_summary.json', summary)
            cfg['seeds'] = [42, 43]
            index = self.load_without_image_reads(cfg, output)
            self.assertEqual(len(index), 84)  # metadata index unions IDs, not duplicate rows
            out = output / 'seed43'
            out.mkdir()
            reused = r.make_splits(index, cfg, 43, out)
            for part, frame in zip(r.MANIFEST_PARTITIONS, reused):
                self.assertEqual(frame.sample_id.tolist(), frames[part].sample_id.tolist()[::-1])

    def test_manifest_package_only_contains_required_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, output, _, source, _ = self.make_fixture(Path(tmp))
            weights = source / 'seed42' / 'clip_gsd'
            weights.mkdir()
            (weights / 'best.pt').write_bytes(b'not-in-bundle')
            destination = package_manifests(source, output / 'manifest-only.zip')
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()), {
                    'config.json', 'seed42/train_manifest.csv', 'seed42/validation_manifest.csv',
                    'seed42/test_manifest.csv', 'seed42/split_summary.json'})
                self.assertIsNone(archive.testzip())
            with self.assertRaises(FileExistsError):
                package_manifests(source, destination)

    def test_training_driver_uses_reused_cohorts_without_gpu_or_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output, _, _, frames = self.make_fixture(Path(tmp))
            cfg.update(output_root=str(output), precision='fp32')
            # Exercise the actual inherited run/scan/split/export plumbing, replacing only
            # CUDA training/inference. Other tests exercise real CPU training separately.
            model = r.nn.Linear(1, 2)
            model.vision = type('VisionMetadata', (), {'config': 'offline test'})()
            def probabilities(model, anchor, frame, config, seed, device, case):
                return np.where(frame.label == 1, .8, .2), np.arange(len(frame)) // config['test_batch']
            with patch.object(r.torch.cuda, 'is_available', return_value=True), \
                    patch.object(r.torch.cuda, 'get_device_name', return_value='mock-only-no-CUDA'), \
                    patch.object(r.torch.cuda, 'empty_cache'), patch.object(r, 'seed_all'), \
                    patch.object(r, 'build_models', return_value=(model, None)), \
                    patch.object(r, 'train_model') as train, \
                    patch.object(r, 'predict', side_effect=probabilities), \
                    patch.object(r, '_full_scan', side_effect=AssertionError('Full scan forbidden')), \
                    patch.object(r.Image, 'open', side_effect=AssertionError('Image decode forbidden')):
                metrics = r.run(cfg)
            args = train.call_args.args
            self.assertEqual(args[2].sample_id.tolist(), frames['train'].sample_id.tolist())
            self.assertEqual(args[3].sample_id.tolist(), frames['validation'].sample_id.tolist())
            self.assertTrue((output / 'completion.json').is_file())
            self.assertTrue((output / 'token_census.csv').is_file())
            self.assertTrue((output / 'seed42' / 'validation_processing_plan.csv').is_file())
            self.assertEqual(len(metrics), 8)


if __name__ == '__main__':
    unittest.main()

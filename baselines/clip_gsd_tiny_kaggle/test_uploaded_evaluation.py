"""CPU-only tests: uploaded checkpoint containers, exact-batch cache and graceful sessions."""
import ast
import copy
import hashlib
import json
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

import evaluate_uploaded_best as r
from build_uploaded_evaluation_notebook import build_notebook, NOTEBOOK


class UploadedEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.cfg = dict(r.DEFAULT_CONFIG, precision='fp32', k=3, micro_batch=1, anchor_micro_batch=1,
                        num_workers=0, train_generators=r.GENS7, expected_generators=r.GENS7,
                        source_generator='all7', train_batch=8, test_batch=2,
                        input_mode='center_crop_max_edge_pad14', crop_max_side=560,
                        gradient_checkpointing=False, smoke=False)
        config = r.CLIPVisionConfig(image_size=28, patch_size=14, hidden_size=16,
            intermediate_size=32, num_hidden_layers=6, num_attention_heads=2, attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        self.vision = r.CLIPVisionModel(config).vision_model
        self.opts = dict(r.UPLOAD_DEFAULTS, expected_signature='unit-signature',
                         save_every_batches=1, log_every_batches=1)

    def model(self, cfg=None):
        model = r.NativeDetector(copy.deepcopy(self.vision), cfg or self.cfg, True).requires_grad_(False).eval()
        anchor = copy.deepcopy(self.vision).requires_grad_(False).eval()
        return model, anchor

    def payload(self):
        cfg = dict(self.cfg, test_batch=8, jpeg_probability=.5, blur_probability=.5,
                   horizontal_flip_probability=0.)
        model, _ = self.model(cfg)
        return dict(model=model.state_dict(), config=cfg, epoch=0,
                    signature='unit-signature', protocol='unit', validation={}, val_loss=.1)

    def images(self, root, n=6):
        rows = []
        for i in range(n):
            relative = f'imagenet_glide/val/{"ai" if i % 2 else "nature"}/{i}.png'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = 28 + 14*(i % 2), 28 + 14*(i % 3)
            image = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            rows.append(dict(path=str(path), relative_path=relative, generator='glide', label=i % 2,
                             split='test', sample_id=hashlib.sha256(relative.encode()).hexdigest(),
                             pixel_hash=hashlib.sha256(('pixel' + relative).encode()).hexdigest(),
                             file_hash=r.file_sha(path), file_bytes=path.stat().st_size, width=w, height=h))
        return pd.DataFrame(rows)

    def test_native_torch_zip_extension_outer_archive_and_unpacked_internals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / 'best.zip'
            payload = self.payload()
            torch.save(payload, native)
            self.assertEqual(r.materialize_checkpoint(native, root/'cache'), native)
            self.assertEqual(r.restricted_checkpoint_load(native)['signature'], 'unit-signature')
            with zipfile.ZipFile(native) as archive:
                names = archive.namelist()
                for item in archive.infolist():
                    target = root / 'unpacked' / item.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(item))
                prefix = r.pytorch_zip_prefix(names)
            folder = root / 'unpacked' / prefix
            restored = r.materialize_checkpoint(folder, root/'cache')
            first_bytes = restored.read_bytes()
            loaded = r.restricted_checkpoint_load(restored)
            r.materialize_checkpoint(folder, root/'cache')
            self.assertEqual(first_bytes, restored.read_bytes())
            for key, value in payload['model'].items():
                torch.testing.assert_close(value, loaded['model'][key], rtol=0, atol=0)
            outer = root / 'outer.zip'
            with zipfile.ZipFile(outer, 'w') as archive:
                archive.write(native, 'some/folder/best.pt')
            extracted = r.materialize_checkpoint(outer, root/'cache')
            self.assertEqual(extracted.read_bytes(), native.read_bytes())
            self.assertEqual(r.validate_uploaded_payload(loaded, self.opts)['jpeg_probability'], .5)

    def test_payload_signature_epoch_and_container_guards(self):
        payload = self.payload()
        for bad in (dict(payload, signature='old-raw-model'), dict(payload, epoch=2),
                    dict(payload, config=dict(payload['config'], smoke=True)),
                    dict(payload, config=dict(payload['config'], test_batch=4))):
            with self.assertRaises(ValueError):
                r.validate_uploaded_payload(bad, self.opts)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'outer.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('readme.txt', 'No model here')
            with self.assertRaisesRegex(ValueError, 'exactly one best.pt'):
                r.materialize_checkpoint(path, root/'cache')

    def test_discovery_selects_new_signature_not_raw_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root/'old', root/'new'
            old.mkdir(); new.mkdir()
            payload = self.payload()
            torch.save(dict(payload, signature='old-raw'), old/'best.pt')
            torch.save(payload, new/'best.zip')
            source, _, loaded = r.find_uploaded_checkpoint(dict(self.opts, input_root=str(root), checkpoint_cache=str(root/'cache')))
            self.assertEqual(source, new/'best.zip')
            self.assertEqual(loaded['epoch'], 0)

    def test_case_aliases_do_not_merge_fake_and_all_jpeg(self):
        plan = r.plan_uploaded_cases(['native', 'png_roundtrip', 'all_jpeg96', 'all_jpeg096',
                                     'fake_jpeg96', 'native'])
        self.assertEqual(plan['cases'], ['native', 'all_jpeg96', 'fake_jpeg96'])
        self.assertEqual(plan['aliases_not_measured'], {'png_roundtrip':'native', 'all_jpeg096':'all_jpeg96'})
        self.assertEqual(plan['duplicate_requests'], ['native'])
        for update in (dict(time_budget_hours=12), dict(max_new_cases_per_run=0), dict(save_every_batches=0)):
            with self.assertRaises(ValueError):
                r.validate_upload_options(dict(self.opts, **update))

    def test_budget_flush_resume_raw_logits_and_cached_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.images(root/'input')
            model, anchor = self.model()
            output = root/'out'
            output.mkdir()
            clock = [0.]
            saved = []
            def callback(case, n):
                saved.append(n)
                clock[0] = 11.  # stop after one completed persisted logical batch
            with patch.object(r.time, 'monotonic', side_effect=lambda: clock[0]):
                partial, complete = r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts,
                    'native', output, 'fp', torch.device('cpu'), deadline=10., callback=callback)
            self.assertFalse(complete)
            self.assertEqual(len(partial), 2)
            self.assertEqual(saved, [2])
            with patch.object(r, 'estimate_bases', wraps=r.estimate_bases) as basis:
                result, complete = r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts,
                    'native', output, 'fp', torch.device('cpu'), deadline=float('inf'))
                self.assertEqual(basis.call_count, 2)
            self.assertTrue(complete)
            self.assertEqual(result.batch_id.tolist(), [0,0,1,1,2,2])
            self.assertEqual(result.sample_id.tolist(), frame.sample_id.tolist())
            probabilities, _ = r.predict(model, anchor, frame, self.cfg, 42, torch.device('cpu'), 'native')
            np.testing.assert_allclose(result.prob_fake, probabilities, atol=1e-7)
            np.testing.assert_allclose(result.logit_fake-result.logit_real, result.logit_margin, atol=1e-6)
            from_margin = torch.sigmoid(torch.tensor(result.logit_margin.to_numpy())).numpy()
            np.testing.assert_allclose(result.prob_fake, from_margin, atol=1e-7)
            with patch.object(r, 'estimate_bases', side_effect=AssertionError('must reuse complete case')):
                cached, done = r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts,
                    'native', output, 'fp', torch.device('cpu'), deadline=float('inf'))
            self.assertTrue(done)
            self.assertEqual(len(cached), 6)
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts,
                    'native', output, 'different', torch.device('cpu'), deadline=float('inf'))

    def test_pending_batches_flush_at_budget_without_full_save_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.images(root/'input')
            model, anchor = self.model()
            out = root/'out'; out.mkdir()
            original = r.estimate_bases
            expired = [False]
            def one_basis(*args, **kwargs):
                value = original(*args, **kwargs)
                expired[0] = True
                return value
            with patch.object(r, 'estimate_bases', side_effect=one_basis), \
                    patch.object(r.time, 'monotonic', side_effect=lambda: 11. if expired[0] else 0.):
                partial, complete = r.evaluate_uploaded_case(model, anchor, frame, self.cfg,
                    dict(self.opts, save_every_batches=10), 'native', out, 'fp', torch.device('cpu'), 10.)
            self.assertFalse(complete)
            self.assertEqual(len(partial), 2)
            self.assertEqual(len(list((out/'progress/native').glob('part_*.csv'))), 1)

    def test_auto_restore_is_fingerprint_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root/'input/previous'
            progress = previous/'progress/native'
            progress.mkdir(parents=True)
            (progress/'part_0000000.csv').write_text('example\n1\n')
            r.atomic_json(previous/'evaluation_state.json', dict(evaluator=r.UPLOAD_EVALUATOR,
                          fingerprint='same', saved_rows={'native':2}))
            out = root/'out'; out.mkdir()
            options = dict(self.opts, input_root=str(root/'input'))
            self.assertEqual(r.restore_uploaded_progress(out, options, 'same'), previous)
            self.assertTrue((out/'progress/native/part_0000000.csv').is_file())
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                r.restore_uploaded_progress(root/'other', dict(options, resume_from=str(previous)), 'wrong')

    def test_progress_without_metadata_or_with_changed_scores_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.images(root/'input', 2)
            model, anchor = self.model()
            out = root/'out'; out.mkdir()
            r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts, 'native', out,
                                      'fp', torch.device('cpu'), float('inf'))
            folder = out/'progress/native'
            part = next(folder.glob('part_*.csv'))
            corrupted = pd.read_csv(part)
            corrupted['prob_fake'] = .12345
            corrupted.to_csv(part, index=False)
            with self.assertRaisesRegex(ValueError, 'scores do not match raw logits'):
                r.read_uploaded_case(folder, frame, self.cfg, 'native')
            (folder/'metadata.json').unlink()
            with self.assertRaisesRegex(ValueError, 'no case fingerprint'):
                r.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.opts, 'native', out,
                                          'fp', torch.device('cpu'), float('inf'))

    def test_test_manifest_matches_real_saved_cohort_without_reading_images(self):
        root = Path(__file__).resolve().parents[2]/'reports/gsd_all7_crop560_analysis/source/seed42'
        if not (root/'test_manifest.csv').is_file():
            self.skipTest('Historical local metadata not available')
        with patch.object(r.Image, 'open', side_effect=AssertionError('must not read images')):
            path, frame, identity = r.choose_test_manifest(dict(self.cfg, manifest_root=None),
                dict(self.opts, test_manifest_path=str(root/'test_manifest.csv')))
        self.assertEqual(len(frame), 7000)
        self.assertEqual(len(identity), 64)
        self.assertEqual(path, root/'test_manifest.csv')

    def test_three_bounded_sessions_resume_without_rerunning_completed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'input'
            frame = self.images(inputs/'tiny-genimage')
            checkpoint = inputs/'uploaded/best.pt'
            checkpoint.parent.mkdir()
            payload = self.payload()
            torch.save(payload, checkpoint)
            manifest = root/'test_manifest.csv'; frame.to_csv(manifest, index=False)
            real_case = r.evaluate_uploaded_case
            cpu = torch.device('cpu')
            def cpu_case(*args, **kwargs):
                values = list(args)
                values[8] = cpu
                return real_case(*values, **kwargs)
            options = dict(self.opts, input_root=str(inputs), checkpoint_path=str(checkpoint),
                test_manifest_path=str(manifest), checkpoint_cache=str(root/'cache'),
                cases=['native','png_roundtrip','all_jpeg96','blur_sigma1'], max_new_cases_per_run=1)
            def build(cfg, device, gsd):
                return self.model(cfg)
            with patch.object(r, 'choose_test_manifest', return_value=(manifest, frame, 'unit-cohort')), \
                    patch.object(r, 'build_models', side_effect=build), \
                    patch.object(r, 'evaluate_uploaded_case', side_effect=cpu_case), \
                    patch.object(r.torch.cuda, 'is_available', return_value=True), \
                    patch.object(r.torch.cuda, 'get_device_name', return_value='mock-GPU'), \
                    patch.object(r.torch.cuda, 'empty_cache'), patch.object(r, 'seed_all'):
                for number in range(3):
                    out = root/f'out{number}'
                    with patch.object(r, 'estimate_bases', wraps=r.estimate_bases) as basis:
                        table = r.run_uploaded_evaluation(dict(options, output_root=str(out)))
                        self.assertEqual(basis.call_count, 1)  # one new case, batch8, six test images
                    state = json.loads((out/'evaluation_state.json').read_text())
                    self.assertEqual(len(state['completed_cases']), number+1)
                    self.assertEqual(state['status'], 'complete' if number == 2 else 'paused')
                    self.assertEqual(len(table[table.generator=='pooled']), number+1)
                    shutil.copytree(out, inputs/f'previous{number}')
            exported = pd.read_csv(root/'out2/predictions.csv')
            self.assertEqual(len(exported), 18)
            self.assertEqual(set(exported.case), {'native','all_jpeg96','blur_sigma1'})
            self.assertTrue({'logit_real','logit_fake','logit_margin'} <= set(exported))

    def test_notebook_is_self_contained_and_only_invokes_evaluator(self):
        notebook = build_notebook()
        source = ''.join(notebook['cells'][3]['source'])
        ast.parse(source)
        namespace = {'__name__':'uploaded_notebook_runtime'}
        exec(compile(source, '<notebook-runtime>', 'exec'), namespace)
        exec(''.join(notebook['cells'][4]['source']), namespace)
        options = namespace['EVAL_CONFIG']
        self.assertIsNone(options['max_new_cases_per_run'])
        self.assertEqual(options['time_budget_hours'], 9.5)
        self.assertFalse(namespace['SMOKE_ONLY'])
        self.assertEqual(options['cases'], list(r.UPLOADED_FULL_SUITE_CASES))
        self.assertNotIn('png_roundtrip', options['cases'])
        launch = ''.join(notebook['cells'][5]['source'])
        self.assertIn('finalize_uploaded_notebook_config', launch)
        captured = []
        namespace['run_uploaded_evaluation'] = lambda config: captured.append(config)
        # Reproduce the actual native-only stale config observed in the user's Kaggle output.
        namespace['EVAL_CONFIG'].update(cases=['native'], max_new_cases_per_run=1, max_batches=2)
        exec(launch, namespace)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]['cases'], list(r.UPLOADED_FULL_SUITE_CASES))
        self.assertIsNone(captured[0]['max_batches'])
        self.assertIsNone(captured[0]['max_new_cases_per_run'])
        self.assertEqual(json.loads((Path(__file__).parent/NOTEBOOK).read_text(encoding='utf-8')), notebook)

    def test_full_suite_guard_requires_explicit_smoke_mode(self):
        stale = dict(r.UPLOAD_DEFAULTS, cases=['native'], max_batches=2, max_new_cases_per_run=1)
        full = r.finalize_uploaded_notebook_config(stale)
        self.assertEqual(full['cases'], list(r.UPLOADED_FULL_SUITE_CASES))
        self.assertEqual(len(full['cases']), 9)
        self.assertIsNone(full['max_batches'])
        self.assertIsNone(full['max_new_cases_per_run'])
        self.assertEqual(full['time_budget_hours'], 9.5)
        self.assertEqual(stale['cases'], ['native'])  # caller's dictionary not mutated
        smoke = r.finalize_uploaded_notebook_config(stale, smoke_only=True)
        self.assertEqual(smoke['cases'], ['native', 'fake_jpeg96'])
        self.assertEqual(smoke['max_batches'], 2)
        self.assertNotEqual(smoke['output_root'], full['output_root'])
        self.assertIsNone(smoke['resume_from'])
        with self.assertRaises(ValueError):
            r.finalize_uploaded_notebook_config(stale, smoke_only=1)

    def test_guarded_full_suite_reaches_all_nine_cases_in_one_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.images(root/'input/tiny', 6)
            checkpoint = root/'input/checkpoint/best.pt'
            checkpoint.parent.mkdir(parents=True)
            torch.save(self.payload(), checkpoint)
            manifest = root/'test_manifest.csv'
            frame.to_csv(manifest, index=False)
            options = r.finalize_uploaded_notebook_config(dict(self.opts,
                input_root=str(root/'input'), checkpoint_path=str(checkpoint),
                test_manifest_path=str(manifest), output_root=str(root/'out'),
                checkpoint_cache=str(root/'cache'), resume_from=None,
                cases=['native'], max_batches=2, max_new_cases_per_run=1))
            called = []
            def synthetic_case(model, anchor, cohort, cfg, opts, name, out, fingerprint, device, deadline, callback):
                # Test orchestration only; actual transforms/model inference have separate tests.
                called.append(name)
                result = cohort.copy()
                margin = np.where(result.label.to_numpy() == 1, 2., -2.)
                result = result.assign(case=name, model='clip_gsd', seed=42,
                    prob_fake=1/(1+np.exp(-margin)), logit_real=0., logit_fake=margin,
                    logit_margin=margin, batch_id=np.arange(len(cohort))//cfg['test_batch'], input_tokens=5)
                callback(name, len(result))
                return result, True
            with patch.object(r, 'choose_test_manifest', return_value=(manifest, frame, 'unit-cohort')), \
                    patch.object(r, 'build_models', side_effect=lambda cfg, device, gsd:self.model(cfg)), \
                    patch.object(r, 'evaluate_uploaded_case', side_effect=synthetic_case), \
                    patch.object(r.torch.cuda, 'is_available', return_value=True), \
                    patch.object(r.torch.cuda, 'get_device_name', return_value='mock-GPU'), \
                    patch.object(r.torch.cuda, 'empty_cache'), patch.object(r, 'seed_all'):
                table = r.run_uploaded_evaluation(options)
            self.assertEqual(called, list(r.UPLOADED_FULL_SUITE_CASES))
            self.assertEqual(len(table[table.generator == 'pooled']), 9)
            state = json.loads((root/'out/evaluation_state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['remaining_cases'], [])
            self.assertEqual(len(state['completed_cases']), 9)


if __name__ == '__main__':
    unittest.main()

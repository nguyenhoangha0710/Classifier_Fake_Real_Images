"""Offline CPU checks for blur/JPEG and deterministic two-view validation."""
import ast
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

import runtime_aug_all7 as r
import runtime_raw_all7 as raw
from build_aug_all7_notebook import build_notebook, NOTEBOOK


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.cfg = dict(r.DEFAULT_CONFIG, precision='fp32', k=3, train_batch=2, test_batch=2,
                        micro_batch=1, anchor_micro_batch=1, num_workers=0,
                        crop_max_side=28, epochs=3, patience=1)
        conf = r.CLIPVisionConfig(image_size=28, patch_size=14, hidden_size=16,
                                 intermediate_size=32, num_hidden_layers=6,
                                 num_attention_heads=2, attention_dropout=0.0)
        conf._attn_implementation = 'sdpa'
        self.vision = r.CLIPVisionModel(conf).vision_model

    def make_images(self, folder, n=4):
        rows = []
        for i in range(n):
            path = folder / f'image{i}.png'
            data = np.random.default_rng(i).integers(0, 256, (51, 65, 3), dtype=np.uint8)
            Image.fromarray(data).save(path)
            rows.append(dict(path=str(path), sample_id=f'id{i}', label=i % 2,
                             generator='glide', pixel_hash=f'pixel{i}'))
        return pd.DataFrame(rows)

    def tiny_model(self):
        model = r.NativeDetector(copy.deepcopy(self.vision), self.cfg, True)
        anchor = copy.deepcopy(self.vision).requires_grad_(False).eval()
        return model, anchor

    def test_default_config_and_raw_protocol_are_separate(self):
        r.validate_raw_config(self.cfg)
        raw.validate_raw_config(dict(raw.DEFAULT_CONFIG))
        self.assertEqual(raw.DEFAULT_CONFIG['jpeg_probability'], 0.0)
        self.assertEqual(raw.DEFAULT_CONFIG['blur_probability'], 0.0)
        self.assertNotEqual(r.protocol_id(self.cfg), raw.protocol_id(self.cfg))
        self.assertNotEqual(r.DEFAULT_CONFIG['output_root'], raw.DEFAULT_CONFIG['output_root'])

    def test_config_rejects_invalid_or_extra_augmentations(self):
        invalid = [dict(jpeg_probability=float('nan')), dict(blur_probability=1.1),
                   dict(jpeg_train_quality=[100, 75]), dict(jpeg_train_quality=[75.5, 100]),
                   dict(blur_sigma=[2, 0]), dict(blur_sigma=[0, float('inf')]),
                   dict(horizontal_flip_probability=.1), dict(jpeg_subsampling=2),
                   dict(validation_processing_seed=True), dict(validation_monitor='native'),
                   dict(input_mode='native_rgb_edge_pad14'), dict(crop_max_side=556)]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                r.validate_raw_config(dict(self.cfg, **update))

    def test_rng_is_local_epoch_aware_and_validation_fixed(self):
        random.seed(123)
        state = random.getstate()
        first = [r.processing_parameters(str(i), self.cfg, 42, train=True, epoch=0) for i in range(20)]
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, [r.processing_parameters(str(i), self.cfg, 42, train=True, epoch=0) for i in range(20)])
        self.assertNotEqual(first, [r.processing_parameters(str(i), self.cfg, 42, train=True, epoch=1) for i in range(20)])
        fixed = r.processing_parameters('one', self.cfg, 42, epoch=0)
        self.assertEqual(fixed, r.processing_parameters('one', self.cfg, 42, epoch=99))
        other = dict(self.cfg, validation_processing_seed=43)
        fixed_many = [r.processing_parameters(str(i), self.cfg, 42) for i in range(20)]
        self.assertNotEqual(fixed_many, [r.processing_parameters(str(i), other, 42) for i in range(20)])

    def test_independent_gates_ranges_and_expected_mix(self):
        params = [r.processing_parameters(str(i), self.cfg, 42, train=True) for i in range(4000)]
        counts = {(a, b): 0 for a in (False, True) for b in (False, True)}
        for p in params:
            counts[p['blur_applied'], p['jpeg_applied']] += 1
            if p['blur_applied']:
                self.assertTrue(0 <= p['blur_sigma'] <= 2)
            else:
                self.assertIsNone(p['blur_sigma'])
            if p['jpeg_applied']:
                self.assertIsInstance(p['jpeg_quality'], int)
                self.assertTrue(75 <= p['jpeg_quality'] <= 100)
            else:
                self.assertIsNone(p['jpeg_quality'])
        for count in counts.values():
            self.assertTrue(850 < count < 1150, counts)

    def test_label_generator_and_row_order_do_not_control_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.make_images(Path(tmp))
            for train, case in ((True, 'native'), (False, 'processed')):
                original = r.Images(frame, self.cfg, 42, train=train, epoch=2, case=case)
                changed = r.Images(frame.assign(label=1-frame.label, generator='sd15'), self.cfg,
                                   42, train=train, epoch=2, case=case)
                reordered = r.Images(frame.iloc[::-1], self.cfg, 42, train=train, epoch=2, case=case)
                for i in range(len(frame)):
                    torch.testing.assert_close(original[i][0], changed[i][0], rtol=0, atol=0)
                    torch.testing.assert_close(original[i][0], reordered[len(frame)-1-i][0], rtol=0, atol=0)

    def test_processed_pixels_equal_blur_then_jpeg_before_crop(self):
        cfg = dict(self.cfg, blur_probability=1.0, blur_sigma=[.7, .7],
                   jpeg_probability=1.0, jpeg_train_quality=[88, 88])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = self.make_images(folder, 1)
            before = Path(frame.path.iloc[0]).read_bytes()
            with Image.open(frame.path.iloc[0]) as im:
                expected = r.jpeg(im.convert('RGB').filter(ImageFilter.GaussianBlur(.7)), 88, 0)
            processed = folder / 'expected.png'
            expected.save(processed)
            reference = frame.assign(path=str(processed))
            want = raw.Images(reference, cfg, 42)[0][0]
            got = r.Images(frame, cfg, 42, train=True)[0][0]
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            # Train and validation with forced parameters coincide, independent of seed/epoch.
            torch.testing.assert_close(got, r.Images(frame, cfg, 99, epoch=8, case='processed')[0][0], rtol=0, atol=0)
            self.assertEqual(before, Path(frame.path.iloc[0]).read_bytes())

    def test_native_is_unchanged_and_processed_is_epoch_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.make_images(Path(tmp))
            for i in range(len(frame)):
                want = raw.Images(frame, self.cfg, 42)[i][0]
                got = r.Images(frame, self.cfg, 42)[i][0]
                torch.testing.assert_close(got, want, rtol=0, atol=0)
                a = r.Images(frame, self.cfg, 42, epoch=0, case='processed')[i][0]
                b = r.Images(frame, self.cfg, 42, epoch=9, case='processed')[i][0]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(tuple(a.shape), (3, 28, 28))
            zero = dict(self.cfg, blur_probability=0.0, jpeg_probability=0.0)
            torch.testing.assert_close(r.Images(frame, zero, 42, train=True)[0][0],
                                       raw.Images(frame, self.cfg, 42)[0][0], rtol=0, atol=0)

    def test_multiworker_loading_does_not_change_validation(self):
        # num_workers=0 remains the Kaggle default; also exercise a spawn worker on Windows.
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.make_images(Path(tmp), 3)
            one = list(r.loader(frame, self.cfg, 42, case='processed'))
            worker_cfg = dict(self.cfg, num_workers=1)
            other = list(r.loader(frame, worker_cfg, 42, case='processed'))
            for (xs, ys, ids), (xx, yy, ii) in zip(one, other):
                torch.testing.assert_close(ys, yy)
                torch.testing.assert_close(ids, ii)
                for x, y in zip(xs, xx):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_validation_plan_has_stable_parameters_and_no_test_ids(self):
        rows = []
        for gen in r.GENS7:
            for split, n in (('train', 20), ('test', 6)):
                for label in (0, 1):
                    for i in range(n):
                        sid = f'{gen}-{split}-{label}-{i}'
                        rows.append(dict(sample_id=sid, pixel_hash=sid, generator=gen,
                                         split=split, label=label))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg = dict(self.cfg, data_mode='full_scan')
            train, val, test = r.make_splits(pd.DataFrame(rows), cfg, 42, folder)
            first = (folder / 'validation_processing_plan.csv').read_bytes()
            r.make_splits(pd.DataFrame(rows), cfg, 42, folder)
            self.assertEqual(first, (folder / 'validation_processing_plan.csv').read_bytes())
            plan = pd.read_csv(folder / 'validation_processing_plan.csv')
            self.assertEqual(plan.sample_id.tolist(), val.sample_id.tolist())
            self.assertFalse(set(plan.sample_id) & set(test.sample_id))
            self.assertFalse(set(train.pixel_hash) & set(val.pixel_hash))
            self.assertEqual(plan.batch_id.tolist(), (np.arange(len(val)) // 2).tolist())
            protocol = json.loads((folder / 'validation_protocol.json').read_text())
            self.assertEqual(protocol['weights'], dict(native=.5, processed=.5))

    def test_validation_ce_uses_logits_without_probability_clipping(self):
        class ExtremeModel(torch.nn.Module):
            def forward(self, images, bases):
                return images.new_tensor([[12.0, -12.0]]).expand(len(images), -1)

        with tempfile.TemporaryDirectory() as tmp:
            frame = self.make_images(Path(tmp))
            p, bids, loss = r.predict(ExtremeModel(), None, frame, self.cfg, 42,
                                      torch.device('cpu'), 'native', return_loss=True)
            expected = float(torch.nn.functional.cross_entropy(
                torch.tensor([[12., -12.]]).expand(4, -1), torch.tensor([0, 1, 0, 1])))
            self.assertAlmostEqual(loss, expected, places=6)
            clipped_loss = -np.mean(np.where(frame.label == 1,
                                   np.log(np.clip(p, 1e-7, 1-1e-7)), np.log(1-np.clip(p, 1e-7, 1-1e-7))))
            self.assertGreater(loss, clipped_loss + 3)
            self.assertEqual(bids.tolist(), [0, 0, 1, 1])

    def test_real_two_view_validation_and_frozen_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.make_images(Path(tmp), 5)
            model, anchor = self.tiny_model()
            before = copy.deepcopy(anchor.state_dict())
            monitor, views, predictions = r.evaluate_validation(model, anchor, frame, self.cfg,
                                                                42, torch.device('cpu'))
            self.assertAlmostEqual(monitor, (views['native']['loss']+views['processed']['loss'])/2)
            self.assertEqual(views['native']['n'], 5)
            self.assertEqual(views['processed']['n'], 5)
            self.assertEqual(predictions['native'][1].tolist(), [0, 0, 1, 1, 2])
            np.testing.assert_array_equal(predictions['native'][1], predictions['processed'][1])
            again = r.evaluate_validation(model, anchor, frame, self.cfg, 42, torch.device('cpu'))
            self.assertEqual(monitor, again[0])
            for key, value in anchor.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_one_epoch_integration_shares_inputs_and_exports_real_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = self.make_images(folder)
            model, anchor = self.tiny_model()
            cfg = dict(self.cfg, epochs=1)
            before = copy.deepcopy(model.head.state_dict())
            anchor_inputs, detector_inputs = [], []
            # Embeddings also run during backward recomputation only inside blocks, so these
            # hooks see exactly one input per image and branch in train and both val passes.
            first = anchor.embeddings.register_forward_pre_hook(
                lambda module, args: anchor_inputs.append(args[0].detach().cpu().clone()))
            second = model.vision.embeddings.register_forward_pre_hook(
                lambda module, args: detector_inputs.append(args[0].detach().cpu().clone()))
            r.train_model(model, anchor, frame, frame, cfg, 42, torch.device('cpu'), folder, 'one-epoch')
            first.remove()
            second.remove()
            self.assertEqual(len(anchor_inputs), 12)  # four train + four native + four processed
            self.assertEqual(len(anchor_inputs), len(detector_inputs))
            for a, b in zip(anchor_inputs, detector_inputs):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.head.state_dict().items()))
            history = json.loads((folder / 'history.json').read_text())
            expected, summaries, _ = r.evaluate_validation(model, anchor, frame, cfg, 42, torch.device('cpu'))
            self.assertAlmostEqual(history[0]['val_loss'], expected)
            self.assertEqual(history[0]['val_native'], summaries['native'])
            self.assertEqual(history[0]['val_processed'], summaries['processed'])
            self.assertTrue(all(p.grad is None for p in anchor.parameters()))

    def test_mean_monitor_selects_checkpoint_and_resume_preserves_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = self.make_images(folder)
            model, anchor = self.tiny_model()

            def evaluation(native_loss, processed_loss):
                p = np.array([.1, .9, .1, .9])
                bids = np.array([0, 0, 1, 1])
                metrics = r.metric(frame.label.to_numpy(), p, .5)
                summaries = dict(native=dict(loss=native_loss, **metrics),
                                 processed=dict(loss=processed_loss, **metrics))
                return (native_loss+processed_loss)/2, summaries, dict(native=(p, bids), processed=(p, bids))

            # Native-only would choose epoch1; mean(native,processed) must choose epoch2.
            evaluations = [evaluation(.2, 1.4), evaluation(.4, .2), evaluation(.4, .2)]
            with patch.object(r, 'evaluate_validation', side_effect=evaluations) as evaluate:
                r.train_model(model, anchor, frame, frame, self.cfg, 42, torch.device('cpu'), folder, 'aug-test')
                before = copy.deepcopy(model.state_dict())
                r.train_model(model, anchor, frame, frame, self.cfg, 42, torch.device('cpu'), folder, 'aug-test')
                self.assertEqual(evaluate.call_count, 3)
            selected = json.loads((folder / 'selected_checkpoint.json').read_text())
            self.assertEqual(selected['epoch'], 2)
            self.assertAlmostEqual(selected['val_loss'], .3)
            self.assertEqual(selected['criterion'], r.VALIDATION_MONITOR)
            history = json.loads((folder / 'history.json').read_text())
            self.assertTrue(history[-1]['early_stopping']['stopped'])
            self.assertEqual(len(history), 3)
            metrics = pd.read_csv(folder / 'validation_metrics.csv')
            self.assertEqual(len(metrics), 6)
            self.assertEqual(set(metrics.case), {'native', 'processed'})
            predictions = pd.read_csv(folder / 'validation_predictions_epoch002.csv')
            self.assertIn('prob_fake_processed', predictions)
            self.assertEqual(predictions.sample_id.tolist(), frame.sample_id.tolist())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in anchor.parameters()))
            with self.assertRaisesRegex(ValueError, 'Resume protocol/config/manifests'):
                r.train_model(model, anchor, frame, frame, dict(self.cfg, blur_probability=.1),
                              42, torch.device('cpu'), folder, 'aug-test')

    def test_reject_raw_checkpoint_even_with_same_supplied_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = self.make_images(folder)
            model, anchor = self.tiny_model()
            torch.save(dict(signature='same', model=model.state_dict(), config=self.cfg), folder / 'last.pt')
            with self.assertRaisesRegex(ValueError, 'Resume protocol/config/manifests'):
                r.train_model(model, anchor, frame, frame, self.cfg, 42, torch.device('cpu'), folder, 'same')

    def test_standalone_notebook_embeds_current_sources_and_correct_recipe(self):
        nb = build_notebook()
        runtime = ''.join(nb['cells'][3]['source'])
        ast.parse(runtime)
        # Execute the embedded runtime in a fresh namespace, with no local file dependency.
        namespace = {'__name__': 'notebook_runtime_test'}
        exec(compile(runtime, '<embedded-runtime>', 'exec'), namespace)
        config_source = ''.join(nb['cells'][4]['source'])
        exec(compile(config_source, '<config>', 'exec'), namespace)
        config = namespace['CONFIG']
        namespace['validate_raw_config'](config)
        self.assertEqual(config['jpeg_probability'], .5)
        self.assertEqual(config['blur_probability'], .5)
        self.assertEqual(config['horizontal_flip_probability'], 0)
        self.assertEqual(config['validation_monitor'], r.VALIDATION_MONITOR)
        self.assertEqual(config['crop_max_side'], 560)
        self.assertEqual(config['data_mode'], 'reuse_manifests')
        path = Path(__file__).parent / NOTEBOOK
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), nb)


if __name__ == '__main__':
    unittest.main()

"""CPU checks for GSD geometry, full logical batches, gradients and leakage."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel

from runtime import (DEFAULT_CONFIG, Detector, SemanticSuppression, top_basis,
                     estimate_bases, make_splits, process, metric, summarize, Images,
                     canonical_generator, train_model, scan_dataset, EarlyStopping)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.cfg = dict(DEFAULT_CONFIG, k=3, micro_batch=2, anchor_micro_batch=2,
                        gradient_checkpointing=False, precision='fp32', num_workers=0,
                        source_generator='sd14', expected_generators=None)
        conf = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                                num_attention_heads=2, image_size=8, patch_size=4, attention_dropout=0.)
        conf._attn_implementation = 'eager'
        self.v = CLIPVisionModel(conf).vision_model

    def test_projection_geometry_and_gradients(self):
        v = top_basis(torch.randn(20, 16), 3)
        f = torch.randn(3, 5, 16, requires_grad=True)
        g = SemanticSuppression()
        result = g(f, v)
        p = f @ v @ v.T
        r = p.square().sum((1, 2)) / f.square().sum((1, 2))
        expected = f - torch.sigmoid(2*r + 1)[:, None, None] * p
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.all((r >= 0) & (r <= 1)))
        result.square().sum().backward()
        for grad in (f.grad, g.a.grad, g.b.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_full_basis_preblock_and_chunk_invariance(self):
        x = torch.randn(4, 3, 8, 8)
        anchor = self.v.requires_grad_(False).eval()
        selected = tuple(range(2, 6))
        actual = estimate_bases(anchor, x, self.cfg, torch.device('cpu'), selected)
        h = anchor.pre_layrnorm(anchor.embeddings(x))
        for l, layer in enumerate(anchor.encoder.layers):
            if l in selected:
                v = top_basis(h[:, 1:].flatten(0, 1), 3)
                torch.testing.assert_close(actual[l] @ actual[l].T, v @ v.T, atol=2e-5, rtol=2e-5)
            h = layer(h, None, None)[0]
        other = estimate_bases(anchor, x, {**self.cfg, 'anchor_micro_batch': 1}, torch.device('cpu'), selected)
        for l in selected:
            torch.testing.assert_close(actual[l] @ actual[l].T, other[l] @ other[l].T, atol=3e-5, rtol=3e-5)

    def test_full_vs_microbatch_loss_gradients(self):
        model = Detector(copy.deepcopy(self.v), self.cfg, True).train()
        other = copy.deepcopy(model)
        x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 1, 0])
        anchor = copy.deepcopy(self.v).requires_grad_(False).eval()
        basis = estimate_bases(anchor, x, self.cfg, torch.device('cpu'), model.selected)
        full = model(x, basis)
        F.cross_entropy(full, y).backward()
        outputs = []
        for a in range(0, 4, 2):
            p = other(x[a:a+2], basis)
            outputs.append(p)
            (F.cross_entropy(p, y[a:a+2], reduction='sum') / 4).backward()
        torch.testing.assert_close(full, torch.cat(outputs), atol=2e-5, rtol=2e-5)
        for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=4e-5, rtol=2e-4)

    def test_checkpointed_block_gradients(self):
        m = Detector(copy.deepcopy(self.v), {**self.cfg, 'gradient_checkpointing': True}, False).train()
        m(torch.randn(2, 3, 8, 8)).sum().backward()
        self.assertIsNotNone(m.vision.encoder.layers[0].self_attn.q_proj.weight.grad)

    def test_plain_baseline_matches_backbone(self):
        model = Detector(self.v, self.cfg, False).eval()
        x = torch.randn(2, 3, 8, 8)
        hidden = self.v(x).last_hidden_state
        expected = model.head(self.v.post_layernorm(hidden)[:, 1:].mean(1))
        torch.testing.assert_close(model(x), expected)

    def test_pixel_leakage_and_balanced_cohort(self):
        rows = []
        for i in range(40):
            rows.append(dict(sample_id=f't{i}', label=i%2, generator='sd14', split='train', pixel_hash=f'p{i}'))
        for gen in ('sd14', 'biggan'):
            for i in range(20):
                rows.append(dict(sample_id=f'{gen}{i}', label=i%2, generator=gen, split='test', pixel_hash=f'e{i}'))
            rows.append(dict(sample_id=f'{gen}leak', label=0, generator=gen, split='test', pixel_hash='p0'))
        with tempfile.TemporaryDirectory() as tmp:
            tr, va, te = make_splits(pd.DataFrame(rows), self.cfg, 42, Path(tmp))
            self.assertFalse(set(tr.pixel_hash) & set(va.pixel_hash))
            self.assertFalse((set(tr.pixel_hash) | set(va.pixel_hash)) & set(te.pixel_hash))
            for _, group in te.groupby('generator'):
                self.assertEqual(sum(group.label == 0), sum(group.label == 1))

    def test_augmentation_pairing_and_eval_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'im.png'
            Image.fromarray(np.random.default_rng(3).integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(path)
            df = pd.DataFrame([dict(path=str(path), sample_id='abc', label=0)])
            a = Images(df, self.cfg, 42, train=True, epoch=1)[0][0]
            b = Images(df.assign(label=1), self.cfg, 42, train=True, epoch=1)[0][0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            image = Image.open(path).convert('RGB')
            np.testing.assert_array_equal(np.asarray(process(image, 'native', self.cfg)), np.asarray(image))
            self.assertEqual(process(image, 'size_jpeg96', self.cfg).size, (256, 256))
            self.assertFalse(np.array_equal(np.asarray(process(image, 'jpeg96', self.cfg)), np.asarray(image)))

    def test_metrics_and_paired_summary(self):
        rows = []
        for model in ('clip_ft', 'clip_gsd'):
            for case in ('native', 'jpeg96'):
                for gen in ('sd14', 'biggan'):
                    for label in (0, 1):
                        p = .2 if label == 0 else .8
                        rows.append(dict(seed=42, model=model, case=case, generator=gen, label=label,
                                         sample_id=f'{gen}{label}', prob_fake=p + (.05 if case != 'native' else 0), batch_id=0))
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize(pd.DataFrame(rows), self.cfg, Path(tmp))
            self.assertTrue((summary.auc == 1).all())
            paired = pd.read_csv(Path(tmp) / 'paired_summary.csv')
            np.testing.assert_allclose(paired.mean_delta, .05)
            self.assertTrue((paired.flip_rate == 0).all())
            self.assertTrue((Path(tmp) / 'macro_unseen.csv').exists())

    def test_seven_generator_sd15_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            cfg = dict(DEFAULT_CONFIG, dataset_root=str(root))
            for gi, gen in enumerate(cfg['expected_generators']):
                for si, split in enumerate(('train', 'val')):
                    for label, name in enumerate(('nature', 'ai')):
                        folder = root / gen / split / name
                        folder.mkdir(parents=True)
                        for i in range(10):
                            Image.new('RGB', (32, 40), (gi*20+si, label*100, i)).save(folder / f'{i}.png')
            out = Path(tmp) / 'reports'
            out.mkdir()
            df = scan_dataset(cfg, out)
            self.assertEqual(set(df[df.split == 'train'].generator), {'sd15'})
            self.assertEqual(df[df.split == 'test'].generator.nunique(), 7)
            tr, va, te = make_splits(df, cfg, 42, out)
            self.assertEqual((len(tr), len(va), len(te)), (18, 2, 140))
            cfg['expected_generators'] = [*cfg['expected_generators'], 'sd14']
            with self.assertRaisesRegex(ValueError, "missing=.*sd14"):
                scan_dataset(cfg, out)
            cfg['expected_generators'] = None
            cfg['source_generator'] = 'sd14'
            with self.assertRaisesRegex(ValueError, 'Source.*sd14'):
                scan_dataset(cfg, out)

    def test_aliases(self):
        for name in ('stable_diffusion_v_1_4', 'SDV4', 'sd14'):
            self.assertEqual(canonical_generator(name), 'sd14')

    def test_early_stop_delta_and_resume_state(self):
        stopper = EarlyStopping(patience=3, min_delta=.01)
        for loss in (.4, .395, .38):
            self.assertFalse(stopper.update(loss))
        self.assertEqual(stopper.state['bad_epochs'], 0)
        self.assertFalse(stopper.update(.39))
        resumed = EarlyStopping(patience=3, min_delta=.01, state=stopper.state)
        self.assertFalse(resumed.update(.40))
        self.assertTrue(resumed.update(.41))
        disabled = EarlyStopping(enabled=False, patience=1)
        disabled.update(.4)
        self.assertFalse(disabled.update(.5))

    def test_training_stops_and_resume_does_not_train(self):
        cfg = {**self.cfg, 'epochs': 7, 'train_batch': 4, 'test_batch': 4, 'patience': 2}
        model = Detector(copy.deepcopy(self.v), cfg, True)
        anchor = copy.deepcopy(self.v).requires_grad_(False).eval()
        x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])
        frame = pd.DataFrame({'label': y.numpy()})
        with tempfile.TemporaryDirectory() as tmp, patch('runtime.loader', return_value=[(x, y, torch.arange(4))]), \
             patch('runtime.predict', return_value=(np.array([.2, .8, .2, .8]), np.zeros(4))) as prediction:
            train_model(model, anchor, frame, frame, cfg, 42, torch.device('cpu'), Path(tmp), 'early-test')
            ck = torch.load(Path(tmp) / 'last.pt', weights_only=False)
            self.assertEqual(ck['epoch'], 2)
            self.assertTrue(ck['early_stopping']['stopped'])
            self.assertEqual(len(ck['history']), 3)
            best = torch.load(Path(tmp) / 'best.pt', weights_only=False)
            self.assertEqual(best['epoch'], 0)
            prediction.reset_mock()
            train_model(model, anchor, frame, frame, cfg, 42, torch.device('cpu'), Path(tmp), 'early-test')
            prediction.assert_not_called()
            for key, value in best['model'].items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_training_checkpoint_resume_both_models(self):
        cfg = {**self.cfg, 'epochs': 1, 'train_batch': 4, 'test_batch': 4}
        x, y = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])
        frame = pd.DataFrame({'label': y.numpy()})
        batch = [(x, y, torch.arange(4))]
        for gsd in (False, True):
            model = Detector(copy.deepcopy(self.v), cfg, gsd)
            anchor = copy.deepcopy(self.v).requires_grad_(False).eval() if gsd else None
            initial = model.vision.embeddings.patch_embedding.weight.detach().clone()
            anchor_before = copy.deepcopy(anchor.state_dict()) if anchor is not None else None
            with tempfile.TemporaryDirectory() as tmp, patch('runtime.loader', return_value=batch):
                train_model(model, anchor, frame, frame, cfg, 42, torch.device('cpu'), Path(tmp), 'test-signature')
                self.assertFalse(torch.equal(initial, model.vision.embeddings.patch_embedding.weight))
                self.assertTrue((Path(tmp) / 'best.pt').exists())
                self.assertTrue((Path(tmp) / 'last.pt').exists())
                best = copy.deepcopy(model.state_dict())
                train_model(model, anchor, frame, frame, cfg, 42, torch.device('cpu'), Path(tmp), 'test-signature')
                for k, v in best.items():
                    torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
            if anchor is not None:
                for k, v in anchor_before.items():
                    torch.testing.assert_close(v, anchor.state_dict()[k], rtol=0, atol=0)

    def test_folder_scan_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            for gen in ('stable_diffusion_v_1_4', 'BigGAN'):
                for split in ('train', 'val'):
                    for label in ('nature', 'ai'):
                        d = root / gen / split / label
                        d.mkdir(parents=True)
                        color = (30, 50, 70) if label == 'nature' else (90, 100, 110)
                        Image.new('RGB', (32, 40), color).save(d / 'sample.png')
            out = Path(tmp) / 'reports'
            out.mkdir()
            df = scan_dataset({**self.cfg, 'dataset_root': str(root), 'strict_eight_generators': False}, out)
            self.assertEqual(len(df), 6)  # SD14 train + SD14/BigGAN val; no BigGAN train
            self.assertTrue((df.width == 32).all())
            self.assertEqual(set(df.format), {'PNG'})
            self.assertTrue((out / 'image_audit.csv').exists())


if __name__ == '__main__':
    unittest.main()

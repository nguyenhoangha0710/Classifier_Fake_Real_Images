"""CPU-executable math/driver tests; deliberately NOT a claim of TPU hardware validation."""
import ast
import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import evaluate_uploaded_best_tpu as t
import test_uploaded_evaluation as fixture
from build_tpu_evaluation_notebook import build_notebook, NOTEBOOK


class CPUAdapterForTests(t.TPUInferenceBackend):
    """Explicit test double; production prepare() never falls back to CPU."""
    def prepare(self, cfg, options):
        t.validate_tpu_options(options)
        self.device = torch.device('cpu')
        self.kind, self.runtime_version, self.hardware = 'unit_cpu', 'test', 'NO TPU'
        self.precision = options['tpu_precision']
        cfg.update(precision=self.precision, attention='eager')
        return self.device


class TPUAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = fixture.UploadedEvaluationTests()
        self.fixture.setUp()
        self.cfg = dict(self.fixture.cfg, attention='eager')
        self.fixture.vision.config._attn_implementation = 'eager'
        # Build actual eager layers, not merely a config string on SDPA instances.
        self.fixture.vision = t.CLIPVisionModel(self.fixture.vision.config).vision_model
        self.backend = CPUAdapterForTests()
        self.options = dict(t.TPU_DEFAULTS, tpu_precision='fp32', require_preflight=False,
            expected_signature='unit-signature', expected_test_order_hash=None)
        self.backend.prepare(self.cfg, self.options)

    def model(self):
        model, anchor = self.fixture.model(self.cfg)
        return model, anchor

    def test_cached_positions_match_huggingface_without_resizing_pixels(self):
        original = copy.deepcopy(self.fixture.vision.embeddings)
        cached = t.CachedPositionEmbeddings(copy.deepcopy(original), max_entries=2)
        for shape in ((28, 28), (42, 28), (56, 56), (28, 70), (28, 28)):
            x = torch.randn(1, 3, *shape)
            torch.testing.assert_close(cached(x), original(x, interpolate_pos_encoding=True), atol=0, rtol=0)
            self.assertLessEqual(len(cached.cache), 2)
        self.assertEqual(set(cached.state_dict()), set(original.state_dict()))

    def test_cpu_eigh_is_full_token_subspace_not_sampled(self):
        torch.manual_seed(3)
        matrix = torch.randn(67, 16)
        gram = sum(chunk.T @ chunk for chunk in matrix.split([7, 20, 40]))
        basis = t.cpu_gram_basis(gram, 67, 3)
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        reference = vh[:3].T
        torch.testing.assert_close(basis @ basis.T, reference @ reference.T, atol=3e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            t.cpu_gram_basis(gram, 2, 3)
        gram[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            t.cpu_gram_basis(gram, 67, 3)

    def test_mixed_shape_gsd_adapter_matches_original_cpu_equations(self):
        model, anchor = self.model()
        images = [torch.randn(3, 28, 28), torch.randn(3, 42, 28), torch.randn(3, 56, 42)]
        basis = t.estimate_bases(anchor, images, self.cfg, torch.device('cpu'), model.selected)
        with torch.no_grad():
            reference = torch.cat([model(image.unsqueeze(0), basis) for image in images]).numpy()
        t.install_cached_positions(model.vision); t.install_cached_positions(anchor)
        output = self.backend.predict_batch(model, anchor, images, self.cfg, torch.device('cpu'))
        np.testing.assert_allclose(output, reference, atol=2e-6, rtol=2e-5)
        self.assertEqual(output.shape, (3, 2))

    def test_changing_other_image_recomputes_common_batch_basis(self):
        model, anchor = self.model()
        t.install_cached_positions(model.vision); t.install_cached_positions(anchor)
        images = [torch.randn(3, 28, 28), torch.randn(3, 42, 28)]
        original = self.backend.estimate(anchor, images, self.cfg, model.selected)
        changed = self.backend.estimate(anchor, [images[0], images[1] * 20 + 5], self.cfg, model.selected)
        self.assertTrue(any(not torch.allclose(original[l] @ original[l].T, changed[l] @ changed[l].T)
                            for l in original))

    def test_toy_preflight_cpu_fp32_and_bf16_paths(self):
        report = self.backend.preflight()
        self.assertEqual(report['status'], 'pass')
        self.assertLess(report['max_abs_logit_delta'], 1e-5)
        self.backend.precision = 'bf16'
        self.assertEqual(self.backend.preflight()['status'], 'pass')

    def test_options_refuse_global_downcast_distributed_and_batch_changes(self):
        for options in (dict(self.options, tpu_precision='fp16'), dict(self.options, micro_batch=2),
                        dict(self.options, anchor_micro_batch=2), dict(self.options, cpu_threads=0)):
            with self.assertRaises(ValueError):
                t.validate_tpu_options(options)
        for env in ({'XLA_USE_BF16':'1'}, {'XLA_DOWNCAST_BF16':'1'}, {'WORLD_SIZE':'8'}, {'XLA_USE_SPMD':'1'}):
            with patch.dict(os.environ, env), self.assertRaises(ValueError):
                t.validate_tpu_options(self.options)

    def test_backend_driver_resume_and_fingerprint_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.fixture.images(root/'input/tiny', 6)
            checkpoint = root/'input/checkpoint/best.pt'
            checkpoint.parent.mkdir(parents=True)
            torch.save(self.fixture.payload(), checkpoint)
            manifest = root/'test_manifest.csv'
            frame.to_csv(manifest, index=False)
            opts = dict(self.options, input_root=str(root/'input'), checkpoint_path=str(checkpoint),
                test_manifest_path=str(manifest), checkpoint_cache=str(root/'cache'),
                output_root=str(root/'out'), cases=['native', 'all_jpeg96'], max_new_cases_per_run=1)
            backend = CPUAdapterForTests()
            def build(cfg, device, state):
                model, anchor = self.fixture.model(cfg)
                model.load_state_dict(state, strict=True)
                t.install_cached_positions(model.vision); t.install_cached_positions(anchor)
                return model, anchor
            with patch.object(t, 'choose_test_manifest', return_value=(manifest, frame, 'unit-cohort')), \
                    patch.object(backend, 'build_models', side_effect=build), \
                    patch.object(t.torch.cuda, 'get_device_name', side_effect=AssertionError('no CUDA')), \
                    patch.object(t.torch.cuda, 'empty_cache', side_effect=AssertionError('no CUDA')), \
                    patch.object(t, 'seed_all'):
                for iteration in range(2):
                    with patch.object(backend, 'predict_batch', wraps=backend.predict_batch) as predict:
                        table = t.run_uploaded_evaluation(opts, backend=backend)
                        self.assertEqual(predict.call_count, 1)
                    state = json.loads((root/'out/evaluation_state.json').read_text())
                    self.assertEqual(len(state['completed_cases']), iteration+1)
                self.assertEqual(state['status'], 'complete')
                identity = json.loads((root/'out/run_fingerprint.json').read_text())
                self.assertEqual(identity['environment']['adapter'], t.TPU_ADAPTER_VERSION)
                self.assertEqual(identity['inference_config']['attention'], 'eager')
                self.assertEqual(identity['inference_config']['precision'], 'fp32')
                with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                    t.run_uploaded_evaluation(dict(opts, tpu_precision='bf16'), backend=backend)

    def test_custom_backend_logits_are_validated_and_saved_as_softmax(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.fixture.images(root/'input', 2)
            model, anchor = self.model()
            with patch.object(self.backend, 'predict_batch', return_value=np.array([[2., 3.], [8., -1.]], dtype=np.float32)):
                result, complete = t.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.options,
                    'native', root/'out', 'unit-fp', torch.device('cpu'), time.monotonic()+60, backend=self.backend)
            self.assertTrue(complete)
            expected = torch.tensor([[2., 3.], [8., -1.]]).softmax(-1)[:, 1].numpy()
            np.testing.assert_allclose(result.prob_fake, expected)
            self.assertEqual(result.batch_id.tolist(), [0, 0])
            self.assertEqual(result.input_tokens.tolist(), [5, 10])
            with patch.object(self.backend, 'predict_batch', return_value=np.full((2, 2), np.nan)), \
                    self.assertRaises(FloatingPointError):
                t.evaluate_uploaded_case(model, anchor, frame, self.cfg, self.options,
                    'native', root/'bad', 'unit-fp', torch.device('cpu'), time.monotonic()+60, backend=self.backend)

    def test_notebook_embeds_all_code_and_smoke_is_default_not_training(self):
        notebook = build_notebook()
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        namespace = {'__name__':'tpu_notebook_test'}
        exec(''.join(notebook['cells'][3]['source']), namespace)
        exec(''.join(notebook['cells'][4]['source']), namespace)
        self.assertFalse(namespace['RUN_FULL_TEST'])
        self.assertEqual(namespace['EVAL_CONFIG']['max_batches'], 2)
        self.assertEqual(namespace['EVAL_CONFIG']['save_every_batches'], 1)
        self.assertIn('smoke', namespace['EVAL_CONFIG']['output_root'])
        call = ''.join(notebook['cells'][5]['source'])
        self.assertEqual(call.strip(), 'metrics = run_uploaded_tpu_evaluation(EVAL_CONFIG)')
        self.assertEqual(json.loads((Path(__file__).parent/NOTEBOOK).read_text(encoding='utf-8')), notebook)


if __name__ == '__main__':
    unittest.main()

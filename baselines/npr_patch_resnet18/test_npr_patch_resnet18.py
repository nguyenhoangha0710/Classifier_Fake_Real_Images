import random
import unittest

import numpy as np
import torch
from PIL import Image

from baselines.npr_patch_resnet18.npr_patch_resnet18 import (
    NPRLayer,
    NPRPatchResNet18,
    extract_native_patches,
    make_sliding_boxes,
    pil_patch_to_normalized_tensor,
)


class NPRPatchUtilitiesTest(unittest.TestCase):
    def test_boxes_cover_image_and_are_even_aligned(self):
        boxes = make_sliding_boxes(510, 382, patch_size=256, stride_ratio=0.5)
        coverage = np.zeros((382, 510), dtype=np.uint8)
        for left, top, width, height in boxes:
            self.assertEqual(left % 2, 0)
            self.assertEqual(top % 2, 0)
            coverage[top : top + height, left : left + width] = 1
        self.assertTrue(np.all(coverage == 1))

    def test_extract_crops_odd_border_without_resize(self):
        image = Image.new("RGB", (301, 259), color=(10, 20, 30))
        patches, info = extract_native_patches(image, patch_sizes=(256, 128, 64))
        self.assertEqual((info["native_width"], info["native_height"]), (300, 258))
        self.assertEqual(info["patch_size"], 256)
        self.assertTrue(patches)
        self.assertTrue(all(patch.size == (256, 256) for patch in patches))

    def test_sampling_is_reproducible(self):
        image = Image.new("RGB", (1024, 1024))
        _, first = extract_native_patches(
            image,
            max_patches=7,
            random_sample=True,
            rng=random.Random(123),
        )
        _, second = extract_native_patches(
            image,
            max_patches=7,
            random_sample=True,
            rng=random.Random(123),
        )
        self.assertEqual(first["boxes"], second["boxes"])

    def test_patch_npr_matches_full_npr_for_even_origin(self):
        generator = torch.Generator().manual_seed(7)
        full = torch.rand((1, 3, 320, 384), generator=generator)
        layer = NPRLayer()
        full_npr = layer(full)
        left, top, size = 64, 32, 128
        patch_npr = layer(full[:, :, top : top + size, left : left + size])
        expected = full_npr[:, :, top : top + size, left : left + size]
        self.assertTrue(torch.equal(patch_npr, expected))

    def test_tensor_conversion_preserves_patch_shape(self):
        patch = Image.new("RGB", (128, 128), color=(127, 63, 31))
        tensor = pil_patch_to_normalized_tensor(patch)
        self.assertEqual(tuple(tensor.shape), (3, 128, 128))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_model_returns_one_image_logit_and_patch_logits(self):
        model = NPRPatchResNet18(num_classes=2)
        patches = torch.randn(3, 3, 64, 64)
        image_logits, patch_logits = model(patches, patch_micro_batch_size=2)
        self.assertEqual(tuple(image_logits.shape), (1, 2))
        self.assertEqual(tuple(patch_logits.shape), (3, 2))
        self.assertTrue(torch.allclose(image_logits, patch_logits.mean(dim=0, keepdim=True)))


if __name__ == "__main__":
    unittest.main()

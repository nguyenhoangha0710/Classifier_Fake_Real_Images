"""CPU smoke tests; run as a script without pytest."""

from __future__ import annotations

import sys
import tempfile
import random
import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.checkpointing import (
    CheckpointManager,
    atomic_torch_save,
    load_checkpoint_cpu,
    restore_manifest_artifacts,
    validate_checkpoint_identity,
)
from src.core import ForensicResidualExtractor, ForensicTwins, barlow_twins_loss
from src.data import (
    TwinViewCollator,
    _extend_imagenet_candidates_from_index,
    _extract_halo,
    _load_coco_index,
    _sample_imagenet_index,
)


def main() -> None:
    fre = ForensicResidualExtractor().freeze()
    model = ForensicTwins(
        fre,
        embed_dim=192,
        depth=2,
        num_heads=3,
        projector_hidden=64,
        projector_out=32,
    )
    view1 = torch.rand(4, 1, 78, 78)  # 64 crop + 7px halo on both sides
    view2 = torch.rand(4, 1, 110, 110)  # 96 crop + halo
    z1, z2, e1, e2 = model(view1, view2)
    assert z1.shape == z2.shape == (4, 192)
    assert e1.shape == e2.shape == (4, 32)
    loss, _ = barlow_twins_loss(e1, e2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.fre.parameters())

    # Cropping a 7px halo before the local FRE must equal FRE-before-crop.
    gray = torch.rand(1, 200, 220)
    box = (31, 47, 64, 64)
    halo = _extract_halo(gray, box, radius=7).unsqueeze(0)
    residual_from_halo = model.residual_from_halo(halo)
    residual_full = model.fre(gray.unsqueeze(0))[:, :, 47:111, 31:95]
    torch.testing.assert_close(residual_from_halo, residual_full)

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        fre_checkpoint = root / "fre.pt"
        torch.save(fre.conv.state_dict(), fre_checkpoint)
        restored_fre = ForensicResidualExtractor.from_pretrained(fre_checkpoint)
        assert restored_fre.conv.w.shape == (8, 1, 15, 15)
        assert all(not parameter.requires_grad for parameter in restored_fre.parameters())

        rows = []
        for index in range(4):
            path = root / f"{index}.jpg"
            pixels = np.random.randint(0, 256, (300, 320, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append({"image_path": str(path), "sample_id": str(index)})
        batch = TwinViewCollator(crop_sizes=(64, 96, 128))(rows)
        assert batch["view1_halo"].shape[0] == 4
        assert batch["view2_halo"].shape[0] == 4
        for first, second in zip(batch["boxes1"], batch["boxes2"]):
            ax, ay, aw, ah = first
            bx, by, bw, bh = second
            assert not (max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh))

        # Official dataset indexes must support deterministic sampling without
        # walking every file in ImageNet/COCO directories.
        indexed_imagenet = root / "indexed_imagenet"
        indexed_imagenet.mkdir()
        for class_id in ("n00000001", "n00000002"):
            (indexed_imagenet / class_id).mkdir()
        imagenet_index = root / "LOC_train_solution.csv"
        imagenet_index.write_text(
            "ImageId,PredictionString\n"
            + "\n".join(
                f"{class_id}_{index},{class_id} 0 0 10 10"
                for class_id in ("n00000001", "n00000002")
                for index in range(10)
            )
            + "\n",
            encoding="utf-8",
        )
        first_sample, rows_read = _sample_imagenet_index(
            imagenet_index,
            indexed_imagenet,
            ("n00000001", "n00000002"),
            candidates_per_class=4,
            rng=random.Random(42),
        )
        second_sample, _ = _sample_imagenet_index(
            imagenet_index,
            indexed_imagenet,
            ("n00000001", "n00000002"),
            candidates_per_class=4,
            rng=random.Random(42),
        )
        assert rows_read == 20
        assert first_sample == second_sample
        assert all(len(paths) == 4 for paths in first_sample.values())
        added = _extend_imagenet_candidates_from_index(
            imagenet_index,
            indexed_imagenet,
            ("n00000001",),
            first_sample,
            random.Random(42),
        )
        assert added == 6
        assert len(first_sample["n00000001"]) == 10

        coco_train = root / "coco2017" / "train2017"
        coco_train.mkdir(parents=True)
        coco_index = root / "coco2017" / "annotations" / "captions_train2017.json"
        coco_index.parent.mkdir()
        coco_index.write_text(
            json.dumps({"images": [{"file_name": f"{index:012d}.jpg"} for index in range(8)]}),
            encoding="utf-8",
        )
        first_coco = _load_coco_index(coco_index, coco_train, random.Random(42))
        second_coco = _load_coco_index(coco_index, coco_train, random.Random(42))
        assert first_coco == second_coco
        assert len(first_coco) == 8

        checkpoint = root / "smoke.ckpt"
        payload = {
            "schema_version": 1,
            "encoder": {},
            "projector": {},
            "optimizer": {},
            "scheduler": {},
            "epoch": 0,
            "next_batch_index": 0,
            "optimizer_step": 0,
            "config": {},
            "config_hash": "config",
            "manifest_hash": "manifest",
            "fre_hash": "fre",
        }
        atomic_torch_save(payload, checkpoint)
        restored = load_checkpoint_cpu(checkpoint)
        assert restored["fre_hash"] == "fre"
        validate_checkpoint_identity(restored, "config", "manifest", "fre")
        try:
            validate_checkpoint_identity(restored, "different", "manifest", "fre")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Mismatched checkpoint identity was not rejected")

        persisted = root / "downloaded_private_dataset"
        persisted.mkdir()
        (persisted / "train_real_50k.csv").write_text("sample_id,label\nreal:1,0\n", encoding="utf-8")
        (persisted / "train_real_50k.summary.json").write_text(
            json.dumps({"seed": 42}), encoding="utf-8"
        )
        (persisted / "train_real_50k.persisted.json").write_text(
            json.dumps({"manifest_sha256": "test"}), encoding="utf-8"
        )
        restored_manifest = root / "restored" / "train_real_50k.csv"
        assert restore_manifest_artifacts(restored_manifest, persisted)
        assert restored_manifest.is_file()
        assert restored_manifest.with_suffix(".summary.json").is_file()
        assert restored_manifest.with_suffix(".persisted.json").is_file()

        upload_checkpoints = root / "upload_checkpoints"
        upload_checkpoint = upload_checkpoints / "epoch_0001.ckpt"
        atomic_torch_save(payload, upload_checkpoint)
        manager = CheckpointManager(
            upload_checkpoints,
            persistent_slug="test/private-checkpoints",
            persistent_files=(
                restored_manifest,
                restored_manifest.with_suffix(".summary.json"),
                restored_manifest.with_suffix(".persisted.json"),
            ),
        )
        completed = mock.Mock(returncode=0, stdout="version created", stderr="")
        with mock.patch("src.checkpointing.shutil.which", return_value="kaggle"), mock.patch(
            "src.checkpointing.subprocess.run", return_value=completed
        ):
            ok, _ = manager.persist_manifest_to_kaggle()
        assert ok
        publish = upload_checkpoints.parent / "kaggle_publish"
        assert (publish / restored_manifest.name).is_file()
        assert (publish / restored_manifest.with_suffix(".summary.json").name).is_file()
        assert (publish / upload_checkpoint.name).is_file()

    print(f"smoke tests passed; loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()

"""CPU-only structural smoke test for the JPEG-bias data pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import test_tiny_jpeg_bias_three_models_kaggle as runner


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "tiny-genimage"
        for generator_index, generator in enumerate(("generator_a", "generator_b")):
            for label_dir in ("nature", "ai"):
                folder = data_root / generator / "val" / label_dir
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(2):
                    pixels = np.full(
                        (64 + 8 * index, 80 + 8 * generator_index, 3),
                        60 + 20 * index,
                        dtype=np.uint8,
                    )
                    image = Image.fromarray(pixels, "RGB")
                    if label_dir == "nature":
                        image.save(
                            folder / f"{index}.jpg",
                            format="JPEG",
                            quality=96,
                            subsampling=0,
                        )
                    else:
                        image.save(folder / f"{index}.png", format="PNG")

        manifest = runner.build_test_manifest(data_root, "val", 42, None)
        audited, invalid = runner.audit_manifest(manifest)
        print(
            audited[
                ["label_name", "pil_format", "jpeg_quality_estimate", "jpeg_quality_mse"]
            ].to_string(index=False)
        )
        if not invalid.empty:
            print(invalid[["image_path", "error_type", "error"]].to_string(index=False))
        controlled, metadata = runner.build_controlled_manifest(
            audited, 94, 98, 42, True
        )
        output_root = Path(temporary) / "results"
        statistics = runner.create_dataset_statistics(
            audited, invalid, controlled, metadata, output_root
        )
        config = dict(runner.DEFAULT_CONFIG)
        for case_name in runner.ALL_CASES:
            frame = runner.case_manifest(case_name, audited, controlled)
            transformed = runner.load_case_image(
                frame.iloc[0].to_dict(), case_name, config
            )
            assert transformed.mode == "RGB"

        assert len(manifest) == 8
        assert invalid.empty
        assert len(controlled) == 8
        assert statistics["num_unique_exact_sizes"] == 4
        assert (output_root / "dataset" / "overall_statistics.json").is_file()
        print("JPEG-bias data-pipeline smoke test passed.")


if __name__ == "__main__":
    main()

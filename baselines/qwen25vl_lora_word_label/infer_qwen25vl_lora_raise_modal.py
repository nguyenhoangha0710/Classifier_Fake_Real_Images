"""Run Qwen2.5-VL LoRA real-only inference on local RAISE images with Modal.

Two-step usage:

    python -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_raise_modal.py --upload-only

    python -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_raise_modal.py --no-upload

One-step usage is also supported. The script uploads the local RAISE folder to
Modal Volume first, then runs inference:

    python -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_raise_modal.py
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "qwen25vl-lora-raise-inference"
GPU_TYPE = "A100-40GB"

OUTPUT_VOLUME_NAME = "qwen25vl-lora-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"
RAISE_VOLUME_NAME = "raise-200-nef-data"

REMOTE_OUTPUT_ROOT = "/outputs/qwen25vl_lora_word_label"
REMOTE_HF_HOME = "/hf-cache"
REMOTE_RAISE_ROOT = "/raise-data"
DEFAULT_REMOTE_RAISE_SUBDIR = "RAISE_200_NEF"
DEFAULT_LOCAL_RAISE_ROOT = r"D:\LLM\LVLM\crawl\RAISE_200_NEF"


image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "torchvision",
    "transformers>=4.49.0,<4.58",
    "accelerate",
    "peft",
    "bitsandbytes",
    "qwen-vl-utils",
    "pandas<3.0",
    "numpy",
    "pillow<12.0",
    "tqdm",
    "safetensors",
    "rawpy",
)
image = image.env(
    {
        "HF_HOME": REMOTE_HF_HOME,
        "HF_DATASETS_CACHE": f"{REMOTE_HF_HOME}/datasets",
        "TRANSFORMERS_CACHE": f"{REMOTE_HF_HOME}/transformers",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
raise_volume = modal.Volume.from_name(RAISE_VOLUME_NAME, create_if_missing=True)


DEFAULT_CONFIG = {
    "base_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_root": REMOTE_OUTPUT_ROOT,
    "adapter_run_id": "20260910_003026",
    "adapter_subdir": "adapter",
    "raise_data_root": f"{REMOTE_RAISE_ROOT}/{DEFAULT_REMOTE_RAISE_SUBDIR}",
    "max_images": None,
    "random_seed": 42,
    "eval_batch_size": 1,
    "load_in_4bit": True,
    "min_pixels": 64 * 28 * 28,
    "max_pixels": 128 * 28 * 28,
    "normalize_candidate_logprob": True,
    "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "save_predictions": True,
    "prompt_template": (
        "Look at the image and decide whether it is real or AI-generated. "
        "Answer with exactly one word: real or fake."
    ),
}

function_options = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 12,
    "memory": 65536,
    "volumes": {
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
        REMOTE_RAISE_ROOT: raise_volume,
    },
}


@app.function(**function_options)
def infer_qwen25vl_lora_raise(config_overrides: dict | None = None) -> dict:
    import gc
    import json
    import random
    import time
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import rawpy
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from PIL import Image, ImageFile
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if isinstance(config.get("thresholds"), str):
        config["thresholds"] = [float(item.strip()) for item in config["thresholds"].split(",") if item.strip()]

    label_id_to_text = {0: "real", 1: "fake"}
    candidate_texts = ["real", "fake"]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".nef"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    run_root = Path(config["output_root"]) / config["adapter_run_id"]
    requested_adapter_dir = run_root / config["adapter_subdir"]
    adapter_candidates = [requested_adapter_dir]
    if config["adapter_subdir"] != "checkpoints/latest/adapter":
        adapter_candidates.append(run_root / "checkpoints" / "latest" / "adapter")
    if config["adapter_subdir"] != "adapter":
        adapter_candidates.append(run_root / "adapter")
    adapter_dir = next((item for item in adapter_candidates if (item / "adapter_config.json").exists()), None)
    if adapter_dir is None:
        checked = "\n".join(str(item) for item in adapter_candidates)
        raise FileNotFoundError(f"Cannot find LoRA adapter. Checked:\n{checked}")

    raise_data_root = Path(config["raise_data_root"])
    if not raise_data_root.exists():
        raise FileNotFoundError(
            f"Cannot find RAISE data at {raise_data_root}. "
            "Run this script once with --upload-only or without --no-upload."
        )

    inference_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = run_root / f"raise_real_only_{inference_id}"
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, data: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def find_image_paths(root: Path) -> list[Path]:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions)
        max_images = config["max_images"]
        if max_images is not None and len(paths) > int(max_images):
            rng = random.Random(config["random_seed"])
            paths = sorted(rng.sample(paths, int(max_images)))
        return paths

    def load_rgb_image(path: Path) -> Image.Image:
        if path.suffix.lower() == ".nef":
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            return Image.fromarray(rgb).convert("RGB")
        return Image.open(path).convert("RGB")

    def load_model_and_processor():
        try:
            processor = AutoProcessor.from_pretrained(
                adapter_dir,
                min_pixels=config["min_pixels"],
                max_pixels=config["max_pixels"],
                trust_remote_code=True,
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                config["base_model_name"],
                min_pixels=config["min_pixels"],
                max_pixels=config["max_pixels"],
                trust_remote_code=True,
            )

        quantization_config = None
        if config["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["base_model_name"],
            torch_dtype=amp_dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        model.eval()
        return model, processor

    def make_user_messages(image_item: Image.Image) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_item},
                    {"type": "text", "text": config["prompt_template"]},
                ],
            }
        ]

    def make_full_messages(image_item: Image.Image, answer: str) -> list[dict[str, Any]]:
        return make_user_messages(image_item) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    def move_inputs_to_device(inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_scores = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1) * mask
        scores = token_scores.sum(dim=-1)
        if config["normalize_candidate_logprob"]:
            scores = scores / mask.sum(dim=-1).clamp_min(1)
        return scores

    @torch.no_grad()
    def score_candidate_batch(images: list[Image.Image], candidate_text: str) -> np.ndarray:
        full_texts = []
        prompt_lengths = []
        for image_item in images:
            prompt_text = processor.apply_chat_template(
                make_user_messages(image_item),
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                make_full_messages(image_item, candidate_text),
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_inputs = processor(text=[prompt_text], images=[image_item], padding=False, return_tensors="pt")
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])
            full_texts.append(full_text)

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        candidate_labels = inputs["input_ids"].clone()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            candidate_labels[row_idx, :prompt_len] = -100
        candidate_labels[candidate_labels == processor.tokenizer.pad_token_id] = -100
        inputs = move_inputs_to_device(inputs)
        candidate_labels = candidate_labels.to(device)
        outputs = model(**inputs)
        return sequence_logprob_from_logits(outputs.logits, candidate_labels).detach().cpu().numpy()

    def iter_batches(items: list[Path], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    def summarize_real_only(pred_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
        fake_probs = pred_df["fake_probability"].to_numpy(dtype=float)
        pred_fake = fake_probs >= threshold
        num_samples = int(len(pred_df))
        num_pred_fake = int(pred_fake.sum())
        num_pred_real = int(num_samples - num_pred_fake)
        return {
            "num_samples": num_samples,
            "threshold": float(threshold),
            "num_pred_real": num_pred_real,
            "num_pred_fake": num_pred_fake,
            "real_accuracy": float(num_pred_real / num_samples) if num_samples else 0.0,
            "false_positive_rate": float(num_pred_fake / num_samples) if num_samples else 0.0,
            "mean_fake_probability": float(np.mean(fake_probs)) if num_samples else None,
            "median_fake_probability": float(np.median(fake_probs)) if num_samples else None,
            "p90_fake_probability": float(np.quantile(fake_probs, 0.90)) if num_samples else None,
            "p95_fake_probability": float(np.quantile(fake_probs, 0.95)) if num_samples else None,
            "p99_fake_probability": float(np.quantile(fake_probs, 0.99)) if num_samples else None,
            "min_fake_probability": float(np.min(fake_probs)) if num_samples else None,
            "max_fake_probability": float(np.max(fake_probs)) if num_samples else None,
        }

    def predict_real_only(image_paths: list[Path]) -> pd.DataFrame:
        rows = []
        errors = []
        progress = tqdm(image_paths, desc="raise_real_only")
        batch_paths = []
        for path in progress:
            batch_paths.append(path)
            if len(batch_paths) < int(config["eval_batch_size"]):
                continue
            rows.extend(predict_path_batch(batch_paths, errors))
            batch_paths = []
        if batch_paths:
            rows.extend(predict_path_batch(batch_paths, errors))
        if errors:
            pd.DataFrame(errors).to_csv(metrics_dir / "raise_read_errors.csv", index=False)
        return pd.DataFrame(rows)

    def predict_path_batch(batch_paths: list[Path], errors: list[dict[str, str]]) -> list[dict[str, Any]]:
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                image_item = load_rgb_image(path)
                images.append(image_item)
                valid_paths.append(path)
            except Exception as exc:
                errors.append({"image_path": str(path), "error": repr(exc)})
        if not images:
            return []

        real_scores = score_candidate_batch(images, "real")
        fake_scores = score_candidate_batch(images, "fake")
        scores = np.stack([real_scores, fake_scores], axis=1)
        probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()
        batch_rows = []
        for path, image_item, fake_prob in zip(valid_paths, images, probs[:, 1]):
            pred_label = int(fake_prob >= 0.5)
            batch_rows.append(
                {
                    "sample_id": None,
                    "image_path": str(path),
                    "relative_path": str(path.relative_to(raise_data_root)),
                    "file_name": path.name,
                    "extension": path.suffix.lower(),
                    "parent_folder": path.parent.name,
                    "width": image_item.width,
                    "height": image_item.height,
                    "label": 0,
                    "label_name": "real",
                    "fake_probability": float(fake_prob),
                    "predicted_label": pred_label,
                    "predicted_label_name": label_id_to_text[pred_label],
                    "model_name": "qwen25vl_lora_word_label",
                    "dataset_tag": "raise_200_nef_real_only",
                    "adapter_run_id": config["adapter_run_id"],
                    "adapter_subdir": config["adapter_subdir"],
                    "adapter_dir": str(adapter_dir),
                }
            )
        return batch_rows

    print("RAISE inference ID:", inference_id)
    print("Device:", device)
    print("Adapter dir:", adapter_dir)
    print("RAISE data root:", raise_data_root)

    save_json(metrics_dir / "inference_config.json", {**config, "adapter_dir": str(adapter_dir)})
    image_paths = find_image_paths(raise_data_root)
    selected_df = pd.DataFrame(
        [
            {
                "sample_id": idx,
                "image_path": str(path),
                "relative_path": str(path.relative_to(raise_data_root)),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "parent_folder": path.parent.name,
                "label": 0,
                "label_name": "real",
            }
            for idx, path in enumerate(image_paths)
        ]
    )
    selected_df.to_csv(metrics_dir / "raise_selected_images.csv", index=False)
    selected_df.groupby("extension").size().reset_index(name="num_samples").to_csv(
        metrics_dir / "raise_extension_counts.csv",
        index=False,
    )
    selected_df.groupby("parent_folder").size().reset_index(name="num_samples").to_csv(
        metrics_dir / "raise_parent_folder_counts.csv",
        index=False,
    )

    print("RAISE samples:", len(image_paths))
    model, processor = load_model_and_processor()
    for text in candidate_texts:
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        print(repr(text), token_ids, "num_tokens=", len(token_ids))

    pred_df = predict_real_only(image_paths)
    if len(pred_df):
        pred_df["sample_id"] = range(len(pred_df))
    if config["save_predictions"]:
        pred_df.to_csv(predictions_dir / "raise_real_predictions.csv", index=False)
        pred_df[pred_df["predicted_label"] == 1].to_csv(
            predictions_dir / "raise_false_positive_predictions.csv",
            index=False,
        )

    metrics = summarize_real_only(pred_df, threshold=0.5)
    metrics.update(
        {
            "model_name": "qwen25vl_lora_word_label",
            "dataset_tag": "raise_200_nef_real_only",
            "adapter_run_id": config["adapter_run_id"],
            "adapter_subdir": config["adapter_subdir"],
            "adapter_dir": str(adapter_dir),
            "raise_data_root": str(raise_data_root),
            "random_seed": config["random_seed"],
            "max_images": config["max_images"],
            "candidate_texts": candidate_texts,
            "normalize_candidate_logprob": config["normalize_candidate_logprob"],
            "inference_id": inference_id,
            "output_dir": str(output_dir),
        }
    )
    save_json(metrics_dir / "raise_real_metrics.json", metrics)
    save_json(metrics_dir / "summary.json", metrics)

    threshold_sweep_df = pd.DataFrame(
        [summarize_real_only(pred_df, threshold) for threshold in config["thresholds"]]
    )
    threshold_sweep_df.to_csv(metrics_dir / "raise_real_threshold_sweep.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_root / f"raise_real_only_summary_{inference_id}.csv", index=False)

    print("raise_200_nef_real_only", json.dumps(metrics, ensure_ascii=False, indent=2))
    output_volume.commit()
    hf_cache_volume.commit()
    raise_volume.commit()

    del model
    cleanup_cuda()
    return metrics


def upload_raise_folder(
    local_data_root: str,
    remote_data_subdir: str,
    force_upload: bool,
    upload_batch_size: int,
) -> None:
    local_root = Path(local_data_root).expanduser().resolve()
    if not local_root.exists():
        raise FileNotFoundError(f"Local RAISE folder does not exist: {local_root}")
    remote_dir = "/" + remote_data_subdir.strip("/\\")
    files = sorted(path for path in local_root.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    total_mb = total_bytes / (1024 * 1024)
    upload_batch_size = max(1, int(upload_batch_size))
    num_batches = (len(files) + upload_batch_size - 1) // upload_batch_size

    print(f"Uploading {local_root} -> {RAISE_VOLUME_NAME}:{remote_dir}", flush=True)
    print(f"Found {len(files)} files, total size {total_mb:.2f} MB", flush=True)
    print(f"Upload batch size: {upload_batch_size} files", flush=True)

    uploaded_files = 0
    uploaded_bytes = 0
    for batch_index in range(num_batches):
        start = batch_index * upload_batch_size
        batch_files = files[start : start + upload_batch_size]
        batch_bytes = sum(path.stat().st_size for path in batch_files)
        batch_mb = batch_bytes / (1024 * 1024)
        print(
            f"Batch {batch_index + 1}/{num_batches}: queueing "
            f"{len(batch_files)} files ({batch_mb:.2f} MB)",
            flush=True,
        )

        with raise_volume.batch_upload(force=force_upload) as batch:
            for local_index, path in enumerate(batch_files, start=1):
                global_index = start + local_index
                relative_path = path.relative_to(local_root).as_posix()
                remote_path = f"{remote_dir}/{relative_path}"
                size_mb = path.stat().st_size / (1024 * 1024)
                print(
                    f"  queued {global_index}/{len(files)}: "
                    f"{relative_path} ({size_mb:.2f} MB)",
                    flush=True,
                )
                batch.put_file(path, remote_path)

        uploaded_files += len(batch_files)
        uploaded_bytes += batch_bytes
        uploaded_mb = uploaded_bytes / (1024 * 1024)
        print(
            f"Batch {batch_index + 1}/{num_batches} committed. "
            f"Progress: {uploaded_files}/{len(files)} files, {uploaded_mb:.2f}/{total_mb:.2f} MB",
            flush=True,
        )

    print(f"Upload finished: {len(files)} files -> {RAISE_VOLUME_NAME}:{remote_dir}", flush=True)


@app.local_entrypoint()
def main(
    adapter_run_id: str = "20260910_003026",
    adapter_subdir: str = "adapter",
    local_data_root: str = DEFAULT_LOCAL_RAISE_ROOT,
    remote_data_subdir: str = DEFAULT_REMOTE_RAISE_SUBDIR,
    upload: bool = True,
    upload_only: bool = False,
    force_upload: bool = False,
    upload_batch_size: int = 20,
    max_images: int = 0,
    random_seed: int = 42,
    eval_batch_size: int = 1,
    thresholds: str = "0.3,0.4,0.5,0.6,0.7,0.8,0.9",
):
    """Upload local RAISE images if requested, then run Modal inference."""

    if upload:
        upload_raise_folder(local_data_root, remote_data_subdir, force_upload, upload_batch_size)

    if upload_only:
        print("Upload-only mode finished. No inference was started.")
        return

    remote_data_subdir = remote_data_subdir.strip("/\\")
    overrides = {
        "adapter_run_id": adapter_run_id,
        "adapter_subdir": adapter_subdir,
        "raise_data_root": f"{REMOTE_RAISE_ROOT}/{remote_data_subdir}",
        "max_images": None if max_images <= 0 else max_images,
        "random_seed": random_seed,
        "eval_batch_size": eval_batch_size,
        "thresholds": thresholds,
    }
    summary = infer_qwen25vl_lora_raise.remote(overrides)
    print("Remote RAISE inference summary:")
    print(summary)

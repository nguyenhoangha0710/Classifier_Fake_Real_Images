"""Run Qwen2.5-VL LoRA real-only inference on CommunityForensics-Eval with Modal."""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "qwen25vl-lora-real-only-inference"
GPU_TYPE = "A100-40GB"

OUTPUT_VOLUME_NAME = "qwen25vl-lora-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_OUTPUT_ROOT = "/outputs/qwen25vl_lora_word_label"
REMOTE_HF_HOME = "/hf-cache"


def find_data_loader_dir() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd() / "data_loader", Path(REMOTE_CODE_ROOT) / "data_loader"]
    candidates.extend(parent / "data_loader" for parent in [file_path.parent, *file_path.parents])
    for candidate in candidates:
        if (candidate / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Cannot find data_loader/. Run from the repository root.")


LOCAL_DATA_LOADER_DIR = find_data_loader_dir()

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "torchvision",
    "transformers>=4.49.0,<4.58",
    "accelerate",
    "peft",
    "bitsandbytes",
    "datasets",
    "qwen-vl-utils",
    "pandas<3.0",
    "pillow<12.0",
    "tqdm",
    "safetensors",
)
image = image.env(
    {
        "HF_HOME": REMOTE_HF_HOME,
        "HF_DATASETS_CACHE": f"{REMOTE_HF_HOME}/datasets",
        "TRANSFORMERS_CACHE": f"{REMOTE_HF_HOME}/transformers",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

function_mounts = []
if hasattr(image, "add_local_dir"):
    image = image.add_local_dir(str(LOCAL_DATA_LOADER_DIR), remote_path=f"{REMOTE_CODE_ROOT}/data_loader")
elif hasattr(modal, "Mount"):
    function_mounts = [modal.Mount.from_local_dir(LOCAL_DATA_LOADER_DIR, remote_path=f"{REMOTE_CODE_ROOT}/data_loader")]
else:
    raise RuntimeError("Modal SDK does not support add_local_dir or Mount.from_local_dir.")

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


DEFAULT_CONFIG = {
    "base_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_root": REMOTE_OUTPUT_ROOT,
    "adapter_run_id": "",
    "adapter_subdir": "adapter",
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "max_real_samples": 1000,
    "shuffle_buffer_size": 1000,
    "random_seed": 42,
    "eval_batch_size": 1,
    "num_workers": 0,
    "load_in_4bit": True,
    "min_pixels": 64 * 28 * 28,
    "max_pixels": 128 * 28 * 28,
    "normalize_candidate_logprob": True,
    "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
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
    },
}
if function_mounts:
    function_options["mounts"] = function_mounts


@app.function(**function_options)
def infer_qwen25vl_lora_commfor_real_only(config_overrides: dict | None = None) -> dict:
    import gc
    import io
    import json
    import sys
    import time
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import PeftModel
    from PIL import Image, ImageFile
    from torch.utils.data import DataLoader, IterableDataset
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from data_loader import UnifiedSample, collate_unified_batch  # noqa: PLC0415

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if isinstance(config.get("thresholds"), str):
        config["thresholds"] = [float(item.strip()) for item in config["thresholds"].split(",") if item.strip()]

    if not config["adapter_run_id"]:
        raise ValueError("adapter_run_id is required. Example: --adapter-run-id 20260910_003026")

    label_id_to_text = {0: "real", 1: "fake"}
    candidate_texts = ["real", "fake"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pin_memory = torch.cuda.is_available()

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

    inference_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = run_root / f"real_only_commfor_{inference_id}"
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

    def first_nonempty(*values: Any, default: str = "unknown") -> str:
        for value in values:
            if value is None:
                continue
            text = str(value)
            if text and text.lower() not in {"none", "nan"}:
                return text
        return default

    def image_from_record(record: dict[str, Any]) -> Image.Image:
        image_data = record.get("image_data", record.get("image"))
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        if isinstance(image_data, dict):
            if image_data.get("bytes") is not None:
                return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
            if image_data.get("path") is not None:
                return Image.open(image_data["path"]).convert("RGB")
        if isinstance(image_data, list):
            return Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
        raise TypeError(f"Unsupported image_data type: {type(image_data)}")

    def generator_from_record(record: dict[str, Any]) -> str:
        return first_nonempty(record.get("model_name"), record.get("architecture"), record.get("real_source"))

    def record_to_sample(record: dict[str, Any], index: int) -> dict[str, Any]:
        label = int(record.get("label"))
        if label != 0:
            raise ValueError(f"Real-only dataset received non-real label: {label}")
        sample = UnifiedSample(
            sample_id=str(record.get("image_name") or f"commfor_real_only:{index}"),
            label=0,
            label_name="real",
            dataset_source=config["commfor_dataset_name"],
            generator=generator_from_record(record),
            split=str(record.get("split") or config["commfor_split"]),
            eval_case="commfor_real_only",
            prompt=record.get("prompt"),
            metadata={
                "image_name": record.get("image_name"),
                "format": record.get("format"),
                "resolution": record.get("resolution"),
                "mode": record.get("mode"),
                "model_name": record.get("model_name"),
                "architecture": record.get("architecture"),
                "real_source": record.get("real_source"),
                "subset": record.get("subset"),
                "nsfw_flag": record.get("nsfw_flag"),
            },
        ).as_dict()
        sample["image_name"] = record.get("image_name")
        sample["model_name"] = record.get("model_name")
        sample["architecture"] = record.get("architecture")
        sample["real_source"] = record.get("real_source")
        sample["subset"] = record.get("subset")
        sample["nsfw_flag"] = record.get("nsfw_flag")
        sample["image"] = image_from_record(record)
        return sample

    class RealOnlyCommunityForensicsDataset(IterableDataset):
        def __init__(self, hf_dataset, max_real_samples: int):
            self.hf_dataset = hf_dataset
            self.max_real_samples = max_real_samples

        def __iter__(self):
            yielded = 0
            scanned = 0
            for record in self.hf_dataset:
                scanned += 1
                if int(record.get("label")) != 0:
                    continue
                yield record_to_sample(record, yielded)
                yielded += 1
                if yielded >= self.max_real_samples:
                    break
            print("Real-only stream finished:", {"scanned": scanned, "yielded_real": yielded})

    def build_real_only_loader():
        hf_dataset = load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=True,
        )
        hf_dataset = hf_dataset.shuffle(
            seed=config["random_seed"],
            buffer_size=config["shuffle_buffer_size"],
        )
        dataset = RealOnlyCommunityForensicsDataset(hf_dataset, int(config["max_real_samples"]))
        return DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )

    def load_model_and_processor():
        quantization_config = None
        if config["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=amp_dtype,
            )
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
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

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

    def make_user_messages(image: Image.Image) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config["prompt_template"]},
                ],
            }
        ]

    def make_full_messages(image: Image.Image, answer: str) -> list[dict[str, Any]]:
        return make_user_messages(image) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

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
    def score_candidate_batch(batch: dict[str, Any], candidate_text: str) -> np.ndarray:
        images = batch["image"]
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

    def threshold_sweep(pred_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([summarize_real_only(pred_df, threshold) for threshold in config["thresholds"]])

    def save_group_tables(pred_df: pd.DataFrame) -> None:
        for column in ["generator", "architecture", "real_source", "subset"]:
            if column not in pred_df.columns:
                continue
            counts = (
                pred_df.groupby(column, dropna=False)
                .size()
                .reset_index(name="num_samples")
                .sort_values("num_samples", ascending=False)
            )
            counts.to_csv(metrics_dir / f"community_forensics_real_only_{column}_counts.csv", index=False)

            rows = []
            for value, part in pred_df.groupby(column, dropna=False):
                metrics = summarize_real_only(part, threshold=0.5)
                metrics[column] = value
                rows.append(metrics)
            pd.DataFrame(rows).sort_values(column).to_csv(
                metrics_dir / f"community_forensics_real_only_{column}_metrics.csv",
                index=False,
            )

    def predict_real_only(loader) -> pd.DataFrame:
        model.eval()
        rows = []
        progress = tqdm(loader, desc="community_forensics_real_only", total=config["max_real_samples"], leave=False)
        for batch in progress:
            real_scores = score_candidate_batch(batch, "real")
            fake_scores = score_candidate_batch(batch, "fake")
            scores = np.stack([real_scores, fake_scores], axis=1)
            probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()
            fake_probs = probs[:, 1]
            for metadata, fake_prob in zip(batch["metadata"], fake_probs):
                row = dict(metadata)
                row["label"] = 0
                row["label_name"] = "real"
                row["fake_probability"] = float(fake_prob)
                row["predicted_label"] = int(fake_prob >= 0.5)
                row["predicted_label_name"] = label_id_to_text[row["predicted_label"]]
                row["model_name"] = "qwen25vl_lora_word_label"
                row["dataset_tag"] = "community_forensics_real_only"
                row["adapter_run_id"] = config["adapter_run_id"]
                row["adapter_subdir"] = config["adapter_subdir"]
                rows.append(row)
            progress.update(0)
        return pd.DataFrame(rows)

    print("Real-only inference ID:", inference_id)
    print("Device:", device)
    print("Adapter dir:", adapter_dir)
    print("Max real samples:", config["max_real_samples"])
    print("Shuffle buffer size:", config["shuffle_buffer_size"])

    save_json(metrics_dir / "inference_config.json", {**config, "adapter_dir": str(adapter_dir)})
    model, processor = load_model_and_processor()
    for text in candidate_texts:
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        print(repr(text), token_ids, "num_tokens=", len(token_ids))

    pred_df = predict_real_only(build_real_only_loader())
    if config["save_predictions"]:
        pred_df.to_csv(predictions_dir / "community_forensics_real_only_predictions.csv", index=False)

    metrics = summarize_real_only(pred_df, threshold=0.5)
    metrics.update(
        {
            "model_name": "qwen25vl_lora_word_label",
            "dataset_tag": "community_forensics_real_only",
            "adapter_run_id": config["adapter_run_id"],
            "adapter_subdir": config["adapter_subdir"],
            "adapter_dir": str(adapter_dir),
            "random_seed": config["random_seed"],
            "shuffle_buffer_size": config["shuffle_buffer_size"],
            "candidate_texts": candidate_texts,
            "normalize_candidate_logprob": config["normalize_candidate_logprob"],
            "inference_id": inference_id,
            "output_dir": str(output_dir),
        }
    )
    save_json(metrics_dir / "community_forensics_real_only_metrics.json", metrics)
    save_json(metrics_dir / "summary.json", metrics)

    threshold_sweep_df = threshold_sweep(pred_df)
    threshold_sweep_df.to_csv(metrics_dir / "community_forensics_real_only_threshold_sweep.csv", index=False)
    save_group_tables(pred_df)
    pd.DataFrame([metrics]).to_csv(run_root / f"real_only_commfor_summary_{inference_id}.csv", index=False)

    print("community_forensics_real_only", json.dumps(metrics, ensure_ascii=False, indent=2))
    output_volume.commit()
    hf_cache_volume.commit()

    del model
    cleanup_cuda()
    return metrics


@app.local_entrypoint()
def main(
    adapter_run_id: str = "",
    adapter_subdir: str = "adapter",
    max_real_samples: int = 1000,
    random_seed: int = 42,
    shuffle_buffer_size: int = 1000,
    eval_batch_size: int = 1,
    thresholds: str = "0.5,0.6,0.7,0.8,0.9",
):
    overrides = {
        "adapter_run_id": adapter_run_id,
        "adapter_subdir": adapter_subdir,
        "max_real_samples": max_real_samples,
        "random_seed": random_seed,
        "shuffle_buffer_size": shuffle_buffer_size,
        "eval_batch_size": eval_batch_size,
        "thresholds": thresholds,
    }
    summary = infer_qwen25vl_lora_commfor_real_only.remote(overrides)
    print("Remote real-only inference summary:")
    print(summary)

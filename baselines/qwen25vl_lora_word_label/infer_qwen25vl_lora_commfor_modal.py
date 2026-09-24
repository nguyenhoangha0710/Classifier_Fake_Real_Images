"""Run Qwen2.5-VL LoRA inference on CommunityForensics-Eval with Modal.

Example:

    modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py \
        --adapter-run-id 20260910_003026 --max-samples 1000
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "qwen25vl-lora-word-label-inference"
GPU_TYPE = "A100-40GB"

OUTPUT_VOLUME_NAME = "qwen25vl-lora-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_OUTPUT_ROOT = "/outputs/qwen25vl_lora_word_label"
REMOTE_HF_HOME = "/hf-cache"


def find_data_loader_dir() -> Path:
    """Find data_loader both locally and inside Modal."""
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd() / "data_loader", Path(REMOTE_CODE_ROOT) / "data_loader"]
    candidates.extend(parent / "data_loader" for parent in [file_path.parent, *file_path.parents])
    for candidate in candidates:
        if (candidate / "__init__.py").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find data_loader/. Run this script from the repository root "
        "or keep HoangHa_Code/data_loader present."
    )


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
    "scikit-learn<1.9",
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
    image = image.add_local_dir(
        str(LOCAL_DATA_LOADER_DIR),
        remote_path=f"{REMOTE_CODE_ROOT}/data_loader",
    )
elif hasattr(modal, "Mount"):
    function_mounts = [
        modal.Mount.from_local_dir(
            LOCAL_DATA_LOADER_DIR,
            remote_path=f"{REMOTE_CODE_ROOT}/data_loader",
        )
    ]
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
    "commfor_streaming": True,
    "max_samples": 1000,
    "sample_strategy": "streaming_shuffle",
    "shuffle_buffer_size": 100,
    "discover_scan_limit": 50000,
    "max_generators": None,
    "target_generators": [
        "IdeogramV1",
        "IdeogramV2",
        "kvikontent_midjourney_v6",
        "DeciDiffusionV2",
        "stable_cascade",
        "MidjourneyV5_2",
        "MidjourneyV6_1",
        "DFGAN",
        "Firefly_Image2",
    ],
    "random_seed": 42,
    "eval_batch_size": 1,
    "num_workers": 0,
    "load_in_4bit": True,
    "min_pixels": 64 * 28 * 28,
    "max_pixels": 128 * 28 * 28,
    "normalize_candidate_logprob": True,
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
def infer_qwen25vl_lora_commfor(config_overrides: dict | None = None) -> dict:
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
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import DataLoader, IterableDataset
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from data_loader import UnifiedSample, collate_unified_batch  # noqa: PLC0415

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if isinstance(config.get("target_generators"), str):
        config["target_generators"] = [
            item.strip() for item in config["target_generators"].split(",") if item.strip()
        ]

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

    adapter_dir = next(
        (candidate for candidate in adapter_candidates if (candidate / "adapter_config.json").exists()),
        None,
    )
    if adapter_dir is None:
        checked = "\n".join(str(candidate) for candidate in adapter_candidates)
        raise FileNotFoundError(
            "Cannot find LoRA adapter. Checked:\n"
            f"{checked}\n"
            "Use --adapter-subdir adapter for final adapter or "
            "--adapter-subdir checkpoints/latest/adapter for latest checkpoint adapter."
        )

    inference_id = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(config["output_root"]) / config["adapter_run_id"] / f"inference_commfor_{inference_id}"
    metrics_dir = run_output_dir / "metrics"
    predictions_dir = run_output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, data: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def image_from_commfor_record(record: dict[str, Any]) -> Image.Image:
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

    def generator_from_commfor_record(record: dict[str, Any]) -> str:
        return str(record.get("model_name") or record.get("architecture") or "unknown")

    def make_balanced_generator_quotas() -> dict[str, int]:
        target_generators = list(config["target_generators"])
        if not target_generators:
            raise ValueError("target_generators must not be empty for balanced_generator sampling.")
        max_samples = int(config["max_samples"])
        base_quota = max_samples // len(target_generators)
        remainder = max_samples % len(target_generators)
        return {
            generator: base_quota + (1 if idx < remainder else 0)
            for idx, generator in enumerate(target_generators)
        }

    def make_quotas_for_generators(target_generators: list[str]) -> dict[str, int]:
        if not target_generators:
            raise ValueError("Cannot build balanced quotas from an empty generator list.")
        max_samples = int(config["max_samples"])
        base_quota = max_samples // len(target_generators)
        remainder = max_samples % len(target_generators)
        return {
            generator: base_quota + (1 if idx < remainder else 0)
            for idx, generator in enumerate(target_generators)
        }

    def build_raw_commfor_stream():
        return load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=config["commfor_streaming"],
        )

    def shuffled_commfor_stream(seed_offset: int = 0):
        hf_dataset = build_raw_commfor_stream()
        if config["commfor_streaming"]:
            hf_dataset = hf_dataset.shuffle(
                seed=config["random_seed"] + seed_offset,
                buffer_size=config["shuffle_buffer_size"],
            )
        elif config["max_samples"] is not None:
            hf_dataset = hf_dataset.shuffle(seed=config["random_seed"] + seed_offset)
        return hf_dataset

    def discover_generators_from_stream() -> list[str]:
        if not config["commfor_streaming"]:
            raise ValueError("discover_then_balance currently expects commfor_streaming=True.")
        counts: dict[str, int] = {}
        label_counts: dict[tuple[str, str], int] = {}
        scan_limit = config["discover_scan_limit"]
        stream = shuffled_commfor_stream(seed_offset=0)
        for scanned, record in enumerate(stream, start=1):
            generator = generator_from_commfor_record(record)
            label_name = label_id_to_text.get(int(record.get("label")), str(record.get("label")))
            counts[generator] = counts.get(generator, 0) + 1
            label_counts[(generator, label_name)] = label_counts.get((generator, label_name), 0) + 1
            if scan_limit is not None and scanned >= int(scan_limit):
                break

        discovered_counts = (
            pd.DataFrame(
                [{"generator": generator, "num_seen": count} for generator, count in counts.items()]
            )
            .sort_values("num_seen", ascending=False)
            .reset_index(drop=True)
        )
        discovered_counts.to_csv(metrics_dir / "discovered_generator_counts.csv", index=False)

        discovered_label_counts = (
            pd.DataFrame(
                [
                    {"generator": generator, "label_name": label_name, "num_seen": count}
                    for (generator, label_name), count in label_counts.items()
                ]
            )
            .sort_values(["generator", "label_name"])
            .reset_index(drop=True)
        )
        discovered_label_counts.to_csv(metrics_dir / "discovered_generator_label_counts.csv", index=False)

        target_generators = discovered_counts["generator"].tolist()
        if config["max_generators"] is not None:
            target_generators = target_generators[: int(config["max_generators"])]
        config["target_generators"] = target_generators
        print("Discovered generators:", target_generators)
        return target_generators

    def commfor_record_to_sample(record: dict[str, Any], index: int, eval_case: str) -> dict[str, Any]:
        image = image_from_commfor_record(record)
        label = int(record.get("label"))
        model_name = generator_from_commfor_record(record)
        sample = UnifiedSample(
            sample_id=str(record.get("image_name") or f"commfor_eval:{index}"),
            label=label,
            label_name="fake" if label == 1 else "real",
            dataset_source=config["commfor_dataset_name"],
            generator=model_name,
            split=str(record.get("split") or config["commfor_split"]),
            eval_case=eval_case,
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
        sample["image"] = image
        return sample

    class CommunityForensicsEvalIterableDataset(IterableDataset):
        def __init__(self, hf_dataset, eval_case: str = "cross_dataset_commfor_eval"):
            self.hf_dataset = hf_dataset
            self.eval_case = eval_case

        def __iter__(self):
            for index, record in enumerate(self.hf_dataset):
                yield commfor_record_to_sample(record, index, self.eval_case)

    class BalancedGeneratorCommunityForensicsEvalIterableDataset(IterableDataset):
        def __init__(self, hf_dataset, quotas: dict[str, int], eval_case: str = "cross_dataset_commfor_eval"):
            self.hf_dataset = hf_dataset
            self.quotas = quotas
            self.eval_case = eval_case

        def __iter__(self):
            counts = {generator: 0 for generator in self.quotas}
            total_target = sum(self.quotas.values())
            yielded = 0
            scanned = 0
            for record in self.hf_dataset:
                generator = generator_from_commfor_record(record)
                scanned += 1
                if generator not in self.quotas or counts[generator] >= self.quotas[generator]:
                    continue
                counts[generator] += 1
                yielded += 1
                yield commfor_record_to_sample(record, yielded - 1, self.eval_case)
                if yielded >= total_target:
                    break
            missing = {
                generator: self.quotas[generator] - count
                for generator, count in counts.items()
                if count < self.quotas[generator]
            }
            if missing:
                print(
                    "Warning: balanced_generator sampling ended before all quotas were filled.",
                    "scanned=",
                    scanned,
                    "yielded=",
                    yielded,
                    "missing=",
                    missing,
                )

    def build_commfor_eval_loader():
        if config["sample_strategy"] == "discover_then_balance":
            target_generators = discover_generators_from_stream()
            quotas = make_quotas_for_generators(target_generators)
            print("Discovered balanced generator quotas:", quotas)
            dataset = BalancedGeneratorCommunityForensicsEvalIterableDataset(
                shuffled_commfor_stream(seed_offset=1),
                quotas,
            )
        elif config["sample_strategy"] == "balanced_generator":
            quotas = make_balanced_generator_quotas()
            print("Balanced generator quotas:", quotas)
            dataset = BalancedGeneratorCommunityForensicsEvalIterableDataset(
                shuffled_commfor_stream(seed_offset=0),
                quotas,
            )
        elif config["sample_strategy"] == "streaming_shuffle":
            hf_dataset = shuffled_commfor_stream(seed_offset=0)
            if config["max_samples"] is not None:
                hf_dataset = hf_dataset.take(config["max_samples"])
            dataset = CommunityForensicsEvalIterableDataset(hf_dataset)
        else:
            raise ValueError(f"Unknown sample_strategy={config['sample_strategy']!r}")
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
        return make_user_messages(image) + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

    def move_inputs_to_device(inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def sequence_logprob_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
        normalize_by_length: bool,
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_scores = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_scores = token_scores * mask
        scores = token_scores.sum(dim=-1)
        if normalize_by_length:
            scores = scores / mask.sum(dim=-1).clamp_min(1)
        return scores

    @torch.no_grad()
    def score_candidate_batch(batch: dict[str, Any], candidate_text: str) -> np.ndarray:
        images = batch["image"]
        prompt_lengths = []
        full_texts = []
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
            prompt_inputs = processor(
                text=[prompt_text],
                images=[image_item],
                padding=False,
                return_tensors="pt",
            )
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
        scores = sequence_logprob_from_logits(
            outputs.logits,
            candidate_labels,
            normalize_by_length=config["normalize_candidate_logprob"],
        )
        return scores.detach().cpu().numpy()

    def compute_metrics(y_true, y_prob) -> dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        return metrics

    def evaluate_by_column(pred_df: pd.DataFrame, column: str) -> pd.DataFrame:
        rows = []
        if column not in pred_df.columns:
            return pd.DataFrame(rows)
        for value, part in pred_df.groupby(column, dropna=False):
            metrics = compute_metrics(part["label"].to_numpy(), part["fake_probability"].to_numpy())
            metrics[column] = value
            metrics["num_samples"] = int(len(part))
            rows.append(metrics)
        return pd.DataFrame(rows).sort_values(column) if rows else pd.DataFrame(rows)

    def save_count_tables(pred_df: pd.DataFrame, dataset_tag: str) -> None:
        for column in ["generator", "architecture"]:
            if column not in pred_df.columns:
                continue
            total_counts = (
                pred_df.groupby(column, dropna=False)
                .size()
                .reset_index(name="num_samples")
                .sort_values("num_samples", ascending=False)
            )
            total_counts.to_csv(metrics_dir / f"{dataset_tag}_{column}_counts.csv", index=False)
            label_counts = (
                pred_df.groupby([column, "label_name"], dropna=False)
                .size()
                .reset_index(name="num_samples")
                .sort_values([column, "label_name"])
            )
            label_counts.to_csv(metrics_dir / f"{dataset_tag}_{column}_label_counts.csv", index=False)

    def predict_word_label(loader, dataset_tag: str):
        model.eval()
        all_labels, all_fake_probs, rows = [], [], []
        for batch in tqdm(loader, desc=dataset_tag, leave=False):
            real_scores = score_candidate_batch(batch, "real")
            fake_scores = score_candidate_batch(batch, "fake")
            scores = np.stack([real_scores, fake_scores], axis=1)
            probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()
            fake_probs = probs[:, 1]
            all_labels.extend(batch["label"].numpy().tolist())
            all_fake_probs.extend(fake_probs.tolist())
            rows.extend(batch["metadata"])
        return np.array(all_labels, dtype=int), np.array(all_fake_probs, dtype=float), pd.DataFrame(rows)

    def evaluate_and_save(loader, dataset_tag: str) -> dict[str, Any]:
        y_true, y_prob, meta_df = predict_word_label(loader, dataset_tag=dataset_tag)
        pred_df = meta_df.copy()
        pred_df["label"] = y_true
        pred_df["label_name"] = pred_df["label"].map(label_id_to_text)
        pred_df["predicted_label"] = (y_prob >= 0.5).astype(int)
        pred_df["predicted_label_name"] = pred_df["predicted_label"].map(label_id_to_text)
        pred_df["fake_probability"] = y_prob
        pred_df["model_name"] = "qwen25vl_lora_word_label"
        pred_df["dataset_tag"] = dataset_tag
        pred_df["adapter_run_id"] = config["adapter_run_id"]
        pred_df["adapter_subdir"] = config["adapter_subdir"]

        metrics = compute_metrics(y_true, y_prob)
        metrics.update(
            {
                "model_name": "qwen25vl_lora_word_label",
                "dataset_tag": dataset_tag,
                "adapter_run_id": config["adapter_run_id"],
                "adapter_subdir": config["adapter_subdir"],
                "adapter_dir": str(adapter_dir),
                "num_samples": int(len(pred_df)),
                "threshold": 0.5,
                "random_seed": config["random_seed"],
                "shuffle_buffer_size": config["shuffle_buffer_size"],
                "sample_strategy": config["sample_strategy"],
                "discover_scan_limit": config["discover_scan_limit"],
                "max_generators": config["max_generators"],
                "target_generators": config["target_generators"],
                "candidate_texts": candidate_texts,
                "normalize_candidate_logprob": config["normalize_candidate_logprob"],
            }
        )

        save_json(metrics_dir / f"{dataset_tag}_overall_metrics.json", metrics)
        by_generator = evaluate_by_column(pred_df, "generator")
        if len(by_generator):
            by_generator.to_csv(metrics_dir / f"{dataset_tag}_generator_metrics.csv", index=False)
        by_architecture = evaluate_by_column(pred_df, "architecture")
        if len(by_architecture):
            by_architecture.to_csv(metrics_dir / f"{dataset_tag}_architecture_metrics.csv", index=False)
        save_count_tables(pred_df, dataset_tag)
        if config["save_predictions"]:
            pred_df.to_csv(predictions_dir / f"{dataset_tag}_predictions.csv", index=False)
        print(dataset_tag, json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    print("Inference ID:", inference_id)
    print("Device:", device)
    print("Adapter dir:", adapter_dir)
    print("CommFor samples:", config["max_samples"])
    print("Sample strategy:", config["sample_strategy"])

    save_json(metrics_dir / "inference_config.json", {**config, "adapter_dir": str(adapter_dir)})
    model, processor = load_model_and_processor()
    for text in candidate_texts:
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        print(repr(text), token_ids, "num_tokens=", len(token_ids))

    commfor_loader = build_commfor_eval_loader()
    metrics = evaluate_and_save(commfor_loader, dataset_tag="community_forensics_eval")

    summary = {
        **metrics,
        "inference_id": inference_id,
        "output_dir": str(run_output_dir),
    }
    save_json(metrics_dir / "summary.json", summary)
    pd.DataFrame([summary]).to_csv(
        Path(config["output_root"])
        / config["adapter_run_id"]
        / f"inference_commfor_summary_{inference_id}.csv",
        index=False,
    )
    output_volume.commit()
    hf_cache_volume.commit()

    del model, commfor_loader
    cleanup_cuda()
    return summary


@app.local_entrypoint()
def main(
    adapter_run_id: str = "",
    adapter_subdir: str = "adapter",
    max_samples: int = 1000,
    sample_strategy: str = "streaming_shuffle",
    discover_scan_limit: int = 50000,
    max_generators: int | None = None,
    target_generators: str = (
        "IdeogramV1,IdeogramV2,kvikontent_midjourney_v6,DeciDiffusionV2,stable_cascade,"
        "MidjourneyV5_2,MidjourneyV6_1,DFGAN,Firefly_Image2"
    ),
    random_seed: int = 42,
    shuffle_buffer_size: int = 100,
    eval_batch_size: int = 1,
):
    """Start Modal inference using a saved LoRA adapter."""

    overrides = {
        "adapter_run_id": adapter_run_id,
        "adapter_subdir": adapter_subdir,
        "max_samples": max_samples,
        "sample_strategy": sample_strategy,
        "discover_scan_limit": discover_scan_limit,
        "max_generators": max_generators,
        "target_generators": target_generators,
        "random_seed": random_seed,
        "shuffle_buffer_size": shuffle_buffer_size,
        "eval_batch_size": eval_batch_size,
    }
    summary = infer_qwen25vl_lora_commfor.remote(overrides)
    print("Remote inference summary:")
    print(summary)

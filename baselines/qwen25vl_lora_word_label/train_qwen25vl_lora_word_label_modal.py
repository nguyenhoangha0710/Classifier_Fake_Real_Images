"""Run Qwen2.5-VL LoRA word-label fine-tuning on Modal.

Run from the repository root:

    modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py

Expected Tiny-GenImage path inside Modal:

    /data/tiny-genimage/
      imagenet_ai_0419_biggan/
        train/ai
        train/nature
        val/ai
        val/nature
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "qwen25vl-lora-word-label"
GPU_TYPE = "A100-40GB"

TINY_VOLUME_NAME = "tiny-genimage-data"
OUTPUT_VOLUME_NAME = "qwen25vl-lora-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_TINY_ROOT = "/data/tiny-genimage"
REMOTE_OUTPUT_ROOT = "/outputs/qwen25vl_lora_word_label"
REMOTE_HF_HOME = "/hf-cache"


def find_data_loader_dir() -> Path:
    """Find data_loader both when launched locally and when imported inside Modal."""
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd() / "data_loader", Path(REMOTE_CODE_ROOT) / "data_loader"]
    candidates.extend(parent / "data_loader" for parent in [file_path.parent, *file_path.parents])
    for candidate in candidates:
        if (candidate / "__init__.py").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find data_loader/. Run this script from the repository root or keep HoangHa_Code/data_loader present."
    )


LOCAL_DATA_LOADER_DIR = find_data_loader_dir()


image = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install(
    "torch",
    "torchvision",
    "transformers>=4.49.0,<4.58",
    "accelerate",
    "peft",
    "bitsandbytes",
    "datasets",
    "kaggle",
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
tiny_volume = modal.Volume.from_name(TINY_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


DEFAULT_CONFIG = {
    "base_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "tiny_dataset_root": REMOTE_TINY_ROOT,
    "output_root": REMOTE_OUTPUT_ROOT,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "download_tiny_from_kaggle": True,
    "kaggle_dataset_slug": "yangsangtai/tiny-genimage",
    "download_only": False,
    "max_commfor_eval_samples": 1000,
    "commfor_shuffle_buffer_size": 100,
    "max_train_samples": 5000,
    "max_tiny_internal_test_samples": 1000,
    "max_val_samples": 1000,
    "balance_real": True,
    "random_seed": 42,
    "val_fraction": 0.2,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_workers": 0,
    "max_epochs": 2,
    "patience": 1,
    "min_delta": 1e-3,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "load_in_4bit": True,
    "use_gradient_checkpointing": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "min_pixels": 64 * 28 * 28,
    "max_pixels": 128 * 28 * 28,
    "normalize_candidate_logprob": True,
    "prompt_template": (
        "Look at the image and decide whether it is real or AI-generated. "
        "Answer with exactly one word: real or fake."
    ),
    "save_predictions": True,
    "checkpoint_every_optimizer_steps": 25,
    "resume_from_latest_checkpoint": True,
    "resume_run_id": None,
}


function_options = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "secrets": [modal.Secret.from_name("kaggle-secret")],
    "volumes": {
        "/data": tiny_volume,
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    function_options["mounts"] = function_mounts


@app.function(**function_options)
def train_qwen25vl_lora_word_label(config_overrides: dict | None = None) -> dict:
    import gc
    import io
    import json
    import os
    import random
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
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
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, IterableDataset
    from tqdm.auto import tqdm
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from data_loader import (  # noqa: PLC0415
        TinyGenImageKaggleConfig,
        TinyGenImageKaggleDataset,
        UnifiedSample,
        build_kaggle_tiny_index,
        build_kaggle_tiny_splits,
        collate_unified_batch,
        find_tiny_genimage_root,
        summarize_index,
    )

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    label_id_to_text = {0: "real", 1: "fake"}
    candidate_texts = ["real", "fake"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pin_memory = torch.cuda.is_available()

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint_run_id() -> str | None:
        if not output_root.exists():
            return None
        candidate_dirs = sorted(
            path
            for path in output_root.iterdir()
            if path.is_dir() and (path / "checkpoints" / "latest" / "training_state.pt").exists()
        )
        return candidate_dirs[-1].name if candidate_dirs else None

    resumed_run_id = config.get("resume_run_id")
    if resumed_run_id is None and config["resume_from_latest_checkpoint"]:
        resumed_run_id = find_latest_checkpoint_run_id()
    run_id = resumed_run_id or time.strftime("%Y%m%d_%H%M%S")
    config["resume_run_id"] = resumed_run_id

    run_dir = output_root / run_id
    for subdir in ["adapter", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_json(path: Path, data: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def looks_like_tiny_genimage_root(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        for generator_dir in root.iterdir():
            if not generator_dir.is_dir():
                continue
            if (generator_dir / "train").exists() and (
                (generator_dir / "train" / "ai").exists()
                or (generator_dir / "train" / "nature").exists()
                or (generator_dir / "train" / "fake").exists()
                or (generator_dir / "train" / "real").exists()
            ):
                return True
        return False

    def normalize_kaggle_secret_env() -> None:
        if "KAGGLE_USERNAME" not in os.environ and "username" in os.environ:
            os.environ["KAGGLE_USERNAME"] = os.environ["username"]
        if "KAGGLE_KEY" not in os.environ and "key" in os.environ:
            os.environ["KAGGLE_KEY"] = os.environ["key"]
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            raise RuntimeError(
                "Missing Kaggle credentials. Create Modal secret 'kaggle-secret' with "
                "KAGGLE_USERNAME/KAGGLE_KEY or from kaggle.json."
            )

    def find_downloaded_tiny_root(staging_dir: Path) -> Path | None:
        direct_candidates = [
            staging_dir / "tiny-genimage",
            staging_dir / "tiny_genimage",
            staging_dir / "Tiny-GenImage",
            staging_dir,
        ]
        for candidate in direct_candidates:
            if looks_like_tiny_genimage_root(candidate):
                return candidate
        for candidate in staging_dir.rglob("*"):
            if looks_like_tiny_genimage_root(candidate):
                return candidate
        return None

    def download_tiny_genimage_from_kaggle_if_needed() -> Path:
        target_root = Path(config["tiny_dataset_root"])
        if looks_like_tiny_genimage_root(target_root):
            print("Tiny-GenImage already exists in Modal Volume:", target_root)
            return target_root

        if not config["download_tiny_from_kaggle"]:
            return find_tiny_genimage_root(str(target_root))

        normalize_kaggle_secret_env()
        staging_dir = Path("/data/_tiny_genimage_kaggle_download")
        staging_dir.mkdir(parents=True, exist_ok=True)

        if not find_downloaded_tiny_root(staging_dir):
            command = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                config["kaggle_dataset_slug"],
                "-p",
                str(staging_dir),
                "--unzip",
                "-o",
            ]
            print("Downloading Tiny-GenImage from Kaggle:", " ".join(command))
            subprocess.run(command, check=True)
        else:
            print("Found existing Kaggle download staging dir:", staging_dir)

        downloaded_root = find_downloaded_tiny_root(staging_dir)
        if downloaded_root is None:
            found_items = [str(path.relative_to(staging_dir)) for path in staging_dir.iterdir()]
            raise FileNotFoundError(
                "Downloaded Kaggle dataset, but Tiny-GenImage folder structure was not recognized. "
                f"Top-level staging items: {found_items[:30]}"
            )

        target_root.parent.mkdir(parents=True, exist_ok=True)
        if downloaded_root.resolve() != target_root.resolve():
            if target_root.exists() and any(target_root.iterdir()):
                raise FileExistsError(
                    f"Target root exists but does not look like Tiny-GenImage: {target_root}"
                )
            if target_root.exists():
                target_root.rmdir()
            if downloaded_root == staging_dir:
                target_root.mkdir(parents=True, exist_ok=True)
                for child in staging_dir.iterdir():
                    if child.resolve() == target_root.resolve():
                        continue
                    shutil.move(str(child), str(target_root / child.name))
            else:
                shutil.move(str(downloaded_root), str(target_root))

        tiny_volume.commit()
        print("Tiny-GenImage is ready:", target_root)
        return find_tiny_genimage_root(str(target_root))

    def stratified_train_val_split(df: pd.DataFrame, val_fraction: float, seed: int):
        if len(df) < 4 or val_fraction <= 0:
            return df.reset_index(drop=True), df.reset_index(drop=True)
        stratify = df["label"].astype(str) + "_" + df["generator"].astype(str)
        if stratify.value_counts().min() < 2:
            stratify = df["label"]
        if pd.Series(stratify).value_counts().min() < 2:
            stratify = None
        train_df, val_df = train_test_split(
            df,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def maybe_limit_df(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
        if max_samples is None or len(df) <= max_samples:
            return df.reset_index(drop=True)
        return df.sample(n=max_samples, random_state=seed, replace=False).reset_index(drop=True)

    def build_tiny_combined_splits(detected_root: Path) -> dict[str, Any]:
        split_config = TinyGenImageKaggleConfig(
            dataset_root=str(detected_root),
            eval_case="combined",
            balance_real=config["balance_real"],
            seed=config["random_seed"],
            max_train_samples=config["max_train_samples"],
            max_eval_samples=config["max_tiny_internal_test_samples"],
        )
        splits = build_kaggle_tiny_splits(split_config)
        print(splits["notes"])
        print("train real/fake:", splits["train_real_count"], splits["train_fake_count"])
        print("tiny test real/fake:", splits["eval_real_count"], splits["eval_fake_count"])
        return splits

    def build_tiny_loaders(splits: dict[str, Any]):
        train_inner_df, val_inner_df = stratified_train_val_split(
            splits["train_df"],
            val_fraction=config["val_fraction"],
            seed=config["random_seed"],
        )
        val_inner_df = maybe_limit_df(val_inner_df, config["max_val_samples"], config["random_seed"])
        tiny_test_df = splits["eval_df"].reset_index(drop=True)

        print("inner train/val:", len(train_inner_df), len(val_inner_df))
        print("tiny internal test:", len(tiny_test_df))

        train_dataset = TinyGenImageKaggleDataset(train_inner_df, eval_case="combined", transform=None)
        val_dataset = TinyGenImageKaggleDataset(val_inner_df, eval_case="combined", transform=None)
        tiny_test_dataset = TinyGenImageKaggleDataset(tiny_test_df, eval_case="combined", transform=None)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )
        tiny_test_loader = DataLoader(
            tiny_test_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )
        return train_loader, val_loader, tiny_test_loader, train_inner_df, val_inner_df, tiny_test_df

    def build_epoch_train_loader(train_inner_df: pd.DataFrame, epoch: int, start_batch_idx: int = 0):
        epoch_df = train_inner_df.sample(
            frac=1,
            random_state=config["random_seed"] + int(epoch),
        ).reset_index(drop=True)
        start_row = int(start_batch_idx) * int(config["train_batch_size"])
        if start_row > 0:
            epoch_df = epoch_df.iloc[start_row:].reset_index(drop=True)
        epoch_dataset = TinyGenImageKaggleDataset(epoch_df, eval_case="combined", transform=None)
        return DataLoader(
            epoch_dataset,
            batch_size=config["train_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )

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

    class CommunityForensicsEvalIterableDataset(IterableDataset):
        def __init__(self, hf_dataset, eval_case: str = "cross_dataset_commfor_eval"):
            self.hf_dataset = hf_dataset
            self.eval_case = eval_case

        def __iter__(self):
            for index, record in enumerate(self.hf_dataset):
                image = image_from_commfor_record(record)
                label = int(record.get("label"))
                model_name = str(record.get("model_name") or record.get("architecture") or "unknown")
                sample = UnifiedSample(
                    sample_id=str(record.get("image_name") or f"commfor_eval:{index}"),
                    label=label,
                    label_name="fake" if label == 1 else "real",
                    dataset_source=config["commfor_dataset_name"],
                    generator=model_name,
                    split=str(record.get("split") or config["commfor_split"]),
                    eval_case=self.eval_case,
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
                yield sample

    def build_commfor_eval_loader():
        hf_dataset = load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=config["commfor_streaming"],
        )
        if config["commfor_streaming"]:
            hf_dataset = hf_dataset.shuffle(
                seed=config["random_seed"],
                buffer_size=config["commfor_shuffle_buffer_size"],
            )
            if config["max_commfor_eval_samples"] is not None:
                hf_dataset = hf_dataset.take(config["max_commfor_eval_samples"])
        elif config["max_commfor_eval_samples"] is not None:
            hf_dataset = hf_dataset.shuffle(seed=config["random_seed"]).select(
                range(config["max_commfor_eval_samples"])
            )

        dataset = CommunityForensicsEvalIterableDataset(hf_dataset)
        return DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )

    def load_qwen_lora_model():
        quantization_config = None
        if config["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=amp_dtype,
            )

        processor = AutoProcessor.from_pretrained(
            config["base_model_name"],
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["base_model_name"],
            torch_dtype=amp_dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if config["use_gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if config["load_in_4bit"]:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config["lora_target_modules"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        ]

    def move_inputs_to_device(inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def build_qwen_train_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"].tolist()
        full_texts = []
        prompt_lengths = []

        for image, label_id in zip(images, labels):
            answer = label_id_to_text[int(label_id)]
            prompt_messages = make_user_messages(image)
            full_messages = make_full_messages(image, answer)
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_inputs = processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])
            full_texts.append(full_text)

        inputs = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        target_labels = inputs["input_ids"].clone()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            target_labels[row_idx, :prompt_len] = -100
        target_labels[target_labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = target_labels
        return move_inputs_to_device(inputs)

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
            lengths = mask.sum(dim=-1).clamp_min(1)
            scores = scores / lengths
        return scores

    @torch.no_grad()
    def score_candidate_batch(batch: dict[str, Any], candidate_text: str) -> np.ndarray:
        images = batch["image"]
        prompt_lengths = []
        full_texts = []
        for image in images:
            prompt_messages = make_user_messages(image)
            full_messages = make_full_messages(image, candidate_text)
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_inputs = processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])
            full_texts.append(full_text)

        inputs = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
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
        for value, part in pred_df.groupby(column):
            if len(part) == 0:
                continue
            metrics = compute_metrics(part["label"].to_numpy(), part["fake_probability"].to_numpy())
            metrics[column] = value
            metrics["num_samples"] = int(len(part))
            rows.append(metrics)
        return pd.DataFrame(rows).sort_values(column) if rows else pd.DataFrame(rows)

    @torch.no_grad()
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

    def evaluate_and_save(loader, dataset_tag: str):
        y_true, y_prob, meta_df = predict_word_label(loader, dataset_tag=dataset_tag)
        pred_df = meta_df.copy()
        pred_df["label"] = y_true
        pred_df["predicted_label"] = (y_prob >= 0.5).astype(int)
        pred_df["fake_probability"] = y_prob
        pred_df["model_name"] = "qwen25vl_lora_word_label"
        pred_df["dataset_tag"] = dataset_tag

        metrics = compute_metrics(y_true, y_prob)
        metrics.update(
            {
                "model_name": "qwen25vl_lora_word_label",
                "dataset_tag": dataset_tag,
                "num_samples": int(len(pred_df)),
                "threshold": 0.5,
                "candidate_texts": candidate_texts,
                "normalize_candidate_logprob": config["normalize_candidate_logprob"],
            }
        )

        save_json(run_dir / "metrics" / f"{dataset_tag}_overall_metrics.json", metrics)
        by_generator = evaluate_by_column(pred_df, "generator")
        if len(by_generator):
            by_generator.to_csv(run_dir / "metrics" / f"{dataset_tag}_generator_metrics.csv", index=False)
        by_architecture = evaluate_by_column(pred_df, "architecture")
        if len(by_architecture):
            by_architecture.to_csv(run_dir / "metrics" / f"{dataset_tag}_architecture_metrics.csv", index=False)
        if config["save_predictions"]:
            pred_df.to_csv(run_dir / "predictions" / f"{dataset_tag}_predictions.csv", index=False)
        print(dataset_tag, json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    print("Run ID:", run_id)
    print("Device:", device)
    print("AMP dtype:", amp_dtype)
    print("Tiny dataset root:", config["tiny_dataset_root"])
    seed_everything(config["random_seed"])

    detected_tiny_root = download_tiny_genimage_from_kaggle_if_needed()
    print("Detected Tiny-GenImage root:", detected_tiny_root)
    index_df = build_kaggle_tiny_index(TinyGenImageKaggleConfig(dataset_root=str(detected_tiny_root)))
    structure_summary = summarize_index(index_df)
    structure_summary_path = run_dir / "metrics" / f"tiny_structure_summary_{run_id}.csv"
    structure_summary.to_csv(structure_summary_path, index=False)
    if config["download_only"]:
        summary = {
            "run_id": run_id,
            "method": "download_tiny_genimage_from_kaggle",
            "tiny_dataset_root": str(detected_tiny_root),
            "kaggle_dataset_slug": config["kaggle_dataset_slug"],
            "structure_summary_path": str(structure_summary_path),
            "num_images": int(len(index_df)),
        }
        save_json(run_dir / "metrics" / "download_summary.json", summary)
        output_volume.commit()
        tiny_volume.commit()
        print("Download-only finished:", json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    tiny_splits = build_tiny_combined_splits(detected_tiny_root)
    train_loader, val_loader, tiny_test_loader, train_inner_df, val_inner_df, tiny_test_df = build_tiny_loaders(
        tiny_splits
    )
    train_inner_df.to_csv(run_dir / "metrics" / "tiny_train_inner_split.csv", index=False)
    val_inner_df.to_csv(run_dir / "metrics" / "tiny_val_inner_split.csv", index=False)
    tiny_test_df.to_csv(run_dir / "metrics" / "tiny_internal_test_split.csv", index=False)

    model, processor = load_qwen_lora_model()
    print("Model device:", next(model.parameters()).device)
    print("Token check:")
    for text in candidate_texts:
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        print(repr(text), token_ids, "num_tokens=", len(token_ids))

    config_summary = {
        **config,
        "run_id": run_id,
        "method": "qwen25vl_lora_word_label_modal",
        "train_dataset": "Tiny-GenImage",
        "train_eval_case": "combined",
        "label_id_to_text": label_id_to_text,
        "candidate_texts": candidate_texts,
        "device": device,
        "amp_dtype": str(amp_dtype),
        "gpu_type": GPU_TYPE,
        "tiny_train_rows": int(len(tiny_splits["train_df"])),
        "tiny_eval_rows": int(len(tiny_splits["eval_df"])),
    }
    save_json(run_dir / "metrics" / "config.json", config_summary)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_metric = -float("inf")
    best_epoch = 0
    best_adapter_state = None
    bad_epochs = 0
    history = []
    global_step = 0
    resume_start_epoch = 1
    resume_start_batch_idx = 0
    latest_checkpoint_dir = run_dir / "checkpoints" / "latest"

    def move_optimizer_state_to_device() -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def save_training_checkpoint(epoch: int, next_batch_idx: int, reason: str) -> None:
        latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lora_state_path = latest_checkpoint_dir / "lora_state_dict.pt"
        training_state_path = latest_checkpoint_dir / "training_state.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "next_batch_idx": int(next_batch_idx),
                "global_step": int(global_step),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
                "history": history,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_summary,
                "reason": reason,
            },
            training_state_path,
        )
        torch.save(get_peft_model_state_dict(model), lora_state_path)
        model.save_pretrained(latest_checkpoint_dir / "adapter")
        processor.save_pretrained(latest_checkpoint_dir / "adapter")
        save_json(
            latest_checkpoint_dir / "checkpoint_info.json",
            {
                "run_id": run_id,
                "epoch": int(epoch),
                "next_batch_idx": int(next_batch_idx),
                "global_step": int(global_step),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
                "reason": reason,
                "resume_command": (
                    "modal run baselines/qwen25vl_lora_word_label/"
                    f"train_qwen25vl_lora_word_label_modal.py --resume-run-id {run_id}"
                ),
            },
        )
        output_volume.commit()
        print(
            "Saved training checkpoint:",
            latest_checkpoint_dir,
            f"epoch={epoch}",
            f"next_batch_idx={next_batch_idx}",
            f"global_step={global_step}",
            f"reason={reason}",
        )

    def load_training_checkpoint_if_available() -> None:
        nonlocal best_metric, best_epoch, bad_epochs, history, global_step
        nonlocal resume_start_epoch, resume_start_batch_idx
        training_state_path = latest_checkpoint_dir / "training_state.pt"
        lora_state_path = latest_checkpoint_dir / "lora_state_dict.pt"
        if not config["resume_from_latest_checkpoint"] or not training_state_path.exists():
            return
        print("Loading training checkpoint:", latest_checkpoint_dir)
        training_state = torch.load(training_state_path, map_location="cpu")
        lora_state = torch.load(lora_state_path, map_location="cpu")
        set_peft_model_state_dict(model, lora_state)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        move_optimizer_state_to_device()
        resume_start_epoch = int(training_state.get("epoch", 1))
        resume_start_batch_idx = int(training_state.get("next_batch_idx", 0))
        global_step = int(training_state.get("global_step", 0))
        best_metric = float(training_state.get("best_metric", -float("inf")))
        best_epoch = int(training_state.get("best_epoch", 0))
        bad_epochs = int(training_state.get("bad_epochs", 0))
        history = list(training_state.get("history", []))
        print(
            "Resumed checkpoint:",
            f"run_id={run_id}",
            f"epoch={resume_start_epoch}",
            f"next_batch_idx={resume_start_batch_idx}",
            f"global_step={global_step}",
            f"best_metric={best_metric}",
        )

    load_training_checkpoint_if_available()

    for epoch in range(resume_start_epoch, config["max_epochs"] + 1):
        model.train()
        epoch_losses = []
        step = 0
        start_batch_idx = resume_start_batch_idx if epoch == resume_start_epoch else 0
        optimizer.zero_grad(set_to_none=True)

        epoch_train_loader = build_epoch_train_loader(train_inner_df, epoch, start_batch_idx=start_batch_idx)
        progress = tqdm(
            epoch_train_loader,
            desc=f"train epoch {epoch}/{config['max_epochs']}",
            leave=False,
        )
        for local_step, batch in enumerate(progress, start=1):
            step = start_batch_idx + local_step
            inputs = build_qwen_train_inputs(batch)
            outputs = model(**inputs)
            loss = outputs.loss / config["gradient_accumulation_steps"]
            loss.backward()

            if step % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if (
                    config["checkpoint_every_optimizer_steps"] is not None
                    and config["checkpoint_every_optimizer_steps"] > 0
                    and global_step % config["checkpoint_every_optimizer_steps"] == 0
                ):
                    save_training_checkpoint(
                        epoch=epoch,
                        next_batch_idx=step,
                        reason="periodic_optimizer_step",
                    )

            epoch_losses.append(float(loss.detach().cpu()) * config["gradient_accumulation_steps"])
            progress.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        if step % config["gradient_accumulation_steps"] != 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            save_training_checkpoint(
                epoch=epoch,
                next_batch_idx=step,
                reason="end_of_epoch_train",
            )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        y_val, p_val, _ = predict_word_label(val_loader, dataset_tag="tiny_val")
        val_metrics = compute_metrics(y_val, p_val)
        current = val_metrics["balanced_accuracy"]
        history_row = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss, **val_metrics}
        history.append(history_row)
        pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
        print(f"epoch {epoch}/{config['max_epochs']} loss={train_loss:.4f} val_bal_acc={current:.4f}")

        if current > best_metric + config["min_delta"]:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            best_adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model).items()
            }
            set_peft_model_state_dict(model, best_adapter_state)
            model.save_pretrained(run_dir / "adapter")
            processor.save_pretrained(run_dir / "adapter")
            output_volume.commit()
            print("Saved best adapter:", run_dir / "adapter")
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break
        save_training_checkpoint(
            epoch=epoch + 1,
            next_batch_idx=0,
            reason="end_of_epoch_validation",
        )
        resume_start_batch_idx = 0

    if best_adapter_state is not None:
        set_peft_model_state_dict(model, best_adapter_state)
    else:
        model.save_pretrained(run_dir / "adapter")
        processor.save_pretrained(run_dir / "adapter")

    print("Evaluating Tiny-GenImage validation split for sanity check...")
    tiny_metrics = evaluate_and_save(tiny_test_loader, dataset_tag="tiny_genimage_validation")

    print("Evaluating CommunityForensics-Eval random streaming sample...")
    commfor_loader = build_commfor_eval_loader()
    commfor_metrics = evaluate_and_save(commfor_loader, dataset_tag="community_forensics_eval")

    summary = {
        **config_summary,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        "adapter_path": str(run_dir / "adapter"),
        "tiny_balanced_accuracy": tiny_metrics.get("balanced_accuracy"),
        "tiny_f1": tiny_metrics.get("f1"),
        "tiny_roc_auc": tiny_metrics.get("roc_auc"),
        "commfor_balanced_accuracy": commfor_metrics.get("balanced_accuracy"),
        "commfor_f1": commfor_metrics.get("f1"),
        "commfor_roc_auc": commfor_metrics.get("roc_auc"),
        "commfor_average_precision": commfor_metrics.get("average_precision"),
    }
    save_json(run_dir / "metrics" / "summary.json", summary)

    summary_df = pd.DataFrame([summary])
    summary_path = output_root / f"summary_qwen25vl_lora_word_label_{run_id}.csv"
    manifest_path = output_root / f"adapter_manifest_{run_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df[
        [
            "run_id",
            "base_model_name",
            "adapter_path",
            "best_epoch",
            "best_val_balanced_accuracy",
            "tiny_balanced_accuracy",
            "tiny_roc_auc",
            "commfor_balanced_accuracy",
            "commfor_roc_auc",
            "commfor_average_precision",
        ]
    ].to_csv(manifest_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved adapter manifest:", manifest_path)
    output_volume.commit()
    hf_cache_volume.commit()

    del train_loader, val_loader, tiny_test_loader, commfor_loader, model, optimizer
    cleanup_cuda()
    return summary


@app.local_entrypoint()
def main(
    max_train_samples: int = 5000,
    full_train: bool = False,
    download_only: bool = False,
    max_epochs: int = 2,
    max_val_samples: int = 1000,
    max_commfor_eval_samples: int = 1000,
    gradient_accumulation_steps: int = 16,
    kaggle_dataset_slug: str = "yangsangtai/tiny-genimage",
    checkpoint_every_optimizer_steps: int = 25,
    resume_run_id: str | None = None,
    no_resume: bool = False,
):
    """Start the Modal training job.

    Examples:

        modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py

        modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --full-train --max-epochs 1

        modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --download-only

        modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --resume-run-id 20260909_184050
    """

    overrides = {
        "max_train_samples": None if full_train else max_train_samples,
        "download_only": download_only,
        "max_epochs": max_epochs,
        "max_val_samples": max_val_samples,
        "max_commfor_eval_samples": max_commfor_eval_samples,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "kaggle_dataset_slug": kaggle_dataset_slug,
        "checkpoint_every_optimizer_steps": checkpoint_every_optimizer_steps,
        "resume_run_id": resume_run_id,
        "resume_from_latest_checkpoint": not no_resume,
    }
    summary = train_qwen25vl_lora_word_label.remote(overrides)
    print("Remote run summary:")
    print(summary)

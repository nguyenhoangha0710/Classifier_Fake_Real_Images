"""Tiny-GenImage loader for Kaggle folder-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from PIL import Image

from .schemas import LABEL_ID_TO_NAME, UnifiedSample


DATASET_SOURCE = "Tiny-GenImage-Kaggle"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("val", "valid", "validation", "test"),
    "val": ("val", "valid", "validation", "test"),
    "test": ("test", "val", "valid", "validation"),
}
LABEL_DIR_TO_ID = {
    "nature": 0,
    "real": 0,
    "0_real": 0,
    "0-real": 0,
    "0": 0,
    "ai": 1,
    "fake": 1,
    "1_fake": 1,
    "1-fake": 1,
    "1": 1,
}
KNOWN_GENERATOR_HINTS = (
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "sd14",
    "sd15",
    "stable",
    "vqdm",
    "wukong",
)
GENERATOR_ALIASES = {
    "adm": ("adm", "guided"),
    "biggan": ("biggan",),
    "glide": ("glide",),
    "midjourney": ("midjourney",),
    "sd14": ("sd14", "sdv4", "stablediffusion14", "stablediffusionv14"),
    "sd15": ("sd15", "sdv5", "stablediffusion15", "stablediffusionv15"),
    "vqdm": ("vqdm",),
    "wukong": ("wukong",),
}


KaggleEvalCase = Literal[
    "combined",
    "in_domain",
    "cross_generator",
    "train_one_generator",
]


@dataclass(frozen=True)
class TinyGenImageKaggleConfig:
    """Config for folder-style Tiny-GenImage splits on Kaggle."""

    dataset_root: str | None = None
    kaggle_input_root: str = "/kaggle/input"
    eval_case: KaggleEvalCase = "combined"
    generator: str | None = None
    heldout_generator: str | None = None
    base_generator: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    balance_real: bool = True
    seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def find_label_dirs(split_dir: Path) -> dict[int, Path]:
    label_dirs: dict[int, Path] = {}
    if not split_dir.exists() or not split_dir.is_dir():
        return label_dirs

    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        label_id = LABEL_DIR_TO_ID.get(child.name.lower())
        if label_id is not None:
            label_dirs[label_id] = child
    return label_dirs


def resolve_split_dir(generator_dir: Path, split_name: str) -> Path | None:
    for alias in SPLIT_ALIASES.get(split_name, (split_name,)):
        candidate = generator_dir / alias
        if candidate.exists() and candidate.is_dir() and find_label_dirs(candidate):
            return candidate
    return None


def looks_like_generator_dir(path: Path) -> bool:
    has_split = any(resolve_split_dir(path, split) is not None for split in ("train", "validation"))
    if not has_split:
        return False
    lowered = path.name.lower()
    return any(hint in lowered for hint in KNOWN_GENERATOR_HINTS) or has_split


def find_tiny_genimage_root(
    dataset_root: str | Path | None = None,
    kaggle_input_root: str | Path = "/kaggle/input",
) -> Path:
    """Find the parent folder that contains generator directories."""

    if dataset_root is not None:
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"dataset_root does not exist: {root}")
        return root

    input_root = Path(kaggle_input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"kaggle_input_root does not exist: {input_root}")

    candidates = [input_root]
    candidates.extend([p for p in input_root.rglob("*") if p.is_dir()])
    best_root = None
    best_count = 0

    for candidate in candidates:
        try:
            children = [p for p in candidate.iterdir() if p.is_dir()]
        except OSError:
            continue
        generator_count = sum(1 for child in children if looks_like_generator_dir(child))
        if generator_count > best_count:
            best_root = candidate
            best_count = generator_count

    if best_root is None or best_count == 0:
        raise FileNotFoundError(
            "Cannot find Tiny-GenImage folder structure. Expected generator/train/ai and generator/train/nature."
        )
    return best_root


def list_generator_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    generator_dirs = [p for p in root.iterdir() if p.is_dir() and looks_like_generator_dir(p)]
    return sorted(generator_dirs, key=lambda p: p.name.lower())


def iter_images(path: Path):
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def build_kaggle_tiny_index(config: TinyGenImageKaggleConfig) -> pd.DataFrame:
    root = find_tiny_genimage_root(config.dataset_root, config.kaggle_input_root)
    generator_dirs = list_generator_dirs(root)
    if not generator_dirs:
        raise FileNotFoundError(f"No generator dirs found under {root}")

    records: list[dict[str, Any]] = []
    split_requests = {
        "train": config.train_split,
        "validation": config.validation_split,
    }

    for generator_dir in generator_dirs:
        for canonical_split, requested_split in split_requests.items():
            split_dir = resolve_split_dir(generator_dir, requested_split)
            if split_dir is None:
                continue

            label_dirs = find_label_dirs(split_dir)
            for label, label_dir in label_dirs.items():
                for image_path in iter_images(label_dir):
                    records.append(
                        {
                            "sample_id": f"{generator_dir.name}:{canonical_split}:{label}:{image_path.name}",
                            "image_path": str(image_path),
                            "label": label,
                            "label_name": LABEL_ID_TO_NAME[label],
                            "generator": generator_dir.name,
                            "split": canonical_split,
                            "dataset_source": DATASET_SOURCE,
                        }
                    )

    if not records:
        raise FileNotFoundError(f"No images found under detected Tiny-GenImage root: {root}")

    df = pd.DataFrame(records)
    df.attrs["dataset_root"] = str(root)
    return df


def resolve_generator_name(df: pd.DataFrame, generator: str) -> str:
    wanted = normalize_name(generator)
    wanted_tokens = {wanted, *GENERATOR_ALIASES.get(wanted, ())}
    names = sorted(df["generator"].unique().tolist())
    exact = [name for name in names if normalize_name(name) in wanted_tokens]
    if not exact:
        exact = [
            name
            for name in names
            if any(token in normalize_name(name) or normalize_name(name) in token for token in wanted_tokens)
        ]
    if len(exact) != 1:
        raise ValueError(f"Generator {generator!r} does not match exactly one generator. Available: {names}")
    return exact[0]


def balance_real_fake(df: pd.DataFrame, seed: int, max_samples: int | None = None) -> pd.DataFrame:
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]
    per_class = min(len(real_df), len(fake_df))
    if max_samples is not None:
        per_class = min(per_class, max_samples // 2)

    real_df = real_df.sample(n=per_class, random_state=seed, replace=False)
    fake_df = fake_df.sample(n=per_class, random_state=seed + 1, replace=False)
    return pd.concat([real_df, fake_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def maybe_limit(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=seed, replace=False).reset_index(drop=True)


def build_kaggle_tiny_splits(config: TinyGenImageKaggleConfig) -> dict[str, Any]:
    df = build_kaggle_tiny_index(config)
    train_all = df[df["split"] == "train"]
    eval_all = df[df["split"] == "validation"]

    if config.eval_case == "combined":
        train_df = train_all
        eval_df = eval_all
        notes = "Train on all generator folders; evaluate on all validation folders."
    elif config.eval_case == "in_domain":
        if config.generator is None:
            raise ValueError("generator is required for eval_case='in_domain'.")
        generator = resolve_generator_name(df, config.generator)
        train_df = train_all[train_all["generator"] == generator]
        eval_df = eval_all[eval_all["generator"] == generator]
        notes = f"Train/evaluate on the same generator folder: {generator}."
    elif config.eval_case == "cross_generator":
        if config.heldout_generator is None:
            raise ValueError("heldout_generator is required for eval_case='cross_generator'.")
        heldout = resolve_generator_name(df, config.heldout_generator)
        train_df = train_all[train_all["generator"] != heldout]
        eval_df = eval_all[eval_all["generator"] == heldout]
        notes = f"Leave-one-generator-out: train excludes {heldout}; eval uses {heldout}."
    elif config.eval_case == "train_one_generator":
        if config.base_generator is None:
            raise ValueError("base_generator is required for eval_case='train_one_generator'.")
        base = resolve_generator_name(df, config.base_generator)
        train_df = train_all[train_all["generator"] == base]
        eval_df = eval_all
        notes = f"Train on one generator folder ({base}); evaluate on all validation folders."
    else:
        raise ValueError(f"Unsupported eval_case: {config.eval_case}")

    if config.balance_real:
        train_df = balance_real_fake(train_df, seed=config.seed, max_samples=config.max_train_samples)
        eval_df = balance_real_fake(eval_df, seed=config.seed, max_samples=config.max_eval_samples)
    else:
        train_df = maybe_limit(train_df, config.max_train_samples, seed=config.seed)
        eval_df = maybe_limit(eval_df, config.max_eval_samples, seed=config.seed)

    return {
        "train_df": train_df,
        "eval_df": eval_df,
        "full_index": df,
        "dataset_root": df.attrs.get("dataset_root"),
        "eval_case": config.eval_case,
        "balance_real": config.balance_real,
        "seed": config.seed,
        "notes": notes,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_real_count": int((train_df["label"] == 0).sum()),
        "train_fake_count": int((train_df["label"] == 1).sum()),
        "eval_real_count": int((eval_df["label"] == 0).sum()),
        "eval_fake_count": int((eval_df["label"] == 1).sum()),
        "generators": sorted(df["generator"].unique().tolist()),
    }


class TinyGenImageKaggleDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that reads images lazily from folder paths."""

    def __init__(self, df: pd.DataFrame, eval_case: str, transform=None, task_type: str = "classification"):
        self.df = df.reset_index(drop=True)
        self.eval_case = eval_case
        self.transform = transform
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        sample = UnifiedSample(
            sample_id=str(row["sample_id"]),
            label=int(row["label"]),
            label_name=str(row["label_name"]),
            dataset_source=str(row["dataset_source"]),
            generator=str(row["generator"]),
            split=str(row["split"]),
            eval_case=self.eval_case,
            image_path=str(row["image_path"]),
        ).as_dict()
        sample["image"] = image
        if self.task_type == "alignment":
            sample["target_text"] = sample["label_name"]
            sample["label_token"] = sample["label_name"]
        return sample


def summarize_index(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "generator", "label_name"])
        .size()
        .reset_index(name="num_images")
        .sort_values(["split", "generator", "label_name"])
    )

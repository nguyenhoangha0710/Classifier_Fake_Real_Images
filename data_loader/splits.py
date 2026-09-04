"""Evaluation split builders for Tiny-GenImage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .tiny_genimage import DATASET_NAME, get_class_names, load_tiny_genimage


TinyEvalCase = Literal[
    "combined",
    "in_domain",
    "cross_generator",
    "train_one_generator",
]


@dataclass(frozen=True)
class TinyGenImageSplitConfig:
    """Config for Tiny-GenImage train/eval splits."""

    eval_case: TinyEvalCase = "combined"
    generator: str | None = None
    heldout_generator: str | None = None
    base_generator: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    cache_dir: str | None = None
    streaming: bool = False
    balance_real: bool = True
    seed: int = 42
    streaming_shuffle_buffer_size: int = 10_000


ACTIVE_FAKE_GENERATORS = ("ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong")
EXPECTED_FAKE_PER_GENERATOR = {
    "train": 2000,
    "validation": 500,
    "val": 500,
}


def _resolve_split_name(dataset_dict, requested: str) -> str:
    if requested in dataset_dict:
        return requested
    aliases = {
        "validation": ["val", "test"],
        "val": ["validation", "test"],
        "test": ["validation", "val"],
    }
    for alias in aliases.get(requested, []):
        if alias in dataset_dict:
            return alias
    raise KeyError(f"Split {requested!r} not found. Available: {list(dataset_dict.keys())}")


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _generator_id(dataset, generator_name: str) -> int:
    names = get_class_names(dataset, "generator")
    if names is None:
        raise ValueError("Cannot resolve generator names because dataset has no ClassLabel names.")

    wanted = _normalize_name(generator_name)
    matches = [idx for idx, name in enumerate(names) if _normalize_name(name) == wanted]
    if not matches:
        contains = [idx for idx, name in enumerate(names) if wanted in _normalize_name(name)]
        matches = contains
    if len(matches) != 1:
        raise ValueError(
            f"Generator {generator_name!r} does not match exactly one generator. "
            f"Available generators: {names}"
        )
    return matches[0]


def _is_real(row) -> bool:
    return int(row["label"]) == 0


def _is_fake_from(row, generator_id: int) -> bool:
    return int(row["label"]) == 1 and int(row["generator"]) == generator_id


def _is_fake_not_from(row, generator_id: int) -> bool:
    return int(row["label"]) == 1 and int(row["generator"]) != generator_id


def _filter_real_or_generator(dataset, generator_id: int):
    return dataset.filter(lambda row: _is_real(row) or _is_fake_from(row, generator_id))


def _filter_fake_from(dataset, generator_id: int):
    return dataset.filter(lambda row: _is_fake_from(row, generator_id))


def _filter_fake_not_from(dataset, generator_id: int):
    return dataset.filter(lambda row: _is_fake_not_from(row, generator_id))


def _filter_fake(dataset):
    return dataset.filter(lambda row: int(row["label"]) == 1)


def _filter_real(dataset):
    return dataset.filter(lambda row: _is_real(row))


def _expected_fake_count(split_name: str, fake_generators: int) -> int:
    if split_name not in EXPECTED_FAKE_PER_GENERATOR:
        raise ValueError(
            f"No expected Tiny-GenImage count is known for split {split_name!r}. "
            "Use streaming=False for exact balancing, or add the split count to EXPECTED_FAKE_PER_GENERATOR."
        )
    return EXPECTED_FAKE_PER_GENERATOR[split_name] * fake_generators


def _balanced_map_dataset(dataset, fake_dataset, seed: int):
    from datasets import concatenate_datasets

    real_dataset = _filter_real(dataset)
    real_count = min(len(real_dataset), len(fake_dataset))
    fake_count = min(len(fake_dataset), real_count)
    real_balanced = real_dataset.shuffle(seed=seed).select(range(real_count))
    fake_balanced = fake_dataset.shuffle(seed=seed).select(range(fake_count))
    return concatenate_datasets([real_balanced, fake_balanced]).shuffle(seed=seed), real_count, fake_count


def _balanced_streaming_dataset(
    dataset,
    fake_dataset,
    real_count: int,
    seed: int,
    buffer_size: int,
):
    from datasets import interleave_datasets

    real_balanced = _filter_real(dataset).shuffle(seed=seed, buffer_size=buffer_size).take(real_count)
    fake_balanced = fake_dataset.shuffle(seed=seed + 1, buffer_size=buffer_size)
    balanced = interleave_datasets(
        [real_balanced, fake_balanced],
        stopping_strategy="all_exhausted",
    )
    return balanced, real_count, real_count


def _make_balanced_or_raw(
    dataset,
    fake_dataset,
    config: TinyGenImageSplitConfig,
    split_name: str,
    fake_generators: int,
):
    if not config.balance_real:
        expected_fake = _expected_fake_count(split_name, fake_generators) if config.streaming else len(fake_dataset)
        return dataset, None, expected_fake

    if config.streaming:
        real_count = _expected_fake_count(split_name, fake_generators)
        return _balanced_streaming_dataset(
            dataset=dataset,
            fake_dataset=fake_dataset,
            real_count=real_count,
            seed=config.seed,
            buffer_size=config.streaming_shuffle_buffer_size,
        )

    return _balanced_map_dataset(dataset, fake_dataset, seed=config.seed)


def _row_counts(real_count: int | None, fake_count: int | None) -> int | None:
    if real_count is None or fake_count is None:
        return None
    return real_count + fake_count


def _active_fake_generator_count(excluded: int = 0) -> int:
    return len(ACTIVE_FAKE_GENERATORS) - excluded


def build_tiny_genimage_splits(config: TinyGenImageSplitConfig | None = None) -> dict:
    """Build Tiny-GenImage splits for the planned evaluation cases.

    Returns a dict with:
    - train: Hugging Face Dataset
    - eval: Hugging Face Dataset
    - eval_case: case name
    - dataset_name: Hugging Face dataset id
    - notes: short description of the split semantics
    """

    config = config or TinyGenImageSplitConfig()
    dataset_dict = load_tiny_genimage(cache_dir=config.cache_dir, streaming=config.streaming)
    train_name = _resolve_split_name(dataset_dict, config.train_split)
    eval_name = _resolve_split_name(dataset_dict, config.validation_split)
    train_ds = dataset_dict[train_name]
    eval_ds = dataset_dict[eval_name]

    if config.eval_case == "combined":
        train_fake = _filter_fake(train_ds)
        eval_fake = _filter_fake(eval_ds)
        train_out, train_real_count, train_fake_count = _make_balanced_or_raw(
            train_ds,
            train_fake,
            config=config,
            split_name=train_name,
            fake_generators=_active_fake_generator_count(),
        )
        eval_out, eval_real_count, eval_fake_count = _make_balanced_or_raw(
            eval_ds,
            eval_fake,
            config=config,
            split_name=eval_name,
            fake_generators=_active_fake_generator_count(),
        )
        return {
            "train": train_out,
            "eval": eval_out,
            "train_split_name": train_name,
            "eval_split_name": eval_name,
            "eval_case": "combined",
            "dataset_name": DATASET_NAME,
            "streaming": config.streaming,
            "balance_real": config.balance_real,
            "seed": config.seed,
            "train_real_count": train_real_count,
            "train_fake_count": train_fake_count,
            "eval_real_count": eval_real_count,
            "eval_fake_count": eval_fake_count,
            "train_rows": _row_counts(train_real_count, train_fake_count),
            "eval_rows": _row_counts(eval_real_count, eval_fake_count),
            "notes": "Train on all Tiny-GenImage train generators; evaluate on all validation generators with balanced real/fake counts.",
        }

    if config.eval_case == "in_domain":
        if config.generator is None:
            raise ValueError("generator is required for eval_case='in_domain'.")
        gid = _generator_id(train_ds, config.generator)
        train_fake = _filter_fake_from(train_ds, gid)
        eval_fake = _filter_fake_from(eval_ds, gid)
        train_out, train_real_count, train_fake_count = _make_balanced_or_raw(
            train_ds,
            train_fake,
            config=config,
            split_name=train_name,
            fake_generators=1,
        )
        eval_out, eval_real_count, eval_fake_count = _make_balanced_or_raw(
            eval_ds,
            eval_fake,
            config=config,
            split_name=eval_name,
            fake_generators=1,
        )
        return {
            "train": train_out,
            "eval": eval_out,
            "train_split_name": train_name,
            "eval_split_name": eval_name,
            "eval_case": "in_domain",
            "dataset_name": DATASET_NAME,
            "streaming": config.streaming,
            "balance_real": config.balance_real,
            "seed": config.seed,
            "generator": config.generator,
            "train_real_count": train_real_count,
            "train_fake_count": train_fake_count,
            "eval_real_count": eval_real_count,
            "eval_fake_count": eval_fake_count,
            "train_rows": _row_counts(train_real_count, train_fake_count),
            "eval_rows": _row_counts(eval_real_count, eval_fake_count),
            "notes": f"Train/evaluate on balanced real images plus fake images from {config.generator}.",
        }

    if config.eval_case == "cross_generator":
        if config.heldout_generator is None:
            raise ValueError("heldout_generator is required for eval_case='cross_generator'.")
        gid = _generator_id(train_ds, config.heldout_generator)
        train_fake = _filter_fake_not_from(train_ds, gid)
        eval_fake = _filter_fake_from(eval_ds, gid)
        train_out, train_real_count, train_fake_count = _make_balanced_or_raw(
            train_ds,
            train_fake,
            config=config,
            split_name=train_name,
            fake_generators=_active_fake_generator_count(excluded=1),
        )
        eval_out, eval_real_count, eval_fake_count = _make_balanced_or_raw(
            eval_ds,
            eval_fake,
            config=config,
            split_name=eval_name,
            fake_generators=1,
        )
        return {
            "train": train_out,
            "eval": eval_out,
            "train_split_name": train_name,
            "eval_split_name": eval_name,
            "eval_case": "cross_generator",
            "dataset_name": DATASET_NAME,
            "streaming": config.streaming,
            "balance_real": config.balance_real,
            "seed": config.seed,
            "heldout_generator": config.heldout_generator,
            "train_real_count": train_real_count,
            "train_fake_count": train_fake_count,
            "eval_real_count": eval_real_count,
            "eval_fake_count": eval_fake_count,
            "train_rows": _row_counts(train_real_count, train_fake_count),
            "eval_rows": _row_counts(eval_real_count, eval_fake_count),
            "notes": f"Leave-one-generator-out: train excludes fake {config.heldout_generator}; eval uses balanced real plus held-out fake.",
        }

    if config.eval_case == "train_one_generator":
        if config.base_generator is None:
            raise ValueError("base_generator is required for eval_case='train_one_generator'.")
        gid = _generator_id(train_ds, config.base_generator)
        train_fake = _filter_fake_from(train_ds, gid)
        eval_fake = _filter_fake(eval_ds)
        train_out, train_real_count, train_fake_count = _make_balanced_or_raw(
            train_ds,
            train_fake,
            config=config,
            split_name=train_name,
            fake_generators=1,
        )
        eval_out, eval_real_count, eval_fake_count = _make_balanced_or_raw(
            eval_ds,
            eval_fake,
            config=config,
            split_name=eval_name,
            fake_generators=_active_fake_generator_count(),
        )
        return {
            "train": train_out,
            "eval": eval_out,
            "train_split_name": train_name,
            "eval_split_name": eval_name,
            "eval_case": "train_one_generator",
            "dataset_name": DATASET_NAME,
            "streaming": config.streaming,
            "balance_real": config.balance_real,
            "seed": config.seed,
            "base_generator": config.base_generator,
            "train_real_count": train_real_count,
            "train_fake_count": train_fake_count,
            "eval_real_count": eval_real_count,
            "eval_fake_count": eval_fake_count,
            "train_rows": _row_counts(train_real_count, train_fake_count),
            "eval_rows": _row_counts(eval_real_count, eval_fake_count),
            "notes": f"Train on balanced real plus fake {config.base_generator}; evaluate on balanced all-generator validation.",
        }

    raise ValueError(f"Unsupported Tiny-GenImage eval_case: {config.eval_case}")

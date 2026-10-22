"""Train the provisional Forensic Twins baseline on Kaggle with 2x T4.

Launch with torchrun so both Kaggle GPUs participate:

    torchrun --standalone --nproc_per_node=2 train_forensic_twins_kaggle.py \
        --config configs/kaggle_2xt4.json

The script is intentionally real-only and refuses manifests containing fake
labels. It saves atomic, resumable checkpoints throughout training.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checkpointing import (
    CheckpointManager,
    find_resume_checkpoint,
    restore_manifest_artifacts,
    sha256_file,
    validate_checkpoint_identity,
)
from src.core import (
    ForensicResidualExtractor,
    ForensicTwins,
    barlow_twins_loss,
    encoder_parameter_count,
)
from src.data import (
    ManifestDataset,
    TwinViewCollator,
    build_real_50k_manifest,
    manifest_hash,
    real_manifest_filename,
)


FRE_RELEASE_URL = (
    "https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/"
    "releases/download/v1.2.0/fre.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "kaggle_2xt4.json")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fre-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", default="auto", help="auto, none, or explicit .ckpt path")
    parser.add_argument(
        "--kaggle-dataset-slug",
        default=None,
        help="Optional USER/DATASET destination used to persist the end-of-epoch checkpoint.",
    )
    parser.add_argument("--build-manifest-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_file"] = str(path.resolve())
    return config


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. Kaggle accelerator must be enabled.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank * 100_003
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def resolve_fre_checkpoint(
    explicit: Optional[Path], input_root: Path, output_root: Path, rank: int
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"FRE checkpoint does not exist: {explicit}")
        return explicit
    matches: List[Path] = []
    if input_root.exists():
        direct = input_root / "fre.pt"
        if direct.is_file():
            matches.append(direct)
        for dataset_root in input_root.iterdir():
            if not dataset_root.is_dir():
                continue
            candidate = dataset_root / "fre.pt"
            if candidate.is_file():
                matches.append(candidate)
            matches.extend(path for path in dataset_root.glob("*/fre.pt") if path.is_file())
    if matches:
        return sorted(set(matches))[0]
    destination = output_root / "cache" / "fre.pt"
    if rank == 0 and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        print(f"[FRE] Downloading official FSD FRE from {FRE_RELEASE_URL}", flush=True)
        try:
            request = urllib.request.Request(FRE_RELEASE_URL, headers={"User-Agent": "forensic-twins"})
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(temporary, destination)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download official fre.pt. Enable Kaggle Internet or add a Kaggle Input "
                f"containing fre.pt from FSD release v1.2.0. Original error: {error}"
            ) from error
    barrier()
    if not destination.exists():
        raise FileNotFoundError(destination)
    return destination


def cosine_warmup_lambda(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1, step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def capture_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def collect_rng_states(world_size: int) -> List[Dict[str, object]]:
    local = capture_rng_state()
    if world_size == 1:
        return [local]
    gathered: List[Optional[Dict[str, object]]] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def unwrap(model: torch.nn.Module) -> ForensicTwins:
    return model.module if isinstance(model, DistributedDataParallel) else model


def append_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def save_training_checkpoint(
    manager: CheckpointManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: Dict[str, object],
    config_hash: str,
    train_manifest_hash: str,
    fre_hash: str,
    epoch: int,
    next_batch_index: int,
    optimizer_step: int,
    rank: int,
    world_size: int,
    tag: str,
    milestone: bool,
    last_metrics: Dict[str, float],
) -> Optional[Path]:
    rng_by_rank = collect_rng_states(world_size)
    if rank != 0:
        return None
    module = unwrap(model)
    payload: Dict[str, object] = {
        "schema_version": 1,
        "encoder": module.encoder.state_dict(),
        "projector": module.projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": optimizer_step,
        "optimizer_step": optimizer_step,
        "rng_by_rank": rng_by_rank,
        "config": config,
        "config_hash": config_hash,
        "manifest_hash": train_manifest_hash,
        "fre_hash": fre_hash,
        "world_size": world_size,
        "micro_batch_size": int(config["training"]["micro_batch_per_gpu"]),
        "effective_batch_size": int(config["training"]["micro_batch_per_gpu"]) * world_size,
        "last_metrics": last_metrics,
        "saved_at_unix": time.time(),
    }
    path = manager.save(payload, tag=tag, milestone=milestone)
    print(f"[checkpoint] saved {path}", flush=True)
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rank, local_rank, world_size, device = setup_distributed()
    seed = int(config["seed"])
    seed_everything(seed, rank)

    input_root = args.input_root or Path(str(config["paths"]["input_root"]))
    output_root = args.output_root or Path(str(config["paths"]["output_root"]))
    config["paths"]["input_root"] = str(input_root)
    config["paths"]["output_root"] = str(output_root)
    if args.kaggle_dataset_slug is not None:
        config["checkpoint"]["kaggle_dataset_slug"] = args.kaggle_dataset_slug
    data_config = config["data"]
    manifest_name = real_manifest_filename(
        int(data_config["imagenet_per_class"]),
        int(data_config["coco_total"]),
    )
    manifest_path = args.manifest or output_root / "manifests" / manifest_name
    manifest_summary_path = manifest_path.with_suffix(".summary.json")
    manifest_persisted_marker = manifest_path.with_suffix(".persisted.json")
    checkpoint_dir = output_root / "artifacts" / "checkpoints"
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        restore_manifest_artifacts(manifest_path, checkpoint_dir)
        summary = build_real_50k_manifest(
            input_root=input_root,
            destination=manifest_path,
            seed=seed,
            imagenet_per_class=int(config["data"]["imagenet_per_class"]),
            coco_total=int(config["data"]["coco_total"]),
            min_side=int(config["data"]["min_side"]),
            workers=int(config["data"]["manifest_workers"]),
        )
        print("[manifest]", json.dumps(summary, indent=2), flush=True)
        print(f"Manifest ready: {manifest_path}", flush=True)

        persistent_slug = config["checkpoint"].get("kaggle_dataset_slug") or None
        if args.build_manifest_only and persistent_slug:
            current_hash = manifest_hash(manifest_path)
            already_persisted = False
            if manifest_persisted_marker.is_file():
                try:
                    marker = json.loads(manifest_persisted_marker.read_text(encoding="utf-8"))
                    already_persisted = (
                        marker.get("manifest_sha256") == current_hash
                        and marker.get("kaggle_dataset_slug") == persistent_slug
                    )
                except Exception:
                    already_persisted = False
            if already_persisted:
                print(
                    f"[persistent-manifest:SKIP] already stored in {persistent_slug}",
                    flush=True,
                )
            else:
                manifest_persisted_marker.write_text(
                    json.dumps(
                        {
                            "manifest_sha256": current_hash,
                            "kaggle_dataset_slug": persistent_slug,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                manifest_manager = CheckpointManager(
                    checkpoint_dir,
                    keep_last=int(config["checkpoint"]["keep_last"]),
                    milestone_every=int(config["checkpoint"]["milestone_every_epochs"]),
                    persistent_slug=persistent_slug,
                    persistent_files=(
                        manifest_path,
                        manifest_summary_path,
                        manifest_persisted_marker,
                    ),
                )
                ok, message = manifest_manager.persist_manifest_to_kaggle()
                status = "OK" if ok else "WARNING"
                print(f"[persistent-manifest:{status}] {message}", flush=True)
                if not ok:
                    manifest_persisted_marker.unlink(missing_ok=True)
        elif args.build_manifest_only and not persistent_slug:
            print(
                "[persistent-manifest:WARNING] No Kaggle Dataset slug is configured; "
                "manifest exists only in /kaggle/working.",
                flush=True,
            )
    barrier()
    if args.build_manifest_only:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    fre_checkpoint = resolve_fre_checkpoint(args.fre_checkpoint, input_root, output_root, rank)
    fre_hash = sha256_file(fre_checkpoint)
    train_manifest_hash = manifest_hash(manifest_path)
    # Paths and retention/upload policy may change when a Kaggle artifact is
    # mounted as an Input. They do not alter optimization; manifest and FRE
    # contents are validated by their own hashes.
    identity_keys = ("run_profile", "seed", "data", "model", "training", "provisional_assumptions")
    config_for_hash = {key: config[key] for key in identity_keys if key in config}
    resolved_config_hash = canonical_hash(config_for_hash)

    dataset = ManifestDataset(manifest_path)
    training = config["training"]
    model_cfg = config["model"]
    micro_batch = int(training["micro_batch_per_gpu"])
    effective_batch = micro_batch * world_size
    expected_world_size = int(training["expected_world_size"])
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Config expects world_size={expected_world_size}, got {world_size}. "
            "Launch with: torchrun --standalone --nproc_per_node=2 ..."
        )
    if effective_batch != int(training["effective_batch_size"]):
        raise RuntimeError("micro_batch_per_gpu * world_size does not match effective_batch_size")

    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True) if world_size > 1 else RandomSampler(dataset)
    collator = TwinViewCollator(
        crop_sizes=tuple(config["data"]["crop_sizes"]),
        fre_radius=7,
        include_identity_rotation=bool(config["data"]["include_identity_rotation"]),
        flip_probability=float(config["data"]["flip_probability"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=int(training["num_workers_per_rank"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(training["num_workers_per_rank"]) > 0,
        prefetch_factor=2 if int(training["num_workers_per_rank"]) > 0 else None,
        collate_fn=collator,
    )

    fre = ForensicResidualExtractor.from_pretrained(fre_checkpoint)
    model = ForensicTwins(
        fre=fre,
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        patch_size=int(model_cfg["patch_size"]),
        projector_hidden=int(model_cfg["projector_hidden"]),
        projector_out=int(model_cfg["projector_out"]),
    ).to(device)
    encoder_params = encoder_parameter_count(model)
    if not (2_500_000 <= encoder_params <= 3_500_000):
        raise RuntimeError(f"Encoder parameter count {encoder_params:,} is inconsistent with paper's ~3.0M")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    epochs = int(training["epochs"])
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(training["warmup_epochs"]) * steps_per_epoch
    base_lr = float(training["learning_rate"])
    min_lr = float(training["min_learning_rate"])
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=base_lr,
        betas=tuple(training["adam_betas"]),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_lambda(step, warmup_steps, total_steps, min_lr / base_lr),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training["amp_fp16"]))

    manager = CheckpointManager(
        checkpoint_dir,
        keep_last=int(config["checkpoint"]["keep_last"]),
        milestone_every=int(config["checkpoint"]["milestone_every_epochs"]),
        persistent_slug=config["checkpoint"].get("kaggle_dataset_slug") or None,
        persistent_files=(
            manifest_path,
            manifest_summary_path,
            manifest_persisted_marker,
        ),
    )
    if rank == 0 and not manager.persistent_slug:
        print(
            "[persistent-checkpoint:WARNING] No Kaggle Dataset slug is configured. "
            "Step/epoch checkpoints are safe only inside the current session. Pass "
            "--kaggle-dataset-slug USER/DATASET to persist every epoch.",
            flush=True,
        )
    start_epoch, start_batch, optimizer_step = 0, 0, 0
    resume_payload = None
    resume_path = None
    diagnostics: List[str] = []
    if args.resume.lower() == "auto":
        resume_path, resume_payload, diagnostics = find_resume_checkpoint(
            checkpoint_dir,
            input_root,
            expected_config_hash=resolved_config_hash,
            expected_manifest_hash=train_manifest_hash,
            expected_fre_hash=fre_hash,
        )
    elif args.resume.lower() != "none":
        from src.checkpointing import load_checkpoint_cpu

        resume_path = Path(args.resume)
        resume_payload = load_checkpoint_cpu(resume_path)
        validate_checkpoint_identity(
            resume_payload,
            expected_config_hash=resolved_config_hash,
            expected_manifest_hash=train_manifest_hash,
            expected_fre_hash=fre_hash,
        )
    if rank == 0:
        for message in diagnostics:
            print(f"[resume] {message}", flush=True)
    if resume_payload is not None:
        module = unwrap(model)
        module.encoder.load_state_dict(resume_payload["encoder"])
        module.projector.load_state_dict(resume_payload["projector"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        if resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = int(resume_payload["epoch"])
        start_batch = int(resume_payload["next_batch_index"])
        optimizer_step = int(resume_payload["optimizer_step"])
        rng_states = resume_payload.get("rng_by_rank", [])
        if rank < len(rng_states):
            restore_rng_state(rng_states[rank])
        if rank == 0:
            print(
                f"RESUMED_FROM={resume_path}\nSTART_EPOCH={start_epoch}\n"
                f"NEXT_BATCH_INDEX={start_batch}\nGLOBAL_STEP={optimizer_step}\n"
                f"LR={optimizer.param_groups[0]['lr']}",
                flush=True,
            )

    if rank == 0:
        metadata = {
            "run_profile": config["run_profile"],
            "provisional": True,
            "world_size": world_size,
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "encoder_parameters": encoder_params,
            "manifest": str(manifest_path),
            "manifest_hash": train_manifest_hash,
            "fre_checkpoint": str(fre_checkpoint),
            "fre_hash": fre_hash,
            "config_hash": resolved_config_hash,
        }
        (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    model.train()
    save_every_steps = int(config["checkpoint"]["save_every_optimizer_steps"])
    milestone_every = int(config["checkpoint"]["milestone_every_epochs"])
    persist_every = int(config["checkpoint"].get("persist_every_epochs", 1))
    if persist_every < 1:
        raise ValueError("checkpoint.persist_every_epochs must be >= 1")
    log_every = int(training["log_every_steps"])
    log_path = output_root / "artifacts" / "logs" / "train_metrics.csv"
    last_metrics: Dict[str, float] = {}

    for epoch in range(start_epoch, epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(loader):
            if batch_index < epoch_start_batch:
                continue
            view1 = batch["view1_halo"].to(device, non_blocking=True)
            view2 = batch["view2_halo"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled(), dtype=torch.float16):
                _, _, embedding1, embedding2 = model(view1, view2)
            loss, metrics = barlow_twins_loss(
                embedding1,
                embedding2,
                redundancy_lambda=float(model_cfg["barlow_lambda"]),
                eps=float(model_cfg["barlow_eps"]),
            )
            scaler.scale(loss).backward()
            if float(training["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updated = scaler.get_scale() >= scale_before_step
            if not optimizer_updated:
                if rank == 0:
                    print(
                        f"[amp-overflow] epoch={epoch} batch_index={batch_index} "
                        f"scale={scale_before_step}->{scaler.get_scale()} update_skipped=true",
                        flush=True,
                    )
                continue
            scheduler.step()
            optimizer_step += 1
            last_metrics = {key: float(value.item()) for key, value in metrics.items()}
            last_metrics["lr"] = float(optimizer.param_groups[0]["lr"])

            if rank == 0 and optimizer_step % log_every == 0:
                row = {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "optimizer_step": optimizer_step,
                    "view1_size": batch["view1_size"][0],
                    "view2_size": batch["view2_size"][0],
                    **last_metrics,
                }
                append_log(log_path, row)
                print("[train] " + " ".join(f"{key}={value}" for key, value in row.items()), flush=True)

            if optimizer_step % save_every_steps == 0:
                save_training_checkpoint(
                    manager,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    resolved_config_hash,
                    train_manifest_hash,
                    fre_hash,
                    epoch,
                    batch_index + 1,
                    optimizer_step,
                    rank,
                    world_size,
                    tag=f"step_{optimizer_step:08d}",
                    milestone=False,
                    last_metrics=last_metrics,
                )
        start_batch = 0
        completed_epoch = epoch + 1
        milestone = completed_epoch % milestone_every == 0 or completed_epoch == epochs
        path = save_training_checkpoint(
            manager,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            resolved_config_hash,
            train_manifest_hash,
            fre_hash,
            completed_epoch,
            0,
            optimizer_step,
            rank,
            world_size,
            tag=f"epoch_{completed_epoch:04d}",
            milestone=milestone,
            last_metrics=last_metrics,
        )
        barrier()
        should_persist = completed_epoch % persist_every == 0 or completed_epoch == epochs
        if rank == 0 and should_persist and path is not None:
            ok, message = manager.persist_to_kaggle(path, completed_epoch)
            status = "OK" if ok else "WARNING"
            print(f"[persistent-checkpoint:{status}] {message}", flush=True)
        barrier()

    if rank == 0:
        module = unwrap(model)
        final_path = output_root / "artifacts" / "checkpoints" / "encoder_final.pt"
        torch.save(
            {
                "encoder": module.encoder.state_dict(),
                "config": config,
                "manifest_hash": train_manifest_hash,
                "fre_hash": fre_hash,
                "optimizer_step": optimizer_step,
            },
            final_path,
        )
        print(f"Training complete. Final encoder: {final_path}", flush=True)
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

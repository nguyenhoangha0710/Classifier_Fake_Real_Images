"""One-command Kaggle launcher for manifest, 2-GPU training and GMM evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "train", "gmm", "all"), nargs="?", default="train")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "kaggle_2xt4.json")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/forensic_twins_original"),
    )
    parser.add_argument("--resume", default="auto", help="auto, none, or an explicit checkpoint path")
    parser.add_argument(
        "--kaggle-dataset-slug",
        default="",
        help="Private checkpoint Dataset as USER/DATASET. Strongly recommended.",
    )
    parser.add_argument("--tiny-root", type=Path, default=None)
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    return parser.parse_args()


def run(command: List[str]) -> None:
    print("[launcher] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def distributed_train_command(args: argparse.Namespace, manifest_only: bool = False) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(BASE_DIR / "train_forensic_twins_kaggle.py"),
        "--config",
        str(args.config),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
        "--resume",
        args.resume,
    ]
    if args.kaggle_dataset_slug:
        command.extend(("--kaggle-dataset-slug", args.kaggle_dataset_slug))
    if manifest_only:
        command.append("--build-manifest-only")
    return command


def manifest_command(args: argparse.Namespace) -> List[str]:
    """Manifest construction is CPU/I/O-only and must not initialize NCCL."""
    command = [
        sys.executable,
        str(BASE_DIR / "train_forensic_twins_kaggle.py"),
        "--config",
        str(args.config),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
        "--resume",
        args.resume,
        "--build-manifest-only",
    ]
    if args.kaggle_dataset_slug:
        command.extend(("--kaggle-dataset-slug", args.kaggle_dataset_slug))
    return command


def gmm_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(BASE_DIR / "fit_gmm_and_evaluate.py"),
        "--config",
        str(args.config),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
    ]
    if args.tiny_root is not None:
        command.extend(("--tiny-root", str(args.tiny_root)))
    if args.max_eval_per_class is not None:
        command.extend(("--max-eval-per-class", str(args.max_eval_per_class)))
    return command


def main() -> None:
    args = parse_args()
    if args.stage == "manifest":
        run(manifest_command(args))
    elif args.stage == "train":
        # Fast no-op when the manifest already exists; otherwise build it before
        # launching NCCL so rank 1 cannot time out while rank 0 audits images.
        run(manifest_command(args))
        run(distributed_train_command(args))
    elif args.stage == "gmm":
        run(gmm_command(args))
    else:
        run(manifest_command(args))
        run(distributed_train_command(args))
        run(gmm_command(args))


if __name__ == "__main__":
    main()

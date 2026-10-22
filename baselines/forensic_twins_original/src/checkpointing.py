"""Atomic checkpoints and conservative automatic resume for Kaggle runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REQUIRED_KEYS = {
    "schema_version",
    "encoder",
    "projector",
    "optimizer",
    "scheduler",
    "epoch",
    "next_batch_index",
    "optimizer_step",
    "config",
    "config_hash",
    "manifest_hash",
    "fre_hash",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return True
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return bool(expected) and sha256_file(path) == expected


def atomic_torch_save(payload: Dict[str, object], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(sidecar_tmp, sidecar)
    return digest


def load_checkpoint_cpu(path: Path) -> Dict[str, object]:
    if not verify_sidecar(path):
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(payload)!r}")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    return payload


def validate_checkpoint_identity(
    payload: Dict[str, object],
    expected_config_hash: str,
    expected_manifest_hash: str,
    expected_fre_hash: str,
) -> None:
    """Refuse a resume whose experiment inputs differ from the current run."""
    mismatches = []
    for key, expected in (
        ("config_hash", expected_config_hash),
        ("manifest_hash", expected_manifest_hash),
        ("fre_hash", expected_fre_hash),
    ):
        actual = str(payload.get(key, ""))
        if actual != str(expected):
            mismatches.append(f"{key}: checkpoint={actual}, current={expected}")
    if mismatches:
        raise RuntimeError("Checkpoint identity mismatch; refusing resume: " + "; ".join(mismatches))


class CheckpointManager:
    def __init__(
        self,
        directory: Path,
        keep_last: int = 3,
        milestone_every: int = 5,
        persistent_slug: Optional[str] = None,
        persistent_files: Optional[Sequence[Path]] = None,
    ):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self.milestone_every = max(1, milestone_every)
        self.persistent_slug = persistent_slug
        self.persistent_files = [Path(path) for path in (persistent_files or ())]
        self.index_path = self.directory / "checkpoint_index.json"

    def _load_index(self) -> Dict[str, object]:
        if not self.index_path.exists():
            return {"schema_version": 1, "checkpoints": []}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "checkpoints": []}

    def _write_index(self, index: Dict[str, object]) -> None:
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def save(self, payload: Dict[str, object], tag: str, milestone: bool = False) -> Path:
        path = self.directory / f"{tag}.ckpt"
        digest = atomic_torch_save(payload, path)

        # last.ckpt is an independently valid atomic copy, never a partial overwrite.
        last_tmp = self.directory / "last.ckpt.tmp"
        shutil.copy2(path, last_tmp)
        os.replace(last_tmp, self.directory / "last.ckpt")
        last_sidecar = self.directory / "last.ckpt.sha256"
        last_sidecar_tmp = self.directory / "last.ckpt.sha256.tmp"
        last_sidecar_tmp.write_text(
            f"{digest}  last.ckpt\n", encoding="utf-8"
        )
        os.replace(last_sidecar_tmp, last_sidecar)

        index = self._load_index()
        entries = [entry for entry in index.get("checkpoints", []) if entry.get("path") != path.name]
        entries.append(
            {
                "path": path.name,
                "sha256": digest,
                "epoch": int(payload["epoch"]),
                "next_batch_index": int(payload["next_batch_index"]),
                "optimizer_step": int(payload["optimizer_step"]),
                "milestone": bool(milestone),
            }
        )
        entries.sort(key=lambda entry: int(entry["optimizer_step"]))
        index["checkpoints"] = self._cleanup(entries)
        self._write_index(index)
        return path

    def _cleanup(self, entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        rolling = [entry for entry in entries if not entry.get("milestone")]
        keep_names = {entry["path"] for entry in rolling[-self.keep_last :]}
        keep_names.update(entry["path"] for entry in entries if entry.get("milestone"))
        for entry in rolling[:-self.keep_last]:
            if entry["path"] in keep_names:
                continue
            path = self.directory / str(entry["path"])
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
        return [entry for entry in entries if entry["path"] in keep_names]

    def _prepare_publish_directory(self) -> Path:
        publish_dir = self.directory.parent / "kaggle_publish"
        publish_dir.mkdir(parents=True, exist_ok=True)
        for old in publish_dir.iterdir():
            if old.is_file():
                old.unlink(missing_ok=True)
        return publish_dir

    @staticmethod
    def _copy_with_sidecar(path: Path, destination: Path) -> None:
        if not path.is_file():
            return
        shutil.copy2(path, destination / path.name)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.is_file():
            shutil.copy2(sidecar, destination / sidecar.name)

    def _newest_local_checkpoint(self) -> Optional[Path]:
        last = self.directory / "last.ckpt"
        if last.is_file():
            return last
        candidates = list(self.directory.glob("*.ckpt"))
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def _version_publish_directory(self, publish_dir: Path, message: str) -> Tuple[bool, str]:
        if not self.persistent_slug:
            return False, "persistent_slug is not configured"
        if shutil.which("kaggle") is None:
            return False, "kaggle CLI is unavailable"
        metadata = {
            "title": self.persistent_slug.split("/", 1)[-1].replace("-", " ").title(),
            "id": self.persistent_slug,
            "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        }
        (publish_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(publish_dir),
            "-m",
            message,
            "-r",
            "zip",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()

    def persist_manifest_to_kaggle(self) -> Tuple[bool, str]:
        """Persist manifest artifacts immediately, preserving the newest checkpoint."""
        publish_dir = self._prepare_publish_directory()
        for path in self.persistent_files:
            self._copy_with_sidecar(path, publish_dir)
        checkpoint = self._newest_local_checkpoint()
        if checkpoint is not None:
            self._copy_with_sidecar(checkpoint, publish_dir)
        if not any(path.name.endswith(".csv") for path in publish_dir.iterdir() if path.is_file()):
            return False, "no manifest CSV was supplied for persistence"
        return self._version_publish_directory(
            publish_dir,
            "Forensic Twins real-50K manifest",
        )

    def persist_to_kaggle(self, checkpoint: Path, epoch: int) -> Tuple[bool, str]:
        """Version the newest checkpoint together with its exact train manifest."""
        publish_dir = self._prepare_publish_directory()
        self._copy_with_sidecar(checkpoint, publish_dir)
        for path in self.persistent_files:
            self._copy_with_sidecar(path, publish_dir)
        return self._version_publish_directory(
            publish_dir,
            f"Forensic Twins checkpoint epoch {epoch}",
        )


def restore_manifest_artifacts(destination: Path, checkpoint_directory: Path) -> bool:
    """Restore a previously persisted manifest before any expensive audit."""
    if destination.is_file():
        return False
    source = checkpoint_directory / destination.name
    source_summary = checkpoint_directory / destination.with_suffix(".summary.json").name
    if not source.is_file() or not source_summary.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    shutil.copy2(source_summary, destination.with_suffix(".summary.json"))
    source_marker = checkpoint_directory / destination.with_suffix(".persisted.json").name
    if source_marker.is_file():
        shutil.copy2(source_marker, destination.with_suffix(".persisted.json"))
    print(f"[manifest-restore] restored {destination} from {checkpoint_directory}", flush=True)
    return True


def _candidate_paths(working_dir: Path, input_root: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    last = working_dir / "last.ckpt"
    if last.exists():
        candidates.append(last)
    candidates.extend(sorted(working_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True))
    if input_root is not None and input_root.exists():
        # Never recursively walk ImageNet/COCO (millions of files). Checkpoint
        # datasets are expected to have a descriptive top-level Kaggle slug.
        for dataset_root in input_root.iterdir():
            if not dataset_root.is_dir():
                continue
            name = dataset_root.name.lower()
            if not any(token in name for token in ("checkpoint", "forensic", "twins")):
                continue
            candidates.extend(
                sorted(dataset_root.glob("**/*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
            )
    seen, unique = set(), []
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_resume_checkpoint(
    working_dir: Path,
    input_root: Optional[Path],
    expected_config_hash: str,
    expected_manifest_hash: str,
    expected_fre_hash: str,
) -> Tuple[Optional[Path], Optional[Dict[str, object]], List[str]]:
    diagnostics: List[str] = []
    valid: List[Tuple[int, Path, Dict[str, object]]] = []
    for path in _candidate_paths(working_dir, input_root):
        try:
            payload = load_checkpoint_cpu(path)
            validate_checkpoint_identity(
                payload,
                expected_config_hash=expected_config_hash,
                expected_manifest_hash=expected_manifest_hash,
                expected_fre_hash=expected_fre_hash,
            )
            valid.append((int(payload["optimizer_step"]), path, payload))
        except Exception as error:
            diagnostics.append(f"reject {path}: {error}")
    if not valid:
        return None, None, diagnostics
    _, path, payload = max(valid, key=lambda item: item[0])
    diagnostics.append(f"resume {path} at optimizer_step={payload['optimizer_step']}")
    return path, payload, diagnostics

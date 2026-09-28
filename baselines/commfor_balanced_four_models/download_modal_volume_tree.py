"""Download a Modal Volume directory without the Modal 1.5.x CLI directory bug."""

from __future__ import annotations

import argparse
import time
from pathlib import Path, PurePosixPath

import modal
from modal.types import FileEntryType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def normalized_remote_path(value: str) -> PurePosixPath:
    cleaned = value.replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("--remote must identify a directory inside the Modal Volume")
    return PurePosixPath(cleaned)


def safe_local_path(destination: Path, relative: PurePosixPath) -> Path:
    target = destination.joinpath(*relative.parts)
    resolved_destination = destination.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_destination and resolved_destination not in resolved_target.parents:
        raise ValueError(f"Unsafe remote path: {relative}")
    return target


def download_file(
    volume: modal.Volume,
    remote_path: str,
    destination: Path,
    expected_size: int,
    retries: int,
) -> str:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return "skipped"
    if destination.exists() and not destination.is_file():
        raise IsADirectoryError(f"Expected a file destination, got directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".modal-part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if partial.exists():
            partial.unlink()
        try:
            written = 0
            with partial.open("wb") as handle:
                for chunk in volume.read_file(remote_path):
                    written += handle.write(chunk)
            if written != expected_size:
                raise IOError(
                    f"Incomplete file {remote_path}: received {written}, expected {expected_size}"
                )
            partial.replace(destination)
            return "downloaded"
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            wait_seconds = min(2 ** (attempt - 1), 16)
            print(
                f"Retry {attempt}/{retries} for {remote_path} after {error!r}; "
                f"waiting {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    if partial.exists():
        partial.unlink()
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")

    remote_root = normalized_remote_path(args.remote)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(args.volume, environment_name=args.environment)
    entries = volume.listdir(remote_root.as_posix(), recursive=True)
    files = [entry for entry in entries if entry.type == FileEntryType.FILE]
    if not files:
        raise FileNotFoundError(
            f"No files found at {args.volume}:{remote_root.as_posix()}"
        )

    downloaded = 0
    skipped = 0
    for index, entry in enumerate(files, start=1):
        entry_path = PurePosixPath(entry.path.lstrip("/"))
        try:
            relative = entry_path.relative_to(remote_root)
        except ValueError as error:
            raise ValueError(
                f"Modal returned a path outside the requested root: {entry.path}"
            ) from error
        local_path = safe_local_path(destination, relative)
        status = download_file(
            volume=volume,
            remote_path=entry.path,
            destination=local_path,
            expected_size=int(entry.size),
            retries=int(args.retries),
        )
        if status == "downloaded":
            downloaded += 1
        else:
            skipped += 1
        print(f"[{index}/{len(files)}] {status}: {relative.as_posix()}")

    print(
        f"Complete: destination={destination}, files={len(files)}, "
        f"downloaded={downloaded}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()

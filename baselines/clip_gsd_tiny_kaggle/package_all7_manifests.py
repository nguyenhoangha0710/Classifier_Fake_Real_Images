"""Package only small saved All7 manifests, never model weights or source images."""
import argparse
from pathlib import Path
import zipfile


def package_manifests(source, destination, seeds=(42,)):
    source, destination = Path(source), Path(destination)
    files = []
    if (source / 'config.json').is_file():
        files.append(Path('config.json'))
    for seed in seeds:
        for part in ('train', 'validation', 'test'):
            relative = Path(f'seed{seed}') / f'{part}_manifest.csv'
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)
            files.append(relative)
        summary = Path(f'seed{seed}') / 'split_summary.json'
        if (source / summary).is_file():
            files.append(summary)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing bundle: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + '.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(source / relative, arcname=relative.as_posix())
    temp.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Previous All7 output containing seed42/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    args = parser.parse_args()
    print(package_manifests(args.source, args.output, args.seeds))

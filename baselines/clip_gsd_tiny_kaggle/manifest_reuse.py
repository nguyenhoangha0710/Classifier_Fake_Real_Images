"""Trusted audit-manifest reuse. Loaded after runtime.py; never decode/hash image content.

The caller deliberately trusts the previous image audit and the same dataset version.
Filesystem stat checks cannot detect content changes that keep the same byte length.
"""
from pathlib import Path, PurePosixPath
import re

MANIFEST_PARTITIONS = ('train', 'validation', 'test')
MANIFEST_COLUMNS = {
    'sample_id', 'relative_path', 'generator', 'split', 'label', 'width', 'height',
    'file_hash', 'pixel_hash', 'file_bytes',
}


def find_manifest_root(cfg):
    def complete(root):
        return all((root / f'seed{seed}' / f'{part}_manifest.csv').is_file()
                   for seed in cfg['seeds'] for part in MANIFEST_PARTITIONS)

    if cfg['manifest_root']:
        root = Path(cfg['manifest_root']).resolve()
        if re.fullmatch(r'seed\d+', root.name):
            root = root.parent
        if not complete(root):
            raise FileNotFoundError(f'Missing saved train/validation/test manifests under {root}/seed<seed>. '
                                    'Attach the previous All7 output; no full-scan fallback is enabled.')
        return root
    search = Path(cfg['input_root'])
    roots = sorted({p.parent.parent.resolve() for p in search.rglob('train_manifest.csv')
                    if p.parent.name == f'seed{cfg["seeds"][0]}' and complete(p.parent.parent)})
    if len(roots) != 1:
        raise ValueError(f'Set manifest_root explicitly. Complete All7-manifest candidates: {[str(p) for p in roots]}. '
                         'Need seed42/train_manifest.csv, validation_manifest.csv and test_manifest.csv; '
                         'weights alone are not enough. No image scan was started.')
    return roots[0]


def validate_manifest_frame(frame, cfg, partition):
    missing = MANIFEST_COLUMNS - set(frame)
    if missing or frame.empty:
        raise ValueError(f'Invalid {partition} manifest: missing columns={sorted(missing)} or empty cohort')
    frame = frame.copy()
    if frame[list(MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError(f'Null required metadata in {partition} manifest')
    if frame.sample_id.duplicated().any() or frame.relative_path.duplicated().any():
        raise ValueError(f'Duplicate IDs/paths in {partition} manifest')
    for column in ('sample_id', 'file_hash', 'pixel_hash'):
        if not frame[column].astype(str).str.fullmatch(r'[0-9a-f]{64}').all():
            raise ValueError(f'Invalid saved {column} in {partition} manifest')
    for column in ('label', 'width', 'height', 'file_bytes'):
        values = pd.to_numeric(frame[column], errors='coerce')
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f'Invalid integer {column} in {partition} manifest')
        frame[column] = values.astype('int64')
    if not frame.label.isin([0, 1]).all() or (frame[['width', 'height', 'file_bytes']] <= 0).any().any():
        raise ValueError(f'Invalid labels/dimensions/byte lengths in {partition} manifest')
    source_split = 'test' if partition == 'test' else 'train'
    if set(frame.split) != {source_split}:
        raise ValueError(f'{partition} manifest must originate only from source {source_split} folders')
    counts = frame.groupby(['generator', 'label']).size()
    expected = {(g, y) for g in cfg['train_generators'] for y in (0, 1)}
    if set(counts.index) != expected or counts.nunique() != 1:
        raise ValueError(f'{partition} manifest must contain balanced real/fake cohorts for all seven generators')
    if frame.duplicated(['generator', 'pixel_hash']).any():
        raise ValueError(f'Duplicate pixels within a generator in {partition} manifest')
    label_names = {'nature': 0, 'real': 0, '0_real': 0, 'ai': 1, 'fake': 1, '1_fake': 1}
    split_names = {'train': {'train', 'training'}, 'test': {'val', 'valid', 'validation', 'test'}}
    for row in frame.itertuples():
        rel = str(row.relative_path)
        parts = PurePosixPath(rel).parts
        if (not parts or len(parts) < 4 or PurePosixPath(rel).is_absolute() or
                '\\' in rel or ':' in rel or '..' in parts or rel != PurePosixPath(rel).as_posix()):
            raise ValueError(f'Unsafe relative image path: {rel}')
        if (canonical_generator(parts[0]) != row.generator or parts[1] not in split_names[source_split] or
                label_names.get(parts[2].lower()) != row.label):
            raise ValueError(f'Path/generator/split/label mismatch: {rel}')
        if row.sample_id != hashlib.sha256(rel.encode()).hexdigest():
            raise ValueError(f'Saved sample_id does not match relative path: {rel}')
    return frame


def read_manifest_cohorts(root, cfg, seed):
    folder = root / f'seed{seed}'
    summary_path = folder / 'split_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
    frames, sources = {}, {}
    for partition in MANIFEST_PARTITIONS:
        path = folder / f'{partition}_manifest.csv'
        data = path.read_bytes()  # Hash only the small CSV, NOT any source image.
        frame = pd.read_csv(io.BytesIO(data), dtype={k: str for k in
            ('sample_id', 'relative_path', 'generator', 'split', 'file_hash', 'pixel_hash')})
        frame = validate_manifest_frame(frame, cfg, partition)
        if summary is not None:
            if (summary.get(partition) != len(frame) or
                    summary.get(partition + '_hash') != digest(frame.sample_id.tolist())):
                raise ValueError(f'{path}: ordered cohort does not match saved split_summary.json')
        frames[partition] = frame
        sources[partition] = dict(path=str(path), csv_sha256=hashlib.sha256(data).hexdigest(),
                                  count=len(frame), ordered_ids_hash=digest(frame.sample_id.tolist()))
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(frames[a].sample_id) & set(frames[b].sample_id):
            raise ValueError(f'Saved IDs overlap across {a}/{b}')
        if set(frames[a].pixel_hash) & set(frames[b].pixel_hash):
            raise ValueError(f'Saved pixel hashes overlap across {a}/{b}')
    joined = pd.concat(list(frames.values()), ignore_index=True)
    if (joined.groupby('pixel_hash').label.nunique() > 1).any():
        raise ValueError('Conflicting saved labels for identical pixel hashes')
    return frames, sources


def resolve_manifest_dataset(frame, cfg):
    first = frame.relative_path.iloc[0]
    if cfg['dataset_root']:
        candidates = [Path(cfg['dataset_root'])]
    else:
        search = Path(cfg['input_root'])
        candidates = [search / 'tiny-genimage', search / 'datasets/yangsangtai/tiny-genimage']
        generator_dir = PurePosixPath(first).parts[0]
        print('[manifest] Locating dataset root using directory metadata only.', flush=True)
        candidates.extend(p.parent for p in search.rglob(generator_dir) if p.is_dir())
    matches = sorted({p.resolve() for p in candidates if (p / first).is_file()})
    if len(matches) != 1:
        raise ValueError(f'Set dataset_root to the directory containing generator folders. Candidates: '
                         f'{[str(p) for p in matches]}. No RGB/hash scan was started.')
    root = matches[0]
    resolved = frame.copy()
    paths = []
    for index, row in enumerate(resolved.itertuples()):
        path = (root / row.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError(f'Image path escapes dataset root: {row.relative_path}')
        if not path.is_file() or path.stat().st_size != row.file_bytes:
            raise ValueError(f'Missing or changed file (path/size check): {path}. '
                             'Use the same dataset version as the saved audit.')
        paths.append(str(path))
        if (index + 1) % 5000 == 0:
            print(f'[manifest] {index + 1:,}/{len(frame):,} file paths/sizes checked; no image decode/hash.', flush=True)
    resolved['path'] = paths  # Ignore stale absolute /kaggle/input paths from the previous run.
    return resolved, root


def load_manifest_index(cfg, out):
    root = find_manifest_root(cfg)
    source_config_path = root / 'config.json'
    if source_config_path.is_file():
        source = json.loads(source_config_path.read_text(encoding='utf-8'))
        if source.get('smoke') and not cfg['smoke']:
            raise ValueError('Cannot use smoke-only source manifests for a full training run')
        if set(source.get('train_generators', [])) != set(cfg['train_generators']):
            raise ValueError('Source config is not an All7 experiment')
        if source.get('validation_fraction') != cfg['validation_fraction']:
            raise ValueError('validation_fraction differs from saved split; reused manifests are never re-split')
    all_frames, cohorts, sources = [], {}, {}
    for seed in cfg['seeds']:
        frames, provenance = read_manifest_cohorts(root, cfg, seed)
        all_frames.extend(frames.values())
        cohorts[str(seed)] = {part: f.sample_id.tolist() for part, f in frames.items()}
        sources[str(seed)] = provenance
    merged = pd.concat(all_frames, ignore_index=True)
    # A sample may occur in several seed experiments; metadata must agree.
    for column in sorted(MANIFEST_COLUMNS - {'sample_id'}):
        if (merged.groupby('sample_id')[column].nunique(dropna=False) > 1).any():
            raise ValueError(f'Conflicting saved {column} across seeds/cohorts')
    merged = merged.drop_duplicates('sample_id', keep='first').reset_index(drop=True)
    merged, dataset = resolve_manifest_dataset(merged, cfg)
    provenance = dict(mode='reuse_manifests', manifest_root=str(root), dataset_root=str(dataset),
                      sources=sources, unique_selected_images=len(merged),
                      checks=['schema', 'ordered_split_summary_if_present', 'saved_pixel_hash_leakage',
                              'label_and_split_path_consistency', 'file_exists', 'file_byte_length'],
                      skipped=['image_decode', 'image_content_hash', 'pixel_hash_recomputation',
                               'discovery_of_unlisted_or_new_images'],
                      trust='Saved audit metadata and unchanged dataset version are trusted. '
                            'Same-size content changes are NOT detected.')
    target = out / 'manifest_reuse.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != provenance:
        raise ValueError('Manifest/input fingerprint changed; choose a new output_root before resume')
    dump(target, provenance)
    save_csv(out / 'reused_manifest_index.csv', merged)
    merged.attrs['reused_cohorts'] = cohorts
    merged.attrs['manifest_root'] = str(root)
    print(f'[manifest] Reused {len(merged):,} selected images. No RGB decoding or image hashing; no re-split.', flush=True)
    return merged


def reuse_saved_splits(df, cfg, seed, out):
    cohorts = df.attrs.get('reused_cohorts', {}).get(str(seed))
    if cohorts is None:
        raise ValueError('Missing validated saved-cohort index; refusing to re-split or scan implicitly')
    indexed = df.set_index('sample_id', drop=False)
    frames = []
    for partition in MANIFEST_PARTITIONS:
        frame = indexed.loc[cohorts[partition]].reset_index(drop=True)
        frame.attrs = {}
        if cfg['smoke']:
            limit = cfg['smoke_train_per_class'] if partition == 'train' else cfg['smoke_eval_per_class']
            frame = frame[frame.groupby(['generator', 'label']).cumcount() < limit].reset_index(drop=True)
        save_csv(out / f'{partition}_manifest.csv', frame)
        frame.groupby(['generator', 'label']).size().rename('n').to_csv(out / f'{partition}_counts.csv')
        frames.append(frame)
    tr, va, te = frames
    dump(out / 'split_summary.json', dict(train=len(tr), validation=len(va), test=len(te),
         train_hash=digest(tr.sample_id.tolist()), validation_hash=digest(va.sample_id.tolist()),
         test_hash=digest(te.sample_id.tolist()), selection='Reused ordered audited All7 cohorts; no re-split',
         manifest_root=df.attrs['manifest_root'], smoke_subset=bool(cfg['smoke'])))
    print(f'[split/reused] train={len(tr):,}, validation={len(va):,}, test={len(te):,}; original ID order retained.', flush=True)
    return tr, va, te

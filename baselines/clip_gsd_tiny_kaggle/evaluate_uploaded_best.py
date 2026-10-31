"""Inference-only uploaded-checkpoint evaluator with bounded, resumable sessions."""
if 'case_spec' not in globals():
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    for _name in ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py', 'evaluate_all7_variants.py'):
        exec(compile((_here / _name).read_text(encoding='utf-8'), str(_here / _name), 'exec'), globals())

import shutil
import zipfile

UPLOAD_EVALUATOR = 'gsd_uploaded_best_variants_v1'
UPLOADED_FULL_SUITE_CASES = (
    'native', 'fake_jpeg96', 'all_jpeg96', 'all_jpeg90', 'all_jpeg80',
    'resize_square256', 'resize_square560', 'blur_sigma1', 'resize256_jpeg96',
)
UPLOAD_DEFAULTS = dict(
    input_root='/kaggle/input', checkpoint_path=None, test_manifest_path=None, dataset_root=None,
    output_root='/kaggle/working/gsd_uploaded_best_variants_v1',
    checkpoint_cache='/kaggle/working/gsd_uploaded_checkpoint_cache',
    pretrained_model_path=None, resume_from='auto',
    expected_signature='4595045b4c6c76c1ad4aa303a7d2f1690cfac8150add4548d24609a6c980dce6',
    expected_epoch=1,
    expected_test_order_hash='c544b38bf4276a2fce06fd205fb4dbb86fa772ac26532509679311e7d8a96aa2',
    cases=['native', 'fake_jpeg96', 'all_jpeg96', 'all_jpeg90', 'all_jpeg80',
           'resize_square256', 'resize_square560', 'blur_sigma1', 'resize256_jpeg96'],
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
    save_every_batches=10, log_every_batches=10,
    max_new_cases_per_run=3, time_budget_hours=9.5, max_batches=None,
    verify_file_sizes=True,
)


def finalize_uploaded_notebook_config(options=None, smoke_only=False):
    """Pin the shipped GPU notebook's full suite at the actual invocation boundary.

    A stale cell's cases=['native'], max_batches=2 or case limit cannot silently
    turn the full evaluation into a successful native-only/smoke run. The general
    run_uploaded_evaluation API still accepts custom cases and case budgets.
    """
    if not isinstance(smoke_only, bool):
        raise ValueError('SMOKE_ONLY must be explicitly True or False')
    opts = dict(UPLOAD_DEFAULTS, **(options or {}))
    if smoke_only:
        opts.update(cases=['native', 'fake_jpeg96'], max_batches=2,
                    max_new_cases_per_run=2, resume_from=None,
                    output_root='/kaggle/working/gsd_uploaded_best_variants_smoke_v2')
    else:
        opts.update(cases=list(UPLOADED_FULL_SUITE_CASES), max_batches=None,
                    max_new_cases_per_run=None)
    validate_upload_options(opts)
    print('[launch] ' + ('SMOKE ONLY: 16 images/case; NOT full test results.' if smoke_only else
          'FULL SUITE: 9 cases, 7,000 images/case; no per-session case-count limit.'), flush=True)
    print('[launch] ACTUAL CASES: ' + ', '.join(opts['cases']), flush=True)
    print(f'[launch] time_budget_hours={opts["time_budget_hours"]}; '
          f'resume_from={opts["resume_from"]}; output={opts["output_root"]}', flush=True)
    return opts


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def bind_upload_metadata(path, metadata):
    metadata = json.loads(json.dumps(metadata, default=str))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != metadata:
            raise ValueError(f'Inference fingerprint changed: {path}. Use a NEW output_root.')
    else:
        atomic_json(path, metadata)


def pytorch_zip_prefix(names):
    prefixes = [name[:-len('data.pkl')] for name in names if name.endswith('/data.pkl')]
    matches = [p for p in prefixes if p + 'version' in names and any(n.startswith(p + 'data/') for n in names)]
    return matches[0] if len(matches) == 1 else None


def materialize_checkpoint(source, cache):
    """Accept .pt, a torch archive named .zip, outer ZIP, or unpacked torch internals.

    No extractall and no direct pickle execution. Only torch.load(weights_only=True)
    interprets checkpoint metadata later. The source input is never modified.
    """
    source, cache = Path(source), Path(cache)
    if source.is_file():
        if not zipfile.is_zipfile(source):
            return source  # legacy torch serialization; restricted loader still applies
        with zipfile.ZipFile(source) as archive:
            if pytorch_zip_prefix(archive.namelist()):
                return source  # best.zip can already BE a PyTorch checkpoint
            entries = [i for i in archive.infolist() if not i.is_dir() and
                       PurePosixPath(i.filename).name in ('best.pt', 'best.pth')]
            if len(entries) != 1:
                raise ValueError(f'{source}: expected torch archive or a ZIP containing exactly one best.pt')
            item = entries[0]
            if item.file_size <= 0 or item.file_size > 4 * 1024**3:
                raise ValueError('Unexpected checkpoint archive entry size')
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / ('extracted_' + digest(str(source.resolve()))[:12] + '.pt')
            temp = target.with_suffix('.tmp')
            with archive.open(item) as inp, temp.open('wb') as dest:
                shutil.copyfileobj(inp, dest, length=8 * 1024 * 1024)
            temp.replace(target)
            return target
    if source.is_dir() and (source / 'data.pkl').is_file() and (source / 'version').is_file():
        # Kaggle may unzip best.zip on upload. Restore the torch container, not a model.
        files = sorted(p for p in source.rglob('*') if p.is_file())
        if not (source / 'data').is_dir():
            raise ValueError('Unpacked checkpoint has no tensor-data directory')
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / ('repacked_' + digest(str(source.resolve()))[:12] + '.pt')
        temp = target.with_suffix('.tmp')
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                relative = path.relative_to(source).as_posix()
                if path.is_symlink() or not (relative in {
                        'data.pkl', 'version', 'byteorder', '.format_version', '.storage_alignment',
                        '.data/serialization_id'} or re.fullmatch(r'data/\d+', relative)):
                    raise ValueError(f'Unexpected unpacked checkpoint file: {relative}')
                info = zipfile.ZipInfo('best/' + relative, date_time=(1980, 1, 1, 0, 0, 0))
                with path.open('rb') as inp, archive.open(info, 'w', force_zip64=True) as dest:
                    shutil.copyfileobj(inp, dest, length=8 * 1024 * 1024)
        temp.replace(target)
        return target
    raise FileNotFoundError(f'No checkpoint at {source}. Select best.pt, best.zip, or unpacked best/ folder.')


def restricted_checkpoint_load(path):
    # File extension is irrelevant. Do not fall back to unrestricted pickle loading.
    return torch.load(path, map_location='cpu', weights_only=True, mmap=zipfile.is_zipfile(path))


def validate_uploaded_payload(payload, options):
    if not isinstance(payload, dict) or not {'model', 'config', 'epoch', 'signature'} <= set(payload):
        raise ValueError('Checkpoint must contain model/config/epoch/signature')
    cfg = payload['config']
    validate_raw_config(dict(cfg, jpeg_probability=0., blur_probability=0., horizontal_flip_probability=0.))
    if cfg['input_mode'] != 'center_crop_max_edge_pad14' or cfg.get('crop_max_side') != 560:
        raise ValueError('Expected All7 crop560 checkpoint')
    if cfg.get('smoke') or len(cfg['seeds']) != 1 or cfg['test_batch'] != 8:
        raise ValueError('Expected full single-seed checkpoint with logical test batch8')
    if options['expected_signature'] and payload['signature'] != options['expected_signature']:
        raise ValueError(f'Wrong checkpoint signature: {payload["signature"]}')
    if options['expected_epoch'] is not None and payload['epoch'] + 1 != options['expected_epoch']:
        raise ValueError('Unexpected checkpoint epoch')
    if not isinstance(payload['model'], dict) or not payload['model']:
        raise ValueError('Missing model state_dict')
    return dict(cfg)


def find_uploaded_checkpoint(options):
    if options['checkpoint_path']:
        candidates = [Path(options['checkpoint_path'])]
    else:
        root = Path(options['input_root'])
        candidates = sorted({p for pattern in ('best.pt', 'best.pth', 'best.zip') for p in root.rglob(pattern)
                             if p.is_file()} | {p.parent for p in root.rglob('data.pkl')
                             if (p.parent / 'version').is_file() and (p.parent / 'data').is_dir()})
    matches, failures = [], []
    for source in candidates:
        try:
            path = materialize_checkpoint(source, options['checkpoint_cache'])
            payload = restricted_checkpoint_load(path)
            validate_uploaded_payload(payload, options)
            matches.append((source, path, payload))
        except Exception as exc:
            failures.append(f'{source}: {type(exc).__name__}: {exc}')
    if not matches:
        raise ValueError('No matching uploaded checkpoint. Upload best.pt (preferred; do not unzip it), '
                         'or set checkpoint_path explicitly. Inspected: ' + '; '.join(failures))
    if len(matches) > 1:
        hashes = {file_sha(path) for _, path, _ in matches}
        if len(hashes) != 1:
            raise ValueError('Multiple matching but different checkpoints; set checkpoint_path explicitly')
        print('[checkpoint] Identical checkpoint copies found; using the first.', flush=True)
    return matches[0]


def choose_test_manifest(cfg, options):
    explicit = options['test_manifest_path']
    if explicit:
        paths = [Path(explicit)]
    else:
        embedded = Path(cfg.get('manifest_root') or '/nonexistent') / f'seed{cfg["seeds"][0]}' / 'test_manifest.csv'
        paths = [embedded] if embedded.is_file() else sorted(Path(options['input_root']).rglob('test_manifest.csv'))
    valid, failures = [], []
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={k: str for k in
                ('sample_id', 'relative_path', 'generator', 'split', 'file_hash', 'pixel_hash')})
            frame = validate_manifest_frame(frame, cfg, 'test')
            if len(frame) != 7000:
                raise ValueError('Expected 7,000 test images')
            order = digest(frame.sample_id.tolist())
            if options['expected_test_order_hash'] and order != options['expected_test_order_hash']:
                raise ValueError('Test ID order does not match the saved All7 cohort')
            summary_path = path.parent / 'split_summary.json'
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                if summary.get('test') != len(frame) or summary.get('test_hash') != order:
                    raise ValueError('Saved test split_summary does not match')
            fields = sorted(MANIFEST_COLUMNS)
            valid.append((path, frame, digest(frame[fields].to_dict('records'))))
        except Exception as exc:
            failures.append(f'{path}: {exc}')
    if not valid or len({identity for _, _, identity in valid}) != 1:
        raise ValueError('Set test_manifest_path to the matching seed42/test_manifest.csv. '
                         'Missing/different cohorts: ' + '; '.join(failures))
    if len(valid) > 1:
        print('[manifest] Multiple identical test manifests; reusing one, no image scan.', flush=True)
    return valid[0]


def canonical_case(name):
    spec = case_spec(name)
    if spec['kind'] in ('native', 'png'):
        return 'native'
    if spec['kind'] in ('all_jpeg', 'fake_jpeg'):
        return spec['kind'] + str(spec['quality'])
    if spec['kind'] == 'resize_square':
        return 'resize_square' + str(spec['side'])
    return name


def plan_uploaded_cases(requested):
    if not requested:
        raise ValueError('Select at least one case')
    cases, aliases, duplicates = ['native'], {}, []
    seen_requests = set()
    for name in requested:
        canonical = canonical_case(name)
        if name in seen_requests:
            duplicates.append(name)
        seen_requests.add(name)
        if canonical != name:
            aliases[name] = canonical
        if canonical not in cases:
            cases.append(canonical)
    return dict(requested=requested, cases=cases, aliases_not_measured=aliases, duplicate_requests=duplicates,
                note='Native always included. PNG roundtrip is an RGB-pixel identity alias, '
                     'not independently measured. Fake-only JPEG is NOT equivalent to all-image JPEG.')


def validate_upload_options(options):
    for key in ('micro_batch', 'anchor_micro_batch', 'save_every_batches', 'log_every_batches'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('max_new_cases_per_run', 'max_batches'):
        if options[key] is not None and (isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1):
            raise ValueError(f'{key} must be None or a positive integer')
    if options['num_workers'] < 0 or not 0 < options['time_budget_hours'] <= 10.5:
        raise ValueError('Nonnegative workers and time_budget_hours in (0,10.5] required for a 12h session')
    return plan_uploaded_cases(options['cases'])


def restore_uploaded_progress(out, options, fingerprint):
    if any((out / 'progress').glob('*/part_*.csv')):
        print('[resume] Existing local progress takes precedence.', flush=True)
        return None
    requested = options['resume_from']
    if not requested:
        return None
    if requested == 'auto':
        states = []
        for path in Path(options['input_root']).rglob('evaluation_state.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
                if state.get('evaluator') == UPLOAD_EVALUATOR and state.get('fingerprint') == fingerprint:
                    states.append(path)
            except (ValueError, OSError):
                continue
        if not states:
            return None
        # Prefer the one with the most persisted rows, then use deterministic path ordering.
        states.sort(key=lambda p: (sum(json.loads(p.read_text(encoding='utf-8')).get('saved_rows', {}).values()), str(p)))
        source = states[-1].parent
    else:
        source = Path(requested)
    state = json.loads((source / 'evaluation_state.json').read_text(encoding='utf-8'))
    if state.get('evaluator') != UPLOAD_EVALUATOR or state.get('fingerprint') != fingerprint:
        raise ValueError('Resume source has a different checkpoint/cohort/processing/environment fingerprint')
    for path in sorted((source / 'progress').rglob('*')):
        if not path.is_file() or path.suffix not in ('.csv', '.json'):
            continue
        relative = path.relative_to(source)
        if path.is_symlink():
            raise ValueError('Refusing a symlink in saved progress')
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(f'[resume] Restored saved complete-batch progress from {source}', flush=True)
    return source


def read_uploaded_case(folder, frame, cfg, name):
    if any(folder.glob('part_*.csv')) and not (folder / 'metadata.json').is_file():
        raise ValueError('Saved progress has no case fingerprint metadata; refusing unverified cache')
    prior, n = load_case_parts(folder, frame, cfg['test_batch'])
    if n:
        required = {'logit_real', 'logit_fake', 'logit_margin', 'input_tokens', 'case', 'seed', 'model'}
        if not required <= set(prior) or not np.isfinite(prior[list(required - {'case', 'model'})].to_numpy()).all():
            raise ValueError('Saved progress lacks finite raw logits/token metadata')
        if set(prior.case) != {name} or set(prior.seed) != {cfg['seeds'][0]} or set(prior.model) != {'clip_gsd'}:
            raise ValueError('Saved case/seed/model mismatch')
        if not np.allclose(prior.logit_fake - prior.logit_real, prior.logit_margin, rtol=1e-5, atol=1e-5):
            raise ValueError('Saved raw-logit margin mismatch')
        expected_scores = torch.from_numpy(prior[['logit_real', 'logit_fake']].to_numpy()).softmax(-1)[:, 1].numpy()
        if not np.allclose(prior.prob_fake, expected_scores, rtol=1e-6, atol=2e-7):
            raise ValueError('Saved scores do not match raw logits')
    return prior, n


@torch.no_grad()
def evaluate_uploaded_case(model, anchor, frame, cfg, options, name, out, fingerprint,
                           device, deadline, callback=None, backend=None):
    folder = out / 'progress' / name
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob('part_*.csv')) and not (folder / 'metadata.json').is_file():
        raise ValueError('Saved progress has no case fingerprint metadata; refusing unverified cache')
    bind_upload_metadata(folder / 'metadata.json', dict(run=fingerprint, case=case_spec(name), evaluator=UPLOAD_EVALUATOR))
    prior, start = read_uploaded_case(folder, frame, cfg, name)
    if start == len(frame):
        return prior, True
    model.eval()
    data = DataLoader(Subset(VariantImages(frame, cfg, case_spec(name)), range(start, len(frame))),
                      batch_size=cfg['test_batch'], shuffle=False, num_workers=options['num_workers'],
                      collate_fn=native_collate, pin_memory=device.type == 'cuda', drop_last=False)
    pending, parts, first_pending, done = [], [prior] if start else [], start, start

    def flush():
        nonlocal pending
        if pending:
            chunk = pd.concat(pending, ignore_index=True)
            save_csv(folder / f'part_{first_pending:07d}.csv', chunk)
            parts.append(chunk)
            pending = []
            if callback:
                callback(name, done)

    try:
        for local_bid, (images, labels, indices) in enumerate(data):
            if time.monotonic() >= deadline:
                break
            bid = start // cfg['test_batch'] + local_bid
            if backend is not None:
                scores = backend.predict_batch(model, anchor, images, cfg, device)
                if scores.shape != (len(images), 2) or not np.isfinite(scores).all():
                    raise FloatingPointError(f'Invalid backend logits: {name}, batch {bid}')
                # Preserve softmax-of-FP32-logits semantics; no TPU host round-trip per scalar.
                probabilities = torch.from_numpy(scores).float().softmax(-1)[:, 1].numpy()
            else:
                bases = estimate_bases(anchor, images, cfg, device, model.selected)
                scores = np.empty((len(images), 2), dtype=np.float32)
                probabilities = np.empty(len(images), dtype=np.float32)
                for offset, x in image_chunks(images, cfg['micro_batch']):
                    with amp(cfg, device):
                        logits = model(x.to(device), bases)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f'Nonfinite logits: {name}, batch {bid}')
                    scores[offset:offset + len(x)] = logits.float().cpu().numpy()
                    probabilities[offset:offset + len(x)] = logits.float().softmax(-1)[:, 1].cpu().numpy()
            item = frame.iloc[indices.tolist()][['sample_id', 'relative_path', 'generator', 'label',
                                                'pixel_hash', 'width', 'height']].copy()
            item = item.assign(case=name, model='clip_gsd', seed=cfg['seeds'][0], prob_fake=probabilities,
                               logit_real=scores[:, 0], logit_fake=scores[:, 1],
                               logit_margin=scores[:, 1] - scores[:, 0], batch_id=bid,
                               input_tokens=[(x.shape[-2] // 14) * (x.shape[-1] // 14) + 1 for x in images])
            if not pending:
                first_pending = int(indices[0])
            pending.append(item)
            done = int(indices[-1]) + 1
            if local_bid == 0 or (bid + 1) % options['log_every_batches'] == 0:
                print(f'[{name}] {done:,}/{len(frame):,} images; batch {bid + 1}/{math.ceil(len(frame)/cfg["test_batch"])}', flush=True)
            if len(pending) >= options['save_every_batches'] or done == len(frame):
                flush()
    finally:
        # Soft interrupts / planned stop retain completed batches. Hard kills cannot run finally.
        flush()
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    complete = len(result) == len(frame)
    if len(result):
        expected = frame.iloc[:len(result)]
        if result.sample_id.tolist() != expected.sample_id.tolist():
            raise ValueError('Incorrect resumed case order')
        save_csv(out / f'predictions_{name}{"" if complete else ".partial"}.csv', result)
    return result, complete


def run_uploaded_evaluation(options=None, backend=None):
    started = time.monotonic()
    opts = dict(UPLOAD_DEFAULTS, **(options or {}))
    plan = validate_upload_options(opts)
    deadline = started + opts['time_budget_hours'] * 3600
    source, checkpoint_path, payload = find_uploaded_checkpoint(opts)
    cfg = validate_uploaded_payload(payload, opts)
    manifest_path, full_frame, manifest_identity = choose_test_manifest(cfg, opts)
    frame = full_frame.iloc[:opts['max_batches'] * cfg['test_batch']].copy() if opts['max_batches'] else full_frame.copy()
    frame, dataset_root = resolve_test_files(frame, opts)
    cfg.update(micro_batch=opts['micro_batch'], anchor_micro_batch=opts['anchor_micro_batch'])
    if backend is not None:
        device = backend.prepare(cfg, opts)
        environment = backend.environment()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            raise RuntimeError('Use Kaggle GPU; offline CPU tests use tiny models only')
        if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('Checkpoint precision bf16 unsupported on this GPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        environment = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                           pillow=PIL.__version__, gpu=torch.cuda.get_device_name(0))
    out = Path(opts['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = file_sha(checkpoint_path)
    critical = {k: cfg[k] for k in ('input_mode', 'crop_max_side', 'test_batch', 'micro_batch',
                'anchor_micro_batch', 'precision', 'basis_method', 'attention', 'pooling', 'k',
                'lambda_max', 'a_init', 'b_init', 'jpeg_subsampling', 'threshold')}
    identity = dict(evaluator=UPLOAD_EVALUATOR, checkpoint_sha256=checkpoint_sha,
        training_signature=payload['signature'], epoch=payload['epoch'] + 1,
        manifest_identity=manifest_identity, sample_order_hash=digest(frame.sample_id.tolist()), n=len(frame),
        inference_config=critical, transform_version='original_rgb_before_crop560_v1',
        pretrained_anchor=opts['pretrained_model_path'] or cfg['model_id'],
        environment=environment)
    fingerprint = digest(identity)
    bind_upload_metadata(out / 'run_fingerprint.json', identity)
    restored = restore_uploaded_progress(out, opts, fingerprint)
    atomic_json(out / 'case_plan.json', plan)
    atomic_json(out / 'evaluation_config.json', opts)
    atomic_json(out / 'checkpoint_inspection.json', dict(
        source=str(source), materialized_checkpoint=str(checkpoint_path), sha256=checkpoint_sha,
        epoch=payload['epoch'] + 1, signature=payload['signature'], protocol=payload.get('protocol'),
        training_config=payload['config'], validation=payload.get('validation'),
        val_loss=payload.get('val_loss'), tensors=len(payload['model']),
        note='Best saved checkpoint among completed validations, NOT proof the full training run finished.'))
    atomic_json(out / 'source_paths.json', dict(manifest=str(manifest_path), dataset_root=str(dataset_root),
                                              checkpoint=str(source), restored_from=str(restored) if restored else None))
    state = dict(evaluator=UPLOAD_EVALUATOR, fingerprint=fingerprint, status='running',
                 planned_cases=plan['cases'], completed_cases=[], saved_rows={}, n_per_case=len(frame),
                 smoke=opts['max_batches'] is not None, training=False)

    def progress(name, done):
        state['saved_rows'][name] = done
        atomic_json(out / 'evaluation_state.json', state)

    atomic_json(out / 'evaluation_state.json', state)
    print(f'[checkpoint] epoch {payload["epoch"] + 1}, signature={payload["signature"]}; INFERENCE ONLY', flush=True)
    print(f'[cases] {plan["cases"]}; aliases skipped={plan["aliases_not_measured"]}', flush=True)
    print(f'[budget] Max {opts["max_new_cases_per_run"]} new cases / {opts["time_budget_hours"]}h; '
          'planned pauses return normally so Kaggle can save outputs.', flush=True)
    complete_predictions, model, anchor, started_cases, reason = [], None, None, 0, 'complete'
    for name in plan['cases']:
        folder = out / 'progress' / name
        prior, n = read_uploaded_case(folder, frame, cfg, name)
        if n:
            bind_upload_metadata(folder / 'metadata.json', dict(run=fingerprint, case=case_spec(name), evaluator=UPLOAD_EVALUATOR))
        if n == len(frame):
            print(f'[{name}] reuse COMPLETE saved case; zero model forward.', flush=True)
            result, complete = prior, True
        else:
            if time.monotonic() >= deadline:
                reason = 'time_budget'
                break
            if opts['max_new_cases_per_run'] is not None and started_cases >= opts['max_new_cases_per_run']:
                reason = 'case_limit'
                break
            if model is None:
                if opts['pretrained_model_path']:
                    cfg.update(model_id=opts['pretrained_model_path'], local_files_only=True)
                seed_all(cfg['seeds'][0])
                if backend is None:
                    model, anchor = build_models(cfg, device, True)
                    model.load_state_dict(payload['model'], strict=True)
                else:
                    model, anchor = backend.build_models(cfg, device, payload['model'])
                del payload
                model.requires_grad_(False).eval()
                anchor.requires_grad_(False).eval()
            started_cases += 1
            arguments = (model, anchor, frame, cfg, opts, name, out, fingerprint, device, deadline, progress)
            result, complete = (evaluate_uploaded_case(*arguments) if backend is None else
                                evaluate_uploaded_case(*arguments, backend=backend))
        progress(name, len(result))
        if not complete:
            reason = 'time_budget'
            if len(result):
                partial_rows = [dict(case=name, generator=gen, complete=False, **eval_metrics(group, cfg['threshold']))
                                for gen, group in [('pooled', result), *result.groupby('generator')]]
                save_csv(out / 'partial_metrics.csv', pd.DataFrame(partial_rows))
            break
        save_csv(out / f'predictions_{name}.csv', result)
        complete_predictions.append(result)
        state['completed_cases'].append(name)
        table = write_eval_reports(pd.concat(complete_predictions, ignore_index=True), cfg, out)
        print(table[(table.case == name) & (table.generator == 'pooled')].to_string(index=False), flush=True)
        gc.collect()
        if backend is None:
            torch.cuda.empty_cache()
        else:
            backend.cleanup()
    all_predictions = pd.concat(complete_predictions, ignore_index=True) if complete_predictions else pd.DataFrame()
    if len(all_predictions):
        save_csv(out / 'predictions.csv', all_predictions)
        table = write_eval_reports(all_predictions, cfg, out)
    else:
        table = pd.DataFrame(columns=['case', 'generator', 'n', 'accuracy', 'balanced_accuracy', 'auc', 'ap'])
    finished = len(state['completed_cases']) == len(plan['cases'])
    state.update(status='complete' if finished else 'paused', reason='complete' if finished else reason,
                 remaining_cases=[c for c in plan['cases'] if c not in state['completed_cases']],
                 elapsed_seconds=time.monotonic() - started)
    atomic_json(out / 'evaluation_state.json', state)
    atomic_json(out / 'completion.json', state)
    print(f'[session] {state["status"]}; completed={state["completed_cases"]}; remaining={state["remaining_cases"]}', flush=True)
    if not finished:
        print('[next] Save this notebook output; attach it as Input next run. resume_from=auto restores '
              'matching progress. DO NOT restart the same 3 cases without attaching saved progress.', flush=True)
    return table

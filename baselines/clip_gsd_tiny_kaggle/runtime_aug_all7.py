"""All7 crop560 + label-independent blur/JPEG + fixed two-view validation.

Append after runtime.py and runtime_raw_all7.py in the standalone notebook.
The raw-only runtimes/notebooks remain unchanged.
"""
if 'GENS7' not in globals():
    from pathlib import Path
    _raw_file = Path(__file__).with_name('runtime_raw_all7.py')
    exec(compile(_raw_file.read_text(encoding='utf-8'), str(_raw_file), 'exec'), globals())

if 'load_manifest_index' not in globals():
    _manifest_file = Path(__file__).with_name('manifest_reuse.py')
    exec(compile(_manifest_file.read_text(encoding='utf-8'), str(_manifest_file), 'exec'), globals())

_raw_images = Images
_raw_validate = validate_raw_config
_raw_splits = make_splits
_raw_run = run
_full_scan = scan_dataset

VALIDATION_MONITOR = 'mean_native_fixed_processed_cross_entropy'
DEFAULT_CONFIG = dict(
    DEFAULT_CONFIG,
    output_root='/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_v2',
    data_mode='reuse_manifests', manifest_root=None,
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], horizontal_flip_probability=0.0,
    validation_processing_seed=42, validation_monitor=VALIDATION_MONITOR,
)


def protocol_id(cfg):
    return f"all7_center_crop{cfg['crop_max_side']}_blur_jpeg_fixedval_v2"


def validate_raw_config(cfg):
    # Reuse geometry, batch, evaluation-case and All7 safeguards, not the raw-only ban.
    _raw_validate(dict(cfg, jpeg_probability=0.0, blur_probability=0.0,
                       horizontal_flip_probability=0.0))
    if cfg['input_mode'] != 'center_crop_max_edge_pad14':
        raise ValueError('Blur/JPEG protocol requires the capped center-crop input mode')
    if cfg['horizontal_flip_probability'] != 0:
        raise ValueError('Only Gaussian blur and JPEG are enabled; flip must remain zero')
    for name in ('jpeg_probability', 'blur_probability'):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite probability in [0, 1]')
    quality = cfg['jpeg_train_quality']
    if (not isinstance(quality, (list, tuple)) or len(quality) != 2 or
            any(isinstance(v, bool) or not isinstance(v, int) for v in quality) or
            not 1 <= quality[0] <= quality[1] <= 100):
        raise ValueError('jpeg_train_quality must be two ordered integers in [1, 100]')
    sigma = cfg['blur_sigma']
    if (not isinstance(sigma, (list, tuple)) or len(sigma) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in sigma) or
            not 0 <= sigma[0] <= sigma[1]):
        raise ValueError('blur_sigma must be two finite ordered nonnegative numbers')
    if isinstance(cfg['jpeg_subsampling'], bool) or cfg['jpeg_subsampling'] != 0:
        raise ValueError('This controlled protocol fixes JPEG subsampling=0 (4:4:4)')
    if isinstance(cfg['validation_processing_seed'], bool) or not isinstance(cfg['validation_processing_seed'], int):
        raise ValueError('validation_processing_seed must be an integer')
    if cfg['validation_monitor'] != VALIDATION_MONITOR:
        raise ValueError('Early stopping must monitor equal-weight native/processed validation CE')
    if cfg['data_mode'] not in ('reuse_manifests', 'full_scan'):
        raise ValueError('data_mode must be reuse_manifests or explicit full_scan')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Need a nonempty list of distinct run seeds')


def scan_dataset(cfg, out):
    if cfg['data_mode'] == 'reuse_manifests':
        return load_manifest_index(cfg, out)
    return _full_scan(cfg, out)


def processing_parameters(sample_id, cfg, seed, train=False, epoch=0):
    """Local RNG: independent of labels, row order, workers and global random state.

    Validation deliberately ignores epoch; training changes its draws each epoch.
    The two Bernoulli gates are independent, including when an operation is disabled.
    """
    identity = ['blur_jpeg_v1', int(seed), str(sample_id)]
    identity += ['train', int(epoch)] if train else ['validation', cfg['validation_processing_seed']]
    rng = random.Random(int(digest(identity)[:16], 16))
    blur_applied = rng.random() < cfg['blur_probability']
    jpeg_applied = rng.random() < cfg['jpeg_probability']
    sigma = rng.uniform(*cfg['blur_sigma'])
    quality = rng.randint(*cfg['jpeg_train_quality'])
    return dict(blur_applied=blur_applied, blur_sigma=sigma if blur_applied else None,
                jpeg_applied=jpeg_applied, jpeg_quality=quality if jpeg_applied else None)


def apply_processing(image, parameters, cfg):
    # Operate on the decoded original, BEFORE crop/padding/normalization.
    if parameters['blur_applied']:
        image = image.filter(ImageFilter.GaussianBlur(parameters['blur_sigma']))
    if parameters['jpeg_applied']:
        image = jpeg(image, parameters['jpeg_quality'], cfg['jpeg_subsampling'])
    return image


class Images(_raw_images):
    def __init__(self, frame, cfg, seed, train=False, epoch=0, case='native'):
        super().__init__(frame, cfg, seed, train, epoch, case)
        self.seed, self.epoch = seed, epoch

    def __getitem__(self, i):
        if not self.train and self.case != 'processed':
            # Native/test cases remain byte-for-byte equivalent to the raw evaluator.
            return super().__getitem__(i)
        row = self.rows[i]
        with Image.open(row['path']) as im:
            image = im.convert('RGB')
        parameters = processing_parameters(row['sample_id'], self.cfg, self.seed,
                                           train=self.train, epoch=self.epoch)
        image = apply_processing(image, parameters, self.cfg)
        geom = input_geometry(*image.size, self.cfg)
        image = image.crop((geom['crop_left'], geom['crop_top'],
                            geom['crop_left'] + geom['crop_width'],
                            geom['crop_top'] + geom['crop_height']))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        h, w = x.shape[-2:]
        x = F.pad(x.unsqueeze(0), (0, (-w) % 14, 0, (-h) % 14), mode='replicate').squeeze(0)
        mean = x.new_tensor([.48145466, .4578275, .40821073])[:, None, None]
        std = x.new_tensor([.26862954, .26130258, .27577711])[:, None, None]
        return (x - mean) / std, int(row['label']), i


def save_csv(path, frame):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def make_splits(df, cfg, seed, out):
    if cfg['data_mode'] == 'reuse_manifests':
        train, val, tests = reuse_saved_splits(df, cfg, seed, out)
    else:
        train, val, tests = _raw_splits(df, cfg, seed, out)
    columns = [c for c in ('sample_id', 'generator', 'label', 'pixel_hash') if c in val]
    plan = pd.concat([
        val[columns].reset_index(drop=True),
        pd.DataFrame([processing_parameters(sid, cfg, seed) for sid in val.sample_id]),
    ], axis=1)
    plan['batch_id'] = np.arange(len(plan)) // cfg['test_batch']
    plan['run_seed'] = seed
    plan['validation_processing_seed'] = cfg['validation_processing_seed']
    save_csv(out / 'validation_processing_plan.csv', plan)
    dump(out / 'validation_protocol.json', dict(
        cases=['native', 'processed'], monitor=VALIDATION_MONITOR,
        weights=dict(native=0.5, processed=0.5),
        processing_order=['decode_rgb', 'blur', 'jpeg', 'center_crop', 'edge_pad14', 'normalize'],
        validation_processing_seed=cfg['validation_processing_seed'], run_seed=seed,
        plan_hash=digest(plan.to_dict('records')),
        source='validation split from train folders only; official test excluded',
        batching='same fixed ID order and membership in both validation views',
        loss='mean per-image cross entropy computed directly from logits',
    ))
    print(f'[validation] Fixed native + processed views; plan: {out / "validation_processing_plan.csv"}', flush=True)
    return train, val, tests


@torch.no_grad()
def predict(model, anchor, frame, cfg, seed, device, case, return_loss=False):
    if len(frame) == 0:
        raise ValueError('Cannot evaluate an empty cohort')
    model.eval()
    result = np.full(len(frame), np.nan)
    batches = np.full(len(frame), -1, dtype=np.int64)
    loss_sum, count = 0.0, 0
    for bid, (images, labels, indices) in enumerate(loader(frame, cfg, seed, case=case)):
        # Both branches see exactly the SAME transformed tensors; no second RNG draw.
        bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
        for start, x in image_chunks(images, cfg['micro_batch']):
            with amp(cfg, device):
                logits = model(x.to(device), bases)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'Nonfinite {case} logits')
            ix = indices[start:start + len(x)].numpy()
            result[ix] = logits.float().softmax(-1)[:, 1].cpu().numpy()
            batches[ix] = bid
            if return_loss:
                loss_sum += float(F.cross_entropy(
                    logits.float(), labels[start:start + len(x)].to(device), reduction='sum'))
                count += len(x)
        if bid % 25 == 0:
            print(f'[{case}] batch {bid + 1}/{math.ceil(len(frame) / cfg["test_batch"])}', flush=True)
    if not np.isfinite(result).all() or (batches < 0).any():
        raise ValueError('Missing/nonfinite prediction')
    if return_loss:
        if count != len(frame) or not math.isfinite(loss_sum):
            raise FloatingPointError('Incomplete/nonfinite validation loss')
        return result, batches, loss_sum / count
    return result, batches


def evaluate_validation(model, anchor, val, cfg, seed, device):
    summaries, predictions = {}, {}
    for case in ('native', 'processed'):
        p, batches, loss = predict(model, anchor, val, cfg, seed, device, case, return_loss=True)
        summaries[case] = dict(loss=loss, **metric(val.label.to_numpy(), p, cfg['threshold']))
        predictions[case] = (p, batches)
    if not np.array_equal(predictions['native'][1], predictions['processed'][1]):
        raise ValueError('Validation batch membership changed across views')
    monitor = (summaries['native']['loss'] + summaries['processed']['loss']) / 2.0
    return monitor, summaries, predictions


def train_model(model, anchor, train, val, cfg, seed, device, folder, signature):
    groups = [dict(params=model.vision.parameters(), lr=cfg['lr']),
              dict(params=model.head.parameters(), lr=cfg['head_lr'])]
    if len(model.gsd):
        groups.append(dict(params=model.gsd.parameters(), lr=cfg['modulation_lr']))
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['precision'] == 'fp16')
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    latest = folder / 'last.pt'
    start, best, history = 0, float('inf'), []
    if cfg['resume'] and latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck['signature'] != signature or ck.get('protocol') != protocol_id(cfg) or ck.get('config') != cfg:
            raise ValueError('Resume protocol/config/manifests changed; use a new output_root')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        stopper.state.update(ck['early_stopping'])
        start, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        del ck
        print(f'[resume] next epoch {start + 1}, stopped={stopper.state["stopped"]}', flush=True)
    for epoch in range(start, cfg['epochs']):
        if stopper.state['stopped']:
            break
        model.train()
        running, count, skipped = 0.0, 0, 0
        tick = time.monotonic()
        for bid, (images, labels, _) in enumerate(loader(train, cfg, seed, True, epoch)):
            bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
            optimizer.zero_grad(set_to_none=True)
            for a, x in image_chunks(images, cfg['micro_batch']):
                with amp(cfg, device):
                    logits = model(x.to(device), bases)
                    loss = F.cross_entropy(logits.float(), labels[a:a + len(x)].to(device), reduction='sum') / len(images)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                running += float(loss.detach()) * len(images)
            old = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old)
            count += len(images)
            if bid % 10 == 0:
                print(f'[epoch {epoch + 1}] batch {bid + 1}/{math.ceil(len(train) / cfg["train_batch"])}; loss={running / count:.5f}', flush=True)
        if count == 0:
            raise ValueError('Empty training cohort')
        vloss, summaries, predictions = evaluate_validation(model, anchor, val, cfg, seed, device)
        stop = stopper.update(vloss)
        entry = dict(epoch=epoch + 1, train_loss=running / count, val_loss=vloss,
                     val_native_loss=summaries['native']['loss'], val_processed_loss=summaries['processed']['loss'],
                     validation_monitor=VALIDATION_MONITOR,
                     val_native=summaries['native'], val_processed=summaries['processed'],
                     seconds=time.monotonic() - tick, amp_skipped_steps=skipped, train_images=count,
                     modulation={l: dict(a=float(g.a.detach()), b=float(g.b.detach())) for l, g in model.gsd.items()},
                     early_stopping=dict(stopper.state))
        history.append(entry)
        print(entry, flush=True)
        state = dict(model=model.state_dict(), epoch=epoch, signature=signature, config=cfg,
                     protocol=protocol_id(cfg), validation=summaries, val_loss=vloss)
        if vloss < best:
            best = vloss
            save_checkpoint(folder / 'best.pt', state)
        save_checkpoint(latest, dict(state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                     best=best, history=history, early_stopping=dict(stopper.state)))
        dump(folder / 'history.json', history)
        dump(folder / 'training_status.json', dict(epoch=epoch + 1, max_epochs=cfg['epochs'],
             validation_monitor=VALIDATION_MONITOR, val_loss=vloss, early_stopping=dict(stopper.state)))
        rows = [dict(epoch=e['epoch'], case=case, monitor_loss=e['val_loss'], **e['val_' + case])
                for e in history for case in ('native', 'processed')]
        save_csv(folder / 'validation_metrics.csv', pd.DataFrame(rows))
        columns = [c for c in ('sample_id', 'generator', 'label', 'pixel_hash') if c in val]
        frame = val[columns].copy()
        for case, (p, batches) in predictions.items():
            frame['prob_fake_' + case] = p
            frame['batch_id_' + case] = batches
        save_csv(folder / f'validation_predictions_epoch{epoch + 1:03d}.csv', frame)
        if stop:
            print('[early stopping] Restore best mean native/processed validation CE checkpoint.', flush=True)
            break
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    dump(folder / 'selected_checkpoint.json', dict(epoch=ck['epoch'] + 1,
         criterion=VALIDATION_MONITOR, val_loss=ck['val_loss'], validation=ck['validation'], signature=signature))


def run(config=None):
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    validate_raw_config(cfg)
    print('[protocol] All7 crop560; train blur/JPEG resampled per epoch, SAME tensors for anchor/detector.', flush=True)
    print('[protocol] Fixed native/processed validation; early stopping = 0.5 * CE(native) + 0.5 * CE(processed).', flush=True)
    print('[protocol] New pretrained run only; do not reuse raw-only checkpoint/output directories.', flush=True)
    print(f'[protocol] Data mode: {cfg["data_mode"]}; manifests may be reused, detector weights are NOT reused.', flush=True)
    return _raw_run(cfg)

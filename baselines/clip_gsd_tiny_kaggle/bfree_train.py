"""GSD training on ONE B-Free branch (default SD2.1_inpainted_samecat_origBG), paired with COCO real.

Appended after runtime.py, runtime_raw_all7.py, manifest_reuse.py, runtime_aug_all7.py and
bfree_data.py. Model, GSD basis, optimizer, precision and input geometry are the SAME code and
values as GSD-aug-Tiny (train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb). What changes:
  * data: B-Free train_list minus valid_list, one fake branch, real/fake of the same COCO ID;
  * batch: 8 = 4 COCO IDs x (real, fake) so the GSD basis is estimated on content-matched pairs;
  * blur/JPEG: SAME recipe, but one shared draw per pair (seed, COCO ID, virtual epoch);
  * schedule: virtual epochs of 12,600 IDs (25,200 images = one Tiny epoch), no ID repeated,
    validation + early stopping after each, mid-epoch resume, Kaggle time budget.
Checkpoints use the evaluator payload format so the existing Tiny 9-case and B-Free-valid
evaluators can score them with identical inference code.
"""
import shutil
import time

BFREE_SESSION_STARTED = time.time()
BFREE_TRAIN_PROTOCOL = 'gsd_bfree_single_branch_paired_v1'
INFERENCE_KEYS = ('model_id', 'input_mode', 'crop_max_side', 'test_batch', 'micro_batch', 'anchor_micro_batch',
                  'precision', 'basis_method', 'attention', 'pooling', 'k', 'lambda_max', 'a_init', 'b_init',
                  'jpeg_subsampling', 'threshold')

BFREE_TRAIN_DEFAULTS = dict(
    training_protocol=BFREE_TRAIN_PROTOCOL,
    input_root='/kaggle/input', bfree_root=None,
    output_root='/kaggle/working/gsd_bfree_samecat_origbg_v1', resume_from='auto',
    train_branch='SD2.1_inpainted_samecat_origBG', real_folder=BFREE_REAL,
    check_list_md5=True, image_side=512,
    mask_diff_threshold=8, min_mask_fraction=0.001, max_mask_fraction=0.95,
    max_unreadable_fraction=0.01, audit_valid_ids=True, audit_workers=8,
    seed=42, val_ids=2000, ve_ids=12600, max_virtual_epochs=3, pairs_per_batch=4,
    # ---- identical to GSD-aug-Tiny -------------------------------------------------------
    model_id='openai/clip-vit-large-patch14', local_files_only=False,
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    train_batch=8, test_batch=8, micro_batch=1, anchor_micro_batch=1,
    lr=1e-6, head_lr=1e-6, modulation_lr=1e-6, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8,
    k=64, lambda_max=1.0, a_init=2.0, b_init=1.0, precision='fp16', gradient_checkpointing=True,
    attention='sdpa', basis_method='gram_eigh', pooling='patch_mean_post_ln', threshold=0.5,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], horizontal_flip_probability=0.0,
    validation_processing_seed=42, validation_monitor=VALIDATION_MONITOR,
    early_stopping=True, patience=2, min_delta=0.001,
    # ---- session handling ------------------------------------------------------------------
    num_workers=2, save_every_steps=250, log_every_steps=50, time_budget_hours=11.0,
    validation_time_reserve_minutes=None, save_virtual_epoch_checkpoints=True,
    keep_last_on_complete=False, smoke=False, allow_cpu=False,
)


def bfree_smoke_overrides(cfg):
    return dict(cfg, smoke=True, output_root=cfg['output_root'].rstrip('/') + '_smoke',
                val_ids=8, ve_ids=8, max_virtual_epochs=2, audit_valid_ids=False,
                save_every_steps=1, log_every_steps=1)


def validate_bfree_config(cfg):
    unknown = set(cfg) - set(BFREE_TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    if cfg['training_protocol'] != BFREE_TRAIN_PROTOCOL:
        raise ValueError('training_protocol is fixed')
    if cfg['train_branch'] not in BFREE_ALL_BRANCHES:
        raise ValueError(f'train_branch must be one of {BFREE_ALL_BRANCHES}')
    if cfg['train_batch'] != 2 * cfg['pairs_per_batch'] or cfg['test_batch'] != cfg['train_batch']:
        raise ValueError('Logical batch must be 2 x pairs_per_batch (complete real/fake pairs) for train and eval')
    if cfg['val_ids'] % cfg['pairs_per_batch'] or cfg['ve_ids'] % cfg['pairs_per_batch']:
        raise ValueError('val_ids and ve_ids must be multiples of pairs_per_batch')
    if cfg['horizontal_flip_probability'] != 0:
        raise ValueError('P2 keeps the GSD-aug-Tiny recipe: no flip')
    if not 0 < cfg['time_budget_hours'] <= 11.5:
        raise ValueError('time_budget_hours must be in (0, 11.5] for a 12 h Kaggle session')
    # Re-use the GSD-aug-Tiny augmentation/geometry/batch validator on an equivalent config.
    validate_raw_config(dict(DEFAULT_CONFIG, **{k: cfg[k] for k in (
        'input_mode', 'crop_max_side', 'train_batch', 'test_batch', 'micro_batch', 'anchor_micro_batch',
        'k', 'precision', 'attention', 'basis_method', 'pooling', 'jpeg_probability', 'jpeg_train_quality',
        'jpeg_subsampling', 'blur_probability', 'blur_sigma', 'horizontal_flip_probability',
        'validation_processing_seed', 'validation_monitor')}, seeds=[cfg['seed']], epochs=1))


def eval_compat_config(cfg):
    """Config the EXISTING evaluators read. Inference keys are copied from this run; the
    train_generators/expected_generators fields only describe the Tiny TEST cohort layout."""
    compat = dict(DEFAULT_CONFIG)
    compat.update({k: cfg[k] for k in INFERENCE_KEYS})
    compat.update(local_files_only=cfg['local_files_only'], gradient_checkpointing=cfg['gradient_checkpointing'],
                  train_batch=cfg['train_batch'], seeds=[cfg['seed']], models=['clip_gsd'], smoke=bool(cfg['smoke']),
                  eval_cases=['native'], manifest_root=None, dataset_root=None, data_mode='reuse_manifests',
                  output_root=cfg['output_root'], lr=cfg['lr'], head_lr=cfg['head_lr'],
                  modulation_lr=cfg['modulation_lr'], weight_decay=cfg['weight_decay'],
                  jpeg_probability=cfg['jpeg_probability'], blur_probability=cfg['blur_probability'],
                  jpeg_train_quality=cfg['jpeg_train_quality'], blur_sigma=cfg['blur_sigma'],
                  horizontal_flip_probability=0.0,
                  compat_note=('Evaluator view of a B-Free-trained checkpoint. Training data: '
                               f"B-Free {cfg['train_branch']} vs {cfg['real_folder']}. "
                               'train_generators here only describes the Tiny TEST manifest.'))
    return compat


def check_evaluator_compatibility(payload, cfg):
    """Mirror of evaluate_uploaded_best.validate_uploaded_payload, applied to the compat view."""
    if not isinstance(payload, dict) or not {'model', 'config', 'epoch', 'signature'} <= set(payload):
        raise ValueError('Checkpoint must contain model/config/epoch/signature')
    compat = payload['config']['eval_compat_config']
    _raw_validate(dict(compat, jpeg_probability=0., blur_probability=0., horizontal_flip_probability=0.))
    if compat['input_mode'] != 'center_crop_max_edge_pad14' or compat.get('crop_max_side') != 560:
        raise ValueError('Evaluators expect the crop560 protocol')
    if len(compat['seeds']) != 1 or compat['test_batch'] != 8:
        raise ValueError('Evaluators expect one seed and logical test batch 8')
    for key in INFERENCE_KEYS:
        if compat[key] != payload['config'][key]:
            raise ValueError(f'compat/training mismatch for {key}')
    if not isinstance(payload['model'], dict) or not payload['model']:
        raise ValueError('Missing model state_dict')
    return True


# ----------------------------------------------------------------------------- data (torch)
class BFreePairImages(Dataset):
    """Rows interleave real/fake of the same ID. mode: 'train' | 'native' | 'processed'."""
    def __init__(self, rows, cfg, seed, mode, virtual_epoch=0):
        if mode not in ('train', 'native', 'processed'):
            raise ValueError(mode)
        self.rows, self.cfg, self.seed, self.mode, self.ve = rows, cfg, seed, mode, virtual_epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.mode == 'native':
            parameters = None
        else:
            parameters = pair_processing_parameters(row['coco_id'], self.cfg, self.seed,
                                                    train=self.mode == 'train', virtual_epoch=self.ve)
        image = load_processed_crop(row['path'], self.cfg, parameters)
        # Tensor path identical to GSD-aug-Tiny Images.__getitem__.
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        h, w = x.shape[-2:]
        x = F.pad(x.unsqueeze(0), (0, (-w) % 14, 0, (-h) % 14), mode='replicate').squeeze(0)
        mean = x.new_tensor([.48145466, .4578275, .40821073])[:, None, None]
        std = x.new_tensor([.26862954, .26130258, .27577711])[:, None, None]
        return (x - mean) / std, int(row['label']), i


def pair_loader(rows, batches, cfg, seed, mode, virtual_epoch=0):
    return DataLoader(BFreePairImages(rows, cfg, seed, mode, virtual_epoch), batch_sampler=batches,
                      num_workers=cfg['num_workers'], collate_fn=native_collate,
                      pin_memory=torch.cuda.is_available())


def check_pair_batch(rows, indices):
    ids = [rows[i]['coco_id'] for i in indices]
    labels = [rows[i]['label'] for i in indices]
    if labels != [0, 1] * (len(indices) // 2) or ids[0::2] != ids[1::2]:
        raise AssertionError('Logical batch is not made of complete real/fake pairs of the same ID')


@torch.no_grad()
def evaluate_pairs(model, anchor, frame, cfg, seed, device, mode):
    model.eval()
    rows = pair_rows(frame)
    batches = pair_batches(len(frame), cfg['pairs_per_batch'])
    probs = np.full(len(rows), np.nan)
    margins = np.full(len(rows), np.nan)
    loss_sum, started = 0.0, time.monotonic()
    for bid, (images, labels, indices) in enumerate(pair_loader(rows, batches, cfg, seed, mode)):
        check_pair_batch(rows, indices.tolist())
        bases = estimate_bases(anchor, images, cfg, device, model.selected)
        for start, x in image_chunks(images, cfg['micro_batch']):
            with amp(cfg, device):
                logits = model(x.to(device), bases)
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'Nonfinite {mode} validation logits')
            ix = indices[start:start + len(x)].numpy()
            probs[ix] = logits.softmax(-1)[:, 1].cpu().numpy()
            margins[ix] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            loss_sum += float(F.cross_entropy(logits, labels[start:start + len(x)].to(device), reduction='sum'))
        if bid % 100 == 0:
            print(f'[val:{mode}] batch {bid + 1}/{len(batches)}', flush=True)
    if not np.isfinite(probs).all():
        raise ValueError('Missing validation predictions')
    seconds = time.monotonic() - started
    return probs, margins, loss_sum / len(rows), seconds / len(rows)


def area_metrics(frame, probs, threshold):
    """Fake recall / AUC by mask-area group (reals shared). Rows: real at 2k, fake at 2k+1."""
    real_p, fake_p = probs[0::2], probs[1::2]
    out = {}
    for group in ('small', 'medium', 'large'):
        sel = (frame.area_group == group).to_numpy()
        if sel.sum() == 0:
            continue
        y = np.r_[np.zeros(len(real_p)), np.ones(sel.sum())]
        p = np.r_[real_p, fake_p[sel]]
        out[group] = dict(n_fake=int(sel.sum()), fake_recall=float((fake_p[sel] >= threshold).mean()),
                          auc_vs_all_real=float(roc_auc_score(y, p)))
    return out


# ----------------------------------------------------------------------------- checkpoints
def evaluator_payload(model, cfg, signature, ve, summaries, val_loss, images_seen):
    config = dict(cfg)
    config['eval_compat_config'] = eval_compat_config(cfg)
    return dict(model={k: v.detach().to('cpu') for k, v in model.state_dict().items()},
                epoch=int(ve), virtual_epoch=int(ve) + 1, signature=signature, config=config,
                protocol=BFREE_TRAIN_PROTOCOL, validation=summaries, val_loss=float(val_loss),
                images_seen=int(images_seen))


def save_and_verify_payload(path, payload, cfg):
    save_checkpoint(Path(path), payload)
    loaded = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    check_evaluator_compatibility(loaded, cfg)
    if loaded['signature'] != payload['signature'] or set(loaded['model']) != set(payload['model']):
        raise ValueError(f'Round-trip check failed for {path}')
    del loaded
    print(f'[checkpoint] {Path(path).name} saved; weights_only reload + evaluator checks OK', flush=True)


def write_state(out, state):
    state = dict(state, updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
    atomic_json(out / 'run_state.json', state)
    return state


def save_last(out, model, optimizer, scaler, stopper, state):
    started = time.monotonic()
    save_checkpoint(out / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                          scaler=scaler.state_dict(), early_stopping=dict(stopper.state),
                                          run_state=dict(state)))
    state = write_state(out, state)
    print(f"[save] last.pt at VE{state['ve'] + 1} step {state['step']} ({state['phase']}); "
          f"{time.monotonic() - started:.0f}s", flush=True)
    return state


# ----------------------------------------------------------------------------- preparation
def copy_previous_output(cfg, run_hash, out):
    if cfg['resume_from'] is None:
        return None
    if cfg['resume_from'] == 'auto':
        matches = find_previous_outputs(cfg, run_hash, exclude=[out])
    else:
        folder = Path(cfg['resume_from'])
        state = json.loads((folder / 'run_state.json').read_text(encoding='utf-8'))
        if state.get('run_hash') != run_hash:
            raise ValueError(f'{folder}: run_state belongs to a different config')
        matches = [(int(state.get('images_seen', 0)), '', folder, state)]
    if not matches:
        print('[resume] No previous output of this run attached; starting fresh.', flush=True)
        return None
    images, _, folder, state = matches[-1]
    local = out / 'run_state.json'
    if local.is_file():
        mine = json.loads(local.read_text(encoding='utf-8'))
        if int(mine.get('images_seen', 0)) >= images:
            print('[resume] Local output is already at least as advanced; not copying.', flush=True)
            return None
    print(f'[resume] Copying previous output {folder} (images_seen={images:,}, status={state.get("status")})', flush=True)
    shutil.copytree(folder, out, dirs_exist_ok=True)
    return str(folder)


def prepare_bfree_data(cfg, out):
    root = find_bfree_root(cfg)
    train_list, train_md5 = read_bfree_list(root / 'train_list.csv', BFREE_TRAIN_LIST_MD5,
                                            BFREE_TRAIN_LIST_COUNT if cfg['check_list_md5'] else None,
                                            cfg['check_list_md5'])
    valid_list, valid_md5 = read_bfree_list(root / 'valid_list.csv', BFREE_VALID_LIST_MD5,
                                            BFREE_VALID_LIST_COUNT if cfg['check_list_md5'] else None,
                                            cfg['check_list_md5'])
    valid_ids = set(valid_list.id)
    if not valid_ids <= set(train_list.id):
        raise ValueError('valid_list must be contained in train_list (official release)')
    train_ids = [i for i in train_list.id if i not in valid_ids]  # 41,213 for the official lists
    real_dir = bfree_folder(root, cfg['real_folder'], train_ids[:20])
    fake_dir = bfree_folder(root, cfg['train_branch'], train_ids[:20])
    meta = dict(bfree_root=str(root), real_dir=str(real_dir), fake_dir=str(fake_dir),
                train_list_md5=train_md5, valid_list_md5=valid_md5,
                train_list_ids=len(train_list), valid_list_ids=len(valid_list), train_pool_ids=len(train_ids),
                branch=cfg['train_branch'], audit_rule={k: cfg[k] for k in (
                    'image_side', 'mask_diff_threshold', 'min_mask_fraction', 'max_mask_fraction')})
    if cfg['smoke']:
        train_ids = sorted(train_ids, key=lambda i: id_rank(cfg['seed'], 'smoke', i))[:64]
        meta['smoke_subset'] = len(train_ids)
    audit_path, meta_path = out / 'audit_train.csv', out / 'audit_meta.json'
    reuse = False
    if audit_path.is_file() and meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        same = {k: old.get(k) for k in ('train_list_md5', 'valid_list_md5', 'branch', 'audit_rule', 'smoke_subset')} == \
               {k: meta.get(k) for k in ('train_list_md5', 'valid_list_md5', 'branch', 'audit_rule', 'smoke_subset')}
        reuse = same
    if reuse:
        audit = restore_audit_types(pd.read_csv(audit_path, dtype={'coco_id': str, 'exclusion_reason': str},
                                                keep_default_na=False))
        audit['real_path'] = [str(real_dir / f'{i}.png') for i in audit.coco_id]
        audit['fake_path'] = [str(fake_dir / f'{i}.png') for i in audit.coco_id]
        if sorted(audit.coco_id) != sorted(train_ids):
            raise ValueError('Saved audit IDs differ from the current train pool')
        print(f'[audit] Reusing saved audit of {len(audit):,} train pairs (paths rebased).', flush=True)
    else:
        print(f'[audit] Decoding {len(train_ids):,} train pairs ({cfg["real_folder"]} vs {cfg["train_branch"]}); '
              'mask = max-channel |fake-real| > threshold.', flush=True)
        audit = mark_eligibility(audit_pairs(train_ids, real_dir, fake_dir, cfg, 'train'), cfg)
        atomic_csv(audit_path, audit)
        atomic_json(meta_path, meta)
    unreadable = float((~audit.ok).mean())
    if unreadable > cfg['max_unreadable_fraction']:
        raise ValueError(f'{unreadable:.2%} of train pairs missing/unreadable: wrong dataset version/layout? '
                         f'See {audit_path}')
    ok = audit[audit.ok]
    summary = dict(pairs=len(audit), unreadable=int((~audit.ok).sum()), eligible=int(audit.eligible.sum()),
                   exclusions=audit.exclusion_reason[audit.exclusion_reason != ''].value_counts().to_dict(),
                   identical_pairs=int(ok.identical.astype(bool).sum()) if len(ok) else 0,
                   mask_fraction_quantiles={str(q): float(ok.mask_fraction.quantile(q)) for q in
                                            (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)} if len(ok) else {},
                   median_changed_any_minus_mask=float((pd.to_numeric(ok.changed_any_fraction) -
                                                        ok.mask_fraction).median()) if len(ok) else None)
    atomic_json(out / 'audit_summary.json', summary)
    print('[audit] ' + json.dumps(summary, indent=1), flush=True)
    valid_path = out / 'audit_valid.csv'
    if cfg['audit_valid_ids'] and not valid_path.is_file():
        vfake = bfree_folder(root, cfg['train_branch'], valid_list.id.tolist()[:20])
        print(f'[audit] Mask areas for {len(valid_list):,} official valid pairs (analysis only, never trained on).',
              flush=True)
        atomic_csv(valid_path, audit_pairs(valid_list.id.tolist(), real_dir, vfake, cfg, 'valid'))
    return audit, meta


# ----------------------------------------------------------------------------- main
def run_bfree_training(config=None, build=None):
    cfg = dict(BFREE_TRAIN_DEFAULTS, **(config or {}))
    if cfg['smoke'] is True and not cfg['output_root'].endswith('_smoke'):
        cfg = bfree_smoke_overrides(cfg)
    validate_bfree_config(cfg)
    deadline = BFREE_SESSION_STARTED + cfg['time_budget_hours'] * 3600
    out = Path(cfg['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    run_hash = run_identity_hash(cfg)
    restored = copy_previous_output(cfg, run_hash, out)
    if (out / 'config.json').is_file():
        previous = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if run_identity_hash(previous) != run_hash:
            raise ValueError('output_root holds a different run; choose a new output_root')
    if (out / 'run_state.json').is_file():
        saved = json.loads((out / 'run_state.json').read_text(encoding='utf-8'))
        if saved.get('run_hash') == run_hash and saved.get('status') == 'complete':
            print('[done] This run is already complete (see completion.json); nothing to train.', flush=True)
            return finish_report(out, saved)
    atomic_json(out / 'config.json', cfg)
    seed = cfg['seed']
    print(f"[protocol] GSD on B-Free {cfg['train_branch']} vs {cfg['real_folder']}; batch8 = 4 COCO-ID pairs; "
          'blur/JPEG shared per pair; crop560/pad14; recipe = GSD-aug-Tiny.', flush=True)

    audit, meta = prepare_bfree_data(cfg, out)
    val, pool, split_summary = make_bfree_splits(audit, cfg, seed)
    atomic_csv(out / 'validation_ids.csv', val[['coco_id', 'mask_fraction', 'area_group']])
    atomic_csv(out / 'train_order.csv', pool[['coco_id', 'virtual_epoch', 'mask_fraction', 'area_group']])
    atomic_json(out / 'split_summary.json', split_summary)
    print('[split] ' + json.dumps(split_summary), flush=True)
    signature = sha(dict(run_hash=run_hash, validation=split_summary['validation_order_hash'],
                         train=split_summary['train_order_hash'], lists=[meta['train_list_md5'], meta['valid_list_md5']],
                         eligible=sha(audit[audit.eligible][['coco_id']].assign(
                             m=audit[audit.eligible].mask_fraction.round(6)).values.tolist())))
    n_ve = split_summary['virtual_epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not cfg['allow_cpu']:
        raise RuntimeError('Select a Kaggle GPU (T4 / P100). This notebook does not run on CPU/TPU.')
    if device.type == 'cuda':
        if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support bf16; keep fp16')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed_all(seed)
    model, anchor = (build or build_models)(cfg, device, True)
    groups = [dict(params=model.vision.parameters(), lr=cfg['lr']),
              dict(params=model.head.parameters(), lr=cfg['head_lr']),
              dict(params=model.gsd.parameters(), lr=cfg['modulation_lr'])]
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['precision'] == 'fp16')
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    state = dict(run_hash=run_hash, signature=signature, status='running', ve=0, phase='train', step=0,
                 images_seen=0, best_val_loss=None, best_ve=None, virtual_epochs=n_ve, history=[],
                 train_seconds_per_image=None, eval_seconds_per_image=None, sessions=[])
    if (out / 'last.pt').is_file():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if ck['run_state']['signature'] != signature:
            raise ValueError('last.pt signature differs from this data/config; use a new output_root')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        stopper.state.update(ck['early_stopping'])
        state = dict(ck['run_state'])
        if state['status'] != 'complete':
            state['status'] = 'running'
        del ck
        print(f"[resume] VE{state['ve'] + 1} step {state['step']} phase={state['phase']}; "
              f"images_seen={state['images_seen']:,}", flush=True)
    state['sessions'] = list(state.get('sessions', [])) + [dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                                 restored_from=restored)]
    state = write_state(out, state)
    if state['status'] == 'complete':
        print('[done] This run is already complete; nothing to train.', flush=True)
        return finish_report(out, state)

    save_reserve = 10 * 60
    while state['ve'] < n_ve and state['status'] != 'complete':
        ve = state['ve']
        frame = pool[pool.virtual_epoch == ve].reset_index(drop=True)
        if state['phase'] == 'train':
            rows = pair_rows(frame)
            batches = pair_batches(len(frame), cfg['pairs_per_batch'])
            start_step = state['step']
            print(f'[train] VE{ve + 1}/{n_ve}: {len(frame):,} IDs = {len(rows):,} images, '
                  f'{len(batches):,} steps, starting at step {start_step}', flush=True)
            model.train()
            tick, window_loss, window_n, skipped = time.monotonic(), 0.0, 0, 0
            log_rows = []
            loader = pair_loader(rows, batches[start_step:], cfg, seed, 'train', ve)
            for offset, (images, labels, indices) in enumerate(loader):
                step = start_step + offset
                if time.time() > deadline - save_reserve:
                    state.update(step=step, status='paused_time_budget')
                    append_train_log(out, log_rows)
                    state = save_last(out, model, optimizer, scaler, stopper, state)
                    print('[session] Time budget reached mid virtual epoch. Save Version, then attach this '
                          'Output as Input and run again: training resumes exactly here.', flush=True)
                    return finish_report(out, state)
                check_pair_batch(rows, indices.tolist())
                step_started = time.monotonic()
                bases = estimate_bases(anchor, images, cfg, device, model.selected)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for a, x in image_chunks(images, cfg['micro_batch']):
                    with amp(cfg, device):
                        logits = model(x.to(device), bases)
                        loss = F.cross_entropy(logits.float(), labels[a:a + len(x)].to(device),
                                               reduction='sum') / len(images)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    batch_loss += float(loss.detach()) * len(images)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
                state['images_seen'] += len(images)
                window_loss += batch_loss
                window_n += len(images)
                spi = (time.monotonic() - step_started) / len(images)
                prev = state.get('train_seconds_per_image')
                state['train_seconds_per_image'] = spi if prev is None else 0.98 * prev + 0.02 * spi
                done = step + 1
                if done % cfg['log_every_steps'] == 0 or done == len(batches):
                    eta_h = (len(batches) - done) * len(images) * state['train_seconds_per_image'] / 3600
                    log_rows.append(dict(ve=ve + 1, step=done, images_seen=state['images_seen'],
                                         loss=window_loss / window_n, sec_per_image=state['train_seconds_per_image'],
                                         amp_skipped=skipped))
                    print(f'[VE{ve + 1}] step {done}/{len(batches)} loss={window_loss / window_n:.5f} '
                          f'{state["train_seconds_per_image"]:.3f}s/img ETA {eta_h:.2f}h', flush=True)
                    window_loss, window_n = 0.0, 0
                if done % cfg['save_every_steps'] == 0 and done < len(batches):
                    state['step'] = done
                    append_train_log(out, log_rows)
                    log_rows = []
                    state = save_last(out, model, optimizer, scaler, stopper, state)
            append_train_log(out, log_rows)
            state.update(phase='validate', step=0)
            state['ve_train_seconds'] = time.monotonic() - tick
            state = save_last(out, model, optimizer, scaler, stopper, state)

        # ---------------- validation for this virtual epoch
        per_image = state.get('eval_seconds_per_image') or (
            0.45 * state['train_seconds_per_image'] if state.get('train_seconds_per_image') else 0.3)
        needed = 2 * 2 * len(val) * per_image * 1.15 + save_reserve
        if cfg['validation_time_reserve_minutes'] is not None:
            needed = cfg['validation_time_reserve_minutes'] * 60
        if time.time() + needed > deadline:
            state['status'] = 'paused_time_budget'
            state = write_state(out, state)
            print(f'[session] Not enough time for validation (~{needed / 60:.0f} min). Save Version, attach '
                  'this Output as Input and run again: validation of this VE runs first.', flush=True)
            return finish_report(out, state)
        summaries, frames = {}, val.copy()
        vtick = time.monotonic()
        for mode in ('native', 'processed'):
            p, m, loss, spi = evaluate_pairs(model, anchor, val, cfg, seed, device, mode)
            y = np.tile([0, 1], len(val))
            summaries[mode] = dict(loss=float(loss), **metric(y, p, cfg['threshold']),
                                   by_area=area_metrics(val, p, cfg['threshold']))
            frames[f'prob_real_{mode}'] = p[0::2]
            frames[f'prob_fake_{mode}'] = p[1::2]
            frames[f'margin_real_{mode}'] = m[0::2]
            frames[f'margin_fake_{mode}'] = m[1::2]
            state['eval_seconds_per_image'] = spi
        vloss = 0.5 * summaries['native']['loss'] + 0.5 * summaries['processed']['loss']
        stop = stopper.update(vloss)
        atomic_csv(out / f'validation_predictions_ve{ve + 1}.csv', frames)
        entry = dict(virtual_epoch=ve + 1, images_seen=state['images_seen'], val_loss=float(vloss),
                     val_native=summaries['native'], val_processed=summaries['processed'],
                     validation_seconds=time.monotonic() - vtick, train_seconds=state.get('ve_train_seconds'),
                     modulation={l: dict(a=float(g.a.detach()), b=float(g.b.detach())) for l, g in model.gsd.items()},
                     early_stopping=dict(stopper.state))
        state['history'] = list(state['history']) + [entry]
        atomic_json(out / 'history.json', state['history'])
        print('[validation] ' + json.dumps({k: entry[k] for k in ('virtual_epoch', 'images_seen', 'val_loss')}) +
              f" native BA={summaries['native']['balanced_accuracy']:.4f} AUC={summaries['native']['auc']:.4f}"
              f" fake_recall={summaries['native']['fake_recall']:.4f} | processed BA="
              f"{summaries['processed']['balanced_accuracy']:.4f}", flush=True)
        payload = evaluator_payload(model, cfg, signature, ve, summaries, vloss, state['images_seen'])
        if cfg['save_virtual_epoch_checkpoints']:
            save_and_verify_payload(out / f'checkpoint_ve{ve + 1}.pt', payload, cfg)
        if state['best_val_loss'] is None or vloss < state['best_val_loss']:
            state.update(best_val_loss=float(vloss), best_ve=ve + 1)
            save_and_verify_payload(out / 'best.pt', payload, cfg)
        del payload
        state.update(ve=ve + 1, phase='train', step=0)
        if stop or state['ve'] >= n_ve:
            state['status'] = 'complete'
            state['stop_reason'] = 'early_stopping' if stop else 'all_virtual_epochs_done'
        state = save_last(out, model, optimizer, scaler, stopper, state)
    if state['status'] == 'complete' and not cfg['keep_last_on_complete'] and (out / 'last.pt').is_file():
        (out / 'last.pt').unlink()
        print('[done] Removed last.pt (optimizer state) to keep the Output small; best.pt is kept.', flush=True)
    return finish_report(out, state)


def append_train_log(out, rows):
    if not rows:
        return
    path = out / 'train_log.csv'
    frame = pd.DataFrame(rows)
    if path.is_file():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    atomic_csv(path, frame)


def finish_report(out, state):
    report = dict(status=state['status'], stop_reason=state.get('stop_reason'),
                  virtual_epochs_validated=len(state.get('history', [])),
                  images_seen=state['images_seen'], best_virtual_epoch=state.get('best_ve'),
                  best_val_loss=state.get('best_val_loss'), signature=state['signature'],
                  best_checkpoint=str(out / 'best.pt') if (out / 'best.pt').is_file() else None)
    atomic_json(out / 'selected_checkpoint.json', report)
    if state['status'] == 'complete':
        atomic_json(out / 'completion.json', report)
    print('[report] ' + json.dumps(report, indent=1), flush=True)
    return report

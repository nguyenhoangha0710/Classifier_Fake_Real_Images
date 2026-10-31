"""CAPD — TN2: train on B-Free `SD2.1_inpainted_samecat_origBG` with the GSD-P2 data protocol.

Appended AFTER runtime.py, runtime_raw_all7.py, manifest_reuse.py, runtime_aug_all7.py,
bfree_data.py, bfree_train.py (GSD-P2) and capd_runtime.py (CAPD model / loss / eval).

Identical to GSD-P2 (same functions, called with the same values):
  audit + mask rule, eligibility, 2,000-ID area-stratified validation, train order, virtual epochs
  of 12,600 IDs, batch 8 = 4 (real, fake) pairs of the same COCO ID, blur/JPEG draw shared per pair,
  val_loss = 0.5 CE(native) + 0.5 CE(processed), early stopping patience 2 / min_delta 0.001.
Different (the method): CAPD model (frozen backbone + LoRA + PatchHead + LSE) and its loss
  L_img + 0.5 L_neg (+ L_gap warm-up), no pixel loss. Geometry: 512 crop, pad to the patch multiple
  (DINOv3: 512 = 32 x 16 -> no pad; CLIP: pad to 518 like GSD, identically for real and fake).
"""
CAPD_BFREE_PROTOCOL = 'capd_bfree_single_branch_paired_v1'

CAPD_BFREE_DEFAULTS = dict(
    task='bfree_train', training_protocol=CAPD_BFREE_PROTOCOL,
    run_name='b2_dinov3', backbone='dinov3', output_root='/kaggle/working/capd_tn2',
    input_root='/kaggle/input', bfree_root=None, gsd_output_dir=None, resume_from=None,
    # ---- data protocol: identical to GSD-P2 (bfree_train.BFREE_TRAIN_DEFAULTS) ----
    train_branch='SD2.1_inpainted_samecat_origBG', real_folder=BFREE_REAL,
    check_list_md5=True, image_side=512,
    mask_diff_threshold=8, min_mask_fraction=0.001, max_mask_fraction=0.95,
    max_unreadable_fraction=0.01, audit_valid_ids=True, audit_workers=8,
    seed=42, val_ids=2000, ve_ids=12600, max_virtual_epochs=3, pairs_per_batch=4,
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], validation_processing_seed=42,
    early_stopping=True, patience=2, min_delta=0.001, threshold=0.5,
    # ---- CAPD (identical to TN1) ----
    clip_model_id='openai/clip-vit-large-patch14', dinov3_weights=None, dinov3_repo=None,
    lora_rank=8, lora_alpha=8.0, head_hidden=256, lse_r=5.0,
    lambda_neg=0.5, lambda_gap_max=0.5, gap_warmup_steps=315,
    lr_lora=1e-4, lr_head=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8,
    warmup_steps=500, grad_clip=None, precision='fp16', gradient_checkpointing=True, max_nonfinite=50,
    # ---- session ----
    num_workers=2, time_budget_hours=10.8, save_every_steps=250, log_every_steps=10, smoke=False,
)
CAPD_BFREE_VOLATILE = {'input_root', 'bfree_root', 'gsd_output_dir', 'resume_from', 'output_root', 'num_workers',
                       'time_budget_hours', 'save_every_steps', 'log_every_steps', 'audit_workers',
                       'dinov3_weights', 'dinov3_repo'}
# Keys GSD-P2 hashes into its audit/split; must match for the GSD audit to be reused.
GSD_P2_DATA_KEYS = ('train_branch', 'real_folder', 'check_list_md5', 'image_side', 'mask_diff_threshold',
                    'min_mask_fraction', 'max_mask_fraction', 'seed', 'val_ids', 've_ids', 'max_virtual_epochs',
                    'pairs_per_batch')


def capd_bfree_smoke(cfg):
    return dict(cfg, smoke=True, val_ids=8, ve_ids=8, max_virtual_epochs=2, audit_valid_ids=False,
                save_every_steps=2, log_every_steps=1, warmup_steps=4, gap_warmup_steps=4)


def validate_capd_bfree(cfg):
    unknown = set(cfg) - set(CAPD_BFREE_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    backbone_spec(cfg)
    if cfg['training_protocol'] != CAPD_BFREE_PROTOCOL or cfg['task'] != 'bfree_train':
        raise ValueError('training_protocol/task are fixed')
    if cfg['train_branch'] not in BFREE_ALL_BRANCHES:
        raise ValueError(f'train_branch must be one of {BFREE_ALL_BRANCHES}')
    if cfg['val_ids'] % cfg['pairs_per_batch'] or cfg['ve_ids'] % cfg['pairs_per_batch']:
        raise ValueError('val_ids and ve_ids must be multiples of pairs_per_batch')
    if cfg['precision'] not in ('fp16', 'fp32'):
        raise ValueError('T4: precision fp16 or fp32')
    gsd = dict(BFREE_TRAIN_DEFAULTS)
    for k in GSD_P2_DATA_KEYS + ('jpeg_probability', 'jpeg_train_quality', 'jpeg_subsampling', 'blur_probability',
                                 'blur_sigma', 'validation_processing_seed', 'patience', 'min_delta',
                                 'input_mode', 'crop_max_side', 'threshold'):
        if not cfg['smoke'] and cfg[k] != gsd[k]:
            raise ValueError(f'{k}={cfg[k]!r} differs from GSD-P2 ({gsd[k]!r}); TN2 must keep the GSD-P2 protocol')


# ----------------------------------------------------------------------------- data
class CapdPairImages(Dataset):
    """Rows interleave real/fake of one COCO ID (bfree_data.pair_rows). mode: train | native | processed."""

    def __init__(self, rows, cfg, seed, mode, virtual_epoch=0):
        if mode not in ('train', 'native', 'processed'):
            raise ValueError(mode)
        self.rows, self.cfg, self.seed, self.mode, self.ve = rows, cfg, seed, mode, virtual_epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        params = None if self.mode == 'native' else pair_processing_parameters(
            row['coco_id'], self.cfg, self.seed, train=self.mode == 'train', virtual_epoch=self.ve)
        image = load_processed_crop(row['path'], self.cfg, params)      # GSD-P2: decode -> blur -> JPEG -> crop560
        x, geom = image_to_tensor(image, self.cfg)                      # CAPD: pad to patch multiple + normalize
        meta = dict(coco_id=row['coco_id'], **geom, **(params or {}))
        return x, int(row['label']), i, meta


def capd_pair_loader(rows, batches, cfg, seed, mode, virtual_epoch=0):
    return DataLoader(CapdPairImages(rows, cfg, seed, mode, virtual_epoch), batch_sampler=batches,
                      num_workers=cfg['num_workers'], collate_fn=capd_collate, pin_memory=True)


def reuse_gsd_audit(cfg, folder):
    """Copy GSD-P2's audit files so prepare_bfree_data() reuses them (it re-checks list MD5, branch,
    mask rule and the ID set before reuse; otherwise it audits from scratch)."""
    src = Path(cfg['gsd_output_dir']) if cfg['gsd_output_dir'] else None
    if src is None or not (src / 'audit_train.csv').is_file():
        log('[audit] GSD-P2 output not attached -> full audit (~40 min)')
        return None
    if (folder / 'audit_train.csv').is_file():
        return str(src)
    for name in ('audit_train.csv', 'audit_meta.json', 'audit_valid.csv'):
        if (src / name).is_file():
            shutil.copyfile(src / name, folder / name)
    log(f'[audit] copied GSD-P2 audit files from {src}')
    return str(src)


def check_gsd_split_reproducible(cfg):
    """Rebuild the FULL GSD-P2 split from GSD's own audit with GSD's own make_bfree_splits and compare the
    order hashes with GSD's split_summary.json. Runs in seconds, also in smoke mode."""
    src = Path(cfg['gsd_output_dir']) if cfg['gsd_output_dir'] else None
    if src is None or not (src / 'audit_train.csv').is_file() or not (src / 'split_summary.json').is_file():
        return None
    audit = restore_audit_types(pd.read_csv(src / 'audit_train.csv', dtype={'coco_id': str, 'exclusion_reason': str},
                                            keep_default_na=False))
    full = dict(cfg, **{k: BFREE_TRAIN_DEFAULTS[k] for k in GSD_P2_DATA_KEYS})
    _, _, mine = make_bfree_splits(audit, full, full['seed'])
    gsd = json.loads((src / 'split_summary.json').read_text(encoding='utf-8'))
    same = {k: mine[k] == gsd.get(k) for k in ('validation_order_hash', 'train_order_hash', 'eligible', 'virtual_epochs')}
    log(f'[split] full GSD-P2 split reproduced from its audit: {same}')
    if not all(same.values()):
        raise ValueError(f'Cannot reproduce the GSD-P2 split: {same}')
    return same


def compare_with_gsd_split(split_summary, gsd_dir):
    if not gsd_dir or not (Path(gsd_dir) / 'split_summary.json').is_file():
        return None
    gsd = json.loads((Path(gsd_dir) / 'split_summary.json').read_text(encoding='utf-8'))
    same = {k: gsd.get(k) == split_summary.get(k) for k in ('validation_order_hash', 'train_order_hash', 'eligible',
                                                              'virtual_epochs', 'ids_per_virtual_epoch')}
    log(f'[split] identical to GSD-P2: {same}')
    if not all(same.values()):
        raise ValueError(f'Split differs from GSD-P2: {same}')
    return same


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def capd_eval_pairs(model, frame, cfg, seed, device, mode):
    model.eval()
    rows = pair_rows(frame)
    batches = pair_batches(len(frame), cfg['pairs_per_batch'])
    z = np.full(len(rows), np.nan); zi = np.full(len(rows), np.nan)
    loss_sum, nonfinite, tick = 0.0, 0, time.monotonic()
    for bi, (images, labels, indices, metas) in enumerate(capd_pair_loader(rows, batches, cfg, seed, mode)):
        check_pair_batch(rows, indices.tolist())
        for x, y, ix, meta in zip(images, labels, indices.tolist(), metas):
            with amp(cfg, device):
                s, _, hw = model(x[None].to(device))
            if not torch.isfinite(s).all():
                nonfinite += 1
                s = torch.nan_to_num(s, nan=0.0, posinf=30.0, neginf=-30.0)
            zz = lse_pool(s, cfg['lse_r'])[0]
            z[ix] = float(zz)
            zi[ix] = score_details(s[0], hw, cfg['lse_r'], meta['pad_right'], meta['pad_bottom'])['z_inner']
            loss_sum += float(F.binary_cross_entropy_with_logits(zz, y.float().to(device)))
        if bi % 100 == 0:
            log(f'[val:{mode}] batch {bi + 1}/{len(batches)}')
    if np.isnan(z).any():
        raise RuntimeError('Missing validation predictions')
    return z, zi, loss_sum / len(rows), nonfinite, (time.monotonic() - tick) / len(rows)


def sigmoid_np(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


# ----------------------------------------------------------------------------- train
def capd_bfree_main(config):
    cfg = dict(CAPD_BFREE_DEFAULTS, **(config or {}))
    if cfg['smoke']:
        cfg = capd_bfree_smoke(cfg)
        cfg['output_root'] = cfg['output_root'].rstrip('/') + '_smoke'
    validate_capd_bfree(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('Needs a Kaggle GPU')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = cfg['seed']
    root_out = Path(cfg['output_root']) / cfg['run_name']
    folder = root_out / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    log(f"[{cfg['run_name']}] CAPD-{cfg['backbone']} on B-Free {cfg['train_branch']} vs {cfg['real_folder']} "
        f"| GSD-P2 data protocol | smoke={cfg['smoke']}")

    # ---- resume: copy previous session output (only this run's folder)
    if cfg['resume_from'] and not (folder / 'last_state.pt').exists():
        src = Path(cfg['resume_from'])
        cand = [src / cfg['run_name'] / f'seed{seed}', src / f'seed{seed}', src]
        src = next((c for c in cand if (c / 'last_state.pt').is_file()), None)
        if src is None:
            raise FileNotFoundError(f"resume_from has no last_state.pt for {cfg['run_name']}: {cand}")
        log(f'[resume] copying previous output from {src}')
        shutil.copytree(src, folder, dirs_exist_ok=True)

    # ---- data: GSD-P2 functions (audit reuse -> identical split)
    check_gsd_split_reproducible(cfg)
    gsd_dir = reuse_gsd_audit(cfg, folder)
    t0 = time.monotonic()
    audit, meta = prepare_bfree_data(cfg, folder)
    val, pool, split_summary = make_bfree_splits(audit, cfg, seed)
    atomic_csv(folder / 'validation_ids.csv', val[['coco_id', 'mask_fraction', 'area_group']])
    atomic_csv(folder / 'train_order.csv', pool[['coco_id', 'virtual_epoch', 'mask_fraction', 'area_group']])
    write_json(folder / 'split_summary.json', split_summary)
    log(f'[split] {json.dumps(split_summary)} ({time.monotonic() - t0:.0f}s)')
    if not cfg['smoke']:
        split_check = compare_with_gsd_split(split_summary, gsd_dir)
        write_json(folder / 'split_vs_gsd.json', dict(gsd_dir=gsd_dir, identical=split_check))
    stable = {k: v for k, v in cfg.items() if k not in CAPD_BFREE_VOLATILE}
    signature = sha(dict(config=stable, val=split_summary['validation_order_hash'],
                         train=split_summary['train_order_hash'], lists=[meta['train_list_md5'], meta['valid_list_md5']]))
    n_ve = split_summary['virtual_epochs']
    write_json(root_out / 'config.json', cfg)

    # ---- model / optimizer (identical to TN1)
    seed_all(seed)
    model = CAPD(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(folder / 'architecture.json', dict(backbone=cfg['backbone'], patch=backbone_spec(cfg)['patch'],
               trainable_parameters=n_train, total_parameters=sum(p.numel() for p in model.parameters()),
               lora_rank=cfg['lora_rank'], lora_targets=list(model.backbone.lora_names),
               head_hidden=cfg['head_hidden'], lse_r=cfg['lse_r'], signature=signature,
               geometry='B-Free 512, crop<=560, replicate pad to patch multiple (dinov3: none; clip: 518)'))
    log(f'[model] trainable params: {n_train:,}')
    params_lora, params_head = model.lora_parameters(), model.head_parameters()
    optimizer = torch.optim.AdamW([dict(params=params_lora, lr=cfg['lr_lora']), dict(params=params_head, lr=cfg['lr_head'])],
                                  betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    st = dict(ve=0, phase='train', step=0, global_step=0, images_seen=0, best=float('inf'), best_ve=None,
              history=[], nonfinite=0, status='running', virtual_epochs=n_ve,
              run=dict(loss=0., l_img=0., l_neg=0., l_gap=0., images=0, skipped=0))
    last = folder / 'last_state.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['signature'] != signature:
            raise ValueError('last_state.pt belongs to a different protocol/config; use a new output_root')
        model.load_state_dict(ck['model'], strict=False)
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']); stopper.state.update(ck['stopper'])
        st = ck['st']
        log(f"[resume] VE{st['ve'] + 1} step {st['step']} phase={st['phase']} status={st['status']}")
        del ck
    if st['status'] == 'complete':
        log('[done] run already complete')
        return 'complete'

    def checkpoint_state(status=None):
        if status:
            st['status'] = status
        save_atomic(last, dict(model=state_of(model), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                               stopper=dict(stopper.state), st=st, signature=signature))
        write_json(folder / 'status.json', dict(status=st['status'], ve=st['ve'] + 1, phase=st['phase'],
                   step=st['step'], global_step=st['global_step'], images_seen=st['images_seen'],
                   hours=elapsed_hours(), best_ve=st['best_ve'], early_stopping=dict(stopper.state)))

    trainables = params_lora + params_head
    while st['ve'] < n_ve and st['status'] != 'complete':
        ve = st['ve']
        frame = pool[pool.virtual_epoch == ve].reset_index(drop=True)
        if st['phase'] == 'train':
            rows = pair_rows(frame)
            batches = pair_batches(len(frame), cfg['pairs_per_batch'])
            log(f"[train] VE{ve + 1}/{n_ve}: {len(frame):,} IDs = {len(rows):,} images, {len(batches):,} steps, "
                f"from step {st['step']}")
            model.train()
            step_rows, tick = [], time.monotonic()
            for images, labels, indices, metas in capd_pair_loader(rows, batches[st['step']:], cfg, seed, 'train', ve):
                if elapsed_hours() > cfg['time_budget_hours']:
                    append_rows(folder / 'train_steps.csv', step_rows)
                    checkpoint_state('paused_time_budget')
                    log('[time budget] state saved; Save Version, attach this output, set RESUME_FROM, run again')
                    return 'paused_time_budget'
                check_pair_batch(rows, indices.tolist())
                lam_gap = lambda_gap(cfg, st['global_step'])
                n = len(images); n_real = int((labels == 0).sum())
                optimizer.zero_grad(set_to_none=True)
                parts = dict(l_img=0., l_neg=0., l_gap=0.)
                did_backward = False
                for x, y in zip(images, labels):
                    yf = y.float().to(device)[None]
                    with amp(cfg, device):
                        s, g, hw = model(x[None].to(device))
                    z = lse_pool(s, cfg['lse_r'])
                    l_img = F.binary_cross_entropy_with_logits(z, yf)
                    l_gap = F.binary_cross_entropy_with_logits(g, yf)
                    l_neg = F.softplus(s).mean() if int(y) == 0 else s.sum() * 0.0
                    loss = (l_img + lam_gap * l_gap) / n
                    if n_real:
                        loss = loss + cfg['lambda_neg'] * l_neg / n_real
                    if not torch.isfinite(loss):
                        st['nonfinite'] += 1
                        if st['nonfinite'] > cfg['max_nonfinite']:
                            checkpoint_state('failed_nonfinite')
                            raise FloatingPointError('Too many nonfinite losses: rerun with precision=fp32')
                        continue
                    scaler.scale(loss).backward()
                    did_backward = True
                    parts['l_img'] += float(l_img.detach()) / n
                    parts['l_gap'] += float(l_gap.detach()) / n
                    parts['l_neg'] += float(l_neg.detach()) / max(n_real, 1)
                st['step'] += 1; st['global_step'] += 1; st['images_seen'] += n
                if not did_backward:
                    continue
                scaler.unscale_(optimizer)
                gnorm = float(torch.nn.utils.clip_grad_norm_(trainables, cfg['grad_clip'] or float('inf')))
                old = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); scheduler.step()
                skipped = int(scaler.get_scale() < old)
                r = st['run']
                total = parts['l_img'] + lam_gap * parts['l_gap'] + cfg['lambda_neg'] * parts['l_neg']
                r['loss'] += total * n; r['l_img'] += parts['l_img'] * n; r['l_neg'] += parts['l_neg'] * n
                r['l_gap'] += parts['l_gap'] * n; r['images'] += n; r['skipped'] += skipped
                step_rows.append(dict(ve=ve + 1, step=st['step'], global_step=st['global_step'], loss=total,
                                      l_img=parts['l_img'], l_neg=parts['l_neg'], l_gap=parts['l_gap'],
                                      lambda_gap=lam_gap, lr_lora=optimizer.param_groups[0]['lr'],
                                      lr_head=optimizer.param_groups[1]['lr'], grad_norm=gnorm,
                                      amp_scale=scaler.get_scale(), amp_skipped=skipped, nonfinite_total=st['nonfinite'],
                                      sec_per_img=(time.monotonic() - tick) / max(1, r['images']),
                                      peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9))
                if st['global_step'] % cfg['log_every_steps'] == 0:
                    eta = (len(batches) - st['step']) * n * (time.monotonic() - tick) / max(1, r['images']) / 3600
                    log(f"[{cfg['run_name']} VE{ve + 1}] step {st['step']}/{len(batches)} "
                        f"loss={r['loss'] / r['images']:.4f} img={r['l_img'] / r['images']:.4f} "
                        f"neg={r['l_neg'] / r['images']:.4f} gap={r['l_gap'] / r['images']:.4f} "
                        f"lam_gap={lam_gap:.3f} gnorm={gnorm:.2f} vram={torch.cuda.max_memory_allocated() / 1e9:.1f}GB "
                        f"ETA {eta:.2f}h")
                    append_rows(folder / 'train_steps.csv', step_rows); step_rows = []
                if st['global_step'] % cfg['save_every_steps'] == 0 and st['step'] < len(batches):
                    checkpoint_state()
            append_rows(folder / 'train_steps.csv', step_rows)
            st.update(phase='validate', step=0)
            checkpoint_state()

        # ---- validation (GSD-P2 monitor) on the 2,000 fixed IDs
        summaries, vf = {}, val[['coco_id', 'mask_fraction', 'area_group']].copy()
        for mode in ('native', 'processed'):
            z, zi, loss, nonf, spi = capd_eval_pairs(model, val, cfg, seed, device, mode)
            p, pi = sigmoid_np(z), sigmoid_np(zi)
            y = np.tile([0, 1], len(val))
            summaries[mode] = dict(loss=loss, nonfinite=nonf, sec_per_img=spi, **metric(y, p, cfg['threshold']),
                                   **extra_metrics(y, p), by_area=area_metrics(val, p, cfg['threshold']),
                                   inner=dict(auc=float(roc_auc_score(y, pi)),
                                              balanced_accuracy=metric(y, pi, cfg['threshold'])['balanced_accuracy'],
                                              fake_recall=float((pi[1::2] >= cfg['threshold']).mean()),
                                              real_recall=float((pi[0::2] < cfg['threshold']).mean())))
            vf[f'z_real_{mode}'], vf[f'z_fake_{mode}'] = z[0::2], z[1::2]
            vf[f'z_inner_real_{mode}'], vf[f'z_inner_fake_{mode}'] = zi[0::2], zi[1::2]
        vloss = 0.5 * summaries['native']['loss'] + 0.5 * summaries['processed']['loss']
        stop = stopper.update(vloss)
        r = st['run']
        entry = dict(virtual_epoch=ve + 1, images_seen=st['images_seen'], val_loss=float(vloss),
                     train_loss=r['loss'] / max(1, r['images']), train_l_img=r['l_img'] / max(1, r['images']),
                     train_l_neg=r['l_neg'] / max(1, r['images']), train_l_gap=r['l_gap'] / max(1, r['images']),
                     amp_skipped=r['skipped'], val_native=summaries['native'], val_processed=summaries['processed'],
                     hours=elapsed_hours(), early_stopping=dict(stopper.state))
        st['history'].append(entry)
        dump(folder / 'history.json', st['history'])
        atomic_csv(folder / f'validation_predictions_ve{ve + 1}.csv', vf)
        pd.DataFrame([dict(ve=e['virtual_epoch'], view=v, monitor_loss=e['val_loss'],
                           **{k: x for k, x in e['val_' + v].items() if not isinstance(x, dict)},
                           **{f'area_{g}_fake_recall': e['val_' + v]['by_area'].get(g, {}).get('fake_recall')
                              for g in ('small', 'medium', 'large')},
                           **{f'inner_{k}': x for k, x in e['val_' + v]['inner'].items()})
                      for e in st['history'] for v in ('native', 'processed')]).to_csv(
            folder / 'validation_metrics.csv', index=False)
        n_, p_ = summaries['native'], summaries['processed']
        log(f"[{cfg['run_name']}] VE{ve + 1}: val_loss={vloss:.5f} native AUC={n_['auc']:.4f} BA={n_['balanced_accuracy']:.4f} "
            f"fake_rec={n_['fake_recall']:.4f} (inner BA={n_['inner']['balanced_accuracy']:.4f}) | processed AUC={p_['auc']:.4f} "
            f"BA={p_['balanced_accuracy']:.4f} | area fake_rec processed "
            + ' '.join(f"{g}={p_['by_area'].get(g, {}).get('fake_recall', float('nan')):.3f}" for g in ('small', 'medium', 'large'))
            + f' stop={stop}')
        payload = dict(model=state_of(model), epoch=ve + 1, virtual_epoch=ve + 1, val_loss=float(vloss),
                       validation=summaries, signature=signature, config=cfg, images_seen=st['images_seen'],
                       protocol=CAPD_BFREE_PROTOCOL)
        save_atomic(folder / f've_{ve + 1:03d}.pt', payload)
        if vloss < st['best']:
            st['best'], st['best_ve'] = float(vloss), ve + 1
            shutil.copyfile(folder / f've_{ve + 1:03d}.pt', folder / 'best.pt')
        del payload
        st.update(ve=ve + 1, phase='train', step=0,
                  run=dict(loss=0., l_img=0., l_neg=0., l_gap=0., images=0, skipped=0))
        if stop or st['ve'] >= n_ve:
            st['status'] = 'complete'
            st['stop_reason'] = 'early_stopping' if stop else 'all_virtual_epochs_done'
        checkpoint_state()
    write_json(folder / 'selected_checkpoint.json', dict(best_ve=st['best_ve'], val_loss=st['best'],
               criterion='0.5 CE native + 0.5 CE processed (GSD-P2)', signature=signature,
               stop_reason=st.get('stop_reason')))
    write_json(folder / 'completion.json', dict(status='complete', hours=elapsed_hours(), best_ve=st['best_ve']))
    log(f"[done] best VE{st['best_ve']} val_loss={st['best']:.5f} ({st.get('stop_reason')})")
    return 'complete'


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1].endswith('.json') \
        and json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('task') == 'bfree_train':
    _cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        log('[exit]', capd_bfree_main(_cfg))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

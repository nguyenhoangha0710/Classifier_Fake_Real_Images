# Kiểm tra checkpoint CAPD TN1 (chạy trên Kaggle, CPU là đủ).
# Add Input: dataset checkpoint (vd hadinho0710/checkpoin-capd-train-tiny) -> dán toàn bộ vào 1 cell -> Run.
# Copy toàn bộ output gửi lại để phân tích.
import json, math
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path('/kaggle/input')
EXPECT = {'r1_clip': dict(lora_layers=144, patch=14), 'r2_dinov3': dict(lora_layers=96, patch=16)}
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)


def mb(p): return f'{p.stat().st_size / 1e6:.1f}MB'


def jload(p):
    try: return json.loads(p.read_text())
    except Exception as e: return f'<{e}>'


def check_state(path, tag):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model']
    lora = [k for k in sd if 'lora_' in k]
    other = [k for k in sd if 'lora_' not in k]
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    n_lora = sum(sd[k].numel() for k in lora); n_other = sum(sd[k].numel() for k in other)
    b_norm = [float(sd[k].float().norm()) for k in lora if k.endswith('lora_B')]
    print(f'  [{tag}] {path.name} {mb(path)} | lora tensors={len(lora)} ({len(lora)//2} layers, {n_lora/1e6:.2f}M) '
          f'| head/gap tensors={len(other)} ({n_other/1e6:.3f}M) | nonfinite={len(bad)} '
          f'| lora_B norm mean={np.mean(b_norm) if b_norm else float("nan"):.4f} min={min(b_norm) if b_norm else float("nan"):.4f}')
    if bad: print('   !! NONFINITE:', bad[:5])
    keys = [k for k in ('epoch', 'val_loss', 'signature') if k in ck]
    for k in keys:
        v = ck[k]; print(f'     {k} = {v if not isinstance(v, str) or len(v) < 80 else v[:16] + "..."}')
    if 'st' in ck:
        st = ck['st']
        print(f'     st: epoch={st["epoch"]} step_in_epoch={st["step_in_epoch"]} global={st["global_step"]} '
              f'status={st["status"]} best={st["best"]} best_epoch={st["best_epoch"]} nonfinite={st["nonfinite"]}')
        print(f'     stopper={ck.get("stopper")}')
    if 'validation' in ck:
        for view, s in ck['validation'].items():
            print(f'     val[{view}] ' + ' '.join(f'{k}={v:.5f}' for k, v in s.items() if isinstance(v, (int, float))))
    cfg = ck.get('config')
    if cfg:
        keep = ['backbone', 'lora_r', 'lora_alpha', 'lse_r', 'lambda_neg', 'lambda_gap_max', 'gap_warmup_steps',
                'lr_lora', 'lr_head', 'batch_size', 'microbatch', 'precision', 'max_crop', 'pad_mode', 'smoke',
                'epochs', 'patience', 'min_delta', 'threshold', 'time_budget_hours']
        print('     config:', {k: cfg[k] for k in keep if k in cfg})
    return sd, ck


runs = sorted({p.parent for p in ROOT.rglob('*.pt') if p.parent.name == 'seed42' or p.name in ('best.pt', 'last_state.pt')})
print('Run folders:', *runs, sep='\n  ')
for folder in runs:
    name = next((n for n in EXPECT if n in str(folder)), folder.parent.name)
    print('\n' + '=' * 100 + f'\n{name}: {folder}')
    for p in sorted(folder.iterdir()):
        print(f'  {p.name:45s} {mb(p) if p.is_file() else "<dir> " + str(len(list(p.iterdir()))) + " files"}')
    for j in ('status.json', 'completion.json', 'selected_checkpoint.json'):
        if (folder / j).exists(): print(f'  {j}: {jload(folder / j)}')

    sds = {}
    for p in sorted(folder.glob('epoch_*.pt')) + [folder / 'best.pt', folder / 'last_state.pt']:
        if p.exists():
            try: sds[p.name], _ = check_state(p, name)
            except Exception as e: print(f'  !! cannot load {p.name}: {e}')
    if name in EXPECT and 'best.pt' in sds:
        n = len([k for k in sds['best.pt'] if 'lora_' in k]) // 2
        print(f'  LoRA layers {n} vs expected {EXPECT[name]["lora_layers"]}:', 'OK' if n == EXPECT[name]['lora_layers'] else '!! MISMATCH')
        same = [e for e in sds if e.startswith('epoch_') and all(torch.equal(sds[e][k], sds['best.pt'][k]) for k in sds['best.pt'])]
        print('  best.pt identical to:', same)

    if (folder / 'validation_metrics.csv').exists():
        v = pd.read_csv(folder / 'validation_metrics.csv')
        cols = [c for c in ['epoch', 'view', 'monitor_loss', 'loss', 'auc', 'ap', 'balanced_accuracy', 'fake_recall', 'real_recall',
                            'best_ba_threshold', 'best_ba', 'tpr_at_fpr1', 'tpr_at_fpr5', 'ece', 'nonfinite'] if c in v]
        print('\n  validation_metrics.csv'); print(v[cols].round(5).to_string(index=False))
    if (folder / 'history.json').exists():
        for e in jload(folder / 'history.json'):
            print(f'  history e{e["epoch"]}: train_loss={e["train_loss"]:.4f} img={e["train_l_img"]:.4f} '
                  f'neg={e["train_l_neg"]:.4f} gap={e["train_l_gap"]:.4f} val_loss={e["val_loss"]:.5f} '
                  f'amp_skipped={e["amp_skipped_steps"]} images={e["train_images"]} hours={e["hours"]:.2f} es={e["early_stopping"]}')

    for p in sorted(folder.glob('validation_predictions_epoch*.csv')):
        f = pd.read_csv(p)
        rows = []
        for gen, g in f.groupby('generator'):
            r = dict(generator=gen, n=len(g))
            for view in ('native', 'processed'):
                if f'prob_fake_{view}' in g:
                    r[f'acc_{view}'] = float(((g[f'prob_fake_{view}'] > .5).astype(int) == g.label).mean())
            rows.append(r)
        print(f'\n  {p.name} per-generator accuracy (thr 0.5)'); print(pd.DataFrame(rows).round(4).to_string(index=False))
        for view in ('native', 'processed'):
            if f'z_{view}' in f:
                zr, zf = f.loc[f.label == 0, f'z_{view}'], f.loc[f.label == 1, f'z_{view}']
                print(f'   z_{view}: real mean={zr.mean():.2f} p99={zr.quantile(.99):.2f} | fake mean={zf.mean():.2f} p1={zf.quantile(.01):.2f}')

    ts = folder / 'train_steps.csv'
    if ts.exists():
        t = pd.read_csv(ts)
        g = t.groupby('epoch').agg(steps=('step', 'count'), l_img=('l_img', 'mean'), l_neg=('l_neg', 'mean'),
                                   l_gap=('l_gap', 'mean'), gnorm_med=('grad_norm', 'median'), gnorm_max=('grad_norm', 'max'),
                                   amp_skipped=('amp_skipped', 'sum'), nonfinite=('nonfinite_total', 'max'),
                                   sec_img=('sec_per_img', 'last'), vram=('peak_vram_gb', 'max'))
        print('\n  train_steps.csv per epoch'); print(g.round(5).to_string())
        t['blk'] = (t.global_step - 1) // 500
        print('  l_img per 500 global steps:', t.groupby('blk').l_img.mean().round(4).tolist())
        dup = t.duplicated(['global_step']).sum()
        print(f'  duplicated global_step rows (resume overlap): {dup}; gaps: {int(t.global_step.max() - t.global_step.nunique())}')

    te = folder / 'tiny_eval' / 'metrics.csv'
    if te.exists():
        m = pd.read_csv(te)
        cols = [c for c in ['case', 'generator', 'auc', 'balanced_accuracy', 'fake_recall', 'real_recall', 'best_ba', 'tpr_at_fpr1'] if c in m]
        print('\n  tiny_eval/metrics.csv'); print(m[cols].round(4).to_string(index=False))

for log in sorted(ROOT.rglob('log_*.txt')):
    lines = log.read_text(errors='ignore').splitlines()
    key = [l for l in lines if any(s in l for s in ('epoch ', '[early', '[exit]', '[time budget]', '[resume]', 'Error', 'Traceback', 'nonfinite', '[eval_tiny]'))
           and 'step ' not in l]
    print('\n' + '=' * 100 + f'\n{log} ({len(lines)} lines) key lines:'); print('\n'.join(key[-40:]))
    print('  last 5 lines:'); print('\n'.join(lines[-5:]))

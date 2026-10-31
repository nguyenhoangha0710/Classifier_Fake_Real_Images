"""Independent read-only audit of CAPD train histories and native predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'source' / 'capd_tn1'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(frame, threshold=.5):
    y = frame.label.to_numpy(dtype=int)
    p = frame.prob_fake.to_numpy(dtype=float)
    pred = p >= threshold
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    order = np.argsort(-p, kind='stable')
    cum_tp = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    ap = float(np.sum((cum_tp[ends] / (ends + 1)) * np.diff(np.r_[0, cum_tp[ends]])) / pos)
    rr, fr = tn / neg, tp / pos
    return dict(n=len(frame), accuracy=(tn + tp) / len(frame), balanced_accuracy=(rr + fr) / 2,
                real_recall=rr, fake_recall=fr, auc=auc, ap=ap, tn=tn, fp=fp, fn=fn, tp=tp)


def main():
    pred_path = ROOT / 'r2_dinov3/seed42/tiny_eval/predictions_native.csv'
    metric_path = ROOT / 'r2_dinov3/seed42/tiny_eval/metrics.csv'
    pred = pd.read_csv(pred_path)
    reported = pd.read_csv(metric_path)
    recomputed = []
    max_delta = 0.0
    for generator, group in [('pooled', pred), *pred.groupby('generator')]:
        row = dict(generator=generator, **metrics(group))
        old = reported[reported.generator == generator].iloc[0]
        for key in ('n', 'auc', 'ap', 'balanced_accuracy', 'accuracy', 'fake_recall', 'real_recall',
                    'tn', 'fp', 'fn', 'tp'):
            max_delta = max(max_delta, abs(float(row[key]) - float(old[key])))
        recomputed.append(row)
    dist = {}
    for (generator, label), group in pred.groupby(['generator', 'label']):
        p = group.prob_fake
        dist[f'{generator}:{label}'] = dict(n=len(group), min=float(p.min()), p01=float(p.quantile(.01)),
            p05=float(p.quantile(.05)), median=float(p.median()), p95=float(p.quantile(.95)),
            p99=float(p.quantile(.99)), max=float(p.max()), mean=float(p.mean()))
    errors = pred[((pred.prob_fake >= .5).astype(int) != pred.label)].copy()
    errors['error_type'] = np.where(errors.label == 0, 'false_positive', 'false_negative')
    runs = {}
    for name in ('r1_clip', 'r2_dinov3'):
        history = read_json(ROOT / name / 'seed42/history.json')
        status = read_json(ROOT / name / 'seed42/status.json')
        architecture = read_json(ROOT / name / 'seed42/architecture.json')
        steps = pd.read_csv(ROOT / name / 'seed42/train_steps.csv')
        runs[name] = dict(status=status, epochs_complete=len(history), history=history,
            architecture=architecture, steps=len(steps), peak_vram_gb=float(steps.peak_vram_gb.max()),
            max_grad_norm=float(steps.grad_norm.max()), final_lr_lora=float(steps.lr_lora.iloc[-1]),
            final_lr_head=float(steps.lr_head.iloc[-1]))
    checks = dict(rows=len(pred), generators=sorted(pred.generator.unique().tolist()),
        balanced=bool(pred.groupby(['generator', 'label']).size().eq(500).all()),
        finite=bool(np.isfinite(pred[['z', 'prob_fake']]).all().all()),
        probability_from_logit_max_error=float(np.max(np.abs(pred.prob_fake - 1 / (1 + np.exp(-pred.z))))),
        checkpoint_epochs=sorted(pred.checkpoint_epoch.unique().tolist()),
        max_reported_metric_delta=max_delta)
    result = dict(checks=checks, metrics=recomputed, distributions=dist,
        errors=errors[['sample_id', 'generator', 'label', 'prob_fake', 'z', 'error_type']].to_dict('records'),
        runs=runs)
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Recompute BFree threshold metrics from version-1 per-image predictions."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source/capd_eval_bfree/r2_dinov3/seed42/bfree_eval'
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

with (SOURCE / 'predictions_bfree.csv').open(encoding='utf8', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 72128
assert len({r['sample_id'] for r in rows}) == len(rows)
branches = sorted({r['branch'] for r in rows} - {'real'})
assert len(branches) == 6
result = {}

for pooling, column in [('lse_full', 'prob_fake'), ('lse_window', 'prob_fake_window')]:
    scores = {b: np.sort(np.array([float(r[column]) for r in rows if r['branch'] == b]))
              for b in ['real'] + branches}
    assert all(len(s) == 10304 and np.isfinite(s).all() for s in scores.values())
    assert all(int(r['label']) == (r['branch'] != 'real') for r in rows)

    def measure(threshold):
        tn = int(np.searchsorted(scores['real'], threshold, side='left'))
        fp = len(scores['real']) - tn
        real_recall = tn / (tn + fp)
        detail = {}
        for branch in branches:
            fn = int(np.searchsorted(scores[branch], threshold, side='left'))
            tp = len(scores[branch]) - fn
            fake_recall = tp / (tp + fn)
            detail[branch] = dict(tn=tn, fp=fp, fn=fn, tp=tp, fake_recall=fake_recall,
                                  real_recall=real_recall, balanced_accuracy=(fake_recall+real_recall)/2)
        macro_fake = float(np.mean([r['fake_recall'] for r in detail.values()]))
        return dict(threshold=float(threshold), macro_ba=(macro_fake+real_recall)/2,
                    macro_fake_recall=macro_fake, real_recall=real_recall,
                    fpr=1-real_recall, real_fp=fp, branches=detail)

    # Exhaustive search over all score boundaries, preserving p >= threshold.
    def best_for(fake_scores):
        candidates = np.unique(np.concatenate([scores['real'], fake_scores, [0., 1.]]))
        tnr = np.searchsorted(scores['real'], candidates, side='left') / len(scores['real'])
        tpr = 1 - np.searchsorted(fake_scores, candidates, side='left') / len(fake_scores)
        ba = (tnr+tpr)/2
        indices = np.flatnonzero(ba == ba.max())
        # If tied, prefer the higher threshold (fewer false positives).
        idx = int(indices[-1])
        return dict(threshold=float(candidates[idx]), ba=float(ba[idx]))

    allfake = np.sort(np.concatenate([scores[b] for b in branches]))
    best = best_for(allfake)
    best_macro = measure(best['threshold'])
    best_branch = {}
    for branch in branches:
        opt = best_for(scores[branch])
        best_branch[branch] = dict(threshold=opt['threshold'], **measure(opt['threshold'])['branches'][branch])
    fixed = [measure(t) for t in THRESHOLDS]
    # Check baseline against the published metrics, not against this implementation alone.
    with (SOURCE / 'metrics_bfree.csv').open(encoding='utf8', newline='') as f:
        published = list(csv.DictReader(f))
    for r in published:
        if r['pooling'] != pooling:
            continue
        if r['branch'] == 'macro':
            assert abs(fixed[0]['macro_ba']-float(r['balanced_accuracy'])) < 1e-12
        else:
            got = fixed[0]['branches'][r['branch']]
            for k in ['tn', 'fp', 'fn', 'tp', 'balanced_accuracy', 'fake_recall', 'real_recall']:
                assert abs(got[k]-float(r[k])) < 1e-12, (pooling,r['branch'],k)
    result[pooling] = dict(fixed_thresholds=fixed, best_shared_threshold=best_macro,
                           oracle_per_branch=best_branch)

(ROOT / 'threshold_sweep.json').write_text(json.dumps(result, indent=2), encoding='utf8')

def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str,r)) + ' |' for r in data])
def pct(v):
    return f'{v*100:.2f}'

lines = ['# BFree: thay đổi ngưỡng dự đoán', '',
         'Nguồn: [Kaggle version 1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).', '',
         'Tính lại từ 72.128 dự đoán: 10.304 real dùng chung + 6 nhánh fake, mỗi nhánh 10.304 ảnh. Dự đoán fake nếu p >= ngưỡng. Macro BA là trung bình BA của 6 cặp real/fake cân bằng; không phải accuracy trên tập gộp mất cân bằng. Đây là BFree validation.', '',
         'Đã kiểm tra các confusion count và metrics tại ngưỡng 0,5 trùng bảng gốc cho cả hai pooling. AUC/AP không thay đổi khi chỉ thay ngưỡng.', '']
for pooling, data in result.items():
    lines += ['## ' + pooling, '',
              table(['Ngưỡng','Macro BA (%)','Recall fake (%)','Recall real (%)','FPR (%)','Real báo nhầm'],
                    [[f"{r['threshold']:.3f}",pct(r['macro_ba']),pct(r['macro_fake_recall']),pct(r['real_recall']),pct(r['fpr']),r['real_fp']] for r in data['fixed_thresholds']]), '']
    opt = data['best_shared_threshold']
    lines += [f"Ngưỡng chung tối ưu trên chính validation: {opt['threshold']:.10f}; macro BA {pct(opt['macro_ba'])}%, fake recall {pct(opt['macro_fake_recall'])}%, real recall {pct(opt['real_recall'])}%, FP={opt['real_fp']}.", '',
              table(['Nhánh','Recall fake @0.5 (%)','@0.1 (%)','@0.02 (%)','@0.01 (%)','BA @0.5 (%)','BA @0.1 (%)','BA @0.02 (%)','BA @0.01 (%)'],
                    [[b.replace('SD2.1_','')] + [pct(next(r for r in data['fixed_thresholds'] if r['threshold']==t)['branches'][b][key]) for key in ['fake_recall','balanced_accuracy'] for t in [0.5,0.1,0.02,0.01]] for b in branches]), '',
              'Ngưỡng tối ưu riêng từng nhánh (oracle, chỉ để phân tích):', '',
              table(['Nhánh','Ngưỡng','BA (%)','Recall fake (%)','Recall real (%)'],
                    [[b.replace('SD2.1_',''),f"{r['threshold']:.8f}",pct(r['balanced_accuracy']),pct(r['fake_recall']),pct(r['real_recall'])] for b,r in data['oracle_per_branch'].items()]), '']
lines += ['## Diễn giải', '',
          'Giảm ngưỡng giúp bắt thêm ảnh giả nhưng làm tăng báo nhầm ảnh thật. Ngưỡng dùng chung được tối ưu trực tiếp trên nhãn validation, nên kết quả tối ưu là mức tham khảo có thiên lệch lựa chọn; cần khóa ngưỡng rồi kiểm tra trên tập độc lập. Ngưỡng riêng từng nhánh còn giả định biết trước loại nhánh, không thể dùng như một bộ phát hiện chung khi nhánh chưa biết.', '']
(ROOT / 'THRESHOLD_REPORT.md').write_text('\n'.join(lines), encoding='utf8')
for pool, data in result.items():
    print(pool)
    for r in data['fixed_thresholds'] + [data['best_shared_threshold']]:
        print('t={:.9f} BA={:.4f}% fake={:.4f}% real={:.4f}% FP={}'.format(r['threshold'],100*r['macro_ba'],100*r['macro_fake_recall'],100*r['real_recall'],r['real_fp']))
    print('origBG at best shared:')
    for b,r in data['best_shared_threshold']['branches'].items():
        if 'origBG' in b:
            print(b, 'BA',pct(r['balanced_accuracy']),'recall',pct(r['fake_recall']))
print('Baseline matches original counts and metrics. Wrote THRESHOLD_REPORT.md and threshold_sweep.json.')

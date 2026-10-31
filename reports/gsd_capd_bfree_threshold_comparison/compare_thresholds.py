"""Compare shared decision thresholds on saved GSD/CAPD BFree predictions.

No inference or source edits. Each branch is a balanced real/fake pair.
GSD real scores remain specific to each inference batch context.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
GSD_ROOT = REPORTS / 'gsd_bfree_valid_analysis/source/gsd_bfree_valid_v1'
CAPD_ROOT = REPORTS / 'capd_tiny_bfree_eval_analysis/source/capd_eval_bfree/r2_dinov3/seed42/bfree_eval'
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]


def read(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def pct(value):
    return f'{100 * value:.2f}'


def evaluate(groups, threshold):
    details = {}
    for branch, (real, fake) in groups.items():
        tn = int(np.searchsorted(real, threshold, side='left'))
        fn = int(np.searchsorted(fake, threshold, side='left'))
        fp, tp = len(real) - tn, len(fake) - fn
        tpr, tnr = tp / len(fake), tn / len(real)
        details[branch] = dict(tn=tn, fp=fp, fn=fn, tp=tp, fake_recall=tpr,
                               real_recall=tnr, fpr=1-tnr, balanced_accuracy=(tpr+tnr)/2)
    macro = {key: float(np.mean([r[key] for r in details.values()]))
             for key in ['fake_recall', 'real_recall', 'fpr', 'balanced_accuracy']}
    return dict(threshold=float(threshold), **macro, branches=details,
                mean_fp_per_branch=float(np.mean([r['fp'] for r in details.values()])))


def optimize(groups, fpr_cap=None):
    # Equal branch/class counts make integer correct-count maximization exactly
    # equivalent to macro BA. Preserve complete tied-score groups and p >= t.
    real = np.sort(np.concatenate([v[0] for v in groups.values()]))
    fake = np.sort(np.concatenate([v[1] for v in groups.values()]))
    thresholds = np.unique(np.concatenate([real, fake, [0., 1.]]))
    tn = np.searchsorted(real, thresholds, side='left')
    tp = len(fake) - np.searchsorted(fake, thresholds, side='left')
    if fpr_cap is None:
        objective = tn + tp
        choices = np.flatnonzero(objective == objective.max())
    else:
        eligible = np.flatnonzero((len(real)-tn)/len(real) <= fpr_cap + 1e-15)
        choices = eligible[tp[eligible] == tp[eligible].max()]
    # Higher threshold breaks ties in favor of fewer false positives.
    return evaluate(groups, thresholds[int(choices[-1])])


def main():
    gsd_path, capd_path = GSD_ROOT/'predictions.csv', CAPD_ROOT/'predictions_bfree.csv'
    gsd, capd = read(gsd_path), read(capd_path)
    branches = sorted({r['case'] for r in gsd})
    assert len(branches) == 6
    assert len(gsd) == 123648 and len(capd) == 72128
    assert len({(r['case'], r['coco_id'], r['label']) for r in gsd}) == len(gsd)
    assert len({(r['branch'], r['coco_id']) for r in capd}) == len(capd)
    assert set(branches) == {r['branch'] for r in capd} - {'real'}
    groups = {model: {} for model in ['GSD', 'CAPD full', 'CAPD window']}
    cohort = {}
    real_by_context = {}
    for branch in branches:
        g = [r for r in gsd if r['case'] == branch]
        c = [r for r in capd if r['branch'] in ['real', branch]]
        counts = {}
        for label in [0, 1]:
            gi = {r['coco_id'].zfill(12) for r in g if int(r['label']) == label}
            ci = {r['coco_id'].zfill(12) for r in c if int(r['label']) == label}
            assert gi == ci and len(gi) == 10304, (branch, label)
            counts[str(label)] = len(gi)
        assert all(r['variant'] == (branch if int(r['label']) else 'COCO_real_512') for r in g)
        assert all(int(r['label']) == (r['branch'] != 'real') for r in c)
        cohort[branch] = counts
        real_by_context[branch] = {r['coco_id'].zfill(12): float(r['prob_fake']) for r in g if int(r['label']) == 0}
        for name, frame, column in [('GSD', g, 'prob_fake'), ('CAPD full', c, 'prob_fake'),
                                     ('CAPD window', c, 'prob_fake_window')]:
            arrays = tuple(np.sort(np.array([float(r[column]) for r in frame if int(r['label']) == label])) for label in [0,1])
            assert all(len(a) == 10304 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in arrays)
            groups[name][branch] = arrays

    original = {'GSD': read(GSD_ROOT/'metrics.csv'),
                'CAPD full': [r for r in read(CAPD_ROOT/'metrics_bfree.csv') if r['pooling']=='lse_full' and r['branch']!='macro'],
                'CAPD window': [r for r in read(CAPD_ROOT/'metrics_bfree.csv') if r['pooling']=='lse_window' and r['branch']!='macro']}
    result = {}
    for name, model_groups in groups.items():
        fixed = [evaluate(model_groups, t) for t in THRESHOLDS]
        for row in original[name]:
            got = fixed[0]['branches'][row['branch']]
            for key in ['tn','fp','fn','tp','balanced_accuracy','fake_recall','real_recall']:
                assert abs(got[key]-float(row[key])) < 1e-12, (name, row['branch'], key)
        result[name] = dict(fixed=fixed, best_shared=optimize(model_groups),
                            at_fpr_cap={str(cap): optimize(model_groups, cap) for cap in [0.01,0.05,0.10]},
                            macro_auc=float(np.mean([float(r['auc']) for r in original[name]])),
                            macro_ap=float(np.mean([float(r['ap']) for r in original[name]])))
    # Independent consistency check against the previously computed CAPD sweep.
    previous = json.loads((REPORTS/'capd_tiny_bfree_eval_analysis/threshold_sweep.json').read_text(encoding='utf8'))
    for model, pool in [('CAPD full','lse_full'),('CAPD window','lse_window')]:
        assert abs(result[model]['best_shared']['balanced_accuracy']-previous[pool]['best_shared_threshold']['macro_ba']) < 1e-12
    common_ids = sorted(next(iter(real_by_context.values())))
    real_matrix = np.array([[real_by_context[b][i] for b in branches] for i in common_ids])
    checks = dict(gsd_rows=len(gsd), capd_rows=len(capd), matched_unique_images=72128,
                  matched_ids_by_branch_label=cohort, baseline_confusion_matrices_verified=18,
                  gsd_real_ids_with_context_dependent_class_at_05=int(np.sum(np.any(real_matrix>=.5,axis=1) != np.all(real_matrix>=.5,axis=1))),
                  identity_limit='COCO IDs, variants and labels match; pixel content hashes were not recomputed.',
                  source_sha256={str(p.relative_to(REPORTS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [gsd_path, capd_path]})
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/'comparison.json').write_text(json.dumps(dict(checks=checks, models=result),indent=2),encoding='utf8')

    lines = ['# GSD và CAPD: quét ngưỡng BFree validation', '',
             'GSD: [test-gsd-bfree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-bfree), checkpoint CLIP+GSD All7 augmented epoch 1. CAPD: [eval-capd-test-tiny-bfree v1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree), checkpoint DINOv3 CAPD Tiny All7 epoch 3.', '',
             '## Giao thức và đối chiếu', '',
             '- Không train lại hoặc chạy inference; tính lại quyết định fake nếu probability >= threshold từ CSV nguồn.',
             '- Hai bên trùng đủ 10.304 COCO IDs cho real và từng nhánh fake, đủ 6 nhánh, tổng 72.128 ảnh riêng biệt. Chưa tính lại hash pixel để xác nhận nội dung ảnh byte-identical.',
             '- GSD có 123.648 dòng vì real được suy luận lại trong từng branch/batch context. CAPD có 72.128 dòng và dùng lại cùng real score cho 6 cặp đánh giá. Giữ nguyên các score GSD theo context, không lấy trung bình rồi thay vào.',
             '- BA = balanced accuracy; macro là trung bình 6 nhánh, mỗi nhánh 10.304 real + 10.304 fake. Recall real/FPR của GSD trong bảng là trung bình 6 context; số FP trung bình không phải số ảnh thật riêng biệt.',
             f"- Đối chiếu lại 18 confusion matrices ở ngưỡng 0,5 khớp metrics gốc. Có {checks['gsd_real_ids_with_context_dependent_class_at_05']} real IDs đổi nhãn dự đoán giữa các context GSD ở ngưỡng 0,5.", '',
             '## Quét cùng các ngưỡng', '',
             table(['Ngưỡng','GSD BA (%)','CAPD full BA (%)','CAPD window BA (%)','GSD recall fake (%)','GSD FPR (%)','CAPD full FPR (%)','CAPD window FPR (%)'],
                   [[f'{t:g}']+[pct(result[m]['fixed'][i]['balanced_accuracy']) for m in groups]+
                    [pct(result['GSD']['fixed'][i]['fake_recall'])]+[pct(result[m]['fixed'][i]['fpr']) for m in groups] for i,t in enumerate(THRESHOLDS)]), '',
             '## Ngưỡng 0,5 và ngưỡng chung tối ưu macro BA', '',
             table(['Model','Cách chọn','Ngưỡng','Macro BA (%)','Recall fake (%)','Recall real (%)','FPR (%)','FP trung bình/nhánh'],
                   [[m,kind,f"{r['threshold']:.10g}",pct(r['balanced_accuracy']),pct(r['fake_recall']),pct(r['real_recall']),pct(r['fpr']),f"{r['mean_fp_per_branch']:.2f}"]
                    for m,d in result.items() for kind,r in [('Mặc định',d['fixed'][0]),('Tối ưu trên validation',d['best_shared'])]]), '',
             'Mỗi model dùng một ngưỡng chung cho cả 6 nhánh. Tìm chính xác trên toàn bộ ranh giới score; giữ nguyên nhóm score bằng nhau. Nếu nhiều ngưỡng cùng tối ưu, chọn ngưỡng cao nhất để ưu tiên ít false positive.', '',
             '## Theo nhánh: dùng ngưỡng chung tối ưu riêng của mỗi model', '',
             table(['Nhánh','GSD BA (%)','CAPD full BA (%)','CAPD window BA (%)','GSD recall fake (%)','CAPD full recall fake (%)','CAPD window recall fake (%)'],
                   [[b.replace('SD2.1_','')]+[pct(result[m]['best_shared']['branches'][b][key]) for key in ['balanced_accuracy','fake_recall'] for m in groups] for b in branches]), '',
             '## So sánh ở cùng giới hạn false positive', '',
             'Mỗi model được chọn một ngưỡng chung tối đa hóa macro recall fake dưới giới hạn macro FPR ghi trong bảng. FPR thực tế có thể thấp hơn giới hạn do score rời rạc; đây là ngưỡng tham khảo chọn trực tiếp trên cùng validation.', '',
             table(['FPR tối đa (%)','Model','Ngưỡng','FPR thực tế (%)','Recall fake (%)','Macro BA (%)'],
                   [[f'{100*cap:g}',m,f"{r['threshold']:.10g}",pct(r['fpr']),pct(r['fake_recall']),pct(r['balanced_accuracy'])]
                    for cap in [.01,.05,.10] for m in groups for r in [result[m]['at_fpr_cap'][str(cap)]]]), '',
             '## AUC/AP không phụ thuộc ngưỡng', '',
             table(['Model','Macro AUC (%)','Macro AP (%)'],[[m,pct(d['macro_auc']),pct(d['macro_ap'])] for m,d in result.items()]), '',
             '## Nhận xét từ kết quả', '',
             f"- GSD: ngưỡng 0,5 đạt BA {pct(result['GSD']['fixed'][0]['balanced_accuracy'])}%; giảm về {result['GSD']['best_shared']['threshold']:.9f} đạt {pct(result['GSD']['best_shared']['balanced_accuracy'])}%, tăng {100*(result['GSD']['best_shared']['balanced_accuracy']-result['GSD']['fixed'][0]['balanced_accuracy']):.2f} điểm phần trăm.",
             '- Trong các ngưỡng số tròn đã thử, GSD 0,05 đạt BA 84,79%, gần tối ưu 84,81%; 0,1 đạt 84,31% và FPR thấp hơn (3,55% so với 6,28%).',
             '- Tối ưu riêng theo BA: GSD cao hơn CAPD 1,15 điểm phần trăm, nhưng FPR 7,29% so với 2,70%; vì thế cần xem thêm cùng ngân sách false positive.',
             '- Ở FPR 1%, CAPD có recall fake cao hơn nhẹ (66,82% so với 66,16%). Ở FPR 5% và 10%, GSD cao hơn (74,33%/79,29% so với 71,82%/74,63%).',
             '- Các nhánh origBG vẫn là điểm yếu. Ở ngưỡng BA tối ưu chung, selfconditioned_origBG recall fake GSD 34,17%, CAPD 16,89%; cả hai còn bỏ sót phần lớn ảnh giả của nhánh này.', '',
             '## Giới hạn diễn giải', '',
             '- Ngưỡng tối ưu và các điểm FPR ở đây được chọn bằng nhãn của chính BFree validation; không phải kết quả test độc lập. Muốn triển khai cần khóa ngưỡng sau calibration và đánh giá trên tập khác.',
             '- So sánh là CLIP+GSD với DINOv3+CAPD; khác cả backbone và cách training/pooling, không đủ để quy toàn bộ chênh lệch cho riêng GSD hay CAPD.',
             '- Một seed và không có ước lượng độ bất định trong báo cáo này. Giá trị probability cùng ngưỡng số học không biểu thị mức calibration tương đương; dùng thêm bảng cùng FPR để đối chiếu.', '',
             '## File chạy', '',
             f"[compare_thresholds.py]({(HERE/'compare_thresholds.py').as_posix()}) — đọc hai CSV local, kiểm tra cohort/baseline, xuất báo cáo và [comparison.json]({(HERE/'comparison.json').as_posix()}).",
             '', 'SHA-256 hai CSV nguồn và các kết quả chi tiết từng nhánh/ngưỡng nằm trong comparison.json.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for name, data in result.items():
        print(name,'AUC',pct(data['macro_auc']))
        for kind,r in [('baseline',data['fixed'][0]),('best',data['best_shared'])]:
            print(kind,'threshold',r['threshold'],'BA',pct(r['balanced_accuracy']),'fake',pct(r['fake_recall']),'real',pct(r['real_recall']))
        for cap,r in data['at_fpr_cap'].items():
            print('FPR cap',cap,'actual',pct(r['fpr']),'fake',pct(r['fake_recall']))
    print('Checks:', json.dumps({k:v for k,v in checks.items() if k not in ['source_sha256','matched_ids_by_branch_label']}))


if __name__ == '__main__':
    main()

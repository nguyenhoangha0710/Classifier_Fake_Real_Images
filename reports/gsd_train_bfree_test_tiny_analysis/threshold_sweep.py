"""Post-hoc threshold analysis of saved Tiny predictions; no inference or retraining."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'source/gsd_bfree_samecat_origbg_best_tiny9_v1'
THRESHOLDS=[.5,.3,.2,.1,.05,.03,.02,.01,.005,.001]

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def pct(v):return f'{100*v:.2f}'

def add_class_metrics(value):
    """Derive both-class precision/F1 from the actual confusion matrix."""
    if not isinstance(value,dict):
        if isinstance(value,list):
            for item in value:add_class_metrics(item)
        return
    for item in list(value.values()):add_class_metrics(item)
    if all(k in value for k in ['tn','fp','fn','tp']):
        tn,fp,fn,tp=[value[k] for k in ['tn','fp','fn','tp']]
        value.update(accuracy=(tn+tp)/(tn+fp+fn+tp),
                     fake_precision=tp/(tp+fp) if tp+fp else 0.,
                     real_precision=tn/(tn+fn) if tn+fn else 0.,
                     fake_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                     real_f1=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.)
    if 'branches' in value:
        for key in ['accuracy','fake_precision','real_precision','fake_f1','real_f1']:
            value[key]=float(np.mean([r[key] for r in value['branches'].values()]))

def main():
    spec=importlib.util.spec_from_file_location('helpers',ROOT.parent/'gsd_capd_bfree_threshold_comparison/compare_thresholds.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    path=RUN/'predictions.csv'
    frame=pd.read_csv(path)
    previous=json.loads((ROOT/'analysis.json').read_text(encoding='utf8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==previous['checks']['source_predictions_sha256']
    state=json.loads((RUN/'completion.json').read_text(encoding='utf8'))
    cases=state['planned_cases'];assert len(frame)==63000 and len(cases)==9
    assert not frame[['case','sample_id']].duplicated().any()
    groups={}
    for case in cases:
        f=frame[frame.case==case]
        arrays=tuple(np.sort(f.loc[f.label==label,'prob_fake'].to_numpy()) for label in [0,1])
        assert all(len(a)==3500 and np.isfinite(a).all() for a in arrays)
        groups[case]=arrays
    fixed=[helpers.evaluate(groups,t) for t in THRESHOLDS]
    for source in previous['pooled']:
        row=fixed[0]['branches'][source['case']]
        for k in ['tn','fp','fn','tp','balanced_accuracy','real_recall','fake_recall']:
            assert abs(row[k]-source[k])<1e-12
    best_native=helpers.optimize({'native':groups['native']})
    native_threshold=best_native['threshold']
    best_shared=helpers.optimize(groups)
    case_optima={c:helpers.optimize({c:groups[c]}) for c in cases}
    shared_native=helpers.evaluate(groups,native_threshold)
    native_generator={}
    for generator,f in frame[frame.case=='native'].groupby('generator'):
        g={generator:tuple(np.sort(f.loc[f.label==label,'prob_fake'].to_numpy()) for label in [0,1])}
        native_generator[generator]={str(t):helpers.evaluate(g,t)['branches'][generator]
                                     for t in [.5,.1,.05,.03,native_threshold,best_shared['threshold']]}
    # Verify optimized native BA by an independent direct-count evaluation.
    f=frame[frame.case=='native'];pred=f.prob_fake>=native_threshold
    assert abs(float((pred==f.label).mean())-best_native['balanced_accuracy'])<1e-12
    result=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fixed_thresholds=fixed,
                best_native=best_native,native_optimum_applied_all_cases=shared_native,
                best_shared_nine_cases=best_shared,per_case_optima=case_optima,
                native_per_generator=native_generator,
                selection='Post-hoc selection using Tiny test labels; not independent calibration.')
    add_class_metrics(result)
    for source in previous['pooled']:
        row=fixed[0]['branches'][source['case']]
        for key in ['accuracy','fake_precision','fake_f1']:
            assert abs(row[key]-source[key])<1e-12
    (ROOT/'threshold_sweep.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    def native_row(r):
        n=r['branches']['native']
        return [f"{r['threshold']:.9g}"]+[pct(n[k]) for k in ['balanced_accuracy','fake_recall','real_recall','fake_precision','real_precision','fake_f1']]+[n['fp'],n['fn']]
    lines=['# GSD train BFree → Tiny: thay đổi ngưỡng', '',
           'Nguồn: [gsd-train-bfree-test-tiny v1](https://www.kaggle.com/datasets/hoangha071005/gsd-train-bfree-test-tiny). Giữ nguyên model, dùng 63.000 dự đoán của 9 case trên cùng 7.000 ảnh nguồn; không train lại hoặc inference lại.', '',
           '**Ngưỡng tối ưu dưới đây được tìm bằng nhãn Tiny test: chỉ là phân tích hậu nghiệm, chưa phải ngưỡng triển khai đã được xác thực.**', '',
           '## 1. Native: quét các ngưỡng cố định', '',
           table(['Ngưỡng','BA/Accuracy (%)','Recall fake (%)','Recall real (%)','Precision fake (%)','Precision real (%)','F1 fake (%)','FP','FN'],[native_row(r) for r in fixed]+[native_row(shared_native),native_row(best_shared)]), '',
           'FP là ảnh thật bị gọi nhầm fake, FN là fake bị bỏ sót; mỗi lớp có 3.500 ảnh. Quy tắc fake nếu prob_fake >= threshold, giữ nguyên các nhóm score bằng nhau.', '',
           '## 2. BA của cả 9 case theo cùng các ngưỡng', '',
           table(['Case']+[f'{t:g}' for t in THRESHOLDS],
                 [[c]+[pct(r['branches'][c]['balanced_accuracy']) for r in fixed] for c in cases]), '',
           '## 3. Hai cách chọn một ngưỡng chung', '',
           table(['Mục tiêu chọn ngưỡng','Ngưỡng','BA native (%)','BA trung bình 9 case (%)','FPR native (%)'],
                 [['Mặc định','0.5',pct(fixed[0]['branches']['native']['balanced_accuracy']),pct(fixed[0]['balanced_accuracy']),pct(fixed[0]['branches']['native']['fpr'])],
                  ['Tối ưu native',f'{native_threshold:.10g}',pct(shared_native['branches']['native']['balanced_accuracy']),pct(shared_native['balanced_accuracy']),pct(shared_native['branches']['native']['fpr'])],
                  ['Tối ưu trung bình 9 case',f"{best_shared['threshold']:.10g}",pct(best_shared['branches']['native']['balanced_accuracy']),pct(best_shared['balanced_accuracy']),pct(best_shared['branches']['native']['fpr'])]]), '',
           'Tối ưu chính xác trên các ranh giới score; khi bằng BA, chọn ngưỡng cao nhất. Trung bình 9 case cân bằng trọng số cho mỗi case; đó là trung bình trên các biến đổi tương quan, không phải 63.000 ảnh độc lập.', '',
           table(['Case','BA @0,5 (%)','BA @ngưỡng native tối ưu (%)','BA @ngưỡng chung 9 case (%)','Ngưỡng tối ưu riêng case','BA tối ưu riêng (%)'],
                 [[c,pct(fixed[0]['branches'][c]['balanced_accuracy']),pct(shared_native['branches'][c]['balanced_accuracy']),pct(best_shared['branches'][c]['balanced_accuracy']),
                   f"{case_optima[c]['threshold']:.9g}",pct(case_optima[c]['balanced_accuracy'])] for c in cases]), '',
           'Ngưỡng riêng từng case là mức oracle tham khảo vì giả định biết trước case và dùng nhãn test để chọn. Không gộp nó thành kết quả của một detector dùng một ngưỡng duy nhất.', '',
           '## 4. Native theo generator, giữ chung một ngưỡng', '',
           table(['Generator','BA @0,5 (%)','BA @0,1 (%)','BA @0,05 (%)','BA @native optimum (%)','Recall fake @native optimum (%)'],
                 [[g]+[pct(d[str(t)]['balanced_accuracy']) for t in [.5,.1,.05,native_threshold]]+[pct(d[str(native_threshold)]['fake_recall'])] for g,d in native_generator.items()]), '',
           '## 5. Giới hạn', '',
           '- AUC/AP không đổi khi chỉ thay ngưỡng, vì score và thứ tự xếp hạng giữ nguyên. Thay ngưỡng có thể sửa một phần lệch calibration nhưng không khôi phục thông tin bị mất do resize/JPEG.',
           '- Đã đối chiếu confusion/recall/BA tại ngưỡng 0,5 của cả 9 case với kết quả đã audit và kiểm tra trực tiếp accuracy ở native optimum. Hash CSV nguồn giữ nguyên.',
           '- Cần chọn threshold trên calibration set tách biệt rồi khóa trước khi đánh giá test; không dùng báo cáo này để tuyên bố accuracy cross-dataset độc lập ở ngưỡng tối ưu.', '',
           '## File chạy', '',
           f"- [threshold_sweep.py]({(ROOT/'threshold_sweep.py').as_posix()}).",
           f"- [threshold_sweep.json]({(ROOT/'threshold_sweep.json').as_posix()}) chứa confusion/recall từng case/ngưỡng và kết quả generator.", '']
    lines+=['## 6. Metrics đầy đủ theo case và ngưỡng', '',
            'Precision fake = TP/(TP+FP); precision real = TN/(TN+FN). Recall fake = TP/(TP+FN); recall real = TN/(TN+FP). F1 fake = 2TP/(2TP+FP+FN). Các tỷ lệ trong bảng là phần trăm. Accuracy và BA bằng nhau vì mỗi case cân bằng hai lớp.', '',
            '**Một ngưỡng chung tối ưu trung bình 9 case:**', '',
            table(['Case','Acc./BA','Recall fake','Recall real','Precision fake','Precision real','F1 fake','FP','FN'],
                  [[c]+[pct(best_shared['branches'][c][k]) for k in ['balanced_accuracy','fake_recall','real_recall','fake_precision','real_precision','fake_f1']]+[best_shared['branches'][c]['fp'],best_shared['branches'][c]['fn']] for c in cases]), '']
    for c in cases:
        candidates=fixed+[shared_native,best_shared,case_optima[c]]
        lines += ['### '+c, '',
                  table(['Ngưỡng','Acc./BA','Recall fake','Recall real','Precision fake','Precision real','F1 fake','TN','FP','FN','TP'],
                        [[f"{r['threshold']:.10g}"]+[pct(r['branches'][c][k]) for k in ['balanced_accuracy','fake_recall','real_recall','fake_precision','real_precision','fake_f1']]+[r['branches'][c][k] for k in ['tn','fp','fn','tp']] for r in candidates]), '']
    lines+=['## 7. Native: metrics đầy đủ theo generator ở ngưỡng chung 9 case', '',
            table(['Generator','Acc./BA','Recall fake','Recall real','Precision fake','Precision real','F1 fake','FP','FN'],
                  [[g]+[pct(d[str(best_shared['threshold'])][k]) for k in ['balanced_accuracy','fake_recall','real_recall','fake_precision','real_precision','fake_f1']]+[d[str(best_shared['threshold'])]['fp'],d[str(best_shared['threshold'])]['fn']] for g,d in native_generator.items()]), '']
    (ROOT/'THRESHOLD_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('NATIVE FIXED')
    for r in fixed:print(native_row(r))
    print('NATIVE BEST',native_threshold,native_row(shared_native))
    print('ALL9 BEST',best_shared['threshold'],'BAmean',pct(best_shared['balanced_accuracy']),native_row(best_shared))
    print('CASE RESULTS: baseline/native-threshold/all9-threshold/per-case-best')
    for c in cases:print(c,pct(fixed[0]['branches'][c]['balanced_accuracy']),pct(shared_native['branches'][c]['balanced_accuracy']),pct(best_shared['branches'][c]['balanced_accuracy']),case_optima[c]['threshold'],pct(case_optima[c]['balanced_accuracy']))

if __name__=='__main__':main()

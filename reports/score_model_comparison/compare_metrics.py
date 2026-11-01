"""Evaluation metrics on the same image/label cohorts used by the score audit."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import compare_scores as source

ROOT=Path(__file__).resolve().parent
KEYS=['accuracy','balanced_accuracy','auc','ap','fake_recall','real_recall','fake_precision','real_precision','fake_f1','real_f1']
DISPLAY=KEYS[1:]
HEAD=['Acc./BA','AUC','AP','Recall fake','Recall real','Precision fake','Precision real','F1 fake','F1 real']

def metric(f,pred_col=None):
    y=f.label.to_numpy();p=f.score.to_numpy();pred=f[pred_col].to_numpy().astype(bool) if pred_col else p>=.5
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    return dict(n=len(f),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(f),balanced_accuracy=.5*(tn/(tn+fp)+tp/(tp+fn)),
                auc=roc_auc_score(y,p),ap=average_precision_score(y,p),fake_recall=tp/(tp+fn),real_recall=tn/(tn+fp),
                fake_precision=tp/(tp+fp) if tp+fp else 0.,real_precision=tn/(tn+fn) if tn+fn else 0.,
                fake_f1=2*tp/(2*tp+fp+fn),real_f1=2*tn/(2*tn+fp+fn))

def summary(details):
    return {stat:{k:float(fn([r[k] for r in details.values()])) for k in KEYS}
            for stat,fn in [('mean',np.mean),('median',np.median)]}

def pct(v):return f'{100*v:.2f}'

def table(data):return source.table(['Model']+HEAD,[[m]+[pct(r[k]) for k in DISPLAY] for m,r in data.items()])

def main():
    d=source.main(return_frames=True)
    native={m:metric(f[f.case=='native']) for m,f in d['tiny'].items()}
    native['AIDE forensic-only ResNet50']=metric(d['forensic'])
    td={m:{c:metric(g) for c,g in f.groupby('case',sort=False)} for m,f in d['tiny'].items()}
    nine={m:summary({c:td[m][c] for c in d['common_core']}) for m in source.CORE}
    three={m:summary({c:td[m][c] for c in d['common_all']}) for m in d['tiny']}
    bd={m:{c:metric(f[f.case==c]) for c in d['branches']} for m,f in d['bfree'].items()}
    bfive={m:summary(rows) for m,rows in bd.items()}
    train_tiny_six={}
    for m in ['CAPD train Tiny','GSD train Tiny']:
        f=d['bfree'][m]
        train_tiny_six[m]=summary({c:metric(g) for c,g in f.groupby('case',sort=False)})
    clean={m:summary({c:metric(f[(f.case==c)&f.id.isin(d['clean'])]) for c in d['branches']}) for m,f in d['bfree'].items()}
    for m,f in d['nll'].items():clean[m]=summary({c:metric(f[f.case==c],'pred_q95') for c in d['branches']})
    partial={}
    last='SD2.1_inpainted_diffcat_origBG'
    ids=set(d['bfree']['GSD train BFree'].loc[d['bfree']['GSD train BFree'].case==last,'id']);assert len(ids)==7024
    for m,f in d['bfree'].items():partial[m]=metric(f[(f.case==last)&f.id.isin(ids)])
    spectral=metric(d['spectral'],'artifact_primary_prediction')
    # Reconcile native against independently audited outputs for the two new models.
    prior=json.loads((source.REPORTS/'capd_train_bfree_eval_analysis/analysis.json').read_text(encoding='utf8'))
    row=next(r for r in prior['tiny'] if r['case']=='native' and r['generator']=='pooled' and r['pooling']=='lse_full')
    for k in KEYS:assert abs(row[k]-native['CAPD train BFree'][k])<1e-12
    prior=json.loads((source.REPORTS/'gsd_train_bfree_test_tiny_analysis/analysis.json').read_text(encoding='utf8'))
    row=next(r for r in prior['pooled'] if r['case']=='native')
    for k in KEYS:
        if k in row:assert abs(row[k]-native['GSD train BFree'][k])<1e-12
    out=dict(tiny_native=native,tiny_case_metrics=td,tiny_common_nine=nine,tiny_common_three=three,
             bfree_common_five=bfive,bfree_train_tiny_complete_six=train_tiny_six,bfree_branch_metrics=bd,bfree_clean1557_common_five=clean,
             bfree_partial7024=partial,spectral_tiny_pipeline=spectral,
             thresholds='Probability models >=0.5; Twins use saved pred_q95; Spectral uses saved artifact_primary_prediction.')
    (ROOT/'metrics_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    lines=['# So sánh metrics CAPD, GSD và baseline', '',
           '**Tất cả bảng là chỉ số đánh giá, đơn vị %.** Probability model dùng ngưỡng0,5; không trộn với kết quả chọn ngưỡng hậu nghiệm. Accuracy=BA trong các cohort cân bằng, nên gộp một cột. Mean/median được tính từ metrics từng case, không phải score.', '',
           '## 1. Tiny native — cùng7.000 ảnh/nhãn', '',table(native),'',
           'Có3.500 real +3.500 fake/model. CAPD là full pooling. CLIP probe là ViT-B/32, AIDE full khác AIDE forensic-only. Model train Tiny đã thấy các generator Tiny, còn model train BFree đang chuyển miền, không phải ablation chỉ thay kiến trúc.', '',
           '## 2. Cùng9 case Tiny có đủ CAPD/GSD', '']
    for stat in ['mean','median']:lines+=['### '+stat,'',table({m:r[stat] for m,r in nine.items()}),'']
    lines+=['Gồm '+', '.join(d['common_core'])+'. Mỗi case được trọng số bằng nhau; không gộp prediction để tính AUC của hỗn hợp9 biến đổi.', '',
            '## 3. Cùng3 case Tiny có đủ baseline CLIP/NPR/AIDE full', '']
    for stat in ['mean','median']:lines+=['### '+stat,'',table({m:r[stat] for m,r in three.items()}),'']
    lines+=['Gồm native, fake_jpeg96, all_jpeg96. Không dùng mean9 case của CAPD/GSD so với mean3 case của baseline. Khớp ảnh gốc/nhãn; pipeline tiền xử lý của từng model vẫn khác nhau.', '',
            '## 4. BFree — cùng5 nhánh hoàn tất và10.304 ID', '']
    for stat in ['mean','median']:lines+=['### '+stat,'',table({m:r[stat] for m,r in bfive.items()}),'']
    lines+=['Mỗi nhánh10.304 real +10.304 fake. GSD train BFree chỉ đủ5 nhánh nên không dùng macro6 của model khác để đối chiếu. Mean precision/F1 được tính từng nhánh trước rồi lấy trung bình.', '',
            '## 5. BFree — từng nhánh đầy đủ', '']
    for case in d['branches']:lines+=['### '+case,'',table({m:rows[case] for m,rows in bd.items()}),'']
    lines+=['## 6. BFree — nhánh diffcat_origBG partial, cùng7.024 ID', '',table(partial),'',
            'Đã lọc các model về đúng ID đã chạy ở GSD train BFree; đây không phải kết quả đủ10.304 ID của nhánh này.', '',
            '## 7. Baseline Twins/GMM — cùng cohort clean1.557 ID ×5 nhánh', '',
            table({m:r['mean'] for m,r in clean.items()}),'',
            'Twins dùng NLL theo chiều đã định cao=fake, quyết định pred_q95 từ real calibration; không áp dụng threshold0,5 cho NLL hoặc đảo score sau khi nhìn nhãn. Tất cả model được lọc đúng cùng ID/nhánh; không so với bảng10.304 ID như cùng cohort.', '',
            '## 8. Spectral/MFM-GMM — Tiny pipeline riêng', '',table({'Spectral/MFM-GMM':spectral}),'',
            'Cùng7.000 ảnh gốc nhưng Spectral có JPEG xử lý trước và fixed-grid tiles. Ngưỡng artifact_primary_prediction được lưu từ pipeline; không phải probability0,5, không phải Tiny native pixel pipeline giống hệt các model trên.', '',
            '## 9. Tiny từng case có prediction', '']
    for c in sorted(set.union(*[set(r) for r in td.values()])):
        lines+=['### '+c,'',table({m:r[c] for m,r in td.items() if c in r}),'']
    lines+=['## File chạy và truy nguồn', '',
            f"- [compare_metrics.py]({(ROOT/'compare_metrics.py').as_posix()}) tính lại metrics từ các frame đã join ID/label.",
            f"- [metrics_comparison.json]({(ROOT/'metrics_comparison.json').as_posix()}) giữ full precision và TN/FP/FN/TP từng case.",
            f"- [Báo cáo score cũ]({(ROOT/'COMPARISON.md').as_posix()}) có thông tin nguồn và cohort; đây là tài liệu riêng về score, không dùng thay báo cáo metrics này.", '']
    # Lead with training source, and show both test datasets within each group.
    tiny_models=['CAPD train Tiny','GSD train Tiny','CLIP ViT-B/32 probe','NPR-ResNet18','AIDE original full','AIDE forensic-only ResNet50']
    bfree_models=['CAPD train BFree','GSD train BFree']
    def named(data,models):
        return {m.replace(' train Tiny','').replace(' train BFree',''):data[m] for m in models}
    organized=['# So sánh metrics theo tập TRAIN → tập TEST', '',
               'Tất cả chỉ số là %. Model xác suất dùng ngưỡng0,5; CAPD dùng full pooling. Accuracy=BA ở từng cohort cân bằng. Không trộn các kết quả tối ưu ngưỡng bằng nhãn test vào bảng mặc định.', '',
               '## I. TRAIN TRÊN TINY-GENIMAGE', '',
               '### I.1. TEST trên Tiny — native, cùng7.000 ảnh', '',table(named(native,tiny_models)), '',
               'Đây là đánh giá trong miền đối với các model train Tiny. AIDE full và forensic-only là hai biến thể riêng. AUC/AP100,00 là làm tròn, không có nghĩa không có lỗi.', '',
               '### I.2. TEST trên BFree — mean trên cùng5 nhánh, cùng10.304 ID', '',
               table(named({m:r['mean'] for m,r in bfive.items()},['CAPD train Tiny','GSD train Tiny'])), '',
               'Đây là chuyển miền Tiny→BFree. Bảng chính giữ cùng5 nhánh như nhóm train BFree bên dưới để đối chiếu nhất quán. Chưa có predictions BFree cùng cohort được xác minh cho CLIP ViT-B/32 probe, NPR, AIDE full/forensic-only ở bảng I.1; không điền điểm Tiny hoặc notebook khác checkpoint/cohort vào đây.', '',
               'Kết quả bổ sung trên đủ6 nhánh của hai model train Tiny (không dùng để so với nhóm chỉ đủ5):', '',
               table(named({m:r['mean'] for m,r in train_tiny_six.items()},list(train_tiny_six))), '',
               '### I.3. TEST Tiny qua các biến đổi — mean/median cùng9 case CAPD/GSD', '']
    for stat in ['mean','median']:organized += [stat,'',table(named({m:r[stat] for m,r in nine.items()},['CAPD train Tiny','GSD train Tiny'])),'']
    organized+=['## II. TRAIN TRÊN BFREE', '',
                'CAPD và GSD trong nhóm này train real COCO + fake SD2.1_inpainted_samecat_origBG. Khác backbone: CAPD-DINOv3 so với CLIP+GSD. Không phải recipe train phối hợp6 nhánh của paper B-Free.', '',
                '### II.1. TEST trên Tiny — native, cùng7.000 ảnh', '',table(named(native,bfree_models)), '',
                'Đây là chuyển miền BFree→Tiny. Không đặt cùng bảng xếp hạng trong miền của nhóm I mà không ghi khác biệt dữ liệu train.', '',
                '### II.2. TEST trên BFree — mean trên cùng5 nhánh hoàn tất, cùng10.304 ID', '',
                f"[Bảng riêng từng case với đầy đủ metrics và nhánh partial matched]({(ROOT/'TRAIN_BFREE_TEST_BFREE_CASES.md').as_posix()}).", '',
                table(named({m:r['mean'] for m,r in bfive.items()},bfree_models)), '',
                'GSD train BFree còn partial ở diffcat_origBG, nên bảng chính I.2 và II.2 đều dùng cùng5 nhánh. Không dùng kết quả đủ6 nhánh bổ sung ở I.2 để trừ trực tiếp với macro5 của nhóm này.', '',
                '### II.3. TEST Tiny qua các biến đổi — mean/median cùng9 case CAPD/GSD', '']
    for stat in ['mean','median']:organized += [stat,'',table(named({m:r[stat] for m,r in nine.items()},bfree_models)),'']
    organized+=['### II.4. TEST BFree — nhánh cuối partial, cùng7.024 ID', '',table(named(partial,bfree_models)),'',
                '## III. PHỤ LỤC ĐỐI CHIẾU COHORT VÀ TỪNG CASE', '',
                'Twins/GMM và Spectral train real-only từ COCO/ImageNet, không tự xếp vào nhóm train Tiny hay train BFree. Các cấu hình, ngưỡng và cohort riêng được ghi dưới đây.', '']
    (ROOT/'METRICS_COMPARISON.md').write_text('\n'.join(organized+lines[1:]),encoding='utf8')
    for label,data in [('NATIVE',native),('TINY9_MEAN',{m:r['mean'] for m,r in nine.items()}),('BFREE5_MEAN',{m:r['mean'] for m,r in bfive.items()}),('CLEAN5',{m:r['mean'] for m,r in clean.items()})]:
        print(label)
        for m,r in data.items():print(m,*[pct(r[k]) for k in DISPLAY])

if __name__=='__main__':main()

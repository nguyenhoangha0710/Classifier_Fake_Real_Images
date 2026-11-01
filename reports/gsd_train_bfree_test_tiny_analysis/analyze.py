"""Audit completed Tiny predictions and summarize cross-dataset generalization."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'source/gsd_bfree_samecat_origbg_best_tiny9_v1'

def metric(f):
    y=f.label.to_numpy();p=f.prob_fake.to_numpy();pred=p>=.5
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    rr=tn/(tn+fp);fr=tp/(tp+fn);precision=tp/(tp+fp)
    return dict(n=len(f),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(f),balanced_accuracy=(rr+fr)/2,
                real_recall=rr,fake_recall=fr,fake_precision=precision,fake_f1=2*tp/(2*tp+fp+fn),
                auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def pct(v):return f'{100*v:.2f}'

def main():
    frame=pd.read_csv(RUN/'predictions.csv')
    saved=pd.read_csv(RUN/'metrics.csv')
    state=json.loads((RUN/'completion.json').read_text(encoding='utf8'))
    ckpt=json.loads((RUN/'checkpoint_inspection.json').read_text(encoding='utf8'))
    cases=state['planned_cases']
    assert state['status']=='complete' and state['completed_cases']==cases and len(cases)==9
    assert len(frame)==63000 and not frame[['case','sample_id']].duplicated().any()
    assert np.isfinite(frame[['prob_fake','logit_real','logit_fake','logit_margin']]).all().all()
    assert frame.prob_fake.between(0,1).all()
    native=frame[frame.case=='native'];reference=native.set_index('sample_id')[['label','generator','pixel_hash']].sort_index()
    computed=[];pooled=[];maxdiff=0
    for case in cases:
        f=frame[frame.case==case]
        assert len(f)==7000 and f.label.value_counts().to_dict()=={0:3500,1:3500}
        assert f.set_index('sample_id')[['label','generator','pixel_hash']].sort_index().equals(reference)
        assert f.sample_id.tolist()==native.sample_id.tolist()
        assert f.batch_id.tolist()==native.batch_id.tolist()
        row=dict(case=case,generator='pooled',**metric(f));pooled.append(row)
        for generator,g in f.groupby('generator'):
            assert g.label.value_counts().to_dict()=={0:500,1:500}
            computed.append(dict(case=case,generator=generator,**metric(g)))
        computed.append(row)
    for r in computed:
        old=saved[(saved.case==r['case'])&(saved.generator==r['generator'])].iloc[0]
        for k in ['n','tn','fp','fn','tp','accuracy','balanced_accuracy','real_recall','fake_recall','fake_precision','fake_f1','auc','ap']:
            maxdiff=max(maxdiff,abs(float(old[k])-r[k]))
    assert maxdiff<1e-12
    macro=pd.read_csv(RUN/'macro_all.csv')
    for case in cases:
        selected=[r for r in computed if r['case']==case and r['generator']!='pooled']
        old=macro[macro.case==case].iloc[0]
        for key in ['accuracy','balanced_accuracy','real_recall','fake_recall','auc','ap']:
            assert abs(np.mean([r[key] for r in selected])-float(old[key]))<1e-12
    tiny_source=ROOT.parent/'gsd_augmented_nine_cases/source/gsd_uploaded_best_variants_v1'
    old_frame=pd.read_csv(tiny_source/'predictions.csv')
    old_native=old_frame[old_frame.case=='native']
    assert old_native.set_index('sample_id')[['label','generator','pixel_hash']].sort_index().equals(reference)
    old_metrics=pd.read_csv(tiny_source/'metrics.csv')
    comparisons=[]
    for row in pooled:
        old=old_metrics[(old_metrics.case==row['case'])&(old_metrics.generator=='pooled')].iloc[0]
        comparisons.append(dict(case=row['case'],gsd_bfree_accuracy=row['accuracy'],gsd_tiny_accuracy=float(old.accuracy),
                                accuracy_change_pp=100*(row['accuracy']-float(old.accuracy))))
    checks=dict(rows=len(frame),cases=len(cases),unique_sample_ids=frame.sample_id.nunique(),
                verified_metric_rows=len(computed),max_metric_difference=maxdiff,
                per_case_ids_labels_generators_pixel_hashes_match=True,sample_and_batch_order_match=True,
                ids_labels_pixel_hashes_match_prior_tiny_evaluation=True,
                source_predictions_sha256=hashlib.sha256((RUN/'predictions.csv').read_bytes()).hexdigest())
    result=dict(checks=checks,checkpoint=dict(epoch=ckpt['epoch'],signature=ckpt['signature'],sha256=ckpt.get('sha256')),
                pooled=pooled,per_generator=[r for r in computed if r['generator']!='pooled'],comparison_to_gsd_tiny=comparisons)
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    lines=['# GSD train BFree → test Tiny-GenImage', '',
           'Nguồn: [hoangha071005/gsd-train-bfree-test-tiny, version 1](https://www.kaggle.com/datasets/hoangha071005/gsd-train-bfree-test-tiny). Đối chiếu ngày 01/11/2026.', '',
           '## Trạng thái và protocol', '',
           '- Evaluation complete, smoke=false; đủ 9 case × 7.000 ảnh = 63.000 dự đoán, không có case partial; khoảng 2,72 giờ. Không có png_roundtrip trong lượt này.',
           '- Model CLIP ViT-L/14 + GSD, train BFree riêng SD2.1_inpainted_samecat_origBG và real COCO, checkpoint best virtual epoch2, không phải model train phối hợp sáu nhánh.',
           f"- Checkpoint source: `{ckpt['source']}`; signature `{ckpt['signature']}`; SHA-256 theo metadata `{ckpt.get('sha256')}`. Không tải weight để tính lại hash trong lượt audit này.",
           '- Test Tiny dùng manifest hadinho0710/mainfest-train-genimage/seed42/test_manifest.csv: 7 generator; mỗi generator 500 real + 500 fake. Đây là cross-dataset evaluation từ BFree/SD2.1 sang Tiny, khác với train Tiny rồi test Tiny.',
           '- Ngưỡng fake probability >=0,5; pooled accuracy = balanced accuracy do hai lớp cân bằng. Các bảng AUC/AP dưới đây dùng pooled 7.000 ảnh, khác macro trung bình AUC từng generator trong macro_all.csv.',
           '- Đã tính lại 72 dòng metrics từ predictions: confusion counts, accuracy, BA, recalls, precision/F1, AUC/AP khớp nguồn trong sai số <1e-12. Đối chiếu cùng ID/label/generator/pixel_hash đã ghi, cùng thứ tự sample/batch giữa 9 case. Pixel_hash là hash nguồn lưu trong manifest, không tính lại ảnh sau biến đổi.',
           '- Cohort native cũng khớp ID/label/pixel_hash với lượt GSD train Tiny augmented trước. Chưa audit near-duplicate/pixel overlap giữa BFree train và Tiny test.', '',
           '## Kết quả pooled theo case', '',
           table(['Case','Accuracy/BA (%)','AUC (%)','AP (%)','Recall fake (%)','Recall real (%)','FP','FN'],
                 [[r['case'],pct(r['accuracy']),pct(r['auc']),pct(r['ap']),pct(r['fake_recall']),pct(r['real_recall']),r['fp'],r['fn']] for r in pooled]), '',
           '## Native theo generator', '',
           table(['Generator','Accuracy/BA (%)','AUC (%)','Recall fake (%)','Recall real (%)','FP','FN'],
                 [[r['generator'],pct(r['accuracy']),pct(r['auc']),pct(r['fake_recall']),pct(r['real_recall']),r['fp'],r['fn']] for r in computed if r['case']=='native' and r['generator']!='pooled']), '',
           '## So với GSD train Tiny All7 augmented', '',
           table(['Case','GSD train BFree acc. (%)','GSD train Tiny acc. (%)','Chênh lệch (điểm %)'],
                 [[r['case'],pct(r['gsd_bfree_accuracy']),pct(r['gsd_tiny_accuracy']),f"{r['accuracy_change_pp']:+.2f}"] for r in comparisons]), '',
           'Hai model dùng cùng tập ảnh, nhưng GSD train Tiny đã thấy các generator Tiny trong training; model BFree chưa train trên Tiny. Bảng thể hiện sự khác nhau giữa khả năng trong miền và chuyển miền, chưa phải bằng chứng nhân quả cho riêng kiến trúc.', '',
           '## Đánh giá', '',
           '- Native: accuracy 81,26%, AUC 97,84%, AP 97,38%; TN=3.461, FP=39, FN=1.273, TP=2.227. Model ít báo nhầm real (1,11%) nhưng bỏ sót 36,37% fake ở ngưỡng 0,5.',
           '- Midjourney khó nhất: BA 67,90%, fake recall 36,60%; ADM fake recall 49,60%. Wukong và SD1.5 đạt BA cao nhất 88,20% và 87,20%.',
           '- JPEG làm giảm accuracy còn khoảng 77–79%. Resize256 và resize256+JPEG96 giảm mạnh còn 69,79% và 68,60%; AUC cũng giảm, nên ảnh hưởng không chỉ do ngưỡng. Blur sigma1 tăng accuracy lên 84,33% nhưng AUC giảm: không suy ra blur luôn cải thiện khả năng xếp hạng.',
           '- Native AUC cao hơn accuracy khá nhiều gợi ý vấn đề calibration/ngưỡng khi chuyển miền; muốn chọn ngưỡng mới cần calibration độc lập, không chọn bằng nhãn Tiny test rồi gọi là kết quả test xác thực.',
           '- Kết quả BFree tốt của cùng checkpoint không tự chuyển thành hiệu suất cao trên Tiny. Cần báo cáo riêng cross-dataset và in-domain; không gộp 9 biến đổi của cùng 7.000 ảnh thành 63.000 ảnh test độc lập.', '',
           '## File chạy local', '',
           f"- [download_source.py]({(ROOT/'download_source.py').as_posix()}) — tải metadata, bảng kết quả và predictions, pin version trong mỗi request.",
           f"- [analyze.py]({(ROOT/'analyze.py').as_posix()}) — tính lại metrics, kiểm tra cohort và xuất báo cáo.",
           f"- [analysis.json]({(ROOT/'analysis.json').as_posix()}) — số liệu chi tiết/hashes.",
           f"- [Notebook test Tiny9]({(ROOT.parent.parent/'baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb').as_posix()}).", '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(checks,indent=2))
    for r in pooled:print(r['case'],'BA',pct(r['accuracy']),'AUC',pct(r['auc']),'fake',pct(r['fake_recall']),'real',pct(r['real_recall']))

if __name__=='__main__':main()

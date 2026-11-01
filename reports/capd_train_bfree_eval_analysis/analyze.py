"""Verify CAPD Tiny/BFree metrics and report all class metrics at threshold .5."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
TINY=SRC/'capd_eval_tiny/b2_dinov3/seed42'
BFREE=SRC/'capd_eval_bfree/b2_dinov3/seed42'
COLS={'lse_full':'prob_fake','lse_window':'prob_fake_window','lse_inner':'prob_fake_inner','lse_nopad':'prob_fake_nopad'}

def metric(f,col):
    y=f.label.to_numpy();p=f[col].to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pred=p>=.5
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    return dict(n=len(f),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(f),balanced_accuracy=(tn/(tn+fp)+tp/(tp+fn))/2,
                fake_recall=tp/(tp+fn),real_recall=tn/(tn+fp),fake_precision=tp/(tp+fp) if tp+fp else 0.,
                real_precision=tn/(tn+fn) if tn+fn else 0.,fake_f1=2*tp/(2*tp+fp+fn),
                real_f1=2*tn/(2*tn+fp+fn),auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def pct(v):return f'{100*v:.2f}'
KEYS=['balanced_accuracy','auc','ap','fake_recall','real_recall','fake_precision','real_precision','fake_f1']
HEAD=['BA/Acc','AUC','AP','Recall fake','Recall real','Precision fake','Precision real','F1 fake']

def main():
    tm=pd.read_csv(TINY/'tiny_eval/metrics.csv');bm=pd.read_csv(BFREE/'bfree_eval/metrics_bfree.csv')
    tiny=[];bfree=[];diff=0.;files=[];native_ids=None
    for case in tm.case.unique():
        p=TINY/f'tiny_eval/predictions_{case}.csv';f=pd.read_csv(p);files.append(p)
        assert len(f)==7000 and not f.sample_id.duplicated().any()
        assert f.label.value_counts().to_dict()=={0:3500,1:3500}
        ident=f.set_index('sample_id')[['label','generator']].sort_index()
        if native_ids is None:native_ids=ident
        else:assert ident.equals(native_ids)
        for _,row in tm[tm.case==case].iterrows():
            g=f if row.generator=='pooled' else f[f.generator==row.generator]
            m=metric(g,COLS[row.pooling]);tiny.append(dict(case=case,generator=row.generator,pooling=row.pooling,**m))
            for k in ['n','tn','fp','fn','tp','accuracy','balanced_accuracy','auc','ap','fake_recall','real_recall']:
                diff=max(diff,abs(float(row[k])-m[k]))
    bfpath=BFREE/'bfree_eval/predictions_bfree.csv';files.append(bfpath)
    bf=pd.read_csv(bfpath,dtype={'coco_id':str});bf['coco_id']=bf.coco_id.str.zfill(12)
    assert len(bf)==72128 and not bf[['branch','coco_id']].duplicated().any()
    official=pd.read_csv(ROOT.parent/'gsd_bfree_valid_analysis/source/gsd_bfree_valid_v1/bfree_valid_ids.csv',dtype=str)
    ids=set(official.id.str.zfill(12));assert len(ids)==10304
    for _,row in bm[bm.branch!='macro'].iterrows():
        f=bf[bf.branch.isin(['real',row.branch])]
        assert len(f)==20608
        for label in [0,1]:assert set(f.loc[f.label==label,'coco_id'])==ids
        m=metric(f,COLS[row.pooling]);bfree.append(dict(branch=row.branch,pooling=row.pooling,**m))
        for k in ['n','tn','fp','fn','tp','accuracy','balanced_accuracy','auc','ap','fake_recall','real_recall']:
            diff=max(diff,abs(float(row[k])-m[k]))
    assert diff<1e-12
    macros={}
    for pool in bm.pooling.unique():
        rows=[r for r in bfree if r['pooling']==pool]
        macros[pool]={k:float(np.mean([r[k] for r in rows])) for k in KEYS+['real_f1']}
        reported=bm[(bm.branch=='macro')&(bm.pooling==pool)].iloc[0]
        for k in ['balanced_accuracy','auc','ap','fake_recall','real_recall']:
            assert abs(macros[pool][k]-float(reported[k]))<1e-12
    train=pd.read_csv(BFREE/'train_order.csv',dtype={'coco_id':str});val=pd.read_csv(BFREE/'validation_ids.csv',dtype={'coco_id':str})
    trainids=set(train.coco_id.str.zfill(12));valids=set(val.coco_id.str.zfill(12))
    assert not trainids&ids and not valids&ids and not trainids&valids
    old=ROOT.parent/'gsd_train_bfree_test_tiny_analysis/source/gsd_bfree_samecat_origbg_best_tiny9_v1/predictions.csv'
    gsd=pd.read_csv(old);gsdnative=gsd[gsd.case=='native'].set_index('sample_id')[['label','generator']].sort_index()
    assert native_ids.equals(gsdnative)
    selected=json.loads((BFREE/'selected_checkpoint.json').read_text(encoding='utf8'))
    states={name:json.loads((root/'completion.json').read_text(encoding='utf8')) for name,root in [('tiny',TINY),('bfree',BFREE)]}
    assert all(s['status']=='complete' for s in states.values())
    oldbf=pd.read_csv(ROOT.parent/'gsd_train_bfree_test_bfree_analysis/source/gsd_bfree_samecat_origbg_best_bfree_valid_v1/metrics.csv')
    five=oldbf.branch.tolist()
    matched_five={pool:{k:float(np.mean([r[k] for r in bfree if r['pooling']==pool and r['branch'] in five])) for k in KEYS} for pool in ['lse_full','lse_window']}
    checks=dict(tiny_cases=len(tm.case.unique()),tiny_prediction_rows=70000,bfree_unique_images=len(bf),bfree_ids=len(ids),
                verified_metric_rows=len(tiny)+len(bfree),max_metric_difference=diff,
                train_pool_ids=len(trainids),internal_validation_ids=len(valids),train_eval_overlap=0,internal_val_eval_overlap=0,
                tiny_ids_labels_generators_match_gsd_bfree_trained=True,
                sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    result=dict(checks=checks,selected_checkpoint=selected,completion=states,tiny=tiny,bfree=bfree,bfree_macro=macros,matched_five_vs_gsd=matched_five)
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    full=[r for r in tiny if r['generator']=='pooled' and r['pooling']=='lse_full']
    bf_full=[r for r in bfree if r['pooling']=='lse_full']
    native=[r for r in tiny if r['case']=='native' and r['generator']!='pooled' and r['pooling']=='lse_full']
    def main_table(label,rows):return table([label]+HEAD,[[r[label]]+[pct(r[k]) for k in KEYS] for r in rows])
    lines=['# CAPD train BFree → Tiny và BFree', '',
           'Nguồn: [hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree, version 1](https://www.kaggle.com/datasets/hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree). Kiểm tra ngày 01/11/2026. Tất cả tỷ lệ trong bảng là phần trăm.', '',
           '## Model và kiểm tra dữ liệu', '',
           '- Run b2_dinov3: DINOv3 ViT-L/16 + LoRA rank8 + patch head CAPD. Train real COCO và nhánh fake SD2.1_inpainted_samecat_origBG, không phải phối hợp sáu nhánh fake.',
           '- Config resume từ hadinho0710/check-point-train-capd-bfree/capd_tn2. Training status complete, images_seen=75.600, global_step=9.450; selected best virtual epoch2, monitor val loss=0,01791314845, stop_reason=all_virtual_epochs_done. Status ve=4 là vị trí vòng lặp sau khi kết thúc, không tự diễn giải thành đã train bốn virtual epoch.',
           '- Cả Tiny và BFree evaluation complete. Tiny đủ 10 case × 7.000 ảnh; mỗi case 3.500 real + 3.500 fake. BFree đủ 10.304 official-valid COCO IDs, mỗi ID có 1 real và 6 fake: 72.128 ảnh riêng biệt.',
           '- Đã đối chiếu trực tiếp ID train_order/validation_ids: không trùng với official-valid BFree. Tập train pool có '+str(len(trainids))+' ID, internal validation '+str(len(valids))+' ID. Kiểm tra theo ID, không tính lại pixel/near-duplicate.',
           '- Tính lại '+str(checks['verified_metric_rows'])+' dòng confusion/accuracy/BA/recalls/AUC/AP từ predictions, khớp bảng nguồn trong sai số <1e-12. Bổ sung precision real/fake và F1 từ confusion counts. Tiny ID/label/generator khớp lượt GSD train BFree → Tiny trước.',
           '- Ngưỡng cố định 0,5; BA bằng accuracy vì mỗi cặp evaluation cân bằng hai lớp. AUC/AP Tiny là pooled 7.000 ảnh; BFree macro là trung bình sáu nhánh, không phải cộng lặp real thành các ảnh độc lập.', '',
           '## Tiny: full pooling, ngưỡng 0,5', '',main_table('case',full), '',
           '### Native theo generator', '',main_table('generator',native), '',
           '## BFree: full pooling, ngưỡng 0,5', '',main_table('branch',bf_full), '',
           '### Macro sáu nhánh theo pooling', '',
           table(['Pooling']+HEAD,[[pool]+[pct(m[k]) for k in KEYS] for pool,m in macros.items()]), '',
           '## Các pooling chẩn đoán', '',
           '- lse_full: LSE trên toàn bộ patch; lse_window: cửa sổ patch, có trong BFree. Tiny nguồn không báo window pooling.',
           '- lse_inner: bỏ hàng cuối và cột cuối patch. lse_nopad: chỉ bỏ patch chứa padding. Đây là các phép can thiệp score, không phải model train riêng.',
           '- BFree ảnh512 chia hết patch16, không padding: full và nopad trùng metrics. inner có BA xấp xỉ50% ở ngưỡng0,5 nhưng AUC vẫn cao; điều này cho thấy score/điểm quyết định rất nhạy với loại patch biên. Chưa đủ để kết luận toàn bộ hiệu suất do shortcut; cần calibration riêng/đối chứng trước khi suy ra nhân quả.', '',
           table(['Tiny case','Pooling']+HEAD,[[r['case'],r['pooling']]+[pct(r[k]) for k in KEYS] for r in tiny if r['generator']=='pooled' and r['pooling']!='lse_full']), '',
           '## Đối chiếu GSD train BFree', '',
           '- Tiny native: CAPD full BA90,31%, AUC98,50%; GSD BA81,26%, AUC97,84% tại cùng ngưỡng0,5 và cùng ID. Khác backbone/training nên không quy toàn bộ chênh lệch cho riêng CAPD/GSD.',
           '- BFree: model CAPD đã có đủ6 nhánh, GSD dataset cũ mới đủ5. Chỉ so matched5, không lấy macro6 CAPD so thẳng macro5 GSD:', '',
           table(['Model','Số nhánh','BA','AUC','Recall fake','Recall real'],
                 [['CAPD full',5]+[pct(matched_five['lse_full'][k]) for k in ['balanced_accuracy','auc','fake_recall','real_recall']],
                  ['CAPD window',5]+[pct(matched_five['lse_window'][k]) for k in ['balanced_accuracy','auc','fake_recall','real_recall']],
                  ['GSD train BFree',5]+[pct(float(oldbf[k].mean())) for k in ['balanced_accuracy','auc','fake_recall','real_recall']]]), '',
           '## Nhận xét', '',
           '- Native Tiny: TP2.838, FN662, TN3.484, FP16; model bỏ sót18,91% fake nhưng chỉ báo nhầm0,46% real ở ngưỡng0,5. Chuyển từ BFree sang Tiny vẫn có sai lệch quyết định; ngưỡng nguồn không tự tối ưu cho miền mới.',
           '- Đọc hiệu suất từng phép biến đổi và từng generator ở bảng trên; không chỉ chọn native hoặc một biến đổi có điểm cao để khẳng định robustness tổng thể.',
           '- Các cột best_ba/best_ba_threshold trong nguồn được tối ưu bằng nhãn evaluation, là diagnostic hậu nghiệm; báo cáo chính giữ ngưỡng0,5, không gọi đó là ngưỡng đã xác thực.', '',
           '## File chạy và nguồn local', '',
           f"- [download_source.py]({(ROOT/'download_source.py').as_posix()}) — metadata/inventory và tải file nhỏ/predictions theo version.",
           f"- [analyze.py]({(ROOT/'analyze.py').as_posix()}) — kiểm tra ID, tính lại metrics và tạo báo cáo.",
           f"- [analysis.json]({(ROOT/'analysis.json').as_posix()}) — đủ metrics, TN/FP/FN/TP, hashes, mỗi generator/case/pooling.",
           f"- [Snapshot capd_all.py từ dataset]({(SRC/'capd_all.py').as_posix()}).", '']
    lines+=['## Phụ lục: BFree đủ từng nhánh và pooling', '']
    for pool in macros:
        lines+=['### '+pool,'',main_table('branch',[r for r in bfree if r['pooling']==pool]),'',
                table(['Nhánh','TN','FP','FN','TP'],[[r['branch'],r['tn'],r['fp'],r['fn'],r['tp']] for r in bfree if r['pooling']==pool]),'']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('CHECKS',json.dumps({k:v for k,v in checks.items() if k!='sha256'}))
    print('TINY FULL')
    for r in full:print(r['case'],*[pct(r[k]) for k in KEYS])
    print('BFREE FULL')
    for r in bf_full:print(r['branch'],*[pct(r[k]) for k in KEYS])
    print('MACRO',json.dumps(macros))
    print('MATCHED5',json.dumps(matched_five))

if __name__=='__main__':main()

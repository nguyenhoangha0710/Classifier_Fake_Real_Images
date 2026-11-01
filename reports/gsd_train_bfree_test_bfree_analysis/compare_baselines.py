"""Compare saved BFree predictions on matched branch/ID cohorts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT.parent
PROJECT=REPORTS.parent
RUN=ROOT/'source/gsd_bfree_samecat_origbg_best_bfree_valid_v1'
REF=ROOT/'baseline_reference_source'


def load(path, kind, score='prob_fake'):
    f=pd.read_csv(path,dtype={'coco_id':str,'bfree_id':str})
    idcol='coco_id' if 'coco_id' in f else 'bfree_id'
    branchcol='case' if kind=='gsd' else ('branch' if kind=='capd' else 'variant')
    out=pd.DataFrame(dict(id=f[idcol].str.zfill(12),branch=f[branchcol],label=f.label.astype(int),score=f[score].astype(float)))
    out['pred']=f.pred_q95.astype(int) if kind=='twins' else (out.score>=.5).astype(int)
    assert not out[['branch','id','label']].duplicated().any()
    assert np.isfinite(out.score).all()
    return out


def metrics(f):
    real=f.label.eq(0);fake=~real
    tn=int((real&f.pred.eq(0)).sum());fp=int((real&f.pred.eq(1)).sum())
    fn=int((fake&f.pred.eq(0)).sum());tp=int((fake&f.pred.eq(1)).sum())
    return dict(n=len(f),tn=tn,fp=fp,fn=fn,tp=tp,balanced_accuracy=.5*(tn/(tn+fp)+tp/(tp+fn)),
                fake_recall=tp/(tp+fn),real_recall=tn/(tn+fp),auc=roc_auc_score(f.label,f.score),
                ap=average_precision_score(f.label,f.score))


def summarize(frame,branches,ids):
    detail={}
    for branch in branches:
        f=frame[(frame.branch.isin(['real',branch])) & frame.id.isin(ids)]
        assert set(f.loc[f.label.eq(0),'id'])==ids and set(f.loc[f.label.eq(1),'id'])==ids
        assert len(f)==2*len(ids)
        detail[branch]=metrics(f)
    macro={key:float(np.mean([r[key] for r in detail.values()])) for key in ['balanced_accuracy','fake_recall','real_recall','auc','ap']}
    return dict(ids_per_branch=len(ids),branches=detail,macro=macro)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def pct(v):return f'{100*v:.2f}'


def main():
    paths={
        'GSD train BFree':RUN/'predictions.csv',
        'GSD train Tiny':REPORTS/'gsd_bfree_valid_analysis/source/gsd_bfree_valid_v1/predictions.csv',
        'CAPD full':REPORTS/'capd_tiny_bfree_eval_analysis/source/capd_eval_bfree/r2_dinov3/seed42/bfree_eval/predictions_bfree.csv',
        'Twins native GMM':REF/'twins_native_predictions.csv',
        'Twins codec GMM':REF/'hoangha071005/eval-forensic-twins-codec/forensic_twins_codec_gmm/bfree/artifacts/metrics/native_png/predictions.csv',
    }
    frames={name:load(p,'gsd' if name.startswith('GSD') else ('capd' if name.startswith('CAPD') else 'twins'),
                      'nll' if name.startswith('Twins') else 'prob_fake') for name,p in paths.items()}
    frames['CAPD window']=load(paths['CAPD full'],'capd','prob_fake_window')
    state=json.loads((RUN/'completion.json').read_text(encoding='utf8'))
    branches=state['completed_branches']
    ids=set(frames['GSD train BFree'].id)
    clean=set(frames['Twins native GMM'].id)
    assert len(ids)==10304 and len(clean)==1557 and clean<=ids
    assert clean==set(frames['Twins codec GMM'].id)
    direct={name:summarize(frames[name],branches,ids) for name in ['GSD train BFree','GSD train Tiny','CAPD full','CAPD window']}
    matched={name:summarize(f,branches,clean) for name,f in frames.items()}
    # Validate downloaded baseline score orientation/identity against their
    # published six-branch summaries before reporting the five-branch subset.
    six=state['planned_branches']
    for model,summary_path in [
        ('Twins native GMM', REF/'hoangha071005/evaluation-twins-bfree/forensic_twins_augmented_gmm_full/artifacts/augmented_bfree_metrics/protocol_native_k1_full/summary.json'),
        ('Twins codec GMM', REF/'hoangha071005/eval-forensic-twins-codec/forensic_twins_codec_gmm/bfree/artifacts/metrics/native_png/summary.json')]:
        published=json.loads(summary_path.read_text(encoding='utf8'))
        recomputed=summarize(frames[model],six,clean)['macro']
        assert abs(recomputed['auc']-published['macro_variant_auroc'])<1e-12
        assert abs(recomputed['ap']-published['macro_variant_average_precision'])<1e-12
        if 'q95_balanced_accuracy' in published:
            assert abs(recomputed['balanced_accuracy']-published['q95_balanced_accuracy'])<1e-12
    spec=importlib.util.spec_from_file_location('threshold_helpers',REPORTS/'gsd_capd_bfree_threshold_comparison/compare_thresholds.py')
    th=importlib.util.module_from_spec(spec);spec.loader.exec_module(th)
    sensitivity={}
    for name in direct:
        f=frames[name]
        groups={b:tuple(np.sort(f.loc[f.branch.isin(['real',b])&f.label.eq(label),'score'].to_numpy()) for label in [0,1]) for b in branches}
        sensitivity[name]=dict(best_shared=th.optimize(groups),at_fpr_05=th.optimize(groups,.05))
    # The unfinished branch is only compared after filtering every model to its 7,024 IDs.
    partial=load(RUN/'predictions_SD2.1_inpainted_diffcat_origBG.partial.csv','gsd')
    partial_ids=set(partial.id);assert len(partial_ids)==7024
    pb=state['remaining_branches']
    partial_comparison={name:summarize(partial if name=='GSD train BFree' else frames[name],pb,partial_ids) for name in direct}
    # Historical CLIP/AIDE only have printed metrics here, not joinable predictions.
    aliases={'selfcond':'SD2.1_selfconditioned','selfcond_bg':'SD2.1_selfconditioned_origBG',
             'samecat':'SD2.1_inpainted_samecat','samecat_bg':'SD2.1_inpainted_samecat_origBG',
             'diffcat':'SD2.1_inpainted_diffcat','diffcat_bg':'SD2.1_inpainted_diffcat_origBG'}
    historical={}
    for name,fn in [('CLIP ViT-L/14 probe','forensight-clip-bfree.ipynb'),('AIDE original full','forensight-aide-bfree.ipynb')]:
        p=PROJECT/'baselines/NhatNguyen'/fn
        notebook=json.loads(p.read_text(encoding='utf8'))
        detail={}
        for cell in notebook['cells']:
            for o in cell.get('outputs',[]):
                for line in ''.join(o.get('text',[])).splitlines():
                    m=re.match(r'\s*T[1-6]_(\w+)\s+2217\s+([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)%\s+([\d.]+)%',line)
                    if m:
                        case,auc,ap,ba,recall=m.groups()
                        detail[aliases[case]]=dict(auc=float(auc),ap=float(ap),balanced_accuracy=float(ba)/100,fake_recall=float(recall)/100)
        assert len(detail)==6
        historical[name]=dict(source=p.as_posix(),ids=2217,macro_five={k:float(np.mean([detail[b][k] for b in branches])) for k in ['auc','ap','balanced_accuracy','fake_recall']},branches=detail)
    result=dict(completed_branches=branches,full_valid_matched=direct,clean1557_matched=matched,
                threshold_sensitivity_five=sensitivity,partial7024_matched=partial_comparison,
                historical_different_cohort=historical,
                sha256={p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()})
    (ROOT/'baseline_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    def summary(data):
        return table(['Model','Macro BA (%)','Macro AUC (%)','Macro AP (%)','Recall fake (%)','Recall real (%)'],
                     [[name]+[pct(d['macro'][k]) for k in ['balanced_accuracy','auc','ap','fake_recall','real_recall']] for name,d in data.items()])
    lines=['# Đánh giá GSD train BFree so với GSD train Tiny và baseline', '',
           '## Phạm vi so sánh', '',
           '- Kết quả mới: CLIP ViT-L/14+GSD train BFree riêng SD2.1_inpainted_samecat_origBG, best virtual epoch2; checkpoint training paused. Evaluation hoàn tất 5/6 nhánh; diffcat_origBG mới có 7.024/10.304 cặp.',
           '- Bảng chính chỉ dùng cùng 5 nhánh đã hoàn tất, cùng COCO IDs/labels. GSD/CAPD dùng probability >=0,5. Twins/GMM dùng score NLL (cao hơn = fake) và pred_q95 từ ngưỡng real-calibration đã lưu, không dùng ngưỡng probability 0,5 cho NLL.',
           '- Đối chiếu theo ID và nhãn, chưa audit lại hash pixel. GSD giữ score theo context gốc; lọc predictions để so cohort không đồng nghĩa chạy lại GSD với batch context mới.',
           '- GSD BFree đã thấy SD2.1/BFree khi training, còn GSD Tiny/CAPD Tiny chuyển miền từ Tiny sang BFree. Đây là so sánh các hệ thống và mức thích nghi dữ liệu, không phải ablation chỉ thay kiến trúc.', '',
           '## 1. Cùng 10.304 ID, cùng 5 nhánh đủ, ngưỡng 0,5', '',summary(direct),'',
           table(['Nhánh','GSD BFree BA (%)','GSD Tiny BA (%)','CAPD full BA (%)','CAPD window BA (%)'],
                 [[b.replace('SD2.1_','')]+[pct(direct[m]['branches'][b]['balanced_accuracy']) for m in direct] for b in branches]), '',
           '### AUC từng nhánh (không phụ thuộc ngưỡng)', '',
           table(['Nhánh','GSD BFree AUC (%)','GSD Tiny AUC (%)','CAPD full AUC (%)','CAPD window AUC (%)'],
                 [[b.replace('SD2.1_','')]+[pct(direct[m]['branches'][b]['auc']) for m in direct] for b in branches]), '',
           '## 2. Cùng cohort clean 1.557 ID, cùng 5 nhánh', '',
           'Lọc predictions GSD/CAPD về đúng tập ID của Twins native và codec. Hai bộ Twins có ID trùng nhau, và đều là tập con của official-valid 10.304 ID. Mỗi nhánh có 1.557 real + 1.557 fake. Đây là cohort loại overlap với real200K của Twins; không dùng kết quả Twins trên tập lớn có overlap để xếp hạng.', '',summary(matched), '',
           'Các cấu hình Twins ở đây là **native K=1 full GMM** và **codec encoder + native GMM**; chưa đại diện mọi lựa chọn mixed/augmented-GMM. AUC <50% là kết quả theo chiều NLL cao=fake đã định trước. Không tự đảo score sau khi nhìn nhãn để báo kết quả đẹp hơn.', '',
           '## 3. Nhạy với ngưỡng: cùng 5 nhánh trên full valid', '',
           table(['Model','Ngưỡng tối ưu BA','BA (%)','Recall fake (%)','FPR (%)','Recall fake @FPR≤5% (%)'],
                 [[m,f"{d['best_shared']['threshold']:.9g}",pct(d['best_shared']['balanced_accuracy']),pct(d['best_shared']['fake_recall']),pct(d['best_shared']['fpr']),pct(d['at_fpr_05']['fake_recall'])] for m,d in sensitivity.items()]), '',
           'Tối ưu dùng một ngưỡng chung trên 5 nhánh và chính nhãn validation. Đây là diagnostic in-sample, không phải kết quả test độc lập hoặc threshold đã được xác thực. Không so các số này với báo cáo tối ưu 6 nhánh trước đó vì tập nhánh khác nhau.', '',
           '## 4. Nhánh diffcat_origBG chưa đủ: cùng 7.024 ID đã có', '',summary(partial_comparison),'',
           'Đã lọc ba baseline về đúng 7.024 ID của partial mới. Đây là tập đã xử lý, không phải mẫu ngẫu nhiên hay kết quả đầy đủ 10.304 ID. Không cộng vào macro 5 nhánh hoặc diễn giải thành macro đủ 6 nhánh.', '',
           '## 5. CLIP probe và AIDE: số tham khảo khác cohort', '',
           'Nguồn là output đã lưu trong notebook baselines/NhatNguyen/. Hai baseline dùng 2.217 ID chọn từ train_list.csv, còn GSD/CAPD dùng official valid 10.304 ID. Chưa có predictions/manifest joinable trong lần tổng hợp này để lọc cùng ID. Bảng dưới chỉ lấy cùng tên 5 nhánh; số gốc đã làm tròn, không xếp hạng head-to-head với bảng 1/2.', '',
           table(['Model','ID/nhánh','BA trung bình 5 nhánh (%)','AUC trung bình 5 nhánh (%)'],
                 [[m,d['ids'],pct(d['macro_five']['balanced_accuracy']),pct(d['macro_five']['auc'])] for m,d in historical.items()]), '',
           '- CLIP notebook này là ViT-L/14 + probe; không nhầm với OpenCLIP ViT-B-32 probe ở benchmark Tiny JPEG-bias.',
           '- AIDE là original full, không phải AIDE forensic-only ResNet50. Không chuyển kết luận nhân quả trong output notebook thành kết luận của báo cáo này.',
           '- Chưa có kết quả BFree được xác minh cho NPR-ResNet18, ResNet50, NPR patch, AIDE+NPR fusion, Qwen hoặc Spectral-GMM trong các artifact đã kiểm tra; không dùng điểm Tiny/CommFor để điền bảng BFree.', '',
           '## Đánh giá kết quả', '',
           '- Trên cùng 5 nhánh và 10.304 ID, GSD train BFree tăng BA 7,55 điểm phần trăm và AUC 8,02 điểm so với GSD train Tiny; hơn CAPD window 6,61 điểm BA. AUC tăng cho thấy chênh lệch không chỉ do chọn ngưỡng 0,5.',
           '- Tăng tập trung ở nhánh train samecat_origBG (+34,87 điểm BA so với GSD Tiny) và selfconditioned_origBG (+17,22 điểm), nhưng selfconditioned giảm 12,98 điểm. Hai nhánh inpainting không origBG cũng thấp hơn GSD Tiny/CAPD tại ngưỡng 0,5; chưa phải model thắng đồng đều.',
           '- Selfconditioned của model mới vẫn có AUC 99,12% nhưng recall fake ở 0,5 chỉ 61,75%: đây là dấu hiệu cần phân tích calibration/ngưỡng, không thể chỉ nhìn accuracy để nói mất hoàn toàn khả năng phân biệt.',
           '- Với một ngưỡng chung tối ưu trên chính 5 nhánh validation, GSD BFree đạt BA 95,72% tại 0,029367857, recall fake 95,24%, FPR 3,79%. Đây chỉ là diagnostic có thiên lệch lựa chọn; không được đưa lên bảng kết quả test chính như threshold cố định đã xác thực.',
           '- Baseline Twins native/codec với NLL và ngưỡng calibration đã lưu chuyển sang BFree kém trên cohort clean. Kết luận chỉ áp dụng các cấu hình đã đọc; cần kiểm tra score/calibration/domain shift trước khi quy lỗi cho kiến trúc, và không đảo score theo nhãn test.',
           '- Muốn chứng minh generalization cần hoàn tất nhánh cuối, đánh giá Tiny cùng checkpoint GSD-BFree và các generator ngoài SD2.1. Bảng hiện tại chứng minh khả năng thích nghi với BFree một nhánh trên ID không trùng training, chưa chứng minh tốt hơn trên mọi miền.', '',
           '## Nguồn và file chạy', '',
           '- [GSD train BFree → BFree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-train-bfree-test-bfree).',
           '- [GSD train Tiny → BFree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-bfree).',
           '- [CAPD train Tiny → BFree v1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).',
           '- [Twins augmented-GMM evaluation v2](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree), model protocol_native_k1_full.',
           '- [Twins codec evaluation v1](https://www.kaggle.com/datasets/hoangha071005/eval-forensic-twins-codec), protocol native_png.',
           f"- [Script so sánh]({(ROOT/'compare_baselines.py').as_posix()}), [JSON chi tiết/hashes]({(ROOT/'baseline_comparison.json').as_posix()}).",
           *[f"- [{m}]({d['source']})." for m,d in historical.items()], '']
    (ROOT/'BASELINE_COMPARISON.md').write_text('\n'.join(lines),encoding='utf8')
    for group,data in [('FULL5',direct),('CLEAN1557_5',matched),('PARTIAL7024',partial_comparison)]:
        print(group)
        for m,d in data.items():print(m,{k:round(v*100,3) for k,v in d['macro'].items()})
    print('THRESHOLD', {m:{'t':d['best_shared']['threshold'],'BA':d['best_shared']['balanced_accuracy'],'FPR':d['best_shared']['fpr']} for m,d in sensitivity.items()})


if __name__=='__main__':main()

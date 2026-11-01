"""Compare class-conditional score means/medians on matched evaluation IDs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT.parent
PROJECT=REPORTS.parent
NEW=REPORTS/'capd_train_bfree_eval_analysis/source'
OLD=REPORTS/'capd_tiny_bfree_eval_analysis/source'
GSD_NEW=REPORTS/'gsd_train_bfree_test_tiny_analysis/source/gsd_bfree_samecat_origbg_best_tiny9_v1'
GSD_OLD=REPORTS/'gsd_augmented_nine_cases/source/gsd_uploaded_best_variants_v1'
CORE=['CAPD train BFree','CAPD train Tiny','GSD train BFree','GSD train Tiny']
BASE=['CLIP ViT-B/32 probe','NPR-ResNet18','AIDE original full']
paths=[]

def read(p,**kwargs):
    paths.append(p)
    # Load only fields needed for cohort joins and score/metric comparisons.
    # Wide CAPD diagnostic CSVs otherwise retain many unused patch statistics.
    wanted={'sample_id','case','label','prob_fake','relative_path','image_path',
            'fake_probability','model','coco_id','branch','bfree_id','variant',
            'nll','pred_q95','nll_anomaly_score','artifact_primary_prediction'}
    kwargs.setdefault('usecols',lambda c:c in wanted)
    return pd.read_csv(p,**kwargs)

def stat(x):
    a=np.asarray(x,dtype=float);assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)))

def summary(f,probability=True):
    assert f.label.isin([0,1]).all()
    if probability:assert f.score.between(0,1).all()
    out={name:stat(f.loc[f.label==label,'score'] if label is not None else f.score)
         for name,label in [('real',0),('fake',1),('all',None)]}
    out['auc']=float(roc_auc_score(f.label,f.score))
    return out

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def cells(r):
    return [f"{r[label][key]:.6f}" for label in ['real','fake','all'] for key in ['mean','median']]

HEAD=['Model','Real mean','Real median','Fake mean','Fake median','All mean','All median']

def st_table(data):return table(HEAD,[[m]+cells(r) for m,r in data.items()])

def main(return_frames=False):
    tiny={}
    frames=[]
    for p in sorted((NEW/'capd_eval_tiny/b2_dinov3/seed42/tiny_eval').glob('predictions_*.csv')):
        frames.append(read(p)[['sample_id','case','label','prob_fake']])
    tiny[CORE[0]]=pd.concat(frames,ignore_index=True).rename(columns={'sample_id':'id','prob_fake':'score'})
    frames=[]
    for p in sorted((ROOT/'source/capd_train_tiny').glob('predictions_*.csv')):
        frames.append(read(p)[['sample_id','case','label','prob_fake']])
    assert len(frames)==10
    tiny[CORE[1]]=pd.concat(frames,ignore_index=True).rename(columns={'sample_id':'id','prob_fake':'score'})
    for name,folder in [(CORE[2],GSD_NEW),(CORE[3],GSD_OLD)]:
        tiny[name]=read(folder/'predictions.csv')[['sample_id','case','label','prob_fake']].rename(columns={'sample_id':'id','prob_fake':'score'})
    ref=read(GSD_NEW/'predictions.csv')
    ref=ref[ref.case=='native'].set_index('sample_id')
    assert len(ref)==7000 and not ref.relative_path.duplicated().any()
    path_to_id={path:sid for sid,path in ref.relative_path.items()}
    baseline=read(PROJECT/'benchmarks/tiny_jpeg_bias_three_models/tiny_jpeg_bias_three_models_results/predictions/all_predictions_long.csv')
    aliases={'clip_linear_probe':BASE[0],'npr_resnet18':BASE[1],'aide_original_full':BASE[2]}
    for source,name in aliases.items():
        f=baseline[(baseline.model==source)&baseline.case.isin(['raw','png_roundtrip','fake_jpeg96','all_jpeg96'])].copy()
        f['id']=f.relative_path.map(path_to_id);assert f.id.notna().all()
        f['case']=f.case.replace({'raw':'native'})
        tiny[name]=f[['id','case','label','fake_probability']].rename(columns={'fake_probability':'score'})
    tiny_details={};tiny_checks={}
    for name,f in tiny.items():
        assert not f[['case','id']].duplicated().any()
        tiny_details[name]={}
        for case,g in f.groupby('case',sort=False):
            assert len(g)==7000
            assert g.set_index('id').label.sort_index().equals(ref.label.sort_index()),(name,case)
            tiny_details[name][case]=summary(g)
        tiny_checks[name]=dict(cases=list(tiny_details[name]),ids_per_case=7000,matched_labels=True)
    common_core=sorted(set.intersection(*[set(tiny_details[m]) for m in CORE]))
    common_all=sorted(set.intersection(*[set(tiny_details[m]) for m in tiny]))
    assert len(common_core)==9 and common_all==['all_jpeg96','fake_jpeg96','native']
    tiny_native={m:tiny_details[m]['native'] for m in tiny}
    # Older forensic-only AIDE has a joinable native evaluation but no variants.
    forensic_path=PROJECT/'benchmarks/commfor_unseen_three_models/artifacts/tiny_genimage_three_models_results/predictions/aide_forensic_resnet50_predictions.csv'
    forensic=read(forensic_path)
    forensic['relative_path']=forensic.image_path.str.split('tiny-genimage/',n=1).str[-1]
    forensic['id']=forensic.relative_path.map(path_to_id)
    assert forensic.id.notna().all() and len(forensic)==7000
    assert forensic.set_index('id').label.sort_index().equals(ref.label.sort_index())
    forensic['score']=forensic.fake_probability
    tiny_native['AIDE forensic-only ResNet50']=summary(forensic)
    tiny_core9={m:summary(tiny[m][tiny[m].case.isin(common_core)]) for m in CORE}
    tiny_all3={m:summary(f[f.case.isin(common_all)]) for m,f in tiny.items()}

    bf={}
    for name,p in [(CORE[0],NEW/'capd_eval_bfree/b2_dinov3/seed42/bfree_eval/predictions_bfree.csv'),
                   (CORE[1],OLD/'capd_eval_bfree/r2_dinov3/seed42/bfree_eval/predictions_bfree.csv')]:
        f=read(p,dtype={'coco_id':str});parts=[]
        for branch in f.branch.unique():
            if branch=='real':continue
            g=f[f.branch.isin(['real',branch])].copy();g['case']=branch;parts.append(g)
        bf[name]=pd.concat(parts,ignore_index=True)[['coco_id','case','label','prob_fake']].rename(columns={'coco_id':'id','prob_fake':'score'})
    folder=REPORTS/'gsd_train_bfree_test_bfree_analysis/source/gsd_bfree_samecat_origbg_best_bfree_valid_v1'
    a=read(folder/'predictions.csv',dtype={'coco_id':str})
    partial=read(folder/'predictions_SD2.1_inpainted_diffcat_origBG.partial.csv',dtype={'coco_id':str})
    bf[CORE[2]]=pd.concat([a,partial],ignore_index=True)[['coco_id','case','label','prob_fake']].rename(columns={'coco_id':'id','prob_fake':'score'})
    bf[CORE[3]]=read(REPORTS/'gsd_bfree_valid_analysis/source/gsd_bfree_valid_v1/predictions.csv',dtype={'coco_id':str})[['coco_id','case','label','prob_fake']].rename(columns={'coco_id':'id','prob_fake':'score'})
    for m,f in bf.items():
        f['id']=f.id.str.zfill(12);assert not f[['case','id','label']].duplicated().any()
    branches=a.case.unique().tolist();assert len(branches)==5
    bfree_details={};bfree_counts={}
    for case in bf[CORE[0]].case.unique():
        ids=set.intersection(*[set(f.loc[f.case==case,'id']) for f in bf.values()])
        assert len(ids)==(10304 if case in branches else 7024)
        bfree_counts[case]=len(ids);bfree_details[case]={}
        for m,f in bf.items():
            g=f[(f.case==case)&f.id.isin(ids)]
            assert len(g)==2*len(ids)
            for y in [0,1]:assert set(g.loc[g.label==y,'id'])==ids
            bfree_details[case][m]=summary(g)
    bf_five={m:summary(f[f.case.isin(branches)]) for m,f in bf.items()}
    # Clean cohort of baseline Twins: raw NLL kept separate from probabilities.
    refs=REPORTS/'gsd_train_bfree_test_bfree_analysis/baseline_reference_source'
    nll={}
    for name,p in [('Twins native GMM',refs/'twins_native_predictions.csv'),
                   ('Twins codec GMM',refs/'hoangha071005/eval-forensic-twins-codec/forensic_twins_codec_gmm/bfree/artifacts/metrics/native_png/predictions.csv')]:
        f=read(p,dtype={'bfree_id':str});parts=[]
        for branch in branches:
            g=f[f.variant.isin(['real',branch])].copy();g['case']=branch;parts.append(g)
        g=pd.concat(parts,ignore_index=True)[['bfree_id','case','label','nll','pred_q95']].rename(columns={'bfree_id':'id','nll':'score'})
        g['id']=g.id.str.zfill(12);nll[name]=g
    clean=set(nll['Twins native GMM'].id);assert len(clean)==1557 and clean==set(nll['Twins codec GMM'].id)
    bf_clean={m:summary(f[f.case.isin(branches)&f.id.isin(clean)]) for m,f in bf.items()}
    nll_clean={m:summary(f,False) for m,f in nll.items()}
    for m,f in bf.items():
        g=f[f.case.isin(branches)&f.id.isin(clean)]
        assert len(g)==1557*2*5
        assert g.label.value_counts().to_dict()=={0:1557*5,1:1557*5}
    spectral=read(PROJECT/'baselines/d2_fad/spectral_gmm_branch/output/results/spectral_gmm_tinygenimage_fixed_grid_test/predictions/tiny_test_predictions.csv')
    spectral['id']=spectral.relative_path.map(path_to_id)
    assert len(spectral)==7000 and spectral.id.notna().all()
    assert spectral.set_index('id').label.sort_index().equals(ref.label.sort_index())
    spectral['score']=spectral.nll_anomaly_score
    spectral_nll=summary(spectral,False)
    if return_frames:
        return dict(tiny=tiny,bfree=bf,nll=nll,forensic=forensic,spectral=spectral,
                    common_core=common_core,common_all=common_all,branches=branches,clean=clean)
    result=dict(score='prob_fake in [0,1], conditional on true real/fake labels; CAPD full pooling.',
                tiny_native=tiny_native,tiny_cases=tiny_details,tiny_common_nine_core=tiny_core9,
                tiny_common_three_all_models=tiny_all3,tiny_checks=tiny_checks,
                bfree_common_five=bf_five,bfree_per_branch_matched=bfree_details,bfree_ids_per_branch=bfree_counts,
                bfree_clean1557_probabilities=bf_clean,bfree_clean1557_nll=nll_clean,tiny_spectral_nll=spectral_nll,
                sources_sha256={p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)})
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    lines=['# So sánh mean/median score CAPD, GSD và baseline', '',
           '**Score trong bảng chính là prob_fake (0–1), không phải accuracy/recall. Real/fake là nhãn thật. CAPD dùng full pooling.**', '',
           'Mỗi số mean/median tính trực tiếp trên prediction thuộc nhóm. Không lấy trung bình các median case; không thay ngưỡng để thay đổi score. Các model có calibration và preprocessing khác nhau, nên score tự tin hơn không tự chứng minh model tốt hơn. Không convert NLL sang probability bằng sigmoid tùy ý.', '',
           '## 1. Tiny native: cùng 7.000 ảnh và nhãn', '',st_table(tiny_native), '',
           'Mỗi model có3.500 real +3.500 fake. Các baseline được join bằng relative_path sang sample_id của GSD/CAPD; ID và nhãn khớp đủ7.000. CLIP ở đây là OpenCLIP ViT-B/32 probe; không phải CLIP ViT-L/14 probe trong notebook BFree NhatNguyen. AIDE full và AIDE forensic-only ResNet50 là hai model khác nhau; bản forensic-only chỉ có native ở lượt so sánh này.', '',
           '## 2. Tiny: cùng9 case có đủ CAPD/GSD', '',st_table(tiny_core9), '',
           'Gồm '+', '.join(common_core)+'. Mỗi lớp31.500 score từ9 biến đổi của cùng3.500 ảnh nguồn; không phải31.500 ảnh độc lập. Không tính PNG roundtrip vì GSD không có case này.', '',
           '## 3. Tiny: cùng3 case có đủ cả ba baseline', '',st_table(tiny_all3), '',
           'Chỉ gộp native, fake_jpeg96, all_jpeg96 cho mọi model:10.500 score/lớp. Không gộp9/10 case CAPD/GSD rồi so với3 case baseline. Tên phép biến đổi chung chưa phải xác minh pixel biến đổi byte-identical; pipeline tiền xử lý model khác nhau.', '',
           '## 4. Tiny: bảng so sánh từng case', '']
    for case in sorted(set.union(*[set(x) for x in tiny_details.values()])):
        lines+=['### '+case,'',st_table({m:d[case] for m,d in tiny_details.items() if case in d}),'']
    lines+=['## 5. BFree: cùng5 nhánh đủ và10.304 ID', '',st_table(bf_five), '',
            'Mỗi model được tính trên5 cặp real/fake cân bằng:51.520 score/lớp. Real CAPD được dùng lại giữa các nhánh; GSD có score real theo batch context. Giữ nguyên các score context, không tự lấy trung bình theo ID trước khi tính median.', '',
            '## 6. BFree: theo nhánh, cùng ID giữa4 model', '']
    for case,d in bfree_details.items():
        lines+=['### '+case+' — '+str(bfree_counts[case])+' ID/lớp','',st_table(d),'']
    lines+=['Nhánh diffcat_origBG chỉ so trên7.024 ID đã có của GSD train BFree, không gọi là nhánh hoàn tất và không đưa vào bảng gộp5 nhánh.', '',
            '## 7. Baseline Twins: cùng cohort clean1.557 ID ×5 nhánh', '',
            '### Probability của CAPD/GSD', '',st_table(bf_clean), '',
            '### NLL nguyên gốc của Twins/GMM — KHÁC THANG với bảng trên', '',st_table(nll_clean), '',
            'NLL có thể âm; theo policy lưu trong pipeline, NLL cao hơn dự đoán fake. Không so khoảng cách mean NLL với khoảng cách probability. Các model trên được lọc về đúng cùng1.557 ID của baseline; đây không phải cohort10.304 ID.', '',
            '## 8. Spectral-GMM trên Tiny: score NLL riêng', '',
            st_table({'Spectral/MFM-GMM':spectral_nll}), '',
            'Đối chiếu đủ7.000 relative paths/labels như Tiny native, nhưng pipeline Spectral nguồn có áp dụng JPEG (jpeg_applied) và fixed-grid tile. Đây không phải probability hay đúng pixel pipeline native của CAPD/GSD; giữ bảng NLL riêng, không gộp cùng các phép biến đổi Tiny ở mục2/3.', '',
            '## 9. Baseline chưa đủ prediction để báo mean/median BFree', '',
            '- CLIP ViT-L/14 probe và AIDE full trong notebook NhatNguyen có metrics/mean score một số nhóm, nhưng chưa có prediction CSV hoặc median đầy đủ để tính thống kê BFree cùng cohort ở đây. Không lấy score Tiny thay cho BFree hoặc suy median từ AUC/accuracy.',
            '- NPR có prediction Tiny đã đối chiếu, chưa có prediction BFree được xác minh. Không điền số giả cho baseline còn thiếu.', '',
            '## File local', '',
            f"- [compare_scores.py]({(ROOT/'compare_scores.py').as_posix()}) — tải nguồn CAPD train Tiny ở bước chuẩn bị, script này chỉ đọc CSV local và lập bảng.",
            f"- [comparison.json]({(ROOT/'comparison.json').as_posix()}) — full-precision statistics, counts, AUC tham khảo và SHA256 nguồn.", '']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf8')
    for title,data in [('TINY NATIVE',tiny_native),('TINY COMMON9',tiny_core9),('BFREE COMMON5',bf_five),('BFREE NLL CLEAN',nll_clean)]:
        print(title)
        for m,r in data.items():print(m,*cells(r))

if __name__=='__main__':main()

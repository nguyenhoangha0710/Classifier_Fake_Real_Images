"""Threshold sweeps and equal-case mean/median metrics for CAPD Tiny/BFree."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
TINY=SRC/'capd_eval_tiny/b2_dinov3/seed42/tiny_eval'
BFREE=SRC/'capd_eval_bfree/b2_dinov3/seed42/bfree_eval'
GRID=[.5,.3,.2,.1,.05,.03,.02,.01,.005,.001]
COLS={'lse_full':'prob_fake','lse_window':'prob_fake_window','lse_inner':'prob_fake_inner','lse_nopad':'prob_fake_nopad'}
KEYS=['accuracy','balanced_accuracy','auc','ap','fake_recall','real_recall','fake_precision','real_precision','fake_f1','real_f1','fpr']
DISPLAY=['balanced_accuracy','auc','ap','fake_recall','real_recall','fake_precision','real_precision','fake_f1','real_f1']
HEAD=['Acc./BA','AUC','AP','Recall fake','Recall real','Precision fake','Precision real','F1 fake','F1 real']

def measure(real,fake,t,auc,ap):
    tn=int(np.searchsorted(real,t,side='left'));fn=int(np.searchsorted(fake,t,side='left'))
    fp=len(real)-tn;tp=len(fake)-fn
    return dict(n=len(real)+len(fake),threshold=float(t),tn=tn,fp=fp,fn=fn,tp=tp,
                accuracy=(tn+tp)/(len(real)+len(fake)),balanced_accuracy=.5*(tn/len(real)+tp/len(fake)),auc=auc,ap=ap,
                fake_recall=tp/len(fake),real_recall=tn/len(real),fpr=fp/len(real),
                fake_precision=tp/(tp+fp) if tp+fp else 0.,real_precision=tn/(tn+fn) if tn+fn else 0.,
                fake_f1=2*tp/(2*tp+fp+fn),real_f1=2*tn/(2*tn+fp+fn))

def best(groups):
    real=np.sort(np.concatenate([g[0] for g in groups.values()]))
    fake=np.sort(np.concatenate([g[1] for g in groups.values()]))
    ts=np.unique(np.concatenate([real,fake,[0.,1.]]))
    correct=np.searchsorted(real,ts,side='left')+len(fake)-np.searchsorted(fake,ts,side='left')
    return float(ts[np.flatnonzero(correct==correct.max())[-1]])

def summarize(rows):
    return {stat:{k:float(fn([r[k] for r in rows])) for k in KEYS}
            for stat,fn in [('mean',np.mean),('median',np.median)]}

def evaluate(groups,metadata,t):
    rows={case:measure(*arrays,t,metadata[case]['auc'],metadata[case]['ap']) for case,arrays in groups.items()}
    return dict(threshold=t,cases=rows,**summarize(list(rows.values())))

def pct(v):return f'{100*v:.2f}'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def detail_table(e):
    return table(['Case/nhánh']+HEAD,[[c]+[pct(r[k]) for k in DISPLAY] for c,r in e['cases'].items()]+
                 [[stat]+[pct(e[stat][k]) for k in DISPLAY] for stat in ['mean','median']])

def main():
    original=json.loads((ROOT/'analysis.json').read_text(encoding='utf8'))
    for name,digest in original['checks']['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    tg={p:{} for p in ['lse_full','lse_inner','lse_nopad']};tm={p:{} for p in tg}
    for case in dict.fromkeys(r['case'] for r in original['tiny']):
        f=pd.read_csv(TINY/f'predictions_{case}.csv')
        for p in tg:
            tg[p][case]=tuple(np.sort(f.loc[f.label==y,COLS[p]].to_numpy()) for y in [0,1])
            tm[p][case]=next(r for r in original['tiny'] if r['case']==case and r['generator']=='pooled' and r['pooling']==p)
    bf=pd.read_csv(BFREE/'predictions_bfree.csv',dtype={'coco_id':str})
    bg={p:{} for p in COLS};bm={p:{} for p in COLS}
    for branch in dict.fromkeys(r['branch'] for r in original['bfree']):
        f=bf[bf.branch.isin(['real',branch])]
        for p in bg:
            bg[p][branch]=tuple(np.sort(f.loc[f.label==y,COLS[p]].to_numpy()) for y in [0,1])
            bm[p][branch]=next(r for r in original['bfree'] if r['branch']==branch and r['pooling']==p)
    opt={p:dict(best_shared=best(g),best_native=best({'native':g['native']})) for p,g in tg.items()}
    primary_t=opt['lse_full']['best_shared']
    thresholds=sorted(set(GRID+[primary_t,opt['lse_full']['best_native']]),reverse=True)
    results={}
    for name,groups,metadata in [('tiny',tg,tm),('bfree',bg,bm)]:
        results[name]={}
        for pool,g in groups.items():
            results[name][pool]=[evaluate(g,metadata[pool],t) for t in thresholds]
            baseline=next(r for r in results[name][pool] if r['threshold']==.5)
            for case,r in baseline['cases'].items():
                for k in KEYS:
                    if k in metadata[pool][case]:assert abs(r[k]-metadata[pool][case][k])<1e-12,(name,pool,case,k)
    def select(dataset,pool,t):return next(r for r in results[dataset][pool] if r['threshold']==t)
    combined=[]
    for t in thresholds:
        rows=list(select('tiny','lse_full',t)['cases'].values())+list(select('bfree','lse_full',t)['cases'].values())
        combined.append(dict(threshold=t,**summarize(rows)))
    out=dict(thresholds=thresholds,tiny_optima=opt,tiny_shared_full_threshold=primary_t,
             results=results,combined_sixteen_cases_equal_weight=combined,
             averaging='Arithmetic mean / median of per-case metrics; Tiny 10 cases, BFree 6 branches, combined 16 equal-weight cases. Not pooled predictions.',
             selection='Optima use Tiny test labels; post-hoc diagnostic, not independent calibration.')
    (ROOT/'threshold_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    lines=['# CAPD train BFree: ngưỡng Tiny và mean/median Tiny–BFree', '',
           'Nguồn: [dataset v1](https://www.kaggle.com/datasets/hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree). Model b2_dinov3, best virtual epoch2. Không train/inference lại; giữ nguyên các prediction đã audit.', '',
           '## Quy ước', '',
           '- Tiny = 10 case, mỗi case 7.000 ảnh; BFree = 6 nhánh, mỗi nhánh 20.608 ảnh cân bằng. Tất cả tỷ lệ trong bảng là %. Fake nếu score probability >= threshold.',
           '- Mean = trung bình số học của chỉ số từng case/nhánh; median = trung vị của chỉ số từng case/nhánh. Tính F1/precision từng case trước rồi mới lấy mean/median, không tính lại F1 từ recall/precision trung bình.',
           '- Accuracy và BA bằng nhau ở từng case cân bằng, nên báo chung một cột. AUC/AP lấy từ bảng đã tính lại từ predictions, không đổi theo ngưỡng.',
           '- Bảng chính dùng full pooling. Pooling window/inner/nopad được ghi riêng; không trộn pooling vào mean/median. Precision bằng0 khi không có dự đoán lớp tương ứng, nên precision của các chẩn đoán gần như chỉ dự đoán một lớp cần đọc cùng recall.',
           '- Có thêm bảng gộp16 case full-pooling với trọng số1/16 mỗi case (Tiny10/16, BFree6/16), không phải cân bằng hai dataset và không phải pooled accuracy trên toàn bộ prediction. Các biến đổi Tiny tương quan và real BFree được dùng lại, không phải các ảnh độc lập.', '',
           '## 1. Quét ngưỡng Tiny full pooling: mean và median trên 10 case', '',
           table(['Ngưỡng','Mean BA','Median BA','Mean recall fake','Median recall fake','Mean recall real','Median recall real','Mean precision fake','Median precision fake','Mean F1 fake','Median F1 fake'],
                 [[f"{r['threshold']:.10g}"]+[pct(r[s][k]) for k in ['balanced_accuracy','fake_recall','real_recall','fake_precision','fake_f1'] for s in ['mean','median']] for r in results['tiny']['lse_full']]), '',
           f"Ngưỡng chung tối ưu mean BA Tiny10 (full): **{primary_t:.10g}**. Ngưỡng tối ưu riêng native full: **{opt['lse_full']['best_native']:.10g}**. Các ngưỡng này được chọn bằng nhãn Tiny test: chỉ là phân tích hậu nghiệm.", '',
           '### BA từng case Tiny theo ngưỡng', '',
           table(['Case']+[f'{t:g}' for t in GRID],[[c]+[pct(select('tiny','lse_full',t)['cases'][c]['balanced_accuracy']) for t in GRID] for c in tg['lse_full']]), '',
           '## 2. Mean/median chỉ số đầy đủ: full pooling', '']
    for t in [.5,primary_t]:
        lines+=['### Ngưỡng '+f'{t:.10g}', '',
                table(['Dataset','Thống kê']+HEAD,
                      [[name,stat]+[pct(select(name,'lse_full',t)[stat][k]) for k in DISPLAY] for name in ['tiny','bfree'] for stat in ['mean','median']]+
                      [['combined16',stat]+[pct(next(r for r in combined if r['threshold']==t)[stat][k]) for k in DISPLAY] for stat in ['mean','median']]),'']
    lines+=['## 3. Tiny từng case: ngưỡng0,5 và ngưỡng tối ưu mean Tiny10', '']
    for t in [.5,primary_t]:lines+=['### '+f'{t:.10g}','',detail_table(select('tiny','lse_full',t)),'']
    lines+=['## 4. BFree từng nhánh ở cùng hai ngưỡng', '',
            'Ngưỡng tối ưu Tiny được áp dụng nguyên sang BFree, không tìm tối ưu bằng nhãn BFree trong báo cáo này.', '']
    for t in [.5,primary_t]:lines+=['### '+f'{t:.10g}','',detail_table(select('bfree','lse_full',t)),'']
    lines+=['## 5. Từng pooling: mean/median và toàn bộ ngưỡng', '',
            'Tiny có full/inner/nopad; BFree có full/window/inner/nopad. Các pooling chẩn đoán có thang probability khác nhau nên cùng ngưỡng số học có thể cho recall rất khác; không suy ra nhân quả chỉ từ chênh lệch này.', '']
    for name in results:
        for pool,sweeps in results[name].items():
            lines+=['### '+name+' — '+pool,'',
                    table(['Ngưỡng','Thống kê']+HEAD,[[f"{r['threshold']:.10g}",stat]+[pct(r[stat][k]) for k in DISPLAY] for r in sweeps for stat in ['mean','median']]),'']
    lines+=['## 6. Toàn bộ chỉ số Tiny theo từng case và từng ngưỡng (full)', '']
    for e in results['tiny']['lse_full']:lines+=['### '+f"{e['threshold']:.10g}",'',detail_table(e),'']
    lines+=['## File chạy', '',
            f"- [threshold_summary.py]({(ROOT/'threshold_summary.py').as_posix()}).",
            f"- [threshold_summary.json]({(ROOT/'threshold_summary.json').as_posix()}) chứa toàn bộ metrics/confusion counts từng ngưỡng, từng pooling, mean và median.",
            '- Đã kiểm tra SHA256 toàn bộ prediction với lần audit trước; các chỉ số tại0,5 khớp bảng đã tính lại.', '']
    (ROOT/'THRESHOLD_SUMMARY.md').write_text('\n'.join(lines),encoding='utf8')
    print('TINY OPTIMA',opt)
    for t in [.5,.1,.05,.03,.01,primary_t]:
        print('THRESHOLD',t)
        for name in ['tiny','bfree']:
            e=select(name,'lse_full',t)
            for stat in ['mean','median']:print(name,stat,{k:round(e[stat][k]*100,3) for k in DISPLAY})
    print('NATIVE OPT/SHARED')
    for t in [opt['lse_full']['best_native'],primary_t]:
        r=select('tiny','lse_full',t)['cases']['native'];print(t,{k:round(r[k]*100,3) for k in DISPLAY})

if __name__=='__main__':main()

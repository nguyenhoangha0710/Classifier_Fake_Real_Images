"""Mean/median of prediction probabilities and raw logits, not evaluation metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
COLS={'lse_full':('prob_fake','z'),'lse_window':('prob_fake_window','z_window'),
      'lse_inner':('prob_fake_inner','z_inner'),'lse_nopad':('prob_fake_nopad','z_nopad')}

def stats(values):
    a=np.asarray(values,dtype=float);assert len(a)>0 and np.isfinite(a).all()
    return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))

def summarize(f,prob,logit):
    assert f[prob].between(0,1).all()
    return {kind:{label:stats(f.loc[f.label==y,column] if y is not None else f[column])
                  for label,y in [('real',0),('fake',1),('all',None)]}
            for kind,column in [('prob_fake',prob),('logit',logit)]}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def cells(summary,kind='prob_fake'):
    return [f"{summary[kind][label][key]:.6f}" for label in ['real','fake','all'] for key in ['mean','median']]

def main():
    prior=json.loads((ROOT/'analysis.json').read_text(encoding='utf8'))
    for name,digest in prior['checks']['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    cases=list(dict.fromkeys(r['case'] for r in prior['tiny']))
    tiny={p:{} for p in ['lse_full','lse_inner','lse_nopad']};bf={p:{} for p in COLS}
    tinyframes=[]
    for case in cases:
        f=pd.read_csv(SRC/f'capd_eval_tiny/b2_dinov3/seed42/tiny_eval/predictions_{case}.csv')
        assert len(f)==7000 and not f.sample_id.duplicated().any()
        for p in tiny:tiny[p][case]=summarize(f,*COLS[p])
        tinyframes.append(f)
    all_tiny=pd.concat(tinyframes,ignore_index=True);assert len(all_tiny)==70000
    b=pd.read_csv(SRC/'capd_eval_bfree/b2_dinov3/seed42/bfree_eval/predictions_bfree.csv')
    assert len(b)==72128 and not b.sample_id.duplicated().any()
    branches=[v for v in b.branch.unique() if v!='real']
    pairs=[]
    for branch in branches:
        f=b[b.branch.isin(['real',branch])];assert len(f)==20608
        for p in bf:bf[p][branch]=summarize(f,*COLS[p])
        pairs.append(f)
    aggregate={}
    for name,frame,pools in [('tiny_10_case_views',all_tiny,tiny),('bfree_unique_images',b,bf),
                             ('bfree_six_balanced_pairs',pd.concat(pairs,ignore_index=True),bf)]:
        aggregate[name]={p:summarize(frame,*COLS[p]) for p in pools}
    out=dict(score_definition='prob_fake in [0,1], higher = more fake; reported real/fake groups are ground-truth labels, not predicted labels.',
             tiny=tiny,bfree=bf,aggregate=aggregate,
             source_predictions_hashes=prior['checks']['sha256'])
    (ROOT/'score_statistics.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    headers=['Case/nhánh','Real mean','Real median','Fake mean','Fake median','All mean','All median']
    lines=['# CAPD train BFree: mean và median score dự đoán', '',
           'Score chính là **prob_fake** trên thang0–1, không phải accuracy/recall/F1. Real/fake ở đây là **nhãn thật** của ảnh. Score càng gần1 càng nghiêng về fake; ảnh real tốt nên có score thấp.', '',
           'Tính trực tiếp từ prediction CSV; kiểm tra SHA256 nguồn khớp lần audit trước. Thay ngưỡng chỉ thay nhãn dự đoán, **không làm mean/median score thay đổi**. Số trong bảng được làm tròn6 chữ số thập phân, JSON giữ độ chính xác số thực.', '',
           '## 1. Tiny, full pooling', '',table(headers,[[c]+cells(r) for c,r in tiny['lse_full'].items()]), '',
           'Mỗi case có3.500 real +3.500 fake; All là gộp7.000 score của case đó. Các case là biến đổi của cùng ảnh nguồn.', '',
           '## 2. BFree, full pooling', '',table(headers,[[c]+cells(r) for c,r in bf['lse_full'].items()]), '',
           'Mỗi nhánh so10.304 real +10.304 fake. Score real full pooling là cùng một tập được dùng lại cho6 nhánh.', '',
           '## 3. Score gộp theo dataset', '',
           table(headers,[[name]+cells(pools['lse_full']) for name,pools in aggregate.items()]), '',
           '- tiny_10_case_views:70.000 score,35.000 real/35.000 fake; cùng7.000 ảnh nguồn qua10 biến đổi, không phải70.000 ảnh độc lập.',
           '- bfree_unique_images:72.128 score,10.304 real/61.824 fake; real chỉ đếm một lần. All mean/median chịu ảnh hưởng tỷ lệ real:fake1:6.',
           '- bfree_six_balanced_pairs:123.648 score theo6 cặp đánh giá cân bằng; score real được đếm lại6 lần,61.824 real/61.824 fake. Không xem là123.648 ảnh độc lập.',
           '- Median tính trực tiếp trên tất cả score thuộc nhóm, không lấy trung bình hoặc trung vị của các median case.', '',
           '## 4. Các pooling và raw logit z', '',
           'Full là kết quả chính. Window/inner/nopad là các cách gộp score khác nhau; không trộn các pooling. Logit z là score trước sigmoid, có thể âm/dương và không cùng thang với prob_fake.', '']
    for name,data in [('Tiny',tiny),('BFree',bf)]:
        for pool,rows in data.items():
            for kind in ['prob_fake','logit']:
                lines+=['### '+name+' — '+pool+' — '+kind,'',table(headers,[[c]+cells(r,kind) for c,r in rows.items()]),'']
    lines+=['## File chạy', '',f"- [score_statistics.py]({(ROOT/'score_statistics.py').as_posix()}).",
            f"- [score_statistics.json]({(ROOT/'score_statistics.json').as_posix()}) chứa count, mean, median, p05/p95 cho score và logit, từng pooling/nhãn.", '']
    (ROOT/'SCORE_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    for name,data in [('TINY',tiny['lse_full']),('BFREE',bf['lse_full'])]:
        print(name)
        for case,r in data.items():print(case,*cells(r))
    print('AGGREGATES')
    for name,data in aggregate.items():print(name,*cells(data['lse_full']))

if __name__=='__main__':main()

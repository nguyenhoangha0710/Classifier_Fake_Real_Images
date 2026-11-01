"""Render saved CAPD patch logits without rerunning inference.

Usage: python baselines/capd/heatmap/render_heatmaps.py
"""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PRED = ROOT / 'reports/capd_train_bfree_eval_analysis/source/capd_eval_tiny/b2_dinov3/seed42/tiny_eval/predictions_all_jpeg80.csv'
if not DEFAULT_PRED.is_file():
    DEFAULT_PRED = HERE / 'predictions_reference.csv'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'smaps_all_jpeg80')
    parser.add_argument('--predictions',type=Path,default=DEFAULT_PRED)
    parser.add_argument('--output',type=Path,default=HERE/'rendered')
    parser.add_argument('--skip-individual',action='store_true',help='Refresh overview/index only; reuse individual PNGs already rendered')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    individual=args.output/'individual';individual.mkdir(exist_ok=True)
    f=pd.read_csv(args.predictions).set_index('sample_id',verify_integrity=True)
    items=[]
    for p in sorted(args.input.glob('*.npy')):
        raw=np.asarray(np.load(p,allow_pickle=False),dtype=np.float64)
        if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError(f'Invalid 2D heatmap: {p}')
        if p.stem not in f.index:raise ValueError(f'No prediction for {p.stem}')
        row=f.loc[p.stem]
        if not np.isclose(raw.max(),row.s_max,atol=1e-6,rtol=0):raise ValueError(f'Patch maximum does not match predictions: {p.stem}')
        if raw.shape!=(int(row.grid_h),int(row.grid_w)):raise ValueError(f'Grid does not match: {p.stem}')
        label=int(row.label);prob=float(row.prob_fake);pred=int(prob>=.5)
        category=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')
        items.append(dict(id=p.stem,raw=raw,map=sigmoid(raw),label=label,pred=pred,
            prob=prob,category=category,generator=str(row.generator),path=str(p)))
    if not items:raise ValueError('No .npy files found')

    def title(item,short=True):
        name=('Fake' if item['label'] else 'Real')
        pred=('Fake' if item['pred'] else 'Real')
        return f"{item['category']} · {name} → {pred} · {item['generator']}\nP(fake ảnh)={item['prob']:.3f} · {item['id'][:12]}"

    def draw(ax,item):
        im=ax.imshow(item['map'],cmap='inferno',vmin=0,vmax=1,interpolation='nearest',aspect='equal')
        ax.set_title(title(item),fontsize=8,color='#9c2e20' if item['category'] in ['FN','FP'] else '#263445')
        ax.set_xticks([]);ax.set_yticks([])
        return im

    for item in items:
        if args.skip_individual:
            if not (individual/f"{item['id']}.png").is_file():raise FileNotFoundError(item['id'])
            continue
        fig,ax=plt.subplots(figsize=(5.8,5),layout='constrained')
        im=draw(ax,item);ax.set_xlabel(f"Lưới patch: {item['raw'].shape[0]} × {item['raw'].shape[1]}")
        fig.colorbar(im,ax=ax,label='Sigmoid của logit patch (0–1)',shrink=.8)
        fig.savefig(individual/f"{item['id']}.png",dpi=160)
        plt.close(fig)

    def panel(subset,dest,title_text,cols=4,dpi=180):
        rows=int(np.ceil(len(subset)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(cols*3.2,rows*3.25+.6),squeeze=False,layout='constrained')
        for ax in axes.flat:ax.set_visible(False)
        for ax,item in zip(axes.flat,subset):ax.set_visible(True);im=draw(ax,item)
        fig.colorbar(im,ax=list(axes.flat),label='Sigmoid(logit patch) · cùng thang màu 0–1',shrink=.7,pad=.015)
        fig.suptitle(title_text,fontsize=14,fontweight='bold',y=1.04)
        fig.supxlabel('CAPD train B-Free → Tiny JPEG80 · Ngưỡng ảnh 0,5 · Sáng: điểm fake patch cao\nĐiểm patch dùng để trực quan hóa, chưa phải xác suất định vị được hiệu chỉnh.',fontsize=9)
        fig.savefig(dest,dpi=dpi,bbox_inches='tight')
        plt.close(fig)

    # Deterministic selection spanning true negatives, false negatives, true positives.
    # This is a qualitative selection, not an unbiased evaluation sample.
    selected=[]
    for category in ['TN','FN','TP','FP']:
        group=sorted([x for x in items if x['category']==category],key=lambda x:x['prob'])
        if group:
            count=min({'TN':1,'FN':2,'TP':2,'FP':1}[category],len(group))
            for idx in np.unique(np.linspace(0,len(group)-1,count,dtype=int)):
                selected.append(group[idx])
    panel(selected,args.output/'preview_selected.png','Heatmap CAPD: ví dụ đúng và sai được chọn theo nhóm',cols=3)
    for start in range(0,len(items),24):
        panel(items[start:start+24],args.output/f'contact_sheet_{start//24+1:02d}.png',
            f'Heatmap CAPD · mẫu {start+1}–{min(start+24,len(items))}/{len(items)}',cols=4,dpi=140)

    # Raw scores versus display transform, preserving the original patch grid.
    chosen=next((x for x in selected if x['category']=='FN'),selected[0])
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,data,cmap,label in [(axes[0],chosen['raw'],'coolwarm','Logit patch gốc'),(axes[1],chosen['map'],'inferno','Sigmoid(logit patch)')]:
        kw={'vmin':0,'vmax':1} if ax is axes[1] else {}
        im=ax.imshow(data,cmap=cmap,interpolation='nearest',**kw);ax.set_title(label);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(title(chosen),fontsize=11)
    fig.savefig(args.output/'raw_vs_sigmoid.png',dpi=200);plt.close(fig)

    rows=[dict({k:v for k,v in x.items() if k not in ['raw','map']},grid_h=x['raw'].shape[0],grid_w=x['raw'].shape[1],
        logit_min=float(x['raw'].min()),logit_max=float(x['raw'].max())) for x in items]
    pd.DataFrame(rows).to_csv(args.output/'index.csv',index=False,encoding='utf-8-sig')
    counts={c:sum(x['category']==c for x in items) for c in ['TP','TN','FP','FN']}
    summary={'input':str(args.input.resolve()),'predictions':str(args.predictions.resolve()),'n':len(items),
        'categories':counts,'verification':'All sample IDs, patch grid shapes and maximum logits match the supplied predictions.',
        'display':'sigmoid(raw patch logits), shared [0,1] color scale, nearest interpolation',
        'selection':[x['id'] for x in selected],
        'limitations':'Saved subset only; not all test images. Patch sigmoid is not calibrated localization probability; no source-image overlay.'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for x in items:
        cards.append(f'<figure><a href="individual/{x["id"]}.png"><img loading="lazy" src="individual/{x["id"]}.png"></a><figcaption>{html.escape(x["id"])}<br>{x["category"]} · {html.escape(x["generator"])} · P(fake)={x["prob"]:.4f}</figcaption></figure>')
    page='<!doctype html><meta charset="utf-8"><title>CAPD heatmaps</title><style>body{font:16px system-ui;margin:24px;background:#eef2f6;color:#263445}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}figure{margin:0;background:white;padding:12px;border-radius:8px}img{width:100%}figcaption{font-size:12px;overflow-wrap:anywhere}</style><h1>CAPD train B-Free → Tiny JPEG80</h1><p>200 bản đồ patch, ngưỡng ảnh 0,5. Sáng: điểm fake patch cao. TP/TN: đúng; FP: real nhận nhầm fake; FN: fake bị bỏ sót. Đây là tập con được lưu, không đại diện toàn bộ test.</p><div class="grid">'+''.join(cards)+'</div>'
    (args.output/'gallery.html').write_text(page,encoding='utf-8')
    print(json.dumps({'rendered':len(items),'categories':counts,'output':str(args.output.resolve())},ensure_ascii=False))

if __name__=='__main__':main()

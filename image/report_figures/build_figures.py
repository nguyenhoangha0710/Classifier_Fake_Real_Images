"""Report figures from saved checkpoint/evaluation artifacts; no model rerun."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from sklearn.metrics import roc_curve, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
R = ROOT / 'reports'
inputs = set()
def source(p):
    p = ROOT / p if not Path(p).is_absolute() else Path(p)
    inputs.add(p)
    return p
def js(p):
    return json.loads(source(p).read_text(encoding='utf-8'))
def csv(p, **kw):
    return pd.read_csv(source(p), **kw)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.labelcolor': '#263445',
    'text.color': '#263445', 'axes.edgecolor': '#b2bcc7',
    'svg.fonttype': 'none', 'pdf.fonttype': 42})
BLUE, ORANGE = '#176B9B', '#D57732'
records = []
def save(fig, stem, caption, sources_note):
    fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    records.append((stem, caption, sources_note))
    plt.close(fig)
def grid(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, alpha=.22)

d = js('reports/score_model_comparison/metrics_comparison.json')

# 1: Report overview; branch means must not be confused with pooled scores.
settings = [('Tiny → Tiny', 'Tiny', 'tiny'), ('Tiny → B-Free', 'Tiny', 'bf'),
            ('B-Free → Tiny', 'BFree', 'tiny'), ('B-Free → B-Free', 'BFree', 'bf')]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
for ax, metric, label in zip(axes, ['balanced_accuracy','fake_recall'], ['Balanced Accuracy (%)','Fake recall (%)']):
    for i, model in enumerate(['CAPD','GSD']):
        vals=[]
        for _, train, test in settings:
            key=f'{model} train {train}'
            row=d['tiny_native'][key] if test=='tiny' else d['bfree_common_five'][key]['mean']
            vals.append(100*row[metric])
        bars=ax.bar(np.arange(4)+(i-.5)*.36, vals, .36, label=model, color=[BLUE,ORANGE][i])
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
    ax.set_xticks(range(4), [s[0] for s in settings], rotation=14)
    ax.set_ylim(0,110); ax.set_ylabel(label); grid(ax); ax.legend(loc='lower left')
fig.suptitle('CAPD và GSD: so sánh theo miền huấn luyện → đánh giá')
fig.supxlabel('Ngưỡng 0,5 · Tiny: 7.000 ảnh · B-Free: macro 5 nhánh, có nhánh cùng miền train',fontsize=10)
save(fig,'01_overview','So sánh BA và fake recall của CAPD và GSD trong bốn thiết lập huấn luyện–đánh giá. Kết quả B-Free là trung bình năm nhánh, bao gồm inpainted_samecat_origBG thuộc miền huấn luyện B-Free.','metrics_comparison.json: tiny_native, bfree_common_five')

# 2: Per-branch matched evaluation, including explicitly partial sixth case.
keys=list(d['bfree_branch_metrics']['CAPD train BFree'])
names=[k.replace('SD2.1_','') for k in keys]+['inpainted_diffcat_origBG *']
names[3]+=' [train]'
fig, axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
for ax, metric, title in zip(axes,['balanced_accuracy','fake_recall'],['Balanced Accuracy (%)','Fake recall (%)']):
    for i,model in enumerate(['CAPD','GSD']):
        m=f'{model} train BFree'
        rows=[d['bfree_branch_metrics'][m][k] for k in keys]+[d['bfree_partial7024'][m]]
        bars=ax.barh(np.arange(6)+(i-.5)*.34,[100*r[metric] for r in rows],.34,color=[BLUE,ORANGE][i],label=model)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_yticks(range(6),names if ax is axes[0] else ['']*6)
    ax.set_ylim(6,-.6); ax.set_xlim(0,113); ax.set_xlabel(title); grid(ax,'x');ax.legend(loc='lower left',ncol=2,fontsize=9)
fig.suptitle('Huấn luyện trên B-Free: kết quả theo từng nhánh')
fig.supxlabel('Ngưỡng 0,5 · [train]: cùng kiểu nhánh huấn luyện · *: 7.024 ID/lớp chung, kết quả một phần',fontsize=10)
save(fig,'02_bfree_branches','BA và fake recall theo nhánh khi huấn luyện trên B-Free. Năm nhánh đầu được đối chiếu trên 10.304 ID mỗi lớp; nhánh thứ sáu chỉ so sánh trên 7.024 ID mỗi lớp đã có kết quả ở cả hai mô hình và không đưa vào trung bình năm nhánh.','metrics_comparison.json: bfree_branch_metrics, bfree_partial7024')

# 3: Robustness at the same fixed threshold, without mixing train domains.
cases=['native','all_jpeg96','all_jpeg90','all_jpeg80','blur_sigma1','resize_square560','resize_square256','resize256_jpeg96']
labels=['Gốc','JPEG96','JPEG90','JPEG80','Blur σ=1','Resize560','Resize256','Resize256\n+ JPEG96']
fig,axes=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
for ax,train in zip(axes,['Tiny','BFree']):
    for model,color in [('CAPD',BLUE),('GSD',ORANGE)]:
        vals=[100*d['tiny_case_metrics'][f'{model} train {train}'][c]['balanced_accuracy'] for c in cases]
        ax.plot(range(len(cases)),vals,'o-',color=color,label=model,lw=2)
        if model=='CAPD':
            for x,y in enumerate(vals):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_title('Train '+('Tiny' if train=='Tiny' else 'B-Free')+' → Test Tiny')
    ax.set_xticks(range(len(cases)),labels,rotation=35,ha='right');ax.set_ylim(65,103);ax.set_ylabel('Balanced Accuracy (%)');grid(ax);ax.legend(loc='lower left')
fig.suptitle('Độ bền trước các phép biến đổi ảnh')
fig.supxlabel('Ngưỡng cố định 0,5 · Cùng 7.000 ảnh Tiny ở mỗi điều kiện · Trục BA hiển thị từ 65%',fontsize=10)
save(fig,'03_robustness','So sánh độ bền của CAPD và GSD trên Tiny theo miền huấn luyện. Mỗi điều kiện sử dụng cùng 7.000 ảnh; ngưỡng được giữ ở 0,5. PNG roundtrip và chỉ fake JPEG96 được lược khỏi hình để tránh trùng lặp.','metrics_comparison.json: tiny_case_metrics')

# 4: Actual ROC curves from image-level scores, matched IDs and labels.
cap_bf='reports/capd_train_bfree_eval_analysis/source/capd_eval_tiny/b2_dinov3/seed42/tiny_eval/predictions_native.csv'
cap_tiny='reports/score_model_comparison/source/capd_train_tiny/predictions_native.csv'
gsd_bf='reports/gsd_train_bfree_test_tiny_analysis/source/gsd_bfree_samecat_origbg_best_tiny9_v1/predictions.csv'
gsd_tiny='reports/gsd_augmented_nine_cases/source/gsd_uploaded_best_variants_v1/predictions.csv'
frames={}
for train,paths in [('Tiny',[cap_tiny,gsd_tiny]),('BFree',[cap_bf,gsd_bf])]:
    for model,p in zip(['CAPD','GSD'],paths):
        f=csv(p,usecols=lambda c:c in ['sample_id','label','prob_fake','case'])
        if 'case' in f:f=f[f.case=='native']
        assert len(f)==7000 and not f.sample_id.duplicated().any()
        frames[(model,train)]=f.set_index('sample_id').sort_index()
    assert frames[('CAPD',train)].label.equals(frames[('GSD',train)].label)
fig,axes=plt.subplots(1,2,figsize=(10.8,4.9),layout='constrained')
for ax,train in zip(axes,['Tiny','BFree']):
    for model,color in [('CAPD',BLUE),('GSD',ORANGE)]:
        f=frames[(model,train)];x,y,_=roc_curve(f.label,f.prob_fake)
        auc=roc_auc_score(f.label,f.prob_fake)
        assert abs(auc-d['tiny_native'][f'{model} train {train}']['auc'])<1e-10
        ax.plot(x,y,label=f'{model}: AUC = {auc:.6f}',color=color,lw=2)
    ax.plot([0,1],[0,1],'--',color='#adb5bd',lw=1)
    ax.set(xlim=(-.015,1),ylim=(0,1.025),xlabel='False positive rate',ylabel='True positive rate',title='Train '+('Tiny' if train=='Tiny' else 'B-Free')+' → Test Tiny')
    grid(ax,'both');ax.legend(loc='lower right',fontsize=9)
fig.suptitle('Đường cong ROC trên ảnh gốc Tiny')
save(fig,'04_roc_tiny','Đường cong ROC được tính từ điểm dự đoán của từng ảnh trên cùng tập test Tiny. AUC được hiển thị sáu chữ số thập phân để phân biệt các kết quả gần 1; hai đường ở thiết lập train Tiny gần như chồng nhau.','Four native prediction CSVs; AUC checked against metrics_comparison.json')

# 5: Difficult branch illustrated through distributions, not invented image examples.
bf=csv('reports/capd_train_bfree_eval_analysis/source/capd_eval_bfree/b2_dinov3/seed42/bfree_eval/predictions_bfree.csv',usecols=['branch','label','prob_fake'])
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
real=bf.loc[bf.label==0,'prob_fake']
for ax,branch in zip(axes.flat,keys+['SD2.1_inpainted_diffcat_origBG']):
    fake=bf.loc[(bf.branch==branch)&(bf.label==1),'prob_fake']; assert len(fake)>0
    bins=np.linspace(0,1,41)
    weights=lambda s:np.ones(len(s))/len(s)*100
    ax.hist(real,bins=bins,weights=weights(real),color=BLUE,alpha=.6,label='Real')
    ax.hist(fake,bins=bins,weights=weights(fake),color=ORANGE,alpha=.65,label='Fake')
    ax.axvline(.5,color='#334155',ls='--',lw=1.2,label='Ngưỡng 0,5')
    ax.set_title(branch.replace('SD2.1_',''),fontsize=10)
    ax.set(xlim=(0,1),yscale='log',ylim=(.005,150),xlabel='Điểm dự đoán fake',ylabel='Tỷ lệ trong lớp (%)')
    grid(ax);ax.text(.04,.90,f'Fake median = {fake.median():.3f}',transform=ax.transAxes,fontsize=9)
axes.flat[0].legend(fontsize=8,loc='center left')
fig.suptitle('CAPD train B-Free: phân bố điểm dự đoán theo nhánh')
fig.supxlabel('Từng lớp được chuẩn hóa riêng · Trục tung log · Toàn bộ dự đoán CAPD đã lưu, không phải cohort ghép CAPD–GSD',fontsize=9)
save(fig,'05_score_distributions','Phân bố điểm dự đoán của CAPD huấn luyện trên B-Free. Các histogram được chuẩn hóa riêng theo lớp, dùng trục tung log; đường đứt biểu diễn ngưỡng 0,5. Nhánh selfconditioned_origBG có nhiều ảnh giả nhận điểm dưới ngưỡng. Hình dùng toàn bộ dự đoán CAPD, khác cohort ghép dùng trong hình so sánh.','CAPD TN2 predictions_bfree.csv, full pooling')

# 6: Train/validation history and checkpoint selection from saved artifacts.
histories=[('Tiny','reports/capd_tiny_checkpoint_analysis/source/capd_tn1/r2_dinov3/seed42/history.json','reports/capd_tiny_checkpoint_analysis/source/capd_tn1/r2_dinov3/seed42/selected_checkpoint.json'),('B-Free','reports/capd_train_bfree_eval_analysis/source/capd_eval_bfree/b2_dinov3/seed42/history.json','reports/capd_train_bfree_eval_analysis/source/capd_eval_bfree/b2_dinov3/seed42/selected_checkpoint.json')]
fig,axes=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
for col,(name,p,selp) in enumerate(histories):
    hist=js(p);sel=js(selp);x=[h.get('epoch',h.get('virtual_epoch')) for h in hist];best=sel.get('best_epoch',sel.get('best_ve'))
    ax=axes[0,col]
    ax.plot(x,[h['train_loss'] for h in hist],'o-',color=BLUE,label='Train loss')
    ax.plot(x,[h['val_loss'] for h in hist],'s-',color=ORANGE,label='Validation loss')
    ax.axvline(best,color='#687787',ls='--',label=f'Checkpoint chọn: {best}')
    ax.set_title(f'CAPD train {name}');ax.set_ylabel('Loss');ax.legend(fontsize=8);grid(ax)
    ax=axes[1,col]
    for key,label,color in [('val_native','Validation native',BLUE),('val_processed','Validation processed',ORANGE)]:
        if all(key in h and 'balanced_accuracy' in h[key] for h in hist):ax.plot(x,[100*h[key]['balanced_accuracy'] for h in hist],'o-',label=label,color=color)
    ax.axvline(best,color='#687787',ls='--');ax.set_ylabel('Validation BA (%)');ax.legend(fontsize=8);grid(ax)
    for ax in axes[:,col]:ax.set_xticks(x);ax.set_xlabel('Epoch' if name=='Tiny' else 'Virtual epoch')
fig.suptitle('Lịch sử huấn luyện và checkpoint được lựa chọn')
fig.supxlabel('Checkpoint chọn theo validation loss đã lưu · Train loss gồm các thành phần phụ nên khác định nghĩa validation loss',fontsize=9)
save(fig,'06_training_history','Diễn biến train loss, validation loss và validation BA của CAPD. Checkpoint train Tiny được chọn tại epoch 3; checkpoint train B-Free tại virtual epoch 2, theo tiêu chí validation loss trong artifact. Train loss và validation loss có thành phần khác nhau nên không dùng trực tiếp khoảng cách giữa chúng để kết luận overfitting.','CAPD R2/B2 history.json and selected_checkpoint.json')

# 7: Concrete counts for the strongest transfer comparison.
fig,axes=plt.subplots(1,2,figsize=(9.3,4.8),layout='constrained')
for ax,model in zip(axes,['CAPD','GSD']):
    r=d['tiny_native'][f'{model} train BFree'];a=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
    ax.imshow(a,cmap='Blues',vmin=0,vmax=3500)
    for (i,j),n in np.ndenumerate(a):ax.text(j,i,f'{n:,}\n({100*n/3500:.2f}%)',ha='center',va='center',color='white' if n>1750 else '#263445',fontsize=12)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Nhãn dự đoán',ylabel='Nhãn thực',title=model)
fig.suptitle('Train B-Free → Test Tiny: ma trận nhầm lẫn')
fig.supxlabel('Ngưỡng 0,5 · 3.500 ảnh/lớp · Phần trăm chuẩn hóa theo hàng',fontsize=10)
save(fig,'07_transfer_confusion','Ma trận nhầm lẫn của CAPD và GSD khi huấn luyện trên B-Free và đánh giá trên Tiny. CAPD bỏ sót 662 ảnh giả, so với 1.273 ảnh ở GSD; số ảnh thật bị nhận nhầm cũng giảm từ 39 xuống 16.','metrics_comparison.json: tiny_native counts')

# 8: Baseline comparison on the same native Tiny cohort.
models=['CAPD train Tiny','GSD train Tiny','NPR-ResNet18','AIDE forensic-only ResNet50','AIDE original full','CLIP ViT-B/32 probe']
short=['CAPD','GSD','NPR-ResNet18','AIDE forensic-only','AIDE full','CLIP ViT-B/32 probe']
fig,ax=plt.subplots(figsize=(9.2,4.8),layout='constrained')
vals=[100*d['tiny_native'][m]['balanced_accuracy'] for m in models]
bars=ax.barh(short,vals,color=[BLUE,ORANGE]+['#94a7b9']*4)
ax.bar_label(bars,fmt='%.2f%%',padding=5);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Balanced Accuracy (%)');grid(ax,'x')
ax.set_title('Train Tiny → Test Tiny: so sánh với các mô hình đối chứng')
fig.supxlabel('Ảnh gốc · Cùng 7.000 ảnh test · Ngưỡng 0,5',fontsize=10)
save(fig,'08_tiny_baselines','So sánh BA trên cùng 7.000 ảnh gốc Tiny của các mô hình huấn luyện trên Tiny. CAPD và GSD đạt kết quả gần tương đương; chênh lệch BA khoảng 0,04 điểm phần trăm.','metrics_comparison.json: tiny_native')

manifest=[]
for p in sorted(inputs):
    manifest.append({'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

intro='''# Bộ hình kết quả thực nghiệm CAPD và GSD

Mỗi hình có PNG 300 dpi để chèn Word, PDF/SVG dạng vector để dàn trang. Số liệu lấy từ bản artifact Kaggle đã tải trong repository; không chạy lại mô hình và không tạo ảnh minh họa giả. Nhãn Tiny giữ đúng tên dataset Tiny-GenImage, không đại diện cho toàn bộ GenImage.

Ưu tiên hình 01, 02, 03, 06, 07 trong báo cáo chính; hình 04, 05, 08 dùng khi cần phân tích thêm. Hình 01/02 dùng ngưỡng 0,5 và CAPD full pooling. B-Free macro năm nhánh bao gồm nhánh cùng miền train, không được mô tả là trung bình thuần held-out. Nhánh thứ sáu trong so sánh là partial; không gộp vào trung bình năm nhánh. Không suy luận ý nghĩa thống kê từ một lần chạy.

Nguồn và phiên bản được lưu trong [bản đồ artifact](../EXPERIMENT_DATA_URLS.md), cùng provenance trong các thư mục source. File sources.json ghi SHA-256 của các file dùng trực tiếp để tạo hình. Các hình được vẽ lại từ số liệu, không phải hình gốc xuất từ notebook.

Các link nguồn chính:
- [CAPD train Tiny, v1](https://www.kaggle.com/datasets/hadinho0710/checkpoin-capd-train-tiny)
- [CAPD test Tiny/B-Free, v1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree)
- [CAPD train B-Free, nguồn checkpoint được evaluator ghi](https://www.kaggle.com/datasets/hadinho0710/check-point-train-capd-bfree)
- [CAPD train B-Free test Tiny/B-Free, v1](https://www.kaggle.com/datasets/hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree)
- GSD và baseline: xem bản đồ artifact và bảng metrics_comparison.json gốc để đối chiếu từng run.

'''
for stem,caption,note in records:
    intro+=f'## {stem}\n\n![{stem}]({stem}.png)\n\n**Chú thích đề xuất:** {caption}\n\nNguồn trực tiếp: {note}.\n\n'
(OUT/'README.md').write_text(intro,encoding='utf-8')

# Compact contact sheet for human visual QA and quick selection.
canvas=Image.new('RGB',(1600,4*530),'#e9edf2')
draw=ImageDraw.Draw(canvas)
for i,(stem,_,_) in enumerate(records):
    im=Image.open(OUT/f'{stem}.png').convert('RGB')
    im=ImageOps.contain(im,(780,485))
    x=(i%2)*800+(800-im.width)//2;y=(i//2)*530+25
    canvas.paste(im,(x,y));draw.text(((i%2)*800+15,(i//2)*530+510),stem,fill='#263445')
canvas.save(OUT/'preview.jpg',quality=93)
with zipfile.ZipFile(OUT/'report_figures.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.png','.pdf','.svg','.md','.json','.py','.jpg']:z.write(p,p.name)
print(f'Created {len(records)} figures in PNG/PDF/SVG, captions, sources and ZIP at {OUT}')

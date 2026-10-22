"""Scientific figure; system matplotlib is used because bundled runtime lacks it."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE/'diagnostics.json').read_text(encoding='utf-8'))
h = d['history']
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
ax = axes[0,0]
ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h], 'o-',label='Train (augmented)',markersize=3)
ax.plot([r['epoch'] for r in h],[r['val_loss'] for r in h], 'o-',label='SD15 validation',markersize=3)
ax.axvline(13,linestyle='--',color='#555555',label='Selected: epoch 13')
ax.set(xlabel='Epoch', ylabel='Cross-entropy (log scale)',yscale='log',title='Training converges; validation worsens after epoch 13')
ax.legend(fontsize=8)
gens = ['sd15','wukong','midjourney','vqdm','adm','glide','biggan']
cases=['native','jpeg96','size','size_jpeg96']
rows={(r['case'],r['generator']):r for r in d['diagnostics']}
ax=axes[0,1]
values=np.array([[rows[c,g]['balanced_accuracy']*100 for c in cases] for g in gens])
im=ax.imshow(values,vmin=45,vmax=100,cmap='YlGnBu',aspect='auto')
for i in range(len(gens)):
    for j in range(len(cases)):
        ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>85 else '#111111',fontsize=9)
ax.set_xticks(range(4),['Native','JPEG96','Resize256','Resize + JPEG96'],rotation=12)
ax.set_yticks(range(len(gens)),['SD15 (seen)','Wukong','Midjourney','VQDM','ADM','GLIDE','BigGAN'])
ax.set_title('Balanced accuracy at threshold 0.5 (%)')
fig.colorbar(im,ax=ax,shrink=.8)
ax=axes[1,0]
x=np.arange(len(gens)); width=.35
ax.bar(x-width/2,[rows['native',g]['fake_recall']*100 for g in gens],width,label='Fake recall',color='#2563eb')
ax.bar(x+width/2,[rows['native',g]['real_recall']*100 for g in gens],width,label='Real recall',color='#ea9800')
ax.set_xticks(x,gens,rotation=18)
ax.set(ylabel='Recall (%)',ylim=(0,108),title='Native: most failures are fake images classified as real')
ax.legend(fontsize=8)
ax=axes[1,1]
unseen=[g for g in gens if g!='sd15']
for g in unseen:
    y=[rows[c,g]['auc']*100 for c in cases]
    ax.plot(range(4),y,'o-',label=g,markersize=4)
ax.set_xticks(range(4),['Native','JPEG96','Resize256','Resize + JPEG96'],rotation=12)
ax.set(ylabel='ROC-AUC (%)',ylim=(48,102),title='Processing effects differ substantially by generator')
ax.legend(ncol=3,fontsize=8)
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('CLIP + GSD | SD15 source | seed 42 | 7,000 test images',fontsize=15)
fig.savefig(HERE/'training_and_test.png',dpi=170)
fig.savefig(HERE/'training_and_test.svg')
print(HERE/'training_and_test.png')
